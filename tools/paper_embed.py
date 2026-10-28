r"""Dense retrieval over the paper corpus, measured on the SAME lexicon-citation benchmark as BM25 (`paper_text bench`):
embed every page window once with a sentence-embedding model (GPU), keep the vectors per paper, then score dense,
hybrid (BM25 + dense, reciprocal-rank fusion) and cross-encoder-reranked rankings, each with a paired bootstrap
interval against BM25 on the same queries.

    python tools/paper_embed.py models                                  # the registry: ids, prompts, sizes
    python tools/paper_embed.py embed --model qwen3-0.6b --bwm          # embed what is new (resumable; GPU)
    python tools/paper_embed.py bench --model qwen3-0.6b --bwm [--rerank bge-v2-m3] [--terms N]
    python tools/paper_embed.py search "sonar odometry without a DVL" --bwm [--model granite-r2] [--mode dense]
    python tools/paper_embed.py compare --runs a.json b.json [--ranker hybrid-term]   # model A vs model B, paired

NEEDS torch + sentence-transformers (a CUDA build for the GPU) -- deliberately NOT in requirements.txt: run it with an
interpreter that has them (the laptop: the py3.12 venv named in report/boat_world_model/README.md). The pure parts
(windows, fusion, scoring, bootstrap) import neither and are tested in the repo's suite.

WHY: LIT_PAPER_MINING.md sec. 2.1 set the bar -- BM25 on the masked definition put 84% of cited papers in the top 50
(R@10 0.642); an embedding model earns its install only by beating BM25 on these queries, by more than the bootstrap
interval. CACHE (outside git): `<pdfs>_embed/<model>/<sha[:2]>/<sha>.npy` (float16, one row per window) beside the
papers folder; windows are recomputed from the text cache (deterministic), so only vectors are stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_text as PT  # noqa: E402
if str(Path(__file__).resolve().parents[1]) not in sys.path:   # run by path: the repo root, for tools/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.atomic_replace import replace_retry  # noqa: E402

_Q_TERM = ("Instruct: Given a technical term from robotics, sonar, underwater vehicles or machine learning, retrieve "
           "passages of scientific papers that define or use it\nQuery:")
_Q_DEF = ("Instruct: Given a definition of a technical concept whose name is hidden, retrieve passages of scientific "
          "papers about that concept\nQuery:")
#: name -> how to load and prompt it. q / d: the prefix a query / passage needs (the model card's own; q_term / q_def
#: per query form); "auto": the model's own configured prompts (encode_query / encode_document); max_len: tokens.
#: The shortlist is the 2026-10-14 research's (LIT_PAPER_MINING.md): on the public SCIENTIFIC-passage tasks (QASPER,
#: PeerQA) Nemotron-3-Embed-1B and pplx-embed-v1 lead the sub-1.2B models and Qwen3-Embedding-0.6B trails.
MODELS: Dict[str, dict] = {
    "minilm": {"id": "sentence-transformers/all-MiniLM-L6-v2", "q": "", "d": "", "max_len": 256,
               "note": "22M, Apache-2.0; the classic small baseline (a floor and a bug detector)"},
    "granite-r2": {"id": "ibm-granite/granite-embedding-english-r2", "q": "", "d": "", "max_len": 512,
                   "note": "149M ModernBERT, Apache-2.0"},
    "qwen3-0.6b": {"id": "Qwen/Qwen3-Embedding-0.6B", "q": _Q_TERM, "q_term": _Q_TERM, "q_def": _Q_DEF, "d": "",
                   "max_len": 512, "note": "596M, Apache-2.0; MTEB eng-v2 retrieval 61.8"},
    "pplx-0.6b": {"id": "perplexity-ai/pplx-embed-v1-0.6B", "auto": True, "q": "", "d": "", "max_len": 512,
                  "trust": True, "dtype": "bfloat16",            # fp16 OVERFLOWS: every vector NaN (2026-10-14)
                  "note": "0.6B, MIT; QASPER 53.4 / PeerQA 32.7"},
    "nemotron-1b": {"id": "nvidia/Nemotron-3-Embed-1B-BF16", "auto": True, "q": "", "d": "", "max_len": 512,
                    "dtype": "bfloat16", "note": "1.1B, OpenMDW-1.1; QASPER 56.0 / PeerQA 35.8 (the best < 8 GB)"},
    "bge-m3": {"id": "BAAI/bge-m3", "q": "", "d": "", "max_len": 512, "note": "568M, MIT; dense head only here"},
    "e5-large-v2": {"id": "intfloat/e5-large-v2", "q": "query: ", "d": "passage: ", "max_len": 512,
                    "note": "335M, MIT"},
    "gte-modernbert": {"id": "Alibaba-NLP/gte-modernbert-base", "q": "", "d": "", "max_len": 512,
                       "note": "149M, Apache-2.0"},
}
RERANKERS: Dict[str, dict] = {
    "minilm-ce": {"id": "cross-encoder/ms-marco-MiniLM-L6-v2", "max_len": 512, "note": "22M, Apache-2.0"},
    "bge-v2-m3": {"id": "BAAI/bge-reranker-v2-m3", "max_len": 512, "note": "568M, Apache-2.0"},
    "qwen3-rr-0.6b": {"id": "Qwen/Qwen3-Reranker-0.6B", "max_len": 512, "note": "596M, Apache-2.0"},
}
WIN, OVERLAP = 1200, 200                              # characters per window, and shared with the next window


def default_cache(pdfs: Path) -> Path:
    p = Path(pdfs)
    return p.parent / (p.name + "_embed")


# ----------------------------------------------------------------------------------------- windows (pure)
def windows(page: str, size: int = WIN, overlap: int = OVERLAP) -> List[Tuple[int, int]]:
    """A page's text -> [(start, end)] character windows of ~size, overlapping by ~overlap, cut at whitespace so no
    word is split. Deterministic: the same text always gives the same windows (vectors are stored, text is not)."""
    n = len(page)
    if n <= size:
        return [(0, n)] if page.strip() else []
    out, a = [], 0
    while a < n:
        b = min(n, a + size)
        if b < n:
            cut = page.rfind(" ", a + size // 2, b)
            b = cut if cut > a else b
        out.append((a, b))
        if b >= n:
            break
        nxt = b - overlap
        sp = page.find(" ", nxt, b)
        a = sp + 1 if sp != -1 else nxt
    return out


def paper_windows(text: str) -> List[Tuple[int, str]]:
    """Cached paper text (pages split by form feeds) -> [(page number, window text)], whitespace collapsed."""
    out = []
    for i, pg in enumerate(text.split(PT.FF), 1):
        flat = PT._flat_ws(pg)
        out += [(i, flat[a:b]) for a, b in windows(flat)]
    return out


# ---------------------------------------------------------------------------------------- ranking (pure)
def papers_from_chunks(scores: Sequence[float], owner: Sequence[str], k: int = 2000) -> List[str]:
    """Window scores -> paper keys ranked by their BEST window (the same 'best page' rule BM25 uses)."""
    best: Dict[str, float] = {}
    for s, o in zip(scores, owner):
        if s > best.get(o, -1e9):
            best[o] = s
    return [o for o, _ in sorted(best.items(), key=lambda x: (-x[1], x[0]))][:k]


def rrf(*rankings: Sequence[str], k: int = 60, depth: int = 1000) -> List[str]:
    """Reciprocal-rank fusion: score(p) = sum over rankings of 1 / (k + rank). Ties broken by key."""
    sc: Dict[str, float] = {}
    for r in rankings:
        for i, p in enumerate(r[:depth]):
            sc[p] = sc.get(p, 0.0) + 1.0 / (k + i + 1)
    return [p for p, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 13) -> Tuple[float, float, float]:
    """Mean of (a - b) over queries and its 95% percentile interval, resampling QUERIES (paired)."""
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return 0.0, 0.0, 0.0
    rnd = random.Random(seed)
    means = sorted(sum(d[rnd.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def term_text(names: Sequence[str]) -> str:
    """The dense TERM query: the term, with its aliases in parentheses."""
    return names[0] + (f" ({', '.join(names[1:])})" if len(names) > 1 else "")


# ------------------------------------------------------------------------------------------- GPU parts
def _st():
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer, CrossEncoder
    except ImportError as e:                          # pragma: no cover - depends on the interpreter
        raise SystemExit("paper_embed needs torch + sentence-transformers: run it with the GPU venv "
                         f"(report/boat_world_model/README.md) -- {e}")
    return SentenceTransformer, CrossEncoder


def gpu_setup(fraction: float = 0.85) -> None:
    """Two Windows fixes for the GPU tools (measured 2026-10-14, RTX 4070 Laptop 8 GB): (1) transformers' SDPA path
    for grouped-query attention (Qwen3 / Qwen3.5 / Nemotron) is slow on native Windows -- repeating K/V instead ran
    Qwen3-Embedding-0.6B at 14.6 vs 4.3 windows/s, same maths; (2) Windows spills an over-sized batch into shared
    memory and runs ~4x slower instead of failing -- cap this process at `fraction` of VRAM so it fails loudly."""
    try:
        import torch
        import transformers.integrations.sdpa_attention as SA
        SA.use_gqa_in_sdpa = lambda *a, **k: False
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
    except (ImportError, AttributeError):
        pass


def load_model(name: str):
    SentenceTransformer, _ = _st()
    gpu_setup()
    import torch
    spec = MODELS[name]
    kw = {"dtype": getattr(torch, spec.get("dtype", "float16")), "attn_implementation": "sdpa"}
    m = SentenceTransformer(spec["id"], device="cuda" if torch.cuda.is_available() else "cpu",
                            trust_remote_code=bool(spec.get("trust")), model_kwargs=kw)
    m.max_seq_length = spec["max_len"]
    return m


def _unit(v):
    """float32 rows of length 1 (pplx returns int8-quantised vectors by default; every model is normalised here)."""
    import numpy as np
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def encode_docs(m, spec: dict, texts: List[str], batch: int = 32):
    if spec.get("auto"):
        return _unit(m.encode_document(texts, batch_size=batch, convert_to_numpy=True, show_progress_bar=False))
    return _unit(m.encode([spec["d"] + t for t in texts], batch_size=batch, convert_to_numpy=True,
                          show_progress_bar=False))


def encode_queries(m, spec: dict, texts: List[str], form: str = "term", batch: int = 32):
    if spec.get("auto"):
        return _unit(m.encode_query(texts, batch_size=batch, convert_to_numpy=True, show_progress_bar=False))
    q = spec.get("q_" + ("def" if form == "definition" else "term"), spec["q"])
    return _unit(m.encode([q + t for t in texts], batch_size=batch, convert_to_numpy=True, show_progress_bar=False))


def text_fp(text: str) -> str:
    """The fingerprint of the cached text a paper's vectors were computed from: its sha256 and its window count. The
    rows of a .npy map to paper_windows(<that text>), so a rewritten text (an OCR merge, a new extractor VERSION, a
    re-extraction on another machine) makes them STALE (review 2026-10-14: Whitall's vector was its title alone, from
    before its OCR)."""
    return f"{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}:{len(paper_windows(text))}"


def vec_paths(store: Path, name: str, sha: str) -> Tuple[Path, Path]:
    base = store / name / sha[:2]
    return base / f"{sha}.npy", base / f"{sha}.fp"


def is_current(store: Path, name: str, sha: str, text: str) -> bool:
    """Stored vectors that match this text: their fingerprint (`<sha>.fp`) equals this text's. A vector file WITHOUT a
    fingerprint is stale -- it cannot show which text it was made from (fix check 2026-10-14: the old rule, 'rows ==
    windows', blessed two minilm papers whose OCR'd pages had changed but whose window count had not)."""
    npy, fp = vec_paths(store, name, sha)
    if not npy.is_file() or not fp.is_file():
        return False
    return fp.read_text(encoding="utf-8").strip() == text_fp(text)


def embed(name: str, cache: Path, store: Path, batch: int = 16, say=print, limit: int = 0) -> dict:
    """Embed every held paper whose vectors are missing or STALE (its text changed since). Papers are encoded 15 at a
    time and saved after each group, so a stop loses at most one group."""
    import numpy as np
    names = PT.bwm_names()
    todo = []
    for sha, n in sorted(names.items()):
        f = PT.cache_file(cache, sha)
        if f.is_file() and not is_current(store, name, sha, f.read_text(encoding="utf-8")):
            todo.append((sha, n))
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"model": name, "new_papers": 0}
    m = load_model(name)
    spec, t0, nw = MODELS[name], time.time(), 0
    group = 15                                         # windows of 15 papers per encode call: the model sorts them
    for g in range(0, len(todo), group):               # by length, so batches pad less than paper by paper
        part = todo[g:g + group]
        texts, cut, fps = [], [0], []
        for sha, n in part:
            text = PT.cache_file(cache, sha).read_text(encoding="utf-8")
            wins = paper_windows(text)
            texts += [w for _, w in wins] or [n.get("title") or ""]
            cut.append(len(texts))
            fps.append(text_fp(text))
        v = encode_docs(m, spec, texts, batch)
        if not np.isfinite(v).all():                   # a dtype the model overflows in writes NaN rows that rank
            raise SystemExit(f"embed {name}: non-finite vectors for {part[0][1].get('key')} .. -- nothing stored "
                             f"(pplx-0.6b gave all-NaN in float16, 2026-10-14; set its 'dtype')")   # as noise
        for (sha, _), a0, a1, fpv in zip(part, cut, cut[1:], fps):
            npy, fp = vec_paths(store, name, sha)
            npy.parent.mkdir(parents=True, exist_ok=True)
            tmp = npy.with_suffix(".tmp.npy")
            np.save(tmp, v[a0:a1].astype(np.float16))
            replace_retry(tmp, npy)
            fp.write_text(fpv, encoding="utf-8")
        nw += len(texts)
        say(f"  [{g + len(part)}/{len(todo)}] {nw} windows, {nw / max(1e-9, time.time() - t0):.0f} windows/s")
    return {"model": name, "new_papers": len(todo), "windows": nw, "seconds": round(time.time() - t0)}


class Index:
    """Every CURRENT stored window vector of one model on the GPU, with the paper key that owns each row and each
    paper's row span. Vectors whose text changed since they were made are left out and counted as stale."""

    def __init__(self, name: str, cache: Path, store: Path):
        import numpy as np
        import torch
        names = PT.bwm_names()
        mats, self.owner, self.rows, self.span = [], [], [], {}
        self.missing = self.stale = 0
        for sha, n in sorted(names.items()):
            npy, _ = vec_paths(store, name, sha)
            f = PT.cache_file(cache, sha)
            if not npy.is_file() or not f.is_file():
                self.missing += 1
                continue
            if not is_current(store, name, sha, f.read_text(encoding="utf-8")):
                self.stale += 1
                continue
            v = np.load(npy)
            self.span[n["key"]] = (len(self.owner), len(self.owner) + len(v))
            mats.append(v)
            self.owner += [n["key"]] * len(v)
            self.rows += [(sha, j) for j in range(len(v))]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if not mats:                                   # no current vector at all (every text changed since): an
            raise SystemExit(f"{name}: no current vectors in {store} ({self.missing} missing, {self.stale} stale) "
                             f"-- run `embed --model {name} --bwm` first")       # empty store, not a crash
        self.M = torch.from_numpy(np.concatenate(mats)).to(dev)
        self.dev = dev

    def scores(self, qv):
        import torch
        return (self.M @ torch.as_tensor(qv, dtype=self.M.dtype, device=self.dev)).float()

    def rank(self, qv, k_chunks: int = 4000):
        """-> (papers ranked by their best window, [(row, score)] of the top k_chunks windows, all scores)."""
        import torch
        s = self.scores(qv)
        top = torch.topk(s, min(k_chunks, s.shape[0]))
        idx, val = top.indices.tolist(), top.values.tolist()
        return papers_from_chunks(val, [self.owner[i] for i in idx]), list(zip(idx, val)), s

    def best_rows(self, s, key: str, k: int) -> List[int]:
        """A paper's k best windows by this query's scores (any paper, ranked or not by the dense top list)."""
        import torch
        a, b = self.span[key]
        top = torch.topk(s[a:b], min(k, b - a))
        return [a + i for i in top.indices.tolist()]


def window_text(cache: Path, sha: str, j: int, memo: Dict[str, list]) -> str:
    if sha not in memo:
        memo[sha] = paper_windows(PT.cache_file(cache, sha).read_text(encoding="utf-8"))
    w = memo[sha]
    return w[j][1] if j < len(w) else ""


def bm25_memo(cache: Path, store: Path, names: Dict[str, dict]):
    """BM25 rankings per FTS query, kept in <store>/bm25_rankings.json.gz and valid only for the same index AND the
    same paper keys (the key is a hash of the index's (sha, extractor) rows and of the sha -> key map): every model's
    bench needs the same ~1,850 BM25 queries (~22 min), so they are asked once."""
    import gzip
    import sqlite3
    con = sqlite3.connect(Path(cache) / PT.INDEX)
    rows = "\n".join(f"{a} {b}" for a, b in con.execute("select sha, extractor from indexed order by sha"))
    con.close()
    keys = "\n".join(f"{s} {names[s]['key']}" for s in sorted(names))
    sig = hashlib.sha256((rows + "\n#\n" + keys).encode()).hexdigest()[:16]
    f = Path(store) / "bm25_rankings.json.gz"
    memo = {}
    if f.is_file():
        d = json.loads(gzip.decompress(f.read_bytes()))
        memo = d.get("rankings", {}) if d.get("corpus") == sig else {}

    def save():
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
        tmp.write_bytes(gzip.compress(json.dumps({"corpus": sig, "rankings": memo}).encode("utf-8")))
        replace_retry(tmp, f, may_drop=True)
    return memo, save


def bench(name: str, cache: Path, store: Path, n_terms: int = 0, rerank: str = "", depth: int = 30,
          per_paper: int = 2, say=print, allow_partial: bool = False) -> dict:
    """Dense, hybrid (RRF with BM25) and optionally reranked rankings on the lexicon-citation queries, each against
    BM25 on the same query form, with paired-bootstrap 95% intervals on the R@10 / R@50 / MRR differences. Refuses a
    store that misses papers or holds stale vectors (dense could not return them, BM25 could) unless allow_partial."""
    PT.index(cache, say=say)
    names = PT.bwm_names()
    qs = PT.bench_queries(n_terms)
    memo, save_memo = bm25_memo(cache, store, names)
    ix = Index(name, cache, store)
    if (ix.missing or ix.stale) and not allow_partial:
        raise SystemExit(f"bench {name}: {ix.missing} papers not embedded and {ix.stale} stale -- run `embed` first "
                         f"(or --allow-partial, recorded in the result)")
    m = load_model(name)
    spec = MODELS[name]
    say(f"{name}: {ix.M.shape[0]} windows of {len(ix.span)} papers ({ix.missing} not embedded, {ix.stale} stale); "
        f"{len(qs)} queries")
    ce = None
    if rerank:
        _, CrossEncoder = _st()
        import torch
        ce = CrossEncoder(RERANKERS[rerank]["id"], device=ix.dev, max_length=RERANKERS[rerank]["max_len"],
                          model_kwargs={"dtype": torch.float16})
    res: Dict[str, list] = {}
    wmemo: Dict[str, list] = {}
    t0 = time.time()
    for form in ("term", "definition"):
        texts = [term_text(q["names"]) if form == "term" else q["masked"] for q in qs]
        qv = encode_queries(m, spec, texts, form)
        for q, t, v in zip(qs, texts, qv):
            fts = PT.bm25_term_query(q["names"]) if form == "term" else PT.fts_query(q["masked"])
            if fts not in memo:
                memo[fts] = [r["key"] for r in PT._rank_q(cache, fts, names)]
            bm = memo[fts]
            dense, chunks, s = ix.rank(v)
            hyb = rrf(bm, dense)
            for nm, ranked in ((f"bm25-{form}", bm), (f"dense-{form}", dense), (f"hybrid-{form}", hyb)):
                res.setdefault(nm, []).append(PT.bench_score(ranked, q["rel"]))
            if ce is not None:
                cand = [o for o in hyb[:depth] if o in ix.span]   # every hybrid candidate gets windows, even one
                pairs, own = [], []                               # only BM25 found (review 2026-10-14)
                for o in cand:
                    for i in ix.best_rows(s, o, per_paper):
                        sha, j = ix.rows[i]
                        pairs.append((t, window_text(cache, sha, j, wmemo)))
                        own.append(o)
                sc = ce.predict(pairs, batch_size=32, show_progress_bar=False) if pairs else []
                rr = papers_from_chunks(list(map(float, sc)), own)
                rr += [o for o in hyb[:depth] if o not in rr]
                res.setdefault(f"hybrid+{rerank}-{form}", []).append(PT.bench_score(rr + hyb[depth:], q["rel"]))
        save_memo()
        say(f"  {form}: done in {time.time() - t0:.0f} s")
    out = PT.bench_summary(res)
    out["queries"] = [q["term"] for q in qs]          # per-query scores in that order, so two MODELS can be compared
    out["per_query"] = {nm: [[round(x, 4) for x in r] for r in v] for nm, v in res.items()}    # paired (bench_pair)
    out.update(model=name, model_id=MODELS[name]["id"], rerank=rerank, seconds=round(time.time() - t0),
               papers_embedded=len(ix.span), papers_missing=ix.missing, papers_stale=ix.stale,
               windows=int(ix.M.shape[0]))
    out["vs_bm25"] = {}
    pos = {mt: PT.BENCH_METRICS.index(mt) for mt in ("R@10", "R@50", "MRR")}
    for nm, v in res.items():
        form = nm.rsplit("-", 1)[1]
        if nm.startswith("bm25"):
            continue
        base = res[f"bm25-{form}"]
        out["vs_bm25"][nm] = {mt: [round(x, 3) for x in paired_bootstrap([r[i] for r in v], [r[i] for r in base])]
                              for mt, i in pos.items()}
    return out


def search(query: str, name: str, cache: Path, store: Path, mode: str = "hybrid", limit: int = 10):
    """Papers for a query, the way the benchmark ranks them: a short query (<= 6 words) is a TERM (BM25 on it as a
    phrase + the model's term prompt), a longer one a DEFINITION (BM25 on its words + the definition prompt); 'hybrid'
    fuses BM25 and dense by reciprocal rank -- the measured winner (LIT_PAPER_MINING.md sec. 7.5) -- 'dense' is the
    model alone. -> [(rank, {key, title, window})], the window being the paper's best-matching passage."""
    form = "definition" if len(query.split()) > 6 else "term"
    ix = Index(name, cache, store)
    m = load_model(name)
    qv = encode_queries(m, MODELS[name], [query], form)[0]
    dense, _, s = ix.rank(qv)
    by_sha = PT.bwm_names()
    if mode == "hybrid":
        PT.index(cache, say=lambda *_: None)
        fts = PT.bm25_term_query([query]) if form == "term" else PT.fts_query(query)
        order = rrf([r["key"] for r in PT._rank_q(cache, fts, by_sha)], dense)
    else:
        order = dense
    names = {n["key"]: n for n in by_sha.values()}
    memo: Dict[str, list] = {}
    out = []
    for k, key in enumerate(order[:limit], 1):
        rows = ix.best_rows(s, key, 1) if key in ix.span else []
        win = window_text(cache, *ix.rows[rows[0]], memo) if rows else ""
        out.append((k, {"key": key, "title": (names.get(key) or {}).get("title", ""), "window": win}))
    return out


def compare_runs(a: dict, b: dict, ranker: str = "hybrid-term") -> dict:
    """Two `bench --out` results -> run A minus run B on the SAME queries, per metric, with a paired-bootstrap 95%
    interval (the vs-BM25 numbers cannot say whether one MODEL beats another). Pure."""
    if a.get("queries") != b.get("queries"):
        raise ValueError("the two runs scored different queries (re-run both benches with the same lexicon)")
    pa, pb = a["per_query"][ranker], b["per_query"][ranker]
    return {mt: [round(x, 3) for x in paired_bootstrap([r[i] for r in pa], [r[i] for r in pb])]
            for i, mt in enumerate(PT.BENCH_METRICS) if mt in ("R@10", "R@50", "MRR")}


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["models", "embed", "bench", "search", "compare"])
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--model", default="nemotron-1b", choices=sorted(MODELS),   # MEASURED best 2026-10-14 (sec. 7.5)
                    help="default nemotron-1b: the best dense model on the benchmark (granite-r2 ties qwen3-0.6b at a "
                         "quarter of the size)")
    ap.add_argument("--rerank", default="", choices=[""] + sorted(RERANKERS))
    ap.add_argument("--bwm", action="store_true")
    ap.add_argument("--pdfs", type=Path, default=PT.DEFAULT_PDFS)
    ap.add_argument("--cache", type=Path, default=None, help="text cache (default <pdfs>_text)")
    ap.add_argument("--store", type=Path, default=None, help="vector store (default <pdfs>_embed)")
    ap.add_argument("--terms", type=int, default=0, help="bench: a seeded sample of N terms (0 = all)")
    ap.add_argument("--batch", type=int, default=16)   # MEASURED 2026-10-14: 16 beat 32 (22.5 vs 16.4 windows/s)
    ap.add_argument("--limit", type=int, default=0, help="embed: at most N papers this run; search: papers shown (10)")
    ap.add_argument("--out", type=Path, default=None, help="bench: also write the result JSON here")
    ap.add_argument("--allow-partial", action="store_true", help="bench: run on a store missing papers (recorded)")
    ap.add_argument("--runs", nargs=2, type=Path, default=None, metavar=("A.json", "B.json"),
                    help="compare: two `bench --out` files; prints A - B per metric with a paired 95%% interval")
    ap.add_argument("--ranker", default="hybrid-term", help="compare: which ranker (hybrid-term, dense-definition ...)")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense"],
                    help="search: BM25 + dense fused by reciprocal rank (the measured winner) or the model alone")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    cache = a.cache or PT.default_cache(a.pdfs)
    store = a.store or default_cache(a.pdfs)
    if a.cmd == "compare":
        if not a.runs:
            ap.error("compare needs --runs A.json B.json")
        ra, rb = (json.loads(f.read_text(encoding="utf-8")) for f in a.runs)
        res = compare_runs(ra, rb, a.ranker)
        print(f"{ra['model']}{'+' + ra['rerank'] if ra.get('rerank') else ''} - {rb['model']}"
              f"{'+' + rb['rerank'] if rb.get('rerank') else ''} ({a.ranker}, {len(ra['queries'])} paired queries):")
        for mt, (d, lo, hi) in res.items():
            print(f"  {mt:5} {d:+.3f}  95% CI {lo:+.3f} .. {hi:+.3f}")
        return 0
    if a.cmd == "models":
        for k, v in MODELS.items():
            print(f"{k:16} {v['id']:45} {v['note']}  q={v['q'][:40]!r} d={v['d']!r}")
        for k, v in RERANKERS.items():
            print(f"{k:16} {v['id']:45} {v['note']} (reranker)")
        return 0
    if not a.bwm:
        ap.error("--bwm is required (the corpus is the BlueBoat index's held papers)")
    if a.cmd == "embed":
        print(json.dumps(embed(a.model, cache, store, batch=a.batch, limit=a.limit)))
        return 0
    if a.cmd == "bench":
        res = bench(a.model, cache, store, n_terms=a.terms, rerank=a.rerank, allow_partial=a.allow_partial)
        print(json.dumps(res, indent=1))
        if a.out:
            a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0
    for k, n in search(a.query, a.model, cache, store, mode=a.mode, limit=a.limit or 10):
        print(f"{k:2}. {n['key']} -- {n['title'][:90]}\n        {n['window'][:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
