r"""paper_find -- find the papers behind a TASK or a few TERMS in the BlueBoat world-model index: one ranked list fused
from the lexicon's verified citations, the measured-best full-text ranker (BM25 + dense, reciprocal-rank fusion), each
lexicon term's precomputed full-text ranking and the index's own words -- and, for every paper, WHY it is there.

    python tools/paper_find.py "train a world model of a USV from logged data"      # task -> papers
    python tools/paper_find.py --term "TD-MPC2"                                     # term -> definition + papers
    python tools/paper_find.py "sonar odometry" "IMU" --all                         # papers that match EVERY term
    python tools/paper_find.py terms "side-scan sonar odometry with an IMU"          # the lexicon terms a text names
    python tools/paper_find.py "..." --keys | --tsv | --json                         # for scripts and pipes
    python tools/paper_find.py "..." --dense                                         # + the dense model (GPU venv, slow start)
    python tools/paper_find.py precompute                                           # papers/_lookup.json (corpus + GPU venv)
    python tools/paper_find.py bench [--split dev|test] [--tune]                     # the task benchmark (the reviews)

    import sys; sys.path.insert(0, "tools"); from paper_find import find
    for hit in find("sonar odometry without a DVL", k=10):
        print(hit["key"], hit["title"], hit["why"])

WHY (the 2026-10-15 task, as the lab session that started this work worded it): "make a way to easily find (grep) a
paper from a task or from a few important terms: inside the atlas, and outside it as a tool future work can call, so
more of our work relies directly on research". The corpus, the lexicon and the rankers already existed
(tools/paper_text.py, tools/paper_embed.py, build_bwm.py); what was missing is ONE entry point that takes a task the
way a person writes it, maps it onto the program's vocabulary and says why each paper is on the list. The design
follows the measurements in report/boat_world_model/LIT_PAPER_LOOKUP.md: hybrid BM25 + dense was the best full-text
ranker here (LIT_PAPER_MINING.md sec. 7.5; rerankers hurt), and a controlled vocabulary -- our lexicon -- is the
classic way to map free text onto concepts (MeSH-style term mapping).

SIGNALS (each a ranked list of paper keys; fused by WEIGHTED reciprocal rank, k = 60):
  lexicon  the lexicon terms the query names (word-boundary, longest match first; acronyms in exact case) -> the papers
           their verified citations quote. The strongest evidence: a verbatim sentence ties the paper to the term.
  table    the same terms -> each term's precomputed hybrid ranking (papers/_lookup.json, made by `precompute` where
           the corpus and the GPU venv are), so the atlas and a machine without the corpus still rank by full text.
  text     BM25 over every page of every held paper (tools/paper_text.py's FTS5 index; needs the text cache), fused
           with the dense model's ranking when --dense (tools/paper_embed.py; needs the GPU venv).
  meta     BM25 over the index's own words for each paper (title twice, summary, why-for-us, tags, full-text facts,
           topics, venue) -- always available, and exactly what the atlas's Find tab computes in the browser.
  graph    optional: papers that cite, are cited by, or share references with the top results (OpenAlex refs).

HONEST SCOPE (do NOT overclaim): the benchmark labels are the reviews' own inline citations (1-8 papers per query), so
recall is a LOWER bound (many more papers are relevant than any paragraph cites) and comparisons between rankers are
the point. The lexicon and the reviews were written by the same research agents, so the `lexicon` signal is partly
correlated with the labels -- the bench reports every ranker with and without it. A paper is found only if the index
holds it (2,813 verified papers; full text for the held ones).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_text as PT  # noqa: E402
import atlas_project as AP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                         # run by path: the repo root, for tools/
    sys.path.insert(0, str(ROOT))
from tools.atomic_replace import replace_retry  # noqa: E402
BWM = AP.project_dir()                               # the atlas project (default: the BlueBoat one) -- atlas_project
LOOKUP_FILE = "_lookup.json"                          # papers/_lookup.json (committed; made by `precompute`)
RRF_K = 60
#: signal weights in the fusion, per query REGIME (`regime`): a TERM query (a lexicon term, more or less) or a TASK
#: (free text). Chosen on the DEV halves of the two benchmarks only (bench --tune, 2026-10-15); each TEST half was
#: scored with these weights only -- twice, before and after a term-matcher fix made from reading the code, weights
#: unchanged (report/boat_world_model/LIT_PAPER_LOOKUP.md sec. 3). TASK weights come from the grid run
#: with the OTHER-AUTHORS control in place of `meta` (the index's prose minus what the review's own agent wrote): with
#: all prose, `meta` looked like the best signal (DEV R@10 0.793) only because the agent who wrote a review paragraph
#: also wrote its papers' summaries -- without that shared phrasing it is 0.541, and a person's own words share none.
#: The LEXICON weight of a TERM query cannot be measured (its citations ARE the term benchmark's labels): 1.0 is a
#: judgment -- a verbatim sentence ties those papers to the term. The citation graph is NOT fused: added at weight 1
#: it cut DEV task R@10 0.790 -> 0.563 and did not even raise R@50 (0.894 -> 0.872; papers/_lookup_bench.json); the
#: atlas offers 'related by citation' lists instead.
WEIGHTS = {"task": {"text": 1.0, "lexicon": 0.25, "table": 0.0, "meta": 0.25, "graph": 0.0},
           "term": {"text": 1.0, "lexicon": 1.0, "table": 0.5, "meta": 0.5, "graph": 0.0}}
#: the atlas (no full text in the browser): the same regimes, weights tuned with `text` absent (task: on the
#: other-authors control, as above)
ATLAS_WEIGHTS = {"task": {"lexicon": 0.5, "table": 0.25, "meta": 1.0},
                 "term": {"lexicon": 1.0, "table": 1.0, "meta": 1.0}}
#: a query whose content words are at least this share inside matched lexicon terms is a TERM query
TERM_COVERAGE = 0.6
TOP_PER_TERM = 30                                     # papers kept per term / task in papers/_lookup.json
ATLAS_TABLE_DEPTH = 20                                # ... of which the atlas page carries the first 20 (its size)
STOP = PT.STOP
RELEVANCE = {"core": 0, "high": 1, "medium": 2, "low": 3}

#: The program's recurring tasks: the atlas offers each as one click, ranked by the full-text hybrid at `precompute`
#: time. They are the project's atlas.toml [[tasks]] (the BlueBoat eighteen: report/boat_world_model/atlas.toml);
#: edit them there, then run `precompute` and `build_bwm.py build`.
TASKS: List[Tuple[str, str, str]] = AP.tasks(BWM)
#: the dense model a project's lookup uses: its atlas.toml [pipeline] embed_model when set, else Nemotron-3-Embed-1B
#: (the measured best, LIT_PAPER_MINING.md sec. 7.5)
DEFAULT_MODEL = AP.pipeline(BWM)["embed_model"] if AP.pipeline(BWM)["embed_model"] != "auto" else "nemotron-1b"


# ------------------------------------------------------------------------------------------------- the index
def _bwm():
    """build_bwm (a script folder, not a package): loaded once under a private name."""
    name = "_paper_find_build_bwm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, AP.ENGINE / "build_bwm.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _load(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def stem(w: str) -> str:
    """Harman's S-stemmer (1991): plural endings only -- 'models' -> 'model', 'policies' -> 'policy', 'analyses' ->
    'analyse'; 'focus' and 'glass' stay -- the SAME rule the atlas's Find tab runs in JavaScript (build_bwm.py
    BROWSER_TEMPLATE, `stemW`)."""
    if len(w) > 4 and w.endswith("ies") and not w.endswith(("eies", "aies")):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and not w.endswith(("aes", "ees", "oes")):
        return w[:-1]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("us", "ss")):
        return w[:-1]
    return w


def tokens(text: str) -> List[str]:
    """Lower-case ASCII letters / digits, stop words out, S-stemmed (the atlas's `toks` is the same function)."""
    return [stem(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOP]


def meta_text(row: dict) -> str:
    """A paper's own words in the index, as the atlas row carries them (build_bwm.render_browser): title twice, the
    summary, the first why-for-us, eight tags, the full-text fact chips, the topics and the venue."""
    return " ".join([row.get("t") or "", row.get("t") or "", row.get("sum") or "", row.get("why") or "",
                     " ".join(row.get("g") or []), " ".join(row.get("fx") or []),
                     " ".join(str(t).replace("_", " ") for t in row.get("top") or []), row.get("v") or ""])


class MetaBM25:
    """Okapi BM25 (k1 1.2, b 0.75) over short per-paper documents -- pure Python, and line for line the atlas's
    in-browser ranker (`MetaIndex` in the page): same tokens, same idf, ties broken by key."""

    def __init__(self, docs: Dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: Dict[str, Dict[str, int]] = {}
        self.dl: Dict[str, int] = {}
        df: Dict[str, int] = {}
        for k in sorted(docs):
            toks = tokens(docs[k])
            c: Dict[str, int] = {}
            for t in toks:
                c[t] = c.get(t, 0) + 1
            self.tf[k], self.dl[k] = c, len(toks)
            for t in c:
                df[t] = df.get(t, 0) + 1
        self.n = len(docs)
        self.avgdl = (sum(self.dl.values()) / self.n) if self.n else 1.0
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.post: Dict[str, List[str]] = {}
        for k in sorted(self.tf):
            for t in self.tf[k]:
                self.post.setdefault(t, []).append(k)

    def scores(self, query: str) -> Dict[str, float]:
        q = list(dict.fromkeys(tokens(query)))
        sc: Dict[str, float] = {}
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for k in self.post[t]:
                f = self.tf[k][t]
                sc[k] = sc.get(k, 0.0) + idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[k] / self.avgdl))
        return sc

    def rank(self, query: str, limit: int = 1000) -> List[str]:
        sc = self.scores(query)
        return [k for k, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]


def case_exact(name: str) -> bool:
    """An acronym-like name -- at most 6 letters / digits and either no lower-case letter ('USBL', 'TD-MPC2') or at
    least two capitals ('mAP', 'iSAM', 'ViT') -- matches only in its exact case: 'ate' (a verb) never finds ATE and
    'map' never finds mAP. The atlas's `caseExact` is the same rule."""
    alnum = "".join(re.findall(r"[A-Za-z0-9]+", str(name)))
    return len(alnum) <= 6 and (not re.search(r"[a-z]", alnum) or sum(c.isupper() for c in alnum) >= 2)


def name_forms(name: str) -> List[str]:
    """A term name as text may write it: the name itself, the name without a parenthetical ('Extended Kalman filter
    (EKF)' -> 'Extended Kalman filter'), and each ABBREVIATION a parenthetical holds ('EKF'; 'USV / ASV' -> both) -- a
    descriptive one ('World model (learned)') is never a name of its own. The atlas's `nameForms` is the same rule."""
    n = str(name or "").strip()
    out = [n]
    core = re.sub(r"\s+", " ", re.sub(r"\s*\([^()]*\)\s*", " ", n)).strip()
    if core and core != n:
        out.append(core)
    for par in re.findall(r"\(([^()]+)\)", n):
        for part in par.split("/"):
            part = part.strip()
            if part and case_exact(part):
                out.append(part)
    return list(dict.fromkeys(out))


class Terms:
    """Find the lexicon's terms in free text: every term name and alias in each of its name_forms, tokenised like the
    text ([A-Za-z0-9]+), longest match first, never overlapping. A case-exact name (case_exact: acronyms) must match in
    exact case; any other name matches its S-stemmed lower-case words ('world models' -> 'world model'). Names under 3
    letters / digits are never matched ('J', 'DT': too ambiguous in prose). The atlas's `TermFinder` is the same
    algorithm in JavaScript."""

    def __init__(self, lex: Sequence[dict]):
        self.by_id = {t["id"]: t for t in lex}
        self.exact: Dict[Tuple[str, ...], List[str]] = {}
        self.folded: Dict[Tuple[str, ...], List[str]] = {}
        self.maxlen = 1
        for t in sorted(lex, key=lambda t: t["id"]):
            for n in [f for x in [t["term"]] + list(t.get("aliases") or []) for f in name_forms(x)]:
                raw = re.findall(r"[A-Za-z0-9]+", str(n))
                if len("".join(raw)) < 3:
                    continue
                acro = case_exact(n)
                key = tuple(raw) if acro else tuple(stem(w.lower()) for w in raw)
                bucket = self.exact if acro else self.folded
                if t["id"] not in bucket.setdefault(key, []):
                    bucket[key].append(t["id"])
                self.maxlen = max(self.maxlen, len(key))

    def find(self, text: str) -> List[dict]:
        """[{id, term, matched, start}] in text order; a span matching several terms (a shared alias) lists each."""
        toks = [(m.group(0), m.start()) for m in re.finditer(r"[A-Za-z0-9]+", text or "")]
        out, i = [], 0
        while i < len(toks):
            for n in range(min(self.maxlen, len(toks) - i), 0, -1):
                raw = tuple(w for w, _ in toks[i:i + n])
                ids = self.exact.get(raw) or self.folded.get(tuple(stem(w.lower()) for w in raw))
                if ids:
                    for tid in ids:
                        out.append({"id": tid, "term": self.by_id[tid]["term"], "start": toks[i][1],
                                    "matched": text[toks[i][1]:toks[i + n - 1][1] + len(toks[i + n - 1][0])]})
                    i += n
                    break
            else:
                i += 1
        return out


def fuse_scores(rankings: Sequence[Tuple[float, Sequence[str]]], k: int = RRF_K,
                depth: int = 1000) -> List[Tuple[str, float]]:
    """WEIGHTED reciprocal-rank fusion: score(p) = sum_i w_i / (k + rank_i(p)), best first, ties broken by key. With
    all weights 1 this is paper_embed.rrf (Cormack et al. 2009)."""
    sc: Dict[str, float] = {}
    for w, r in rankings:
        if not w:
            continue
        for i, p in enumerate(list(r)[:depth]):
            sc[p] = sc.get(p, 0.0) + w / (k + i + 1)
    return sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))


def fuse(rankings: Sequence[Tuple[float, Sequence[str]]], k: int = RRF_K, depth: int = 1000) -> List[str]:
    """fuse_scores, keys only."""
    return [p for p, _ in fuse_scores(rankings, k, depth)]


class Index:
    """Everything the lookup reads from the repo (no corpus needed): the index's papers and aliases, the lexicon, the
    atlas-row words of every paper, papers/_lookup.json and the in-index citation graph."""

    def __init__(self, base: Path = BWM):
        self.base = Path(base)
        P = _load(self.base / "index" / "papers.json", {"papers": [], "aliases": {}})
        self.papers: List[dict] = P["papers"]
        self.by_key = {p["key"]: p for p in self.papers}
        self.alias: Dict[str, str] = P.get("aliases") or {}
        self.lex: List[dict] = _load(self.base / "index" / "lexicon.json", {"terms": []})["terms"]
        self.term_by_id = {t["id"]: t for t in self.lex}
        self.terms = Terms(self.lex)
        B = _bwm()
        facts = B.facts_rows(self.base, self.alias) if (self.base / "papers").is_dir() else {}
        self.rows = {p["key"]: atlas_row(p, facts.get(p["key"]) or {}) for p in self.papers}
        self.meta = MetaBM25({k: meta_text(r) for k, r in self.rows.items()})
        # the papers' OWN words only (title, full-text fact chips, venue): the bench's control for `meta`
        self.titles = MetaBM25({k: " ".join([r["t"], " ".join(r["fx"]), r["v"]]) for k, r in self.rows.items()})
        self.lookup = _load(self.base / "papers" / LOOKUP_FILE, {})
        held = _load(self.base / "papers" / "_held.json", {}).get("files") or []
        rej = B.rejected_ids(self.base, self.alias)[0] if (self.base / "papers").is_dir() else {}
        self.files = {self.alias.get(r["key"], r["key"]): r["file"] for r in held
                      if self.alias.get(r["key"], r["key"]) not in rej.get(str(r.get("sha256") or "").lower(), ())}
        self._graph = None

    # ---- signals that need no corpus
    def cited_by_terms(self, ids: Sequence[str]) -> Dict[str, List[str]]:
        """paper key -> the matched term ids whose verified citations name it."""
        out: Dict[str, List[str]] = {}
        for tid in dict.fromkeys(ids):
            for c in (self.term_by_id.get(tid) or {}).get("citations") or []:
                k = self.alias.get(c["key"], c["key"])
                if k in self.by_key and tid not in out.setdefault(k, []):
                    out[k].append(tid)
        return out

    def lexicon_rank(self, ids: Sequence[str]) -> List[str]:
        """Papers cited for the matched terms: most matched terms first, then relevance, newest, key."""
        cb = self.cited_by_terms(ids)
        rel = lambda k: RELEVANCE.get(self.by_key[k].get("relevance"), 4)   # noqa: E731
        return sorted(cb, key=lambda k: (-len(cb[k]), rel(k), -(self.by_key[k].get("year") or 0), k))

    def table_rank(self, ids: Sequence[str], depth: int = 0) -> List[str]:
        """The matched terms' precomputed full-text rankings (papers/_lookup.json), fused with equal weights; `depth`
        keeps only each list's first papers (the atlas carries ATLAS_TABLE_DEPTH)."""
        lists = [((self.lookup.get("terms") or {}).get(t) or []) for t in dict.fromkeys(ids)]
        lists = [[self.alias.get(k, k) for k in r if self.alias.get(k, k) in self.by_key][:depth or None] for r in lists]
        return fuse([(1.0, r) for r in lists if r])

    def graph(self) -> Dict[str, Dict[str, set]]:
        """{'out': cites, 'in': cited by, 'couple': shares >= 3 references} per key, from papers/_meta.json."""
        if self._graph is None:
            self._graph = citation_graph(self.base, self.alias)
        return self._graph

    def graph_rank(self, seeds: Sequence[str]) -> List[str]:
        """Papers linked to the seeds (cites / cited by / >= 3 shared references), most linked seeds first."""
        G, sc = self.graph(), {}
        for s in seeds:
            for kind in ("out", "in", "couple"):
                for c in G.get(kind, {}).get(s, ()):
                    sc[c] = sc.get(c, 0) + 1
        return [k for k, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]


def atlas_row(p: dict, fx: dict) -> dict:
    """The fields of the atlas's paper row that the Find tab searches (build_bwm.render_browser builds the same)."""
    return {"t": p.get("title") or "", "sum": p.get("summary") or "",
            "why": next(iter((p.get("why_for_us") or {}).values()), ""), "g": (p.get("tags") or [])[:8],
            "fx": fact_chips(fx), "top": p.get("topics") or [], "v": p.get("venue") or ""}


def fact_chips(fx: dict) -> List[str]:
    """The full-text facts the atlas shows as chips: simulators, world models, planners, RL, platform (<= 10)."""
    chips: List[str] = []
    for k in ("simulators", "world_models", "planners", "rl", "platform"):
        for v in (fx or {}).get(k) or []:
            if v.lower() not in ("world model", "world models") and v not in chips:
                chips.append(v)
    return chips[:10]


def citation_graph(base: Path, alias: Dict[str, str], min_shared: int = 3) -> Dict[str, Dict[str, set]]:
    """In-index citations (OpenAlex, papers/_meta.json) as sets per key, plus bibliographic coupling: two index papers
    whose OpenAlex reference lists share >= min_shared works (a reference cited by > 200 index papers is too generic
    to count)."""
    B = _bwm()
    meta = B.meta_rows(Path(base), alias)
    out: Dict[str, set] = {}
    inn: Dict[str, set] = {}
    refs: Dict[str, set] = {}
    for k, r in meta.items():
        for x in r.get("refs_in_corpus") or []:
            if x != k:
                out.setdefault(k, set()).add(x)
                inn.setdefault(x, set()).add(k)
        rl = (r.get("openalex") or {}).get("refs") or []
        if rl:
            refs[k] = set(rl)
    inv: Dict[str, List[str]] = {}
    for k in sorted(refs):
        for w in refs[k]:
            inv.setdefault(w, []).append(k)
    shared: Dict[Tuple[str, str], int] = {}
    for w, ks in inv.items():
        if 2 <= len(ks) <= 200:
            for i, a in enumerate(ks):
                for b in ks[i + 1:]:
                    shared[(a, b)] = shared.get((a, b), 0) + 1
    couple: Dict[str, set] = {}
    for (a, b), n in shared.items():
        if n >= min_shared:
            couple.setdefault(a, set()).add(b)
            couple.setdefault(b, set()).add(a)
    return {"out": out, "in": inn, "couple": couple}


@lru_cache(maxsize=4)
def load_index(base: str = str(BWM)) -> Index:
    return Index(Path(base))


# ------------------------------------------------------------------------------------------ full text (optional)
def text_fts(query: str, phrases: Sequence[str] = ()) -> str:
    """The FTS5 query of the text signal: each given term as one exact phrase, then the free text's content words, all
    OR'ed (BM25 ranks pages with more and rarer of them first); every token quoted, so no operator can be injected."""
    parts = [PT.fts_query(p, phrase=True) for p in phrases if str(p).strip()]
    words = PT.fts_query(query)
    return " OR ".join(x for x in dict.fromkeys(parts + ([words] if words else [])) if x)


def text_door():
    """The lab's library door (tools/library/remote.py) when this machine has one set up, else None. It serves the lab's
    paper text caches (`rank` / `pages`, tools/remote_access/library_door.py)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from tools.library import remote
        return remote.default()
    except Exception:                                  # noqa: BLE001 -- no door: this machine's text only
        return None


def collection_of(base: Path, pdfs: Path) -> str:
    """The name the lab's library door knows a project's text cache by: its PDF folder's name (`<papers>/<name>`,
    cache `<name>_text` -- the BlueBoat atlas's `boat_wm`), or, for a project whose PDFs default to its own `pdfs/`,
    its atlas short_name (on the lab those live in `<papers>/<short_name>`: report/*/atlas.toml [paths])."""
    return Path(pdfs).name if Path(pdfs).name != "pdfs" else AP.config(Path(base))["atlas"]["short_name"]


class Corpus:
    """The held papers' full text (tools/paper_text.py) and, with dense=True, the dense model (tools/paper_embed.py).
    The papers are the `base` atlas project's (its papers/_held.json; default the BlueBoat one). `available` is False
    when no text can be reached: the lookup then runs without the text signal.

    WHERE THE TEXT COMES FROM (2026-10-19, `source`): this machine's cache (`<pdfs>_text`) when it holds a text for
    EVERY paper the project holds; else the lab's cache through its library door (tools/library/remote.py: `rank` and
    `pages`, one round trip each, ~1.2 s) when one is set up here -- `door=None` never asks it; else this machine's
    cache as far as it goes. A door that fails mid-run is dropped for the rest of the run and this machine's cache, if
    any, answers (`door_error` says why). The dense model never runs through the door."""

    def __init__(self, pdfs: Optional[Path] = None, dense: bool = False, model: str = DEFAULT_MODEL,
                 base: Optional[Path] = None, door="auto"):
        self.base = Path(base or BWM)
        self.pdfs = Path(pdfs or PT.DEFAULT_PDFS)
        self.cache = PT.default_cache(self.pdfs)
        held = (self.base / "papers" / "_held.json").is_file()
        self.names = PT.bwm_names(self.base) if held else {}
        self.by_key = {n["key"]: sha for sha, n in self.names.items()}
        self.local = held and (self.cache / PT.MANIFEST).is_file()
        have = set(PT.read_manifest(self.cache)) if self.local else set()
        self.missing = len(set(self.names) - have)
        self.door, self.door_error, self.collection = None, "", ""
        if held and self.missing and door is not None:
            self.door = text_door() if door == "auto" else door
            self.collection = collection_of(self.base, self.pdfs) if self.door is not None else ""
        self.dense_name, self._dense, self.want_dense = model, None, dense

    @property
    def available(self) -> bool:
        return self.local or self.door is not None

    @property
    def source(self) -> str:
        n = len(self.names)
        if self.door is not None:
            return f"the lab's text cache '{self.collection}' through {self.door.alias} ({self.missing} of {n} held " \
                   f"papers have no text on this machine)"
        if self.local:
            return f"this machine's text cache ({n - self.missing} of {n} held papers)" + \
                (f"; the lab's door failed: {self.door_error[:160]}" if self.door_error else "")
        return "no full text (no text cache here, no library door)"

    def _drop_door(self, e: Exception) -> None:
        self.door_error, self.door = str(e), None

    def bm25(self, query: str, phrases: Sequence[str] = (), limit: int = 1000) -> List[str]:
        """Pages ranked by BM25 for the query's content words OR'ed (the benchmark's 'definition' form, which a task
        is), plus each explicitly given term as an exact phrase."""
        if not self.available:
            return []
        fts = text_fts(query, phrases)
        if not fts:
            return []
        if self.door is not None:
            try:
                return [self.names[sha]["key"] for sha, _ in self.door.rank(self.collection, fts, limit)
                        if sha in self.names]
            except Exception as e:                     # noqa: BLE001 -- the door failed: this machine's text, if any
                self._drop_door(e)
                if not self.local:
                    return []
        PT.index(self.cache, say=lambda *_: None)
        return [r["key"] for r in PT._rank_q(self.cache, fts, self.names, limit)]

    def dense(self, query: str) -> List[str]:
        if not (self.local and self.want_dense):
            return []
        import paper_embed as PE
        if self._dense is None:
            store = PE.default_cache(self.pdfs)
            self._dense = (PE.Index(self.dense_name, self.cache, store), PE.load_model(self.dense_name))
        ix, m = self._dense
        form = "definition" if len(query.split()) > 6 else "term"
        qv = PE.encode_queries(m, PE.MODELS[self.dense_name], [query], form)[0]
        return ix.rank(qv)[0]

    def pages_many(self, keys: Sequence[str], query: str, n: int = 2) -> Dict[str, List[dict]]:
        """{key: its best pages} for several papers: through the door in ONE round trip, else `pages` each."""
        keys = [k for k in keys if self.by_key.get(k)]
        if self.door is not None and keys:
            try:
                got = self.door.pages(self.collection, PT.fts_query(query), [self.by_key[k] for k in keys], n)
                return {k: got.get(self.by_key[k], []) for k in keys}
            except Exception as e:                     # noqa: BLE001
                self._drop_door(e)
        return {k: self.pages(k, query, n) for k in keys}

    def pages(self, key: str, query: str, n: int = 2) -> List[dict]:
        """A paper's best pages for the query, with highlighted snippets (BM25 over its pages) -- this machine's cache;
        `pages_many` asks the door."""
        sha = self.by_key.get(key)
        if not (self.local and sha):
            return []
        import sqlite3
        q = PT.fts_query(query)
        if not q:
            return []
        con = sqlite3.connect(self.cache / PT.INDEX, timeout=120)
        try:
            rows = con.execute("with p as materialized (select page, bm25(pages) as s, snippet(pages, 2, '[[', ']]', "
                               "'...', 18) as snip from pages where pages match ? and sha = ?) select page, snip from p "
                               "order by s limit ?", (q, sha, n)).fetchall()
        finally:
            con.close()
        return [{"page": pg, "snippet": sn} for pg, sn in rows]


# ---------------------------------------------------------------------------------------------- the lookup
def signals(ix: Index, query: str, corpus: Optional[Corpus] = None, mode: str = "auto",
            term_ids: Optional[Sequence[str]] = None, phrases: Sequence[str] = ()) -> Dict[str, List[str]]:
    """Each available signal's ranking for the query. mode 'atlas' = what the atlas's Find tab can compute (no full
    text); 'auto' adds the full text when this machine has it. `phrases`: terms the caller named one by one."""
    ids = list(term_ids) if term_ids is not None else [m["id"] for m in ix.terms.find(query)]
    out = {"lexicon": ix.lexicon_rank(ids), "table": ix.table_rank(ids, ATLAS_TABLE_DEPTH if mode == "atlas" else 0),
           "meta": ix.meta.rank(query)}
    if mode != "atlas" and corpus is not None and corpus.available:
        bm = corpus.bm25(query, phrases)
        dn = corpus.dense(query) if corpus.want_dense else []
        out["text"] = fuse([(1.0, bm), (1.0, dn)]) if dn else bm
    return out


def regime(query: str, matches: Sequence[dict]) -> str:
    """'term' when the query is (mostly) lexicon terms -- at least TERM_COVERAGE of its content words lie inside
    matched term spans ('TD-MPC2', 'side-scan sonar odometry') -- else 'task' (free text). Each regime has its own
    measured weights: a term's precomputed full-text ranking is the best evidence for a term, not for a paragraph."""
    q = tokens(query)
    if not q or not matches:
        return "task"
    spans = {m["start"]: m["matched"] for m in matches}
    covered = len(tokens(" ".join(spans.values())))
    return "term" if covered / len(q) >= TERM_COVERAGE else "task"


def weights_for(reg: str, mode: str) -> Dict[str, float]:
    """The fusion weights of a regime in a mode ('atlas' = no full text)."""
    return dict((ATLAS_WEIGHTS if mode == "atlas" else WEIGHTS)[reg])


def rank_scored(sig: Dict[str, List[str]], weights: Dict[str, float], ix: Optional[Index] = None,
                graph_seeds: int = 10) -> List[Tuple[str, float]]:
    """Fuse the signals with `weights` (a signal absent from them weighs 0); with a graph weight, the papers linked
    to the fused top `graph_seeds` join as one more list. -> [(key, fused score)], best first."""
    lists = [(weights.get(n, 0.0), r) for n, r in sorted(sig.items())]
    fused = fuse_scores(lists)
    if weights.get("graph") and ix is not None and fused:
        fused = fuse_scores(lists + [(weights["graph"], ix.graph_rank([k for k, _ in fused[:graph_seeds]]))])
    return fused


def rank(sig: Dict[str, List[str]], weights: Dict[str, float], ix: Optional[Index] = None,
         graph_seeds: int = 10) -> List[str]:
    """rank_scored, keys only."""
    return [k for k, _ in rank_scored(sig, weights, ix, graph_seeds)]


def find(query: str = "", terms: Sequence[str] = (), k: int = 20, mode: str = "auto", dense: bool = False,
         require_all: bool = False, topic: str = "", relevance: str = "", held_only: bool = False,
         year_from: int = 0, base: Path = BWM, pdfs: Optional[Path] = None, snippets: bool = True,
         weights: Optional[Dict[str, float]] = None, corpus: Optional[Corpus] = None, door="auto") -> List[dict]:
    """The papers for a task (free text) and / or a few terms, best first: [{rank, key, title, year, venue, status,
    relevance, topics, held, pdf, url, score, why: [...], pages: [{page, snippet}]}]. `require_all` keeps only papers
    whose full text (or index words, without the corpus) contains EVERY term -- the grep-like AND. `corpus`: one made
    already (the CLI prints its `source`); else one for `base` (`door`: Corpus's -- None = this machine's text only)."""
    ix = load_index(str(base))
    terms = [t for t in terms if str(t).strip()]
    text = " ".join([query] + list(terms)).strip()
    if not text:
        return []
    if mode == "atlas":
        corpus = None
    elif corpus is None:
        corpus = Corpus(pdfs, dense=dense, base=base, door=door)
    matches = ix.terms.find(text)
    ids = list(dict.fromkeys(m["id"] for m in matches))
    for t in terms:                                    # a term given on its own: exact name / alias first
        ids += [i for i in exact_term_ids(ix, t) if i not in ids]
    reg = "term" if (terms and not query.strip()) else regime(text, matches)
    sig = signals(ix, text, corpus, mode, ids, phrases=terms)
    w = weights_for(reg, "atlas" if "text" not in sig else "auto")
    w.update(weights or {})
    scored = rank_scored(sig, w, ix)
    keep = _filters(ix, topic, relevance, held_only, year_from)
    order = [p for p, _ in scored if keep(p)]
    if require_all and terms:                          # checked lazily, best first, until k papers pass
        passed = []
        for p in order:
            if _has_all(ix, corpus, p, terms):
                passed.append(p)
                if len(passed) >= k:
                    break
        order = passed
    order = order[:k]
    cb = ix.cited_by_terms(ids)
    pos = {n: {p: i + 1 for i, p in enumerate(r[:200])} for n, r in sig.items()}
    sc = dict(scored)
    pages = corpus.pages_many([p for p in order if ix.files.get(p)], text) \
        if snippets and corpus is not None and corpus.available else {}
    out = []
    for i, key in enumerate(order, 1):
        p = ix.by_key[key]
        why = []
        for tid in cb.get(key, []):
            t = ix.term_by_id[tid]
            q = next((c for c in t["citations"] if ix.alias.get(c["key"], c["key"]) == key), None)
            why.append(f"lexicon '{t['term']}': \"{q['quote']}\" ({q.get('where') or 'n.p.'})" if q else
                       f"lexicon '{t['term']}'")
        for n in ("table", "text", "meta"):
            if key in pos.get(n, {}):
                why.append({"table": "full-text rank {r} for the matched term(s)", "text": "full-text rank {r}",
                            "meta": "index words rank {r}"}[n].format(r=pos[n][key]))
        rec = {"rank": i, "key": key, "title": p.get("title") or "", "year": p.get("year"), "venue": p.get("venue") or "",
               "status": p.get("status") or "", "relevance": p.get("relevance") or "", "topics": p.get("topics") or [],
               "held": bool(p.get("pdf_held")), "file": ix.files.get(key, ""), "url": p.get("url") or "",
               "score": round(sc[key], 6), "regime": reg, "why": why, "pages": []}
        if corpus is not None and corpus.available and ix.files.get(key):
            f = corpus.pdfs / ix.files[key]
            rec["pdf"] = str(f) if f.is_file() else ""
            rec["pages"] = pages.get(key, [])
        out.append(rec)
    return out


def exact_term_ids(ix: Index, name: str) -> List[str]:
    """Lexicon terms whose name or alias IS this string (case-insensitive, punctuation-insensitive)."""
    n = PT._flat_ws(re.sub(r"[^A-Za-z0-9]+", " ", name)).strip().lower()
    out = []
    for t in ix.lex:
        for x in [t["term"]] + list(t.get("aliases") or []):
            if PT._flat_ws(re.sub(r"[^A-Za-z0-9]+", " ", x)).strip().lower() == n:
                out.append(t["id"])
                break
    return out


def _filters(ix: Index, topic: str, relevance: str, held_only: bool, year_from: int):
    def keep(k: str) -> bool:
        p = ix.by_key.get(k)
        if not p:
            return False
        if topic and not any(topic in t for t in p.get("topics") or []):
            return False
        if relevance and RELEVANCE.get(p.get("relevance"), 9) > RELEVANCE.get(relevance, 9):
            return False
        if held_only and not p.get("pdf_held"):
            return False
        return not (year_from and (p.get("year") or 0) < year_from)
    return keep


def _has_all(ix: Index, corpus: Optional[Corpus], key: str, terms: Sequence[str]) -> bool:
    """Every term occurs in the paper: in its full text when this machine has it, else in its index words."""
    if corpus is not None and corpus.local and key in corpus.by_key:
        f = PT.cache_file(corpus.cache, corpus.by_key[key])
        if f.is_file():
            body = PT._flat_ws(f.read_text(encoding="utf-8")).lower()
            return all(re.search(r"(?<![a-z0-9])" + re.escape(t.lower()), body) for t in terms)
    words = meta_text(ix.rows[key]).lower()
    return all(t.lower() in words for t in terms)


# ------------------------------------------------------------------------------------------- precompute
def precompute(base: Path = BWM, pdfs: Optional[Path] = None, model: str = DEFAULT_MODEL, say=print,
               today: str = "") -> dict:
    """papers/_lookup.json: for EVERY lexicon term (name + aliases) and every TASKS entry, the full-text hybrid ranking
    the benchmark measured best (BM25 + dense, RRF), top TOP_PER_TERM. Needs the text cache, the dense vectors and
    the GPU venv; deterministic for one corpus and model."""
    import paper_embed as PE
    corpus = Corpus(pdfs, dense=True, model=model, base=base, door=None)
    if not corpus.available:
        raise SystemExit(f"no text cache at {corpus.cache}: run `python tools/paper_text.py build` first")
    ix = Index(base)
    store = PE.default_cache(corpus.pdfs)
    PT.index(corpus.cache, say=say)
    dix = PE.Index(model, corpus.cache, store)
    if dix.missing or dix.stale:
        raise SystemExit(f"{model}: {dix.missing} held papers not embedded, {dix.stale} stale -- run "
                         f"`python tools/paper_embed.py embed --model {model} --bwm` first")
    m = PE.load_model(model)
    spec = PE.MODELS[model]
    terms: Dict[str, List[str]] = {}
    names = [[t["term"]] + list(t.get("aliases") or []) for t in ix.lex]
    vecs = PE.encode_queries(m, spec, [PE.term_text(n) for n in names], "term")
    for i, (t, nm, v) in enumerate(zip(ix.lex, names, vecs), 1):
        bm = [r["key"] for r in PT._rank_q(corpus.cache, PT.bm25_term_query(nm), corpus.names, 1000)]
        terms[t["id"]] = PE.rrf(bm, dix.rank(v)[0])[:TOP_PER_TERM]
        if i % 200 == 0:
            say(f"  {i}/{len(ix.lex)} terms")
    tasks, todo = [], AP.tasks(base)
    tv = PE.encode_queries(m, spec, [q for _, _, q in todo], "definition") if todo else []
    for (tid, title, q), v in zip(todo, tv):
        bm = [r["key"] for r in PT._rank_q(corpus.cache, PT.fts_query(q), corpus.names, 1000)]
        tasks.append({"id": tid, "title": title, "query": q,
                      "papers": PE.rrf(bm, dix.rank(v)[0])[:TOP_PER_TERM]})
    doc = {"schema": "svfr.bwm.lookup/1", "made": today or "",
           "ranker": f"hybrid: BM25 (SQLite FTS5) + dense {model} ({spec['id']}), reciprocal-rank fusion k=60",
           "corpus": {"papers": len(dix.span), "windows": int(dix.M.shape[0])},
           "terms": dict(sorted(terms.items())), "tasks": tasks}
    out = Path(base) / "papers" / LOOKUP_FILE
    out.write_text(_dump_lookup(doc), encoding="utf-8", newline="\n")
    say(f"wrote {out}: {len(terms)} terms, {len(tasks)} tasks ({doc['corpus']['papers']} papers)")
    return doc


def _dump_lookup(doc: dict) -> str:
    """One term per line (a changed term is a one-line diff), sorted keys."""
    one = lambda o: json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))   # noqa: E731
    head = {k: v for k, v in doc.items() if k not in ("terms", "tasks")}
    lines = [one(head)[:-1] + ',"tasks":[']
    lines.append(",\n".join(one(t) for t in doc["tasks"]))
    lines.append('],"terms":{')
    lines.append(",\n".join(f"{json.dumps(k)}:{one(v)}" for k, v in sorted(doc["terms"].items())))
    lines.append("}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- benchmark
KEY_REF = re.compile(r"\[([a-z][a-z0-9_]*)\]")


def _units(md: str) -> List[str]:
    """A review split into its citing units: table rows, list items (with their wrapped lines), paragraphs; a blank
    line, a heading or a table rule ends a unit."""
    out, cur = [], []

    def flush():
        if cur:
            out.append(" ".join(cur))
            cur.clear()
    for ln in md.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or re.match(r"^\|?\s*:?-{3,}", s):
            flush()
            continue
        if s.startswith("|"):
            flush()
            out.append(s)
            continue
        if re.match(r"^([-*+]|\d+[.)])\s+", s):
            flush()                                    # a list item starts a unit; its wrapped lines join it
        cur.append(s)
    flush()
    return out


def _clean_query(u: str) -> str:
    u = KEY_REF.sub(" ", u)
    u = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", u)              # markdown links -> their text
    u = re.sub(r"https?://\S+", " ", u)
    u = re.sub(r"[`*_|>#]", " ", u)
    u = re.sub(r"^\s*([-+]|\d+[.)])\s+", " ", u)
    return re.sub(r"\s+", " ", u).strip()


def task_queries(base: Path = BWM, min_words: int = 8, max_keys: int = 8) -> List[dict]:
    """The task benchmark, built the way LitSearch built its inline-citation questions: every unit (paragraph, list
    item, table row) of a LIT_*.md review that cites 1..max_keys index papers by [key] is a query -- its text with the
    keys removed -- and the HELD papers it cites are the relevant set. The reference list at the end is not a query."""
    ix = load_index(str(base))
    held = {p["key"] for p in ix.papers if p.get("pdf_held")}
    out = []
    for f in sorted(Path(base).glob("LIT_*.md")):
        md = re.split(r"(?im)^#+\s*(references|bibliography)\b.*$", f.read_text(encoding="utf-8"))[0]
        for n, u in enumerate(_units(md)):
            keys = [ix.alias.get(k, k) for k in KEY_REF.findall(u)]
            keys = list(dict.fromkeys(k for k in keys if k in ix.by_key))
            rel = sorted(set(keys) & held)
            if not rel or len(keys) > max_keys:
                continue
            q = _clean_query(u)
            if len(q.split()) < min_words:
                continue
            out.append({"id": f"{f.stem}:{n}", "doc": f.name, "query": q, "rel": rel})
    return out


def review_topics(base: Path = BWM) -> Dict[str, str]:
    """Each LIT_*.md review's topic key: the topic file (papers/<topic>.json) holding most of the papers it cites."""
    ix = load_index(str(base))
    rows = _bwm().topic_rows(Path(base))
    by_topic: Dict[str, set] = {}
    for t, r in rows:
        by_topic.setdefault(t, set()).add(ix.alias.get(r["key"], r["key"]))
    out = {}
    for f in sorted(Path(base).glob("LIT_*.md")):
        cited = {ix.alias.get(k, k) for k in KEY_REF.findall(f.read_text(encoding="utf-8"))}
        best = sorted(by_topic, key=lambda t: (-len(by_topic[t] & cited), t))
        if best and by_topic[best[0]] & cited:
            out[f.name] = best[0]
    return out


def meta_other_authors(base: Path, exclude_topic: str) -> MetaBM25:
    """The index's words for every paper WITHOUT the prose the given topic's research agent wrote (its summaries,
    why-for-us and tags): title twice, fact chips, venue, topics, and the other topics' prose. The fair control for
    a review's paragraphs, which that same agent wrote -- shared phrasing cannot help a ranker here."""
    ix = load_index(str(base))
    prose: Dict[str, List[str]] = {}
    for t, r in _bwm().topic_rows(Path(base)):
        if t == exclude_topic:
            continue
        k = ix.alias.get(r["key"], r["key"])
        prose.setdefault(k, []).extend([str(r.get("summary") or ""), str(r.get("why_for_us") or ""),
                                        " ".join(str(x) for x in r.get("tags") or [])])
    docs = {}
    for k, row in ix.rows.items():
        docs[k] = " ".join([row["t"], row["t"], " ".join(row["fx"]), row["v"],
                            " ".join(str(t).replace("_", " ") for t in row["top"])] + prose.get(k, []))
    return MetaBM25(docs)


def split_of(doc: str, seed: int = 13, base: Path = BWM) -> str:
    """DEV or TEST by REVIEW (units of one review never straddle the split): the reviews ordered by a seeded hash of
    their names, the first half DEV -- balanced, and deterministic for a given set of reviews (a review added later
    can move one other review across the middle)."""
    docs = sorted((f.name for f in Path(base).glob("LIT_*.md")),
                  key=lambda d: hashlib.sha256(f"{seed}:{d}".encode()).hexdigest())
    if doc not in docs:
        docs = sorted(docs + [doc], key=lambda d: hashlib.sha256(f"{seed}:{d}".encode()).hexdigest())
    return "dev" if docs.index(doc) < (len(docs) + 1) // 2 else "test"


def term_queries(base: Path = BWM, seed: int = 13) -> List[dict]:
    """The TERM benchmark: tools/paper_text.py's lexicon-citation queries (every term whose verified citations point at
    a held paper; rel = those papers), the query being the term's NAME as a person would type it, split DEV / TEST by
    a seeded hash of the term id (first half of the ordering DEV)."""
    ix = load_index(str(base))
    by_name = {t["term"]: t["id"] for t in ix.lex}
    qs = [{"id": by_name[q["term"]], "doc": by_name[q["term"]], "query": q["term"], "names": q["names"],
           "rel": q["rel"]} for q in PT.bench_queries() if q["term"] in by_name]
    order = sorted((q["id"] for q in qs), key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    half = set(order[:(len(order) + 1) // 2])
    for q in qs:
        q["split"] = "dev" if q["id"] in half else "test"
    return qs


def drop_terms(qs: List[dict], lexicon_file: Path) -> List[dict]:
    """`qs` without the TERM queries of the terms one lexicon file defines (bench --skip-terms). The control for terms
    whose labels a LEXICAL search found -- lexicon/fulltext_terms.json was mined from the full text, so its citations
    flatter the full-text rankers on exactly those queries."""
    d = json.loads(Path(lexicon_file).read_text(encoding="utf-8"))
    drop = {t["term"] for t in (d["terms"] if isinstance(d, dict) else d)}
    return [q for q in qs if q["query"] not in drop]


def bench(base: Path = BWM, pdfs: Optional[Path] = None, split: str = "all", n: int = 0, dense: bool = True,
          model: str = "nemotron-1b", tune: bool = False, say=print, seed: int = 13, kind: str = "tasks",
          skip_terms: Optional[Path] = None) -> dict:
    """Score every ranker on the TASK benchmark (the reviews' citing paragraphs) or the TERM benchmark (lexicon terms)
    -- R@10 / R@50 / MRR / Hit@10 / Hit@50, macro over queries -- each paired against the full-text ranker with a
    bootstrap 95% interval. Full-text rankings are memoised beside the text cache
    (`<pdfs>_text/paper_find_memo.json.gz`, keyed by the index, the model and the query): a re-run is quick."""
    import paper_embed as PE
    ix = load_index(str(base))
    if kind == "terms":
        qs = [q for q in term_queries(base, seed) if split == "all" or q["split"] == split]
        if skip_terms:
            qs = drop_terms(qs, skip_terms)
    else:
        qs = [q for q in task_queries(base) if split == "all" or split_of(q["doc"], seed) == split]
    if n and n < len(qs):
        qs = random.Random(seed).sample(qs, n)
        qs.sort(key=lambda q: q["id"])
    corpus = Corpus(pdfs, dense=dense, model=model, base=base, door=None)
    if not corpus.available:
        raise SystemExit(f"bench needs the text cache ({corpus.cache})")
    PT.index(corpus.cache, say=say)
    memo, save = _memo(corpus, model if dense else "")
    say(f"{len(qs)} {kind} queries ({split}); dense {'on: ' + model if dense else 'off'}")
    form = "term" if kind == "terms" else "definition"
    qtext = (lambda q: PE.term_text(q["names"])) if kind == "terms" else (lambda q: q["query"])   # noqa: E731
    fts_of = (lambda q: PT.bm25_term_query(q["names"])) if kind == "terms" else (lambda q: text_fts(q["query"]))  # noqa
    dense_rank = None
    if dense:
        todo = [q for q in qs if f"dense-{form}:" + qtext(q) not in memo]
        if todo:
            store = PE.default_cache(corpus.pdfs)
            dix = PE.Index(model, corpus.cache, store)
            m = PE.load_model(model)
            vecs = PE.encode_queries(m, PE.MODELS[model], [qtext(q) for q in todo], form)
            for q, v in zip(todo, vecs):
                memo[f"dense-{form}:" + qtext(q)] = dix.rank(v)[0][:1000]
            save()
        dense_rank = lambda q: memo[f"dense-{form}:" + qtext(q)]   # noqa: E731
    rows: Dict[str, list] = {}
    sigs = []
    rtopic = review_topics(base) if kind == "tasks" else {}
    others: Dict[str, MetaBM25] = {}
    for i, q in enumerate(qs, 1):
        kb = "bm25:" + fts_of(q)
        if kb not in memo:
            memo[kb] = [r["key"] for r in PT._rank_q(corpus.cache, fts_of(q), corpus.names, 1000)]
            if i % 50 == 0:
                save()
        if kind == "terms":                            # the term itself, as the lookup would match it
            ids = [q["id"]]
        else:
            ids = list(dict.fromkeys(m_["id"] for m_ in ix.terms.find(q["query"])))
        s = {"bm25": memo[kb], "lexicon": ix.lexicon_rank(ids), "table": ix.table_rank(ids),
             "table_atlas": ix.table_rank(ids, ATLAS_TABLE_DEPTH),
             "meta": ix.meta.rank(q["query"]), "titles": ix.titles.rank(q["query"])}
        s["text"] = fuse([(1.0, s["bm25"]), (1.0, dense_rank(q))]) if dense_rank else s["bm25"]
        if dense_rank:
            s["dense"] = dense_rank(q)
        if kind == "tasks":                            # the control: no prose by the agent who wrote the query
            t = rtopic.get(q["doc"], "")
            if t not in others:
                others[t] = meta_other_authors(base, t)
            s["meta_other"] = others[t].rank(q["query"])
        sigs.append(s)
    save()
    reg = "term" if kind == "terms" else "task"
    rankers = ranker_set(ix, bool(dense_rank), reg, other=kind == "tasks")
    for q, s in zip(qs, sigs):
        for name, fn in rankers.items():
            rows.setdefault(name, []).append(PT.bench_score(fn(s), q["rel"]))
    out = PT.bench_summary(rows)
    out.update(kind=kind, queries=len(qs), split=split, dense_model=model if dense_rank else "",
               mean_relevant=round(sum(len(q["rel"]) for q in qs) / max(1, len(qs)), 2),
               weights={"full": WEIGHTS[reg], "atlas": ATLAS_WEIGHTS[reg]})
    if kind == "tasks":
        out["reviews"] = sorted({q["doc"] for q in qs})
    if skip_terms:
        out["skipped_terms_of"] = Path(skip_terms).name
    out["vs_text"] = {}
    for name, v in rows.items():
        if name != "text":
            out["vs_text"][name] = {mt: [round(x, 3) for x in PE.paired_bootstrap([r[i] for r in v],
                                                                                   [r[i] for r in rows["text"]])]
                                    for i, mt in enumerate(PT.BENCH_METRICS) if mt in ("R@10", "R@50", "MRR")}
    if tune:
        skip = ("lexicon",) if kind == "terms" else ()
        out["tuning"] = {"full": tune_weights(sigs, qs, ("text", "lexicon", "table", "meta"), fixed="text", skip=skip),
                         "atlas": tune_weights(sigs, qs, ("lexicon", "table", "meta"), fixed="meta", skip=skip)}
        if kind == "tasks":                            # the same grids with the other-authors control as `meta`
            alt = [dict(s, meta=s["meta_other"]) for s in sigs]
            out["tuning"]["full-other-authors"] = tune_weights(alt, qs, ("text", "lexicon", "table", "meta"),
                                                               fixed="text")
            out["tuning"]["atlas-other-authors"] = tune_weights(alt, qs, ("lexicon", "table", "meta"), fixed="meta")
    return out


def ranker_set(ix: Index, has_dense: bool, reg: str = "task", other: bool = False) -> Dict[str, callable]:
    """The rankers the bench compares, as functions of one query's signals. 'full' / 'atlas' are what `find` does in
    this regime with and without the full text. Two CONTROLS for `meta`, whose prose the research agents wrote -- the
    same agents who wrote the reviews the task queries come from: `titles` (the papers' own words only: title, fact
    chips, venue) and, on tasks, `meta-other` (meta without the prose of the agent who wrote that review)."""
    W, A = WEIGHTS[reg], ATLAS_WEIGHTS[reg]
    lex0 = lambda w: dict(w, lexicon=0.0)               # noqa: E731
    atl = lambda s: {n: s[n] for n in ("lexicon", "meta")} | {"table": s.get("table_atlas", s["table"])}  # noqa: E731
    r = {
        "bm25": lambda s: s["bm25"],
        "text": lambda s: s["text"],
        "meta": lambda s: s["meta"],
        "titles": lambda s: s["titles"],
        "lexicon": lambda s: s["lexicon"],
        "table": lambda s: s["table"],
        "atlas": lambda s: rank(atl(s), A),             # exactly the page's signals: 20-deep term lists, no text
        "atlas-no-lexicon": lambda s: rank(atl(s), lex0(A)),
        "full": lambda s: rank(s, W),
        "full-no-lexicon": lambda s: rank(s, lex0(W)),
        "full+graph": lambda s: rank(s, dict(W, graph=1.0), ix),
        "full-titles-for-meta": lambda s: rank(dict(s, meta=s["titles"]), W),
    }
    if other:
        r["meta-other"] = lambda s: s["meta_other"]
        r["atlas-meta-other"] = lambda s: rank(dict(atl(s), meta=s["meta_other"]), A)
        r["full-meta-other"] = lambda s: rank(dict(s, meta=s["meta_other"]), W)
    if has_dense:
        r["dense"] = lambda s: s["dense"]
    return r


def tune_weights(sigs: List[dict], qs: List[dict], names: Sequence[str], fixed: str, skip: Sequence[str] = ()) -> dict:
    """A small grid over the weights of `names` (`fixed` held at 1 -- fusion is invariant to scaling all weights; a
    `skip` signal held at 0: the lexicon on TERM queries, whose labels ARE its citations), scored by mean R@10, then
    MRR. DEV split only: the chosen weights are copied into WEIGHTS / ATLAS_WEIGHTS by hand; TEST is scored with
    them only."""
    grid = (0.0, 0.25, 0.5, 1.0, 2.0)
    free = [n for n in names if n != fixed and n not in skip]
    best, table = None, []

    def combos(i, cur):
        if i == len(free):
            yield dict(cur)
            return
        for g in grid:
            cur[free[i]] = g
            yield from combos(i + 1, cur)
    for w in combos(0, {}):
        w = dict(w, **{fixed: 1.0}, **{s: 0.0 for s in skip})
        sc = [PT.bench_score(fuse([(w[n], s[n]) for n in names]), q["rel"]) for s, q in zip(sigs, qs)]
        r10 = sum(x[0] for x in sc) / len(sc)
        mrr = sum(x[2] for x in sc) / len(sc)
        table.append((round(r10, 4), round(mrr, 4), w))
        if best is None or (r10, mrr) > (best[0], best[1]):
            best = (r10, mrr, w)
    table.sort(key=lambda x: (-x[0], -x[1]))
    return {"best": {"R@10": round(best[0], 4), "MRR": round(best[1], 4), "weights": best[2]},
            "top10": [{"R@10": a, "MRR": b, "weights": w} for a, b, w in table[:10]]}


def _memo(corpus: Corpus, model: str):
    """Full-text rankings per query, valid for one text index + key map (+ model): asked once."""
    import sqlite3
    con = sqlite3.connect(corpus.cache / PT.INDEX)
    rows = "\n".join(f"{a} {b}" for a, b in con.execute("select sha, extractor from indexed order by sha"))
    con.close()
    keys = "\n".join(f"{s} {corpus.names[s]['key']}" for s in sorted(corpus.names))
    sig = hashlib.sha256((rows + "\n#\n" + keys + "\n#\n" + model).encode()).hexdigest()[:16]
    f = corpus.cache / "paper_find_memo.json.gz"
    memo = {}
    if f.is_file():
        try:
            d = json.loads(gzip.decompress(f.read_bytes()))
            memo = d.get("rankings", {}) if d.get("corpus") == sig else {}
        except (OSError, ValueError):
            memo = {}

    def save():
        tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
        tmp.write_bytes(gzip.compress(json.dumps({"corpus": sig, "rankings": memo}).encode("utf-8")))
        replace_retry(tmp, f, may_drop=True)           # a cache: a memo held open past the bound is re-asked next run
    return memo, save


# --------------------------------------------------------------------------------------------------- CLI
def _print_hits(hits: List[dict], fmt: str) -> None:
    if fmt == "keys":
        for h in hits:
            print(h["key"])
        return
    if fmt == "json":
        print(json.dumps(hits, ensure_ascii=False, indent=1))
        return
    if fmt == "tsv":
        print("rank\tkey\tyear\theld\trelevance\ttitle\tfile\twhy")
        for h in hits:
            print(f"{h['rank']}\t{h['key']}\t{h['year'] or ''}\t{'held' if h['held'] else ''}\t{h['relevance']}\t"
                  f"{h['title']}\t{h['file']}\t{' | '.join(h['why'])}")
        return
    for h in hits:
        print(f"{h['rank']:3}. {h['key']} ({h['year'] or 'n.d.'}{', PDF held' if h['held'] else ''}, {h['relevance']}) "
              f"-- {h['title'][:110]}")
        for w in h["why"][:4]:
            print(f"       why: {w[:200]}")
        for pg in h["pages"]:
            print(f"       p.{pg['page']}: {pg['snippet'][:220]}")
        if h.get("pdf"):
            print(f"       pdf: {h['pdf']}")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = argv[0] if argv and argv[0] in ("terms", "precompute", "bench", "tasks") else "find"
    if sub != "find":
        argv = argv[1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query", nargs="*", help="find: the task (free text) and / or terms; terms: the text to read")
    ap.add_argument("--term", action="append", default=[], help="a term (repeatable): its definition, then its papers")
    ap.add_argument("--all", dest="require_all", action="store_true",
                    help="only papers that contain EVERY term / query word given (grep-like AND)")
    ap.add_argument("-k", "--k", type=int, default=15, help="papers listed (default 15)")
    ap.add_argument("--mode", choices=["auto", "atlas"], default="auto",
                    help="atlas = exactly what the atlas's Find tab computes (no full text)")
    ap.add_argument("--dense", action="store_true", help="add the dense model to the full text (GPU venv; slow start)")
    ap.add_argument("--topic", default="", help="only papers of this index topic (substring)")
    ap.add_argument("--relevance", choices=["core", "high", "medium", "low"], default="")
    ap.add_argument("--held", action="store_true", help="only papers whose PDF is held")
    ap.add_argument("--year-from", type=int, default=0)
    ap.add_argument("--keys", dest="fmt", action="store_const", const="keys", default="text")
    ap.add_argument("--json", dest="fmt", action="store_const", const="json")
    ap.add_argument("--tsv", dest="fmt", action="store_const", const="tsv")
    ap.add_argument("--no-snippets", action="store_true")
    ap.add_argument("--local-text", action="store_true",
                    help="full text from this machine's cache only: never ask the lab's library door")
    ap.add_argument("--pdfs", type=Path, default=None, help="the papers folder (default: as tools/paper_text.py)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all", help="bench: which half")
    ap.add_argument("--kind", choices=["tasks", "terms"], default="tasks",
                    help="bench: the reviews' citing paragraphs (tasks) or the lexicon's terms (terms)")
    ap.add_argument("--n", type=int, default=0, help="bench: a seeded sample of N queries")
    ap.add_argument("--no-dense", action="store_true", help="bench: BM25 only for the full text")
    ap.add_argument("--tune", action="store_true", help="bench: grid the fusion weights (DEV only!)")
    ap.add_argument("--out", type=Path, default=None, help="bench: also write the result JSON here")
    ap.add_argument("--skip-terms", type=Path, default=None,
                    help="bench --kind terms: leave out the terms one lexicon file defines (a control)")
    ap.add_argument("--today", default="", help="precompute: the date stamped in the file")
    a = ap.parse_args(argv)
    ix = load_index()
    if sub == "terms":
        text = " ".join(a.query)
        for m in ix.terms.find(text):
            t = ix.term_by_id[m["id"]]
            print(f"{m['matched']!r} -> {t['term']} [{m['id']}] ({len(t['citations'])} cited papers)")
        return 0
    if sub == "tasks":
        for tid, title, q in AP.tasks(BWM):
            n = len(next((t["papers"] for t in ix.lookup.get("tasks") or [] if t["id"] == tid), []))
            print(f"{tid:18} {title}  [{n} precomputed]\n{'':18} {q}")
        return 0
    if sub == "precompute":
        import datetime
        precompute(pdfs=a.pdfs, model=a.model, today=a.today or datetime.date.today().isoformat())
        return 0
    if sub == "bench":
        if a.tune and a.split != "dev":
            ap.error("--tune runs on --split dev only (TEST is scored only with the weights DEV chose)")
        if a.skip_terms and a.kind != "terms":
            ap.error("--skip-terms applies to --kind terms")
        res = bench(pdfs=a.pdfs, split=a.split, n=a.n, dense=not a.no_dense, model=a.model, tune=a.tune, kind=a.kind,
                    skip_terms=a.skip_terms)
        print(json.dumps(res, indent=1))
        if a.out:
            a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
        return 0
    if not (a.query or a.term):
        ap.error("give a task, terms, or --term NAME")
    for name in a.term:                                # a named term: what the lexicon says first
        ids = exact_term_ids(ix, name) or [m["id"] for m in ix.terms.find(name)]
        if not ids:
            print(f"'{name}' is not a lexicon term (searching the papers for it anyway)")
        for tid in ids[:3]:
            t = ix.term_by_id[tid]
            if a.fmt == "text":
                print(f"{t['term']}" + (f" (also {', '.join(t['aliases'][:5])})" if t["aliases"] else ""))
                print(f"  {t['definition']}")
                for c in t["citations"][:6]:
                    print(f"  - \"{c['quote']}\" -- {c['key']} ({c.get('year') or 'n.d.'}; {c.get('where') or ''})")
                print()
    query, terms = " ".join(a.query), list(a.term)
    if a.require_all:                                  # --all: every argument is a term the paper must contain
        query, terms = "", list(a.query) + list(a.term)
    corpus = None if a.mode == "atlas" else Corpus(a.pdfs, dense=a.dense, door=None if a.local_text else "auto")
    hits = find(query, terms, k=a.k, mode=a.mode, dense=a.dense, require_all=a.require_all,
                topic=a.topic, relevance=a.relevance, held_only=a.held, year_from=a.year_from, pdfs=a.pdfs,
                snippets=not a.no_snippets, corpus=corpus)
    _print_hits(hits, a.fmt)
    if corpus is not None and a.fmt == "text":
        print(f"(full text: {corpus.source})".encode("ascii", "replace").decode("ascii"))
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
