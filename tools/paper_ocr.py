r"""OCR for pages a PDF's text layer cannot give (scans, image-only pages), run ONCE per page and cached; the text cache
of `tools/paper_text.py` then reads the OCR text for those pages, so search, grep, facts and embeddings see them.

    python tools/paper_ocr.py need --bwm                         # which held PDFs have pages with no usable text layer
    python tools/paper_ocr.py run --bwm [--engine lighton|rapid] [--keys k1,k2] [--pdf FILE]   # OCR them (cached)
    python tools/paper_ocr.py bench --bwm [--n 30] [--dpi 200]   # engines on born-digital pages vs their text layer
    python tools/paper_ocr.py agree --bwm [--out FILE]           # real scans: how far the two engines' texts agree

ENGINES (chosen by the 2026-10-14 research, LIT_PAPER_MINING.md): `lighton` = LightOnOCR-3-0.8B (Apache-2.0, a 0.85B
vision-language model; transformers, bf16, 1.7 GB of VRAM; pages rendered at 400 DPI, longest side 2048 px, the
card's settings) -- the strongest English / old-scan evidence among models that fit 8 GB, but a VLM can write FLUENT
TEXT THAT IS NOT ON THE PAGE; `rapid` = RapidOCR (PP-OCR models on ONNX Runtime, Apache-2.0, CPU, pip-only) -- a
classic line recogniser that cannot invent sentences. `bench` measures both on born-digital pages whose text layer is
the ground truth: CER, WER and the UNSUPPORTED-WORD rate (OCR words that are not on the page at all -- the
hallucination proxy CER hides). Both engines run on real scans; `agree` scores every page both read (order-free
word F1 between them, and the share of LightOn's words far from every word RapidOCR read) -- the pages to read by eye.

MEASURED 2026-10-14 (`bench`, the same 30 seeded born-digital pages at 200 DPI, paired): RapidOCR CER 0.140 (median
0.080), word F1 0.955, unsupported 0.9%, 10.7 s/page (CPU); LightOnOCR-3 CER 0.151 (median 0.098; difference +0.011,
95% CI -0.026..+0.048), word F1 0.936, unsupported 2.3% (+1.4 points, CI +0.15..+2.8: MORE words on no page, up to 11%
on one page), 81 s/page (GPU, with the cache on -- the checkpoint's config turns it OFF, 18x slower). So the text cache
prefers RapidOCR (paper_text.OCR_ENGINES) and LightOn was not run over the scans; its claimed edge is on degraded old
scans, which this bench (born-digital pages) cannot score -- `agree` on real scans is the next measurement.

NEEDS torch + transformers (lighton) or rapidocr + onnxruntime (rapid), and pypdfium2: the GPU venv
(report/boat_world_model/README.md). CACHE (outside git): `<pdfs>_ocr/<engine>/<sha[:2]>/<sha>.json` -- every OCR'd
page's text, the engine + model revision, render size, seconds, and a loop / truncation flag.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_text as PT  # noqa: E402
if str(Path(__file__).resolve().parents[1]) not in sys.path:   # run by path: the repo root, for tools/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.atomic_replace import replace_retry  # noqa: E402

LIGHTON = "lightonai/LightOnOCR-3-0.8B"
needs_ocr = PT.needs_ocr                                # one definition, shared with the text cache's OCR merge
default_store = PT.ocr_store


# ------------------------------------------------------------------------------------------ text (pure)
def norm_words(t: str) -> List[str]:
    t = re.sub(r"-\s*\n\s*(?=[a-z])", "", t or "")              # re-join hyphenated line breaks
    t = re.sub(r"[#*_`|>\[\]]", " ", t)                          # Markdown markup a VLM adds
    return re.findall(r"[a-z0-9]+", t.lower())


def _lev(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _near(w: str, vocab) -> bool:
    """Is w a MISREADING of some word on the page (normalised edit distance <= 0.25; one edit for <= 5 letters)?"""
    lim = 1 if len(w) <= 5 else max(1, int(0.25 * len(w)))
    return any(abs(len(v) - len(w)) <= lim and _lev(w, v) <= lim for v in vocab)


def ocr_scores(ocr: str, truth: str) -> dict:
    """CER (characters of the normalised word stream) and WER -- both ORDER-sensitive, so two columns read across the
    page score badly even when every word is right; bow_f1 -- the order-free share of words recovered (multiset F1:
    recognition alone); `misread` -- OCR words not on the page but within a small edit distance of a word that is (a
    recogniser's letter errors: 's0nar'); and `unsupported` -- OCR words FAR from every word on the page: what a
    recogniser cannot produce and a hallucinating model can (review 2026-10-14: counting misreads as invented made a
    classic engine look like it hallucinates). Pure."""
    import collections
    o, t = norm_words(ocr), norm_words(truth)
    os_, ts = " ".join(o), " ".join(t)
    vocab = set(t)
    common = sum((collections.Counter(o) & collections.Counter(t)).values())
    off = [w for w in o if w not in vocab]
    near = {w: _near(w, vocab) for w in set(off)}
    return {"cer": round(_lev(os_, ts) / max(1, len(ts)), 4), "wer": round(_lev(o, t) / max(1, len(t)), 4),
            "bow_f1": round(2 * common / max(1, len(o) + len(t)), 4),
            "misread": round(sum(1 for w in off if near[w]) / max(1, len(o)), 4),
            "unsupported": round(sum(1 for w in off if not near[w]) / max(1, len(o)), 4),
            "words_ocr": len(o), "words_truth": len(t)}


def reading_order(lines, page_w: float) -> list:
    """[(x0, y0, x1, y1, text)] line boxes -> texts in reading order: full-width lines (titles, wide captions) cut
    the page into bands; within a band the left column is read before the right. Pure."""
    lines = sorted(lines, key=lambda b: (b[1], b[0]))
    out, band = [], []

    def flush():
        left = [b for b in band if (b[0] + b[2]) / 2 < page_w / 2]
        right = [b for b in band if (b[0] + b[2]) / 2 >= page_w / 2]
        out.extend(b[4] for b in sorted(left, key=lambda b: (b[1], b[0])) + sorted(right, key=lambda b: (b[1], b[0])))
        band.clear()
    for b in lines:
        if b[2] - b[0] > 0.55 * page_w and b[0] < 0.3 * page_w:           # spans both columns
            flush()
            out.append(b[4])
        else:
            band.append(b)
    flush()
    return out


def looped(text: str) -> bool:
    """A generation that repeats itself compresses far better than prose (zlib ratio < 0.13 -- LightOn's own check).
    The ratio depends on the zlib BUILD near that line (MEASURED 2026-10-19: one synthetic text is 0.1257 on zlib 1.3.1
    -- Linux, Windows py3.12 -- and 0.1524 on zlib-ng 2.2.4, Windows py3.14), so a text close to 0.13 can flip between
    machines; a real loop (~0.014) and prose (~0.3 and up) sit far from it on both."""
    b = (text or "").encode("utf-8")
    return len(b) > 2000 and len(zlib.compress(b)) / len(b) < 0.13


# ------------------------------------------------------------------------------------------------ engines
class Engine:
    def __init__(self, name: str):
        self.name = name
        self.last: dict = {}
        if name == "lighton":
            import paper_embed
            paper_embed.gpu_setup()                         # LightOnOCR-3 is a Qwen3.5: the same GQA slow path
            import torch
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
            self.torch = torch
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(LIGHTON, dtype=torch.bfloat16,
                                                                          device_map="cuda")
            self.proc = AutoProcessor.from_pretrained(LIGHTON)
            self.revision = getattr(self.model.config, "_commit_hash", "") or ""
        elif name == "rapid":
            from rapidocr import RapidOCR
            self.model = RapidOCR()
            self.revision = "rapidocr"
        else:
            raise ValueError(name)

    def page_image(self, pdf_page):
        dpi = 400 if self.name == "lighton" else 300
        img = pdf_page.render(scale=dpi / 72).to_pil()
        if self.name == "lighton":
            img.thumbnail((2048, 2048))
        return img

    def __call__(self, img, seed: int = 13) -> str:
        self.last = {"size": list(img.size), "truncated": False}
        if self.name == "rapid":
            import numpy as np
            r = self.model(np.asarray(img.convert("RGB"))[:, :, ::-1].copy())    # RapidOCR takes an array as BGR
            boxes, txts = getattr(r, "boxes", None), getattr(r, "txts", None)
            if boxes is None or txts is None:
                return ""
            boxes4 = [(min(p[0] for p in b), min(p[1] for p in b), max(p[0] for p in b), max(p[1] for p in b), t)
                      for b, t in zip(boxes, txts)]
            return "\n".join(reading_order(boxes4, img.width))
        torch = self.torch
        torch.manual_seed(seed)
        conv = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        inp = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt", enable_thinking=False).to(self.model.device)
        with torch.no_grad():                          # use_cache: the checkpoint's config says FALSE, so generate
            out = self.model.generate(**inp, max_new_tokens=4096, do_sample=True, temperature=0.2, top_p=0.9,
                                      use_cache=True)  # re-ran the whole page every token (256 in 491 s vs 27 s)
        gen = out[0, inp["input_ids"].shape[1]:]
        self.last["truncated"] = int(gen.shape[0]) >= 4096           # hit the cap: the page may be cut short
        return self.proc.decode(gen, skip_special_tokens=True)


def ocr_pdf(pdf: Path, eng: Engine, pages: Optional[Sequence[int]] = None, say=print) -> List[dict]:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    out = []
    for n in [x for x in (pages or range(1, len(doc) + 1)) if 1 <= x <= len(doc)]:
        t0 = time.time()
        txt = eng(eng.page_image(doc[n - 1]))
        out.append({"page": n, "text": txt, "seconds": round(time.time() - t0, 1), "looped": looped(txt),
                    "truncated": bool(eng.last.get("truncated")), "image": eng.last.get("size")})
        say(f"    p.{n}: {len(txt)} chars, {out[-1]['seconds']} s" + (" LOOPED" if out[-1]["looped"] else ""))
    doc.close()
    return out


# ------------------------------------------------------------------------------------------------ corpus
def store_file(store: Path, engine: str, sha: str) -> Path:
    return store / engine / sha[:2] / f"{sha}.json"


def _n_pages(pdf: Path, cache: Path, sha: str) -> int:
    """A PDF's page count: the text cache's, else the PDF's own (an any-folder `--pdf`, a machine whose text cache is
    not built yet -- fix check 2026-10-14: an empty count made a partial store look complete)."""
    n = (PT.read_manifest(cache).get(sha) or {}).get("pages") or 0
    if n:
        return n
    try:
        import pypdfium2 as pdfium
        return len(pdfium.PdfDocument(str(pdf)))
    except Exception:                                    # noqa: BLE001 -- unreadable: nothing to complete
        return 0


def _page(cache: Path, sha: str, n: int) -> str:
    f = PT.cache_file(cache, sha)
    pages = f.read_text(encoding="utf-8").split(PT.FF) if f.is_file() else []
    return pages[n - 1] if 0 < n <= len(pages) else ""


def pages_needing_ocr(cache: Path, sha: str) -> List[int]:
    f = PT.cache_file(cache, sha)
    if not f.is_file():
        return []
    return [i for i, pg in enumerate(f.read_text(encoding="utf-8").split(PT.FF), 1) if needs_ocr(pg)]


def tried_pages(store: Path, sha: str, engine: Optional[str] = None) -> set:
    """Pages NOT to offer again. Some engine read it usably (`paper_text.usable_ocr`): done. Else, for `engine`, the
    pages ITS store holds; with no engine (the plain listing), the pages EVERY engine has tried -- a page one engine
    botched stays offered to the other (fix check 2026-10-14: 'tried by any' broke the LightOn / RapidOCR fallback)."""
    per, usable = {}, set()
    for e in PT.OCR_ENGINES:
        try:
            pgs = json.loads(store_file(store, e, sha).read_text(encoding="utf-8")).get("pages") or []
        except (OSError, ValueError):
            pgs = []
        per[e] = {int(pg.get("page") or 0) for pg in pgs}
        usable |= {int(pg.get("page") or 0) for pg in pgs if PT.usable_ocr(pg)}
    if engine:
        return usable | per.get(engine, set())
    return usable | set.intersection(*per.values())


def needs_listing(cache: Path, store: Path, sha: str, man: Dict[str, dict], engine: Optional[str] = None
                  ) -> Optional[tuple]:
    """(pages, pages still to OCR) when `need` lists this paper, else None -- the rule `need` applies per paper (and
    tools/atlas_coverage.py, per paper and machine)."""
    tried = tried_pages(store, sha, engine)
    bad = [x for x in pages_needing_ocr(cache, sha) if x not in tried]
    tot = (man.get(sha) or {}).get("pages") or 0
    title = 1 in bad and PT.stamp_only(_page(cache, sha, 1))
    begun = any(store_file(store, e, sha).is_file() for e in PT.OCR_ENGINES)   # a paper OCR has begun on: every
    if bad and tot and (len(bad) >= 3 or len(bad) / tot > 0.2 or title or begun):   # page still to do is offered
        return tot, len(bad)
    return None


def need(pdfs: Path, say=print, store: Optional[Path] = None, engine: Optional[str] = None) -> List[dict]:
    """Held PDFs with pages the text layer cannot give, worst first -- minus the pages any engine read usably and,
    plain, the pages EVERY engine has tried (with `engine`: the pages THAT engine tried), so a page one engine botched
    stays offered to the other (`tried_pages`). A blank page or two is just a blank page -- but a TITLE page holding
    nothing (or only a download stamp) is always listed (2026-10-14: the JSTOR scans' page 1), and so is a paper OCR
    has begun on; a page every engine has tried is never listed again (review 2026-10-14: a blank cover looped)."""
    cache = PT.default_cache(pdfs)
    store = Path(store) if store else PT.ocr_store(pdfs)
    man = PT.read_manifest(cache)
    rows = []
    for sha, n in PT.bwm_names().items():
        got = needs_listing(cache, store, sha, man, engine)
        if got:
            rows.append({"key": n["key"], "sha": sha, "pages": got[0], "need": got[1], "file": n.get("file", "")})
    rows.sort(key=lambda r: (-r["need"] / max(1, r["pages"]), -r["need"]))
    for r in rows:
        say(f"  {r['need']:>4}/{r['pages']:<4} pages  {r['key']}  ({r['file'][:70]})")
    return rows


def run(pdfs: Path, store: Path, engine: str, keys: Optional[Sequence[str]] = None, pdf: Optional[Path] = None,
        all_pages: bool = False, say=print) -> dict:
    """OCR the pages that need it (every page with --pdf / all_pages) of the chosen papers; cached per engine."""
    import hashlib
    cache = PT.default_cache(pdfs)
    jobs = []
    if pdf is not None:
        sha = hashlib.sha256(Path(pdf).read_bytes()).hexdigest()
        jobs.append((Path(pdf).stem, sha, Path(pdf), None))
    else:
        rows = need(pdfs, say=lambda *_: None, store=store, engine=engine)
        for r in rows:
            if keys is None or r["key"] in keys:
                tried = tried_pages(store, r["sha"], engine)   # as `need` counted them: never a page another
                jobs.append((r["key"], r["sha"], Path(pdfs) / r["file"], None if all_pages else   # engine READ
                             [x for x in pages_needing_ocr(cache, r["sha"]) if x not in tried]))
    def stored(sha) -> Optional[List[dict]]:           # a half-written store file is redone, not trusted
        f = store_file(store, engine, sha)
        try:
            got = json.loads(f.read_text(encoding="utf-8")).get("pages") if f.is_file() else None
        except (OSError, ValueError):
            return None
        return got if isinstance(got, list) else None

    todo = []
    for key, sha, f, pages in jobs:                    # a store made before a page counted as needing OCR is
        have = stored(sha)                             # COMPLETED with the pages it lacks, never redone or skipped
        if have is None:
            todo.append((key, sha, f, pages, []))
        else:
            done = {pg.get("page") for pg in have}
            want = pages or list(range(1, _n_pages(f, cache, sha) + 1))
            if [x for x in want if x not in done]:      # `--pdf` (every page) completes a partial store too
                todo.append((key, sha, f, [x for x in want if x not in done], have))
    if not todo:
        return {"engine": engine, "papers": 0}
    eng, t0, n = Engine(engine), time.time(), 0
    for key, sha, f, pages, have in todo:
        say(f"  {key}: {len(pages) if pages else 'all'} pages with {engine}")
        res = ocr_pdf(f, eng, pages, say=say)
        out = store_file(store, engine, sha)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        merged = sorted({pg["page"]: pg for pg in list(have) + list(res)}.values(), key=lambda pg: pg["page"])
        tmp.write_text(json.dumps({"schema": "svfr.ocr/1", "key": key, "sha256": sha, "engine": engine,
                                   "model": LIGHTON if engine == "lighton" else "rapidocr", "revision": eng.revision,
                                   "pages": merged}, ensure_ascii=False), encoding="utf-8")
        replace_retry(tmp, out)
        n += len(res)
    return {"engine": engine, "papers": len(todo), "pages": n, "seconds": round(time.time() - t0)}


def agreement(store: Path, engines: Sequence[str] = ("lighton", "rapid")) -> List[dict]:
    """Every page BOTH engines OCR'd -> how far their texts agree: `agree_f1`, the order-free word F1 between them
    (1 = the same words), and `unsupported_vs_other`, the share of the first engine's words FAR from every word the
    second read (the hallucination proxy with the other engine as the reference). A real scan has no ground truth, so
    this is not accuracy: a low score marks a page to read by eye -- one engine misread it, or the VLM wrote words
    that are not there. Reads the store only."""
    out = []
    for fa in sorted((Path(store) / engines[0]).glob("*/*.json")):
        try:
            a = json.loads(fa.read_text(encoding="utf-8"))
            b = json.loads(store_file(store, engines[1], fa.stem).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        other = {q["page"]: q for q in b.get("pages") or []}
        for pg in a.get("pages") or []:
            q = other.get(pg["page"])
            if q is None or not ((pg.get("text") or "").strip() or (q.get("text") or "").strip()):
                continue                               # (both blank: nothing to agree on -- fix check 2026-10-14)
            sc = ocr_scores(pg.get("text") or "", q.get("text") or "")
            out.append({"key": a.get("key"), "sha256": fa.stem, "page": pg["page"], "agree_f1": sc["bow_f1"],
                        "unsupported_vs_other": sc["unsupported"], "words": [sc["words_ocr"], sc["words_truth"]],
                        "looped": bool(pg.get("looped") or q.get("looped"))})
    return out


def bench(pdfs: Path, engines: Sequence[str], n: int = 30, dpi: int = 200, seed: int = 13, say=print) -> dict:
    """Born-digital pages (seeded sample, >= 1,500 text-layer characters) rendered to images and OCR'd: the text layer is
    the ground truth. dpi: the RENDER resolution fed to the engine's own resizing (scans are often 200-300 DPI)."""
    import pypdfium2 as pdfium
    cache = PT.default_cache(pdfs)
    names = PT.bwm_names()
    rnd = random.Random(seed)
    man = PT.read_manifest(cache)                      # the truth must be a TEXT LAYER, never earlier OCR
    shas = sorted(s for s in names if PT.cache_file(cache, s).is_file()
                  and "+ocr-" not in (man.get(s) or {}).get("extractor", ""))
    rnd.shuffle(shas)
    picks = []
    for sha in shas:
        pages = PT.cache_file(cache, sha).read_text(encoding="utf-8").split(PT.FF)
        good = [i for i, p in enumerate(pages, 1) if len(p) >= 1500 and not needs_ocr(p)]
        if good:
            picks.append((sha, rnd.choice(good)))
        if len(picks) >= n:
            break
    res = {e: [] for e in engines}
    for e in engines:
        eng = Engine(e)
        for sha, pg in picks:
            f = Path(pdfs) / names[sha]["file"]
            doc = pdfium.PdfDocument(str(f))
            truth = PT.cache_file(cache, sha).read_text(encoding="utf-8").split(PT.FF)[pg - 1]
            img = doc[pg - 1].render(scale=dpi / 72).to_pil()
            if e == "lighton":
                img.thumbnail((2048, 2048))
            t0 = time.time()
            txt = eng(img)
            s = ocr_scores(txt, truth)
            s.update(key=names[sha]["key"], page=pg, seconds=round(time.time() - t0, 1), looped=looped(txt))
            res[e].append(s)
            doc.close()
            say(f"  {e} {names[sha]['key']} p.{pg}: CER {s['cer']:.3f} WER {s['wer']:.3f} unsupported "
                f"{s['unsupported']:.3f} ({s['seconds']} s)")
        del eng
    summ = {}
    for e, rows in res.items():
        k = len(rows) or 1
        summ[e] = {m: round(sum(r[m] for r in rows) / k, 4) for m in ("cer", "wer", "bow_f1", "misread",
                                                                           "unsupported", "seconds")}
        summ[e]["median_cer"] = sorted(r["cer"] for r in rows)[len(rows) // 2] if rows else None
        summ[e]["looped"] = sum(r["looped"] for r in rows)
    return {"pages": len(picks), "dpi": dpi, "summary": summ, "rows": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["need", "run", "bench", "agree"])
    ap.add_argument("--bwm", action="store_true")
    ap.add_argument("--engine", default=PT.OCR_ENGINES[0], choices=["lighton", "rapid"])   # the MEASURED choice
    ap.add_argument("--engines", default="rapid,lighton", help="bench: comma list")
    ap.add_argument("--pdfs", type=Path, default=PT.DEFAULT_PDFS)
    ap.add_argument("--store", type=Path, default=None, help="OCR cache (default <pdfs>_ocr)")
    ap.add_argument("--keys", default="")
    ap.add_argument("--pdf", type=Path, default=None, help="run: one PDF (any folder), every page")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None, help="bench / agree: result JSON")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    if a.pdf is None and not Path(a.pdfs).is_dir():    # review 2026-10-14: an absent folder reported '0 PDFs'
        raise SystemExit(f"no papers folder at {a.pdfs} -- pass --pdfs or set SVFR_BWM_PAPERS")
    if a.pdf is None and not (PT.default_cache(a.pdfs) / PT.MANIFEST).is_file():   # else 'need' said '0 PDFs'
        raise SystemExit(f"no text cache beside {a.pdfs} -- run `python tools/paper_text.py build --pdfs {a.pdfs}`")
    store = a.store or default_store(a.pdfs)
    if a.cmd == "need":
        rows = need(a.pdfs, store=store)
        print(f"{len(rows)} held PDFs have pages the text layer cannot give ({sum(r['need'] for r in rows)} pages)")
        return 0
    if a.cmd == "run":
        keys = [k for k in a.keys.split(",") if k] or None
        print(json.dumps(run(a.pdfs, store, a.engine, keys, a.pdf, all_pages=a.pdf is not None)))
        return 0
    if a.cmd == "agree":
        rows = agreement(store)
        f1 = sorted(r["agree_f1"] for r in rows)
        print(f"{len(rows)} pages read by both engines; word F1 between them: median "
              f"{f1[len(f1) // 2] if f1 else '-'}, {sum(1 for x in f1 if x < 0.8)} pages below 0.8")
        for r in sorted(rows, key=lambda r: r["agree_f1"])[:10]:
            print(f"  {r['agree_f1']:.2f}  {r['key']} p.{r['page']}  words {r['words'][0]} vs {r['words'][1]}, "
                  f"lighton far from rapid {r['unsupported_vs_other']:.1%}{'  LOOPED' if r['looped'] else ''}")
        if a.out:
            a.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return 0
    res = bench(a.pdfs, [e for e in a.engines.split(",") if e], a.n, a.dpi)
    print(json.dumps(res["summary"], indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
