"""paper_text -- the FULL TEXT of every paper PDF, extracted once into a content-addressed cache, and a grep over it.

    python tools/paper_text.py build --pdfs C:/svfr/papers/boat_wm                 # extract what is new (parallel)
    python tools/paper_text.py grep "MPPI|path integral" --pdfs C:/svfr/papers/boat_wm --bwm
    python tools/paper_text.py grep "side-?scan" -i --topic sonar_nav --max 3 --bwm  # filter by the bwm index
    python tools/paper_text.py count "BlueBoat" --bwm                              # papers ranked by hits
    python tools/paper_text.py stats --pdfs C:/svfr/papers/boat_wm

WHY (2026-10-12, owner: "make / upgrade existing extraction tools to best grep info from the large number of
papers"). With ~1,000 full texts on disk, reading PDFs -- or feeding them to a model -- is the expensive way to
answer "which papers run MPPI on a boat?". Extract every PDF ONCE, keep the text as plain files, and grep them in
seconds; every hit names the paper and the PAGE, so a person can open the PDF at the right place.

CACHE. `<cache>/<sha256[:2]>/<sha256>.txt`, pages separated by a form feed (\\f), keyed by the PDF's CONTENT: a
renamed file is never re-read, a changed one is. `<cache>/manifest.tsv` = sha256, pages, chars, extractor, file
(the last name it was seen under). Default cache: `<pdfs>_text` beside the PDF folder -- outside it, so the lab
library's ingest of the papers never copies a cache. Extractor: pypdfium2 (Apache-2.0 / BSD-3, pulled in by the
declared pdfplumber; ~20x faster than pdfminer), pdfminer.six as the fallback; worker processes in parallel.

NAMES. `--bwm` resolves each PDF to its BlueBoat-world-model index paper (key, title, year, topics, relevance) through
`report/boat_world_model/papers/_held.json` -- by sha256, so it survives renames -- and allows `--topic` /
`--relevance` / `--key` filters; without it a hit is named by its file.

HONEST SCOPE: text extraction loses tables, equations and figure text, and a scanned PDF has no text (0 chars in the
manifest -- `stats` lists them). A regex over extracted text MISSES what extraction mangled (ligatures are folded,
hyphenated line breaks re-joined, but a column split mid-sentence still breaks a phrase).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "manifest.tsv"
FF = "\f"
#: bump when the text cleaning changes: a text cached by an older version is re-extracted
VERSION = "2"
try:
    import data_roots as _data_roots                  # run from tools/ (python tools/<tool>.py, the tests)
    import atlas_project as _AP
except ImportError:                                   # pragma: no cover - imported as tools.paper_text
    from tools import data_roots as _data_roots
    from tools import atlas_project as _AP
if str(ROOT) not in sys.path:                         # run by path: the repo root, for tools/
    sys.path.insert(0, str(ROOT))
from tools.atomic_replace import replace_retry  # noqa: E402
#: the atlas project every paper tool's `--bwm` means (tools/atlas_project.py: $ATLAS_PROJECT, else the nearest
#: atlas.toml, else the BlueBoat world-model atlas, report/boat_world_model)
BWM = _AP.project_dir()
#: its papers folder (outside git); every paper tool's --pdfs default: $ATLAS_PAPERS, else atlas.toml [paths] papers;
#: for the BlueBoat atlas $SVFR_BWM_PAPERS, else svfr/papers/boat_wm on the first local drive that has it (the
#: laptop's C:, the lab's D: -- as tools/paper_drop does)
DEFAULT_PDFS = _AP.papers_dir(BWM, legacy=lambda: Path(os.environ.get("SVFR_BWM_PAPERS")
                                                       or _data_roots.locate("C:/svfr/papers/boat_wm")))


# ------------------------------------------------------------------------------------------------ extraction
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_page(t: str) -> str:
    """One page's text: line endings normalised, form feeds removed (they separate pages), ligature glyphs folded
    (pdf text carries the fi / fl / ffl ligatures as single characters, which a search for 'offline' would miss)."""
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n").replace(FF, " ")
    t = re.sub("[" + chr(0xFFFE) + chr(0xAD) + "]\\s*", "", t)   # pdfium's line-end hyphenation mark, soft hyphens
    return "".join(unicodedata.normalize("NFKC", c) if 0xFB00 <= ord(c) <= 0xFB06 else c for c in t)


def extract_pages(path) -> Tuple[List[str], str]:
    """PDF -> ([page text, ...], extractor). ([], 'failed: ...') when no extractor can read it."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        try:
            pages = []
            for i in range(len(pdf)):
                page = pdf[i]
                tp = page.get_textpage()
                pages.append(_clean_page(tp.get_text_range()))
                tp.close()
                page.close()
            return pages, f"pypdfium2/{VERSION}"
        finally:
            pdf.close()
    except Exception as e:                                       # noqa: BLE001 -- fall back, then report
        first = f"{type(e).__name__}: {str(e)[:80]}"
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(path)) or ""
        pages = text.split(FF)
        if len(pages) > 1 and not pages[-1].strip():
            pages = pages[:-1]                       # pdfminer ENDS every page with a form feed: no phantom last page
        return [_clean_page(p) for p in pages], f"pdfminer/{VERSION}"
    except Exception as e:                                       # noqa: BLE001
        return [], f"failed: {first}; pdfminer {type(e).__name__}"


def _work(path: str) -> Tuple[str, str, List[str], str]:
    sha = sha256_file(path)
    pages, how = extract_pages(path)
    return path, sha, pages, how


def cache_file(cache: Path, sha: str) -> Path:
    return Path(cache) / sha[:2] / f"{sha}.txt"


def read_manifest(cache: Path) -> Dict[str, dict]:
    f = Path(cache) / MANIFEST
    out: Dict[str, dict] = {}
    if f.is_file():
        for ln in f.read_text(encoding="utf-8").splitlines()[1:]:
            parts = ln.split("\t")
            if len(parts) >= 5:
                out[parts[0]] = {"sha256": parts[0], "pages": int(parts[1] or 0), "chars": int(parts[2] or 0),
                                 "extractor": parts[3], "file": parts[4],
                                 "bytes": int(parts[5]) if len(parts) > 5 and parts[5] else None,
                                 "mtime_ns": int(parts[6]) if len(parts) > 6 and parts[6] else None}
    return out


def write_manifest(cache: Path, rows: Dict[str, dict]) -> None:
    lines = ["sha256\tpages\tchars\textractor\tfile\tbytes\tmtime_ns"] + [
        f"{r['sha256']}\t{r['pages']}\t{r['chars']}\t{r['extractor']}\t{r['file']}\t{r.get('bytes') or ''}\t"
        f"{r.get('mtime_ns') or ''}" for r in sorted(rows.values(), key=lambda r: r["file"].lower())]
    tmp = Path(cache) / (MANIFEST + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    replace_retry(tmp, Path(cache) / MANIFEST)         # Windows: a reader holding the manifest open (WinError 5 / 32)


def pdfs_in(folder: Path) -> List[Path]:
    return sorted((f for f in Path(folder).iterdir() if f.is_file() and f.suffix.lower() == ".pdf"),
                  key=lambda f: f.name.lower())


def build(pdfs: Path, cache: Path, workers: int = 0, say=print, extract=None) -> dict:
    """Extract every PDF in `pdfs` whose content is not cached yet (or was cached by an older extractor VERSION);
    everything else is only re-listed. A file whose name, size and mtime match the manifest keeps its recorded sha256
    (not even hashed); a renamed or touched file is hashed, and only NEW content is read. `extract` (tests) replaces
    the worker."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    man = read_manifest(cache)
    last_seen = {r["file"]: r for r in man.values()}
    files = pdfs_in(pdfs)
    by_size: Dict[str, str] = {}
    todo, seen = [], 0
    for f in files:
        st = f.stat()
        r = last_seen.get(f.name)
        sha = r["sha256"] if r and r.get("bytes") == st.st_size and r.get("mtime_ns") == st.st_mtime_ns else \
            sha256_file(f)
        row = man.get(sha)
        if row and row["extractor"].endswith("/" + VERSION) and cache_file(cache, sha).exists():
            row.update(file=f.name, bytes=st.st_size, mtime_ns=st.st_mtime_ns)    # known content: listed, not read
            by_size[f.name] = sha
            seen += 1
        else:
            todo.append(str(f))
    if todo:
        say(f"  {len(todo)} of {len(files)} PDFs to read")
    results = []
    work = extract or _work
    if workers == 1 or extract is not None or len(todo) < 4:
        results = [work(p) for p in todo]
    else:
        with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 2)) as ex:
            for i, r in enumerate(ex.map(work, todo, chunksize=2), 1):
                results.append(r)
                if i % 100 == 0:
                    say(f"  read {i}/{len(todo)} PDFs")
    new = 0
    for path, sha, pages, how in results:
        cf = cache_file(cache, sha)
        stale = man.get(sha, {}).get("extractor", "") != how
        if pages and (stale or not cf.exists()):
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_name(cf.name + ".tmp")
            tmp.write_text(FF.join(pages), encoding="utf-8", newline="\n")
            # a RECORD, never dropped like a cache: the manifest row below names the extractor that wrote THIS text,
            # and a text the manifest calls current is never re-extracted
            replace_retry(tmp, cf)
            new += 1
        st = Path(path).stat()
        man[sha] = {"sha256": sha, "pages": len(pages), "chars": sum(len(p) for p in pages), "extractor": how,
                    "file": Path(path).name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        by_size[Path(path).name] = sha
        seen += 1
    ocr = merge_ocr(cache, man, ocr_store(pdfs), pdfs, work=extract)
    write_manifest(cache, man)
    empty = sorted((n for n, sha in by_size.items() if man[sha]["chars"] < 200), key=str.lower)
    return {"pdfs": seen, "new_texts": new, "empty": empty, "files": by_size, "ocr_merged": ocr}


#: fewer text-layer characters than this on a page: the page needs OCR (tools/paper_ocr.py)
OCR_MIN_CHARS = 80
# preference order when a PDF was OCR'd by both. MEASURED 2026-10-14 on 30 born-digital pages (paper_ocr bench, paired):
# LightOnOCR-3 was no more accurate than RapidOCR (CER +0.011, CI -0.026..+0.048), wrote MORE words that are on no
# page (2.3% vs 0.9%, CI of the gap +0.15..+2.8 points) and took 7.6x as long -- the engine that cannot invent text
# comes first; LightOn fills a page RapidOCR botched
OCR_ENGINES = ("rapid", "lighton")


#: download stamps a library or publisher writes into the text layer of an otherwise SCANNED page -- not the page's
#: text (MEASURED 2026-10-14: page 1 of both held JSTOR scans, Efron 1979 and Wald 1945, held only the JSTOR banner, so
#: their title pages were never OCR'd and `confirm` could not find their titles)
_STAMPS = re.compile(                                   # anchored at a line / sentence start: unanchored, the
    r"(?<![^\n.])[^\n.]*?collaborating\s+with\s+JSTOR\s+to\s+digitize.*?www\.jstor\.org\W*"   # leading run was QUADRATIC
    # on long lines without '.' (fix check 2026-10-14: `need` 9 s -> 78 s on the corpus)
    r"|This content downloaded from[^\n]*(?:\n\s*All use subject to[^\n]*)?"
    r"|Authorized licensed use limited to:.*?Restrictions apply\.?", re.I | re.S)


#: bump when needs_ocr's RULE -- or which OCR reading counts (`usable_ocr`) -- changes: it is part of every OCR merge
#: signature, so a text merged under the old rule is merged again (review 2026-10-14: a page whose OCR was stored but
#: never merged stayed un-merged forever). "3" (2026-10-15): a reading with no letters of its own is not usable
NEEDS_OCR_RULE = "3"


_STAMP_WORDS = ("jstor", "this content downloaded from", "authorized licensed use")


def _unstamped(t: str) -> str:
    """The text without its download stamps -- the regex runs only on a page that names one (cheap)."""
    low = (t or "").lower()
    return _STAMPS.sub(" ", t) if any(w in low for w in _STAMP_WORDS) else (t or "")


def stamp_only(page_text: str) -> bool:
    """A page with NO text of its own: an empty text layer, or one holding nothing but a download stamp."""
    t = (page_text or "").strip()
    return not t or (_unstamped(t) != t and len(_unstamped(t).strip()) < OCR_MIN_CHARS)


def _letters(s: str) -> int:
    return sum(c.isalnum() for c in _unstamped(s or ""))


def usable_ocr(pg: dict) -> bool:
    """An OCR reading of a page that counts: text with letters of its own (not only noise or the download stamp read
    off the image), not looped, not truncated. The ONE rule for the merge and for `paper_ocr need` (third fix check
    2026-10-15: '| . -' counted as read, so the other engine was never offered the page, and the merge dropped it)."""
    t = pg.get("text") or ""
    return bool(t.strip()) and not pg.get("looped") and not pg.get("truncated") and _letters(t) > 0


def _richer(ocr_text: str, layer: str) -> bool:
    """OCR replaces a page's text layer only where it says clearly MORE: anything at all where the layer has no letters
    of its own, else 20 % and 20 letters more (review 2026-10-14: a short, correct half-title became OCR misreads)."""
    have = _letters(layer)
    return _letters(ocr_text) > (0 if have == 0 else 1.2 * have + 20)


def needs_ocr(page_text: str) -> bool:
    """A page whose text layer cannot be used: (almost) empty once download stamps are set aside (_STAMPS), or more
    than 30% replacement / private-use glyphs (at 5% the review found 15 held pages that are 84-95% good text -- a text
    layer beats OCR there)."""
    t = _unstamped(page_text or "").strip()
    if len(t) < OCR_MIN_CHARS:
        return True
    bad = sum(1 for c in t if c == "\ufffd" or "\ue000" <= c <= "\uf8ff")
    return bad / len(t) > 0.30


def ocr_store(pdfs: Path) -> Path:
    p = Path(pdfs)
    return p.parent / (p.name + "_ocr")


def ocr_inputs(store: Path, sha: str) -> Tuple[str, Dict[int, Tuple[str, str]]]:
    """(signature, {page: (engine, text)}) of every OCR made for this content: per page the FIRST engine in
    OCR_ENGINES with usable text (`usable_ocr`: letters of its own, not looped or truncated) -- a page one engine
    botched falls back to the other. The signature hashes the store files (engine, size, mtime), so new OCR for a paper re-merges it. An
    unreadable store file (a half-written one) is skipped, never fatal."""
    pages: Dict[int, Tuple[str, str]] = {}
    sig = []
    for e in OCR_ENGINES:
        f = Path(store) / e / sha[:2] / f"{sha}.json"
        if not f.is_file():
            continue
        st = f.stat()
        sig.append(f"{e}:{st.st_size}:{st.st_mtime_ns}")
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for x in d.get("pages") or []:
            n, t = int(x.get("page") or 0), x.get("text") or ""
            if n and usable_ocr(x) and n not in pages:
                pages[n] = (e, t)
    if sig:
        sig.append(f"rule:{NEEDS_OCR_RULE}")
    return (hashlib.sha256("|".join(sig).encode()).hexdigest()[:8] if sig else ""), pages


def ocr_hits(layer: List[str], ocr: Dict[int, Tuple[str, str]]) -> List[int]:
    """The pages whose OCR goes into the text: a page the layer cannot give where the OCR says clearly more, or one
    already holding exactly its OCR text. Pure; merge_ocr's rule, shared with `ocr_merge_pending`."""
    return [n for n in sorted(ocr) if 1 <= n <= len(layer)
            and ((needs_ocr(layer[n - 1]) and _richer(ocr[n][1], layer[n - 1]))
                 or layer[n - 1] == _clean_page(ocr[n][1]))]


def ocr_merge_pending(cache: Path, row: dict, store: Path) -> bool:
    """True when the next `build` would merge OCR into this paper's text (merge_ocr's test, without the merge): OCR is
    stored for it, and the text was merged from OTHER OCR inputs, or never merged though some page would take it.
    (tools/atlas_coverage.py: OCR read but not in the text counts as not done.)"""
    sha = row["sha256"]
    cf = cache_file(cache, sha)
    if not cf.is_file():
        return False
    sig, ocr = ocr_inputs(store, sha)
    if not ocr:
        return False
    m = re.search(r"\+ocr-[a-z-]+\.([0-9a-f]+)/", row["extractor"])
    if m:
        return m.group(1) != sig
    return bool(ocr_hits(cf.read_text(encoding="utf-8").split(FF), ocr))


def merge_ocr(cache: Path, man: Dict[str, dict], store: Path, pdfs: Path, work=None) -> int:
    """Fill the pages a text layer cannot give with cached OCR, ONLY those pages. The extractor becomes
    'pypdfium2+ocr-<engines>.<signature>/<VERSION>': the FTS index re-reads the paper, a later build skips it while
    its OCR inputs are unchanged, and NEW OCR (a better engine later) re-merges it from a fresh text-layer extraction.
    The manifest is saved after every merge, so a crash cannot leave merged text under an un-merged label."""
    merged = 0
    if not Path(store).is_dir():
        return 0
    for sha, row in man.items():
        cf = cache_file(cache, sha)
        if not cf.is_file():
            continue
        sig, ocr = ocr_inputs(store, sha)
        if not ocr:
            continue
        m = re.search(r"\+ocr-[a-z-]+\.([0-9a-f]+)/", row["extractor"])
        if m and m.group(1) == sig:
            continue                                     # merged from exactly these inputs
        if "+ocr-" in row["extractor"]:                  # merged from OLDER inputs: start again from the text layer
            f = Path(pdfs) / row["file"]
            if not f.is_file():
                continue
            _, got, layer, how = (work or _work)(str(f))
            if got != sha or not layer:                  # the name now holds OTHER content (fix check 2026-10-14:
                continue                                 # this text was overwritten with the new file's text)
        else:
            layer, how = cf.read_text(encoding="utf-8").split(FF), row["extractor"]
        # a page already holding exactly its OCR text counts too: a build that died between writing the merged text and
        # writing the manifest is re-labelled here, so the index re-reads it (fix check 2026-10-14)
        # OCR replaces a text layer only where it is clearly RICHER: a short but correct page (a half-title) is never
        # swapped for OCR misreads of itself (review 2026-10-14)
        hit = ocr_hits(layer, ocr)
        if not hit and "+ocr-" not in row["extractor"]:
            continue
        for n in hit:
            layer[n - 1] = _clean_page(ocr[n][1])
        tmp = cf.with_name(cf.name + ".tmp")
        tmp.write_text(FF.join(layer), encoding="utf-8", newline="\n")
        replace_retry(tmp, cf)                           # a RECORD: the manifest row below says it holds the OCR
        head, _, ver = how.rpartition("/")
        base = re.sub(r"\+ocr-.*$", "", head or how)
        engines = "-".join(sorted({ocr[n][0] for n in hit}))
        row["extractor"] = f"{base}+ocr-{engines}.{sig}/{ver or VERSION}" if hit else f"{base}/{ver or VERSION}"
        row["chars"] = sum(len(x) for x in layer)
        write_manifest(cache, man)
        merged += 1
    return merged


def _quick_sha_map(pdfs: Path, cache: Path) -> Dict[str, str]:
    """file name -> sha256 from the manifest (the last name each text was seen under)."""
    return {r["file"]: sha for sha, r in read_manifest(cache).items()}


# ----------------------------------------------------------------------------------------------- the corpus
def default_cache(pdfs: Path) -> Path:
    p = Path(pdfs)
    return p.parent / (p.name + "_text")


def bwm_names(base: Optional[Path] = None) -> Dict[str, dict]:
    """sha256 -> {key, title, year, topics, relevance, file} from an atlas project's index (held PDFs only): `base`,
    default the BlueBoat world-model one (BWM)."""
    b = Path(base) if base is not None else BWM
    held = json.loads((b / "papers" / "_held.json").read_text(encoding="utf-8"))
    papers = {p["key"]: p for p in json.loads((b / "index" / "papers.json").read_text(encoding="utf-8"))["papers"]}
    out = {}
    for r in held.get("files") or []:
        p = papers.get(r["key"], {})
        out[r["sha256"]] = {"key": r["key"], "title": p.get("title", ""), "year": p.get("year"),
                            "topics": p.get("topics", []), "relevance": p.get("relevance", ""), "file": r["file"]}
    return out


def corpus(cache: Path, names: Optional[Dict[str, dict]] = None) -> Iterable[Tuple[dict, List[str]]]:
    """(paper info, [page texts]) for every cached text; info from `names` (by sha) else the manifest's file name."""
    man = read_manifest(cache)
    for sha, row in sorted(man.items(), key=lambda kv: kv[1]["file"].lower()):
        cf = cache_file(cache, sha)
        if not cf.is_file() or (names is not None and sha not in names):   # --bwm: index papers only (a text the
            continue                                     # cache keeps for a file no longer held is not one -- review)
        info = dict((names or {}).get(sha) or {"key": Path(row["file"]).stem, "title": "", "year": None,
                                                 "topics": [], "relevance": "", "file": row["file"]})
        info["sha256"] = sha
        yield info, cf.read_text(encoding="utf-8").split(FF)


def _flat_ws(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*(?=[a-z])", "", s))


def search(cache: Path, pattern: str, ignore_case: bool = False, names=None, keep=None, window: int = 110,
           max_per_paper: int = 5) -> List[dict]:
    """Every paper whose text matches `pattern` (a Python regex): [{paper info, hits: [{page, snippet}], n}],
    most hits first. Hyphenated line breaks are re-joined and whitespace collapsed before matching, so a phrase
    broken across lines still matches. `keep(info)` filters papers."""
    rx = re.compile(pattern, re.I if ignore_case else 0)
    out = []
    for info, pages in corpus(cache, names):
        if keep and not keep(info):
            continue
        hits, n = [], 0
        for i, page in enumerate(pages, 1):
            text = _flat_ws(page)
            for m in rx.finditer(text):
                n += 1
                if len(hits) < max_per_paper:
                    a, b = max(0, m.start() - window), min(len(text), m.end() + window)
                    hits.append({"page": i, "snippet": ("..." if a else "") + text[a:m.start()] + "[[" +
                                 m.group(0) + "]]" + text[m.end():b] + ("..." if b < len(text) else "")})
        if n:
            out.append({**info, "n": n, "hits": hits})
    out.sort(key=lambda r: (-r["n"], (r.get("key") or "")))
    return out


# ------------------------------------------------------------------------------------------ ranked search (BM25)
#: The FTS5 index beside the text cache. Keyed by sha256 like the cache: a renamed PDF is never re-indexed, only NEW
#: content is added. Porter stemming + unicode61, so 'models' finds 'model'. The review that chose it:
#: report/boat_world_model/LIT_PAPER_MINING.md (BM25 is the strong baseline on scientific text; MEASURED here in sec. 2.1).
INDEX = "search.db"
STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were with which "
                 "into than then via using used use can may each per other not over under between also more most such "
                 "these those their there they when where while both all any how what who whom whose".split())


def index(cache: Path, say=print) -> Path:
    """Bring the FTS5 page index up to date with the text cache: add texts it lacks, RE-INDEX a text the cache rewrote
    (a new extractor VERSION keeps the sha but changes the text -- found by review 2026-10-13: search kept quoting the
    old text while grep read the new), drop texts no longer in the manifest. Idempotent. The write takes the lock first
    and re-reads what is indexed inside it, so two concurrent searches never both write (the loser waits, then skips)."""
    import sqlite3
    db = Path(cache) / INDEX
    con = sqlite3.connect(db, timeout=120)
    con.execute("create virtual table if not exists pages using fts5(sha UNINDEXED, page UNINDEXED, body, "
                "tokenize='porter unicode61')")
    con.execute("create table if not exists indexed(sha text primary key, extractor text)")
    cols = [r[1] for r in con.execute("pragma table_info(indexed)")]
    if "extractor" not in cols:                       # an index written before 2026-10-13: rebuild what it holds
        con.execute("alter table indexed add column extractor text")
    con.commit()
    man = read_manifest(cache)
    want = {sha: r["extractor"] for sha, r in man.items() if cache_file(cache, sha).is_file()}
    have = dict(con.execute("select sha, extractor from indexed"))
    if all(have.get(sha) == ex for sha, ex in want.items()) and set(have) <= set(want):
        con.close()
        return db                                     # current: no write, no lock
    con.execute("begin immediate")                    # the write lock, then re-read under it
    have = dict(con.execute("select sha, extractor from indexed"))
    gone = [sha for sha in have if sha not in want]
    todo = [sha for sha, ex in want.items() if have.get(sha) != ex]
    drop = gone + [x for x in todo if x in have]
    if drop:                                          # ONE pass: sha is UNINDEXED, so each `where sha = ?` scanned every
        con.execute("create temp table if not exists _drop(sha text primary key)")    # page (MEASURED 2026-10-13: 200
        con.executemany("insert or ignore into _drop values (?)", [(s,) for s in drop])   # of 400 texts 13.2 s -> 2.2 s;
        con.execute("delete from pages where sha in (select sha from _drop)")          # the first full re-index spent
        con.execute("delete from indexed where sha in (select sha from _drop)")        # ~13 min deleting, one by one)
    for i, sha in enumerate(todo, 1):
        for n, pg in enumerate(cache_file(cache, sha).read_text(encoding="utf-8").split(FF), 1):
            if pg.strip():
                con.execute("insert into pages values (?,?,?)", (sha, n, _flat_ws(pg)))
        con.execute("insert into indexed values (?,?)", (sha, want[sha]))
        if i % 200 == 0:
            say(f"  indexed {i}/{len(todo)} texts")
    con.commit()
    con.close()
    return db


def fts_query(text: str, phrase: bool = False) -> str:
    """Plain words -> an FTS5 query: the content words OR'ed (BM25 ranks pages with more, rarer of them first), or ONE
    exact phrase -- with every word kept, stop words included ('structure from motion' must stay adjacent tokens; the
    first version dropped 'from' and the phrase could never match). Words are Unicode letters / digits ('naive',
    'Muller' with their diacritics; FTS5 folds them); quoting keeps every token literal, so no operator can be injected."""
    toks = re.findall(r"[^\W_]+", text or "")
    if phrase:
        return ('"' + " ".join(toks) + '"') if toks else ""
    words = [w for w in toks if w.lower() not in STOP]
    return " OR ".join('"' + w + '"' for w in dict.fromkeys(words)) if words else ""


def rank(cache: Path, query: str, names=None, keep=None, limit: int = 50, pages_per_paper: int = 3,
         phrase: bool = False) -> List[dict]:
    """Papers ranked by BM25 (a paper scores its best page), each with its best pages and highlighted snippets. Papers
    are ranked in SQL over EVERY matching page (no page cut: a filtered search must still find a paper whose best page is
    far down), the --key / --topic filters apply to papers, and snippets are fetched only for the papers shown."""
    import sqlite3
    q = fts_query(query, phrase)
    if not q:
        return []
    db = index(cache, say=lambda *_: None)
    con = sqlite3.connect(db, timeout=120)
    man = read_manifest(cache)
    ranked = con.execute("with hits as materialized (select sha, bm25(pages) as s from pages where pages match ?) "
                         "select sha, min(s) as best from hits group by sha order by best", (q,)).fetchall()
    out = []
    for sha, best in ranked:
        if names is not None and sha not in names:       # --bwm: a text no longer held is not an index paper
            continue
        info = dict((names or {}).get(sha) or {"key": Path(man.get(sha, {}).get("file", sha)).stem, "title": "",
                                                "year": None, "topics": [], "relevance": "",
                                                "file": man.get(sha, {}).get("file", "")})
        if keep and not keep(info):
            continue
        hits = con.execute("with p as materialized (select page, bm25(pages) as s, snippet(pages, 2, '[[', ']]', '...', "
                           "24) as snip from pages where pages match ? and sha = ?) select page, s, snip from p "
                           "order by s limit ?", (q, sha, pages_per_paper)).fetchall()
        out.append({**info, "sha256": sha, "score": -best,
                    "hits": [{"page": pg, "score": round(-sc, 3), "snippet": sn} for pg, sc, sn in hits]})
        if len(out) >= limit:
            break
    con.close()
    return out


BENCH_METRICS = ("R@10", "R@50", "MRR", "Hit@10", "Hit@50")


def bench_queries(n_terms: int = 0, seed: int = 13) -> List[dict]:
    """The lexicon-citation benchmark's queries: [{term, names, masked, rel}] for every lexicon term whose verified
    citations point at a HELD paper (rel = those papers' keys; names = the term and its aliases; masked = its definition
    with every name removed). With n_terms: a seeded sample, so every ranker scores the SAME terms (paired)."""
    held = {n["key"] for n in bwm_names().values()}
    alias = json.loads((BWM / "index" / "papers.json").read_text(encoding="utf-8"))["aliases"]
    lex = json.loads((BWM / "index" / "lexicon.json").read_text(encoding="utf-8"))["terms"]
    rel_of = lambda t: {alias.get(c["key"], c["key"]) for c in t["citations"]} & held
    if n_terms:
        import random
        lex = [t for t in lex if rel_of(t)]
        lex = random.Random(seed).sample(lex, min(n_terms, len(lex)))
    out = []
    for t in lex:
        rel = rel_of(t)
        if not rel:
            continue
        names_t = [t["term"]] + list(t.get("aliases") or [])
        masked = t["definition"]
        for n in names_t:
            masked = re.sub(re.escape(n), " ", masked, flags=re.I)
        out.append({"term": t["term"], "names": names_t, "masked": masked, "rel": sorted(rel)})
    return out


def bench_score(ranked: List[str], rel) -> Tuple[float, ...]:
    """One query: (R@10, R@50, MRR, Hit@10, Hit@50) of a ranked list of paper keys against the relevant set."""
    rel = set(rel)
    mrr = next((1 / (i + 1) for i, k in enumerate(ranked) if k in rel), 0.0)
    return (len(set(ranked[:10]) & rel) / len(rel), len(set(ranked[:50]) & rel) / len(rel), mrr,
            float(bool(set(ranked[:10]) & rel)), float(bool(set(ranked[:50]) & rel)))


def bench_summary(res: Dict[str, list]) -> dict:
    """{ranker: [per-query score tuples]} -> {"terms": n, ranker: {metric: macro mean}}."""
    out = {"terms": len(next(iter(res.values()), []))}
    for k, v in res.items():
        out[k] = {m: round(sum(x[i] for x in v) / max(1, len(v)), 3) for i, m in enumerate(BENCH_METRICS)}
    return out


def bm25_term_query(names_t: List[str]) -> str:
    """The term query: the term and each alias as an exact phrase, OR'ed."""
    return " OR ".join(q for q in (fts_query(n, phrase=True) for n in names_t) if q)


def bench(cache: Path, with_regex: bool = False, say=print, n_terms: int = 0, seed: int = 13) -> dict:
    """The lexicon-citation retrieval benchmark (--bwm): for each lexicon term, the HELD papers its verified citations
    come from are the known-relevant set. Rankers: BM25 on the term (+ aliases, as phrases), BM25 on its DEFINITION
    with the term masked (a meaning-level query), optionally the regex count ranking the grep tool uses (slow:
    one pass over the corpus per term). Macro R@10 / R@50 (the share of a term's cited papers in the top k), MRR, and
    Hit@10 / Hit@50 (the share of terms with at least one cited paper in the top k). Labels are incomplete (1-2 cited
    papers per term), so absolute values are lower bounds; the comparison between rankers is the point.
    `tools/paper_embed.py bench` scores dense / hybrid / reranked rankers on the same queries (bench_queries)."""
    names = bwm_names()
    held = {n["key"]: sha for sha, n in names.items()}
    qs = bench_queries(n_terms, seed)
    index(cache, say=say)
    texts = {}
    if with_regex:                                    # the same text grep matches: pages with hyphen breaks re-joined
        texts = {k: " ".join(_flat_ws(pg) for pg in cache_file(cache, sha).read_text(encoding="utf-8").split(FF)).lower()
                 for k, sha in held.items() if cache_file(cache, sha).is_file()}

    def keys_of(ranked):
        return [r["key"] for r in ranked]
    res: Dict[str, list] = {"bm25-term": [], "bm25-definition": []}
    if with_regex:
        res["regex-count"] = []
    for q in qs:
        rel = set(q["rel"])
        res["bm25-term"].append(bench_score(keys_of(_rank_q(cache, bm25_term_query(q["names"]), names)), rel))
        res["bm25-definition"].append(bench_score(keys_of(_rank_q(cache, fts_query(q["masked"]), names)), rel))
        if with_regex:                               # the grep tool's ranking: occurrences of the term / an alias
            subs = [n.lower() for n in q["names"] if n.strip()]
            pats = [re.compile(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])") for n in subs]
            counts = []
            for k, x in texts.items():               # substring prefilter first (C speed): the same ranking, ~100x
                if any(n in x for n in subs):         # faster than a regex pass over all 118 M characters per term
                    c = sum(len(p.findall(x)) for p in pats)
                    if c:
                        counts.append((c, k))
            counts.sort(key=lambda c: (-c[0], c[1]))
            res["regex-count"].append(bench_score([k for c, k in counts], rel))
    return bench_summary(res)


def _rank_q(cache: Path, fts: str, names, limit: int = 2000) -> List[dict]:
    """Rank papers for a ready FTS5 query string (bench uses raw OR-of-phrases)."""
    import sqlite3
    if not fts:
        return []
    con = sqlite3.connect(Path(cache) / INDEX, timeout=120)
    # bm25() cannot sit inside an aggregate ('unable to use function bm25 in the requested context' -- MEASURED
    # 2026-10-13: a GROUP BY version scored every query 0.0 behind a swallowed error); score pages first.
    # No except: every query here is quoted, so an error is a bug and must surface.
    rows = con.execute("with hits as materialized (select sha, bm25(pages) as s from pages where pages match ?) "
                       "select sha, min(s) as best from hits group by sha order by best limit ?",
                       (fts, limit)).fetchall()             # MATERIALIZED: else SQLite flattens bm25 into the aggregate
    con.close()
    return [{"key": (names or {}).get(sha, {}).get("key", sha)} for sha, _ in rows
            if names is None or sha in names]


def _keep_fn(a, names) -> Optional[callable]:
    keys = set(a.key or [])
    if not (keys or a.topic or a.relevance or a.year_from):
        return None
    rel_rank = {"core": 0, "high": 1, "medium": 2, "low": 3}

    def keep(info):
        if keys and info.get("key") not in keys:
            return False
        if a.topic and not any(a.topic in t for t in info.get("topics") or []):
            return False
        if a.relevance and rel_rank.get(info.get("relevance"), 9) > rel_rank.get(a.relevance, 9):
            return False
        if a.year_from and (info.get("year") or 0) < a.year_from:
            return False
        return True
    return keep


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["build", "grep", "count", "stats", "search", "bench"])
    ap.add_argument("pattern", nargs="?", default="",
                    help="grep / count: a Python regular expression; search: plain words (ranked by BM25)")
    ap.add_argument("--phrase", action="store_true", help="search: the words as ONE exact phrase")
    ap.add_argument("--with-regex", action="store_true", help="bench: also score the regex count ranking (slow)")
    ap.add_argument("--terms", type=int, default=0, help="bench: a seeded random sample of this many terms (paired)")
    ap.add_argument("--pdfs", type=Path, default=DEFAULT_PDFS, help="the folder of paper PDFs")
    ap.add_argument("--cache", type=Path, default=None, help="text cache (default: <pdfs>_text beside the folder)")
    ap.add_argument("--workers", type=int, default=0, help="build: worker processes (default: cores - 2)")
    ap.add_argument("-i", "--ignore-case", action="store_true")
    ap.add_argument("--bwm", action="store_true", help="name hits by the BlueBoat world-model index (and filter by it)")
    ap.add_argument("--key", action="append", help="only this index key (repeatable; needs --bwm)")
    ap.add_argument("--topic", default="", help="only papers of this bwm topic (substring; needs --bwm)")
    ap.add_argument("--relevance", choices=["core", "high", "medium", "low"], default=None,
                    help="only papers at least this relevant (needs --bwm)")
    ap.add_argument("--year-from", type=int, default=0)
    ap.add_argument("--max", type=int, default=3, help="grep: snippets shown per paper")
    ap.add_argument("--window", type=int, default=110, help="grep: characters of context either side")
    ap.add_argument("--limit", type=int, default=50, help="papers listed")
    ap.add_argument("--json", action="store_true", help="grep / count: machine-readable output")
    a = ap.parse_args(argv)
    cache = a.cache or default_cache(a.pdfs)
    if a.cmd == "build":
        if not a.pdfs.is_dir():
            print(f"no PDF folder at {a.pdfs}")
            return 2
        res = build(a.pdfs, cache, a.workers)
        print(f"text: {res['pdfs']} PDFs, {res['new_texts']} newly extracted -> {cache}; "
              f"{len(res['empty'])} with (almost) no text")
        for n in res["empty"][:20]:
            print(f"  no text: {n} (a scan? the matcher and grep cannot see it)")
        return 0
    if a.cmd == "stats":
        man = read_manifest(cache)
        pages = sum(r["pages"] for r in man.values())
        chars = sum(r["chars"] for r in man.values())
        by = {}
        for r in man.values():
            by[r["extractor"].split(":")[0]] = by.get(r["extractor"].split(":")[0], 0) + 1
        print(f"{len(man)} texts, {pages} pages, {chars / 1e6:.1f} M characters in {cache}; extractors {by}; "
              f"empty: {sum(r['chars'] < 200 for r in man.values())}")
        return 0
    if a.cmd == "bench":
        if not a.bwm:
            ap.error("bench needs --bwm (its relevance labels are the BlueBoat lexicon's citations)")
        res = bench(cache, a.with_regex, n_terms=a.terms)
        print(json.dumps(res, indent=1))
        return 0
    if a.cmd == "search":
        if not a.pattern:
            ap.error("search needs words")
        names = bwm_names() if a.bwm else None
        if (a.key or a.topic or a.relevance or a.year_from) and not a.bwm:
            ap.error("--key / --topic / --relevance / --year-from need --bwm")
        res = rank(cache, a.pattern, names, _keep_fn(a, names), a.limit, a.max, a.phrase)
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=1))
            return 0 if res else 1
        print(f"{len(res)} papers ranked by BM25 for: {a.pattern}" + (" (phrase)" if a.phrase else ""))
        for r in res:
            label = f"{r['key']} ({r.get('year') or 'n.d.'})" if names else r.get("file") or r["key"]
            print(f"\n{r['score']:6.2f}  {label}" + (f" -- {r['title'][:100]}" if r.get("title") else ""))
            for h in r["hits"]:
                print(f"        p.{h['page']}: {h['snippet']}")
        return 0 if res else 1
    if not a.pattern:
        ap.error("grep / count need a pattern")
    try:
        re.compile(a.pattern)
    except re.error as e:
        ap.error(f"bad regular expression: {e}")
    names = bwm_names() if a.bwm else None
    if (a.key or a.topic or a.relevance or a.year_from) and not a.bwm:
        ap.error("--key / --topic / --relevance / --year-from need --bwm")
    res = search(cache, a.pattern, a.ignore_case, names, _keep_fn(a, names), a.window, a.max)
    if a.json:
        print(json.dumps(res[:a.limit], ensure_ascii=False, indent=1))
        return 0 if res else 1
    total = sum(r["n"] for r in res)
    print(f"{len(res)} papers, {total} hits for /{a.pattern}/" + (" (case-insensitive)" if a.ignore_case else ""))
    for r in res[:a.limit]:
        label = f"{r['key']} ({r.get('year') or 'n.d.'})" if names else r["file"]
        print(f"\n{label}: {r['n']} hit(s)" + (f" -- {r['title'][:100]}" if r.get("title") else ""))
        if a.cmd == "grep":
            for h in r["hits"]:
                print(f"  p.{h['page']}: {h['snippet']}")
    if len(res) > a.limit:
        print(f"\n... {len(res) - a.limit} more papers (--limit)")
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
