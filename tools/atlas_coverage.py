r"""atlas_coverage -- PER PAPER, which held papers of an atlas were never completely feature-mined ON THIS MACHINE, and
which stage each one is missing (`python tools/atlas.py coverage`).

    python tools/atlas.py coverage                       # every atlas in this repository (report/*/atlas.toml)
    python tools/atlas.py coverage --project report/uf_atlas [--pdfs <its PDF folder>]
    python tools/atlas.py coverage --missing figures     # the paper keys missing that stage, with why
    python tools/atlas.py coverage --json                # everything, per paper and stage

EXIT 0 = nothing is missing, 1 = some stage of some paper is missing, 2 = an atlas could not be measured (its
papers/_held.json or _rejected.json unreadable, its engine not loadable).

WHY (owner, 2026-10-27, on the lab: "should we make a tool that is able to easily find what papers were never completely
feature mined?"). `atlas.py status` answers per STAGE -- ran / up to date -- and a stage that is "up to date" can still
have papers it never reached: the session building the Atlas Browser found by hand that the lab's uf_atlas figure bank
held images for 126 of the 708 figures its index lists, and that 158 held PDFs had never reached the library. Papers
move laptop -> lab with tools/paper_drop.py and their caches with tools/paper_caches.py, so each machine's caches are
partial in their own way: this says, per paper and machine, what is there.

THE STAGES -- each one's evidence is what ITS OWN TOOL counts as done (read from the tool, never re-guessed):
    pdf      a file in the PDF folder whose sha256 is the held row's (named in the text manifest with the same size and
             mtime: trusted, as `paper_text build` does; else hashed)
    text     a <pdfs>_text manifest row for that sha256 written by the CURRENT extractor version, and its text file
             (tools/paper_text.py::build)
    ocr      no page still offered to RapidOCR (tools/paper_ocr.py::needs_listing, the engine `atlas update` runs) and no
             OCR read but not yet merged into the text (tools/paper_text.py::ocr_merge_pending); n/a when no page needs it
    html     arXiv papers: the <pdfs>_html manifest says ok and parsed/<id>.json.gz holds the current parser's output
             (tools/arxiv_html.py::parse_all); n/a when arXiv has no HTML for it, or it is not an arXiv paper
    facts    a papers/_facts.json row for the paper, mined from this sha256 (tools/paper_facts.py::main_bwm; committed)
    embed    current vectors for the embedding model `atlas update` would use here (tools/paper_embed.py::is_current);
             with no torch here (update skips the stage), current vectors of any model already stored
    figures  <pdfs>_figures/<sha[:2]>/<sha>/figures.json and every image it lists -- and every image
             index/atlas_figures.js lists for the paper (tools/figure_bank.py::build_all, build_bwm.py figures_digest)
    meta     papers/_meta.json measured OpenAlex for it (build_bwm.py cmd_meta; committed); n/a when it has no DOI to
             ask with
A stage switched off in atlas.toml [pipeline] reads `off`, never `missing`. Held papers = papers/_held.json, keys
folded through the index's aliases, files _rejected.json says are not that paper left out (build_bwm.py::
figures_digest's rule); one row per held FILE.

WHERE (found as tools/paper_drop.py::atlas_corpus does): in this repository the BlueBoat atlas's PDFs are the working
corpus ($SVFR_BWM_PAPERS, else svfr/papers/boat_wm on the first local drive with it) and any other atlas's sit BESIDE
it, <corpus parent>/<short_name>; elsewhere (a kit) the project's own [paths] papers; $ATLAS_PAPERS or --pdfs win.

HONEST SCOPE: this reads the caches; it never opens a PDF except to hash one the text manifest does not vouch for. A
stage's evidence can be present and still wrong (a figure detector that missed figures; facts from a garbled text) --
that is each tool's own measured quality, not this check. facts and meta are COMMITTED files, so they read the same on
every machine; the rest are this machine's caches. The embedding model and the figure arm are what `atlas update`
would pick HERE (atlas.py::context: GPU, torch, doclayout_yolo); a bank built by another arm still counts as done.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas_project as AP  # noqa: E402
import paper_text as PT  # noqa: E402

STAGES = ("pdf", "text", "ocr", "html", "facts", "embed", "figures", "meta")
STATES = ("done", "missing", "n/a", "off")
#: the OCR engine `atlas update` runs (atlas.py STAGES "ocr": paper_ocr.py run --engine rapid)
OCR_ENGINE = "rapid"
SCHEMA = "svfr.atlas.coverage/1"


class CannotMeasure(RuntimeError):
    """An atlas whose record cannot be read: its coverage is unknown, not zero (exit 2)."""


# ------------------------------------------------------------------------------------------------- where
def in_repository(project: Path) -> bool:
    """An atlas project of THIS repository (report/<name>/), where the PDFs live outside the checkout."""
    try:
        return AP.BLUEBOAT.is_dir() and Path(project).resolve().parent == (AP.ROOT / "report").resolve()
    except OSError:
        return False


def working_corpus() -> Path:
    """The BlueBoat atlas's PDF folder on this machine: tools/paper_drop.py::corpus_dir (no --pdfs)."""
    v = os.environ.get("SVFR_BWM_PAPERS", "").strip()
    if v:
        return Path(v)
    from data_roots import locate
    return Path(locate("C:/svfr/papers/boat_wm"))


def corpus_of(project: Path, explicit: Optional[str] = None) -> Path:
    """The PDF folder of `project` on this machine: --pdfs, else $ATLAS_PAPERS, else (this repository)
    tools/paper_drop.py::atlas_corpus, else the project's [paths] papers."""
    if explicit:
        return Path(explicit).expanduser()
    v = os.environ.get(AP.ENV_PAPERS, "").strip()
    if v:
        return Path(v).expanduser()
    project = Path(project)
    if in_repository(project):
        if AP.is_blueboat(project):
            return working_corpus()
        return working_corpus().parent / AP.config(project)["atlas"]["short_name"]
    return AP.papers_dir(project)


def repository_atlases() -> List[Path]:
    """Every atlas project in this repository: report/*/atlas.toml."""
    return sorted(p.parent for p in (AP.ROOT / "report").glob("*/atlas.toml"))


# ------------------------------------------------------------------------------------------------- record
def _load(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def held_papers(base: Path, B) -> tuple:
    """([{key, sha256, file, arxiv_id, work_doi}], alias_map, notes) -- one row per held, not-rejected FILE, its key
    folded through the aliases (build_bwm.py::figures_digest's rule). Raises CannotMeasure on an unreadable record."""
    held_f = Path(base) / "papers" / "_held.json"
    if held_f.is_file():
        try:
            doc = json.loads(held_f.read_text(encoding="utf-8"))
        except ValueError as e:
            raise CannotMeasure(f"{held_f} is not valid JSON ({e})") from None
    else:
        doc = {}
    papers, alias_map, _ = B.build_papers(Path(base))
    rej, _, problems = B.rejected_ids(Path(base), alias_map)
    if any(p.startswith(B.UNREADABLE_REJECTED) for p in problems):
        raise CannotMeasure(f"{Path(base) / 'papers' / '_rejected.json'}: {problems[0]}")
    by_key = {p["key"]: p for p in papers}
    ML = B._sibling("meta_links")
    rows, seen, notes = [], set(), [f"_rejected.json: {p}" for p in problems]
    for r in doc.get("files") or []:
        if not (isinstance(r, dict) and r.get("key") and r.get("sha256")):
            continue
        k, sha = alias_map.get(r["key"], r["key"]), str(r["sha256"]).strip().lower()
        if k in rej.get(sha, ()) or sha in seen:
            continue
        seen.add(sha)
        p = by_key.get(k) or {}
        rows.append({"key": k, "sha256": sha, "file": str(r.get("file") or ""),
                     "arxiv_id": re.sub(r"v\d+$", "", str(p.get("arxiv_id") or "").strip()),
                     "work_doi": ML.work_doi(p) if p else ""})
    rows.sort(key=lambda r: (r["key"], r["sha256"]))
    return rows, alias_map, notes


def pdfs_here(pdfs: Path, manifest: Dict[str, dict], want: Sequence[dict]) -> tuple:
    """({sha256: file name} of the PDFs in `pdfs` the held rows want, number of files hashed). A file the text manifest
    names with the same size and mtime is that row's content (paper_text.build's rule); the rest are hashed only while
    some wanted sha256 is still unfound -- each row's own file first."""
    if not Path(pdfs).is_dir():
        return {}, 0
    by_name = {r["file"]: r for r in manifest.values()}
    wanted = {r["sha256"] for r in want}
    found: Dict[str, str] = {}
    unvouched = []
    for f in PT.pdfs_in(pdfs):
        st, r = f.stat(), by_name.get(f.name)
        if r and r.get("bytes") == st.st_size and r.get("mtime_ns") == st.st_mtime_ns:
            if r["sha256"] in wanted:
                found.setdefault(r["sha256"], f.name)
        else:
            unvouched.append(f)
    own = {r["file"] for r in want if r["sha256"] not in found}
    unvouched.sort(key=lambda f: (f.name not in own, f.name.lower()))
    hashed = 0
    for f in unvouched:
        if wanted <= set(found):
            break
        sha = PT.sha256_file(f)
        hashed += 1
        if sha in wanted:
            found.setdefault(sha, f.name)
    return found, hashed


# ------------------------------------------------------------------------------------------------- stages
def _text_state(row: Optional[dict], cache: Path, sha: str, have_pdf: bool) -> tuple:
    if row and row["extractor"].endswith("/" + PT.VERSION) and PT.cache_file(cache, sha).is_file():
        return "done", ""
    if row and not row["extractor"].endswith("/" + PT.VERSION):
        return "missing", f"text by an older or failed extractor ({row['extractor'][:40]})"
    return "missing", "not extracted" if have_pdf else "not extracted (no PDF here)"


def _figures_state(store: Path, sha: str, listed: Sequence[str]) -> tuple:
    """(state, why, images the bank should hold, of them present)."""
    d = Path(store) / sha[:2] / sha
    fj = d / "figures.json"
    names = list(dict.fromkeys(listed))
    if not fj.is_file():
        return "missing", "no figures.json", len(names), sum((d / n).is_file() for n in names)
    try:
        rec = json.loads(fj.read_text(encoding="utf-8"))
    except ValueError:
        return "missing", "figures.json unreadable", len(names), 0
    names = list(dict.fromkeys([str(x.get("image")) for x in rec.get("figures") or [] if x.get("image")] + names))
    have = sum((d / n).is_file() for n in names)
    if have < len(names):
        return "missing", f"{len(names) - have} of {len(names)} figure images missing", len(names), have
    return "done", "", len(names), have


def figures_listed(base: Path, B) -> Dict[str, List[str]]:
    """{sha256: image names} index/atlas_figures.js lists (absent / unreadable: {})."""
    doc = B._figures_doc(Path(base)) or {}
    out: Dict[str, List[str]] = {}
    for row in doc.get("papers") or []:
        try:
            _, sha, figs = row[0], str(row[1]).lower(), row[2]
        except (IndexError, TypeError):
            continue
        out[sha] = [f[3] if len(f) > 3 else B.figure_image_name(i, str(f[0] or ""), int(f[1] or 0))
                    for i, f in enumerate(figs, 1)]
    return out


def measure(base: Path, pdfs: Path, B, model: str = "", arm: str = "auto", cfg: Optional[dict] = None,
            notes: Sequence[str] = ()) -> dict:
    """One atlas's coverage on this machine: per held paper and stage (done / missing / n/a / off, with why), the
    counts, and the figure images index/atlas_figures.js lists vs present. `model` / `arm`: the embedding model and
    figure arm `atlas update` would use here ('' model = none; arm 'off' = figures off)."""
    import arxiv_html as AH
    import paper_embed as PE
    import paper_ocr as PO
    base, pdfs = Path(base), Path(pdfs)
    cfg = cfg or AP.config(base)
    pl = cfg["pipeline"]
    rows, alias_map, rec_notes = held_papers(base, B)
    cache, ocr_store = PT.default_cache(pdfs), PT.ocr_store(pdfs)
    html, embed, figs = AH.default_cache(pdfs), PE.default_cache(pdfs), AP.beside(pdfs, "figures")
    man = PT.read_manifest(cache)
    here, hashed = pdfs_here(pdfs, man, rows)
    html_man = _load(html / "manifest.json", {}) if pl.get("arxiv_html") else {}
    facts: Dict[str, set] = {}
    for r in (_load(base / "papers" / "_facts.json", {}).get("papers") or []):
        if isinstance(r, dict) and r.get("key"):
            facts.setdefault(alias_map.get(r["key"], r["key"]), set()).add(str(r.get("sha256") or "").lower())
    meta = B.meta_rows(base, alias_map)
    listed = figures_listed(base, B)
    off = {"ocr": not pl.get("ocr", True), "html": not pl.get("arxiv_html", False),
           "facts": not pl.get("facts", True), "figures": (arm or "").lower() == "off" or pl.get("figures") == "off",
           "meta": not pl.get("meta", True)}
    models = [model] if model else sorted(m for m in PE.MODELS if (embed / m).is_dir())
    if not model and models:          # `update` cannot embed here (no torch); vectors made elsewhere still count
        rec_notes.append(f"no torch for the embed stage here: counting the vectors already stored ({', '.join(models)})")
    out_rows, img_want, img_have = [], 0, 0
    for p in rows:
        sha, st = p["sha256"], {}
        have_pdf = sha in here
        st["pdf"] = ("done", "") if have_pdf else ("missing", f"no PDF with this sha256 in {pdfs}")
        mrow = man.get(sha)
        st["text"] = _text_state(mrow, cache, sha, have_pdf)
        text_ok = st["text"][0] == "done"
        if off["ocr"]:
            st["ocr"] = ("off", "")
        elif not text_ok:
            st["ocr"] = ("missing", "no text yet")
        else:
            todo = PO.needs_listing(cache, ocr_store, sha, man, OCR_ENGINE)
            if todo:
                st["ocr"] = ("missing", f"{todo[1]} of {todo[0]} pages still to OCR")
            elif PT.ocr_merge_pending(cache, mrow, ocr_store):
                st["ocr"] = ("missing", "OCR read but not merged into the text (paper_text.py build)")
            elif any(PO.store_file(ocr_store, e, sha).is_file() for e in PT.OCR_ENGINES):
                st["ocr"] = ("done", "")
            else:
                st["ocr"] = ("n/a", "no page needs OCR")
        if off["html"]:
            st["html"] = ("off", "")
        elif not p["arxiv_id"]:
            st["html"] = ("n/a", "not an arXiv paper")
        else:
            hr = html_man.get(p["arxiv_id"]) or {}
            if hr.get("status") == "ok":
                try:
                    parsed = json.loads(gzip.decompress(
                        (html / "parsed" / f"{AH.flat(p['arxiv_id'])}.json.gz").read_bytes()))
                except (OSError, ValueError, EOFError):
                    parsed = {}
                st["html"] = (("done", "") if parsed.get("parser") == AH.PARSER_VERSION else
                              ("missing", "fetched, not parsed by the current parser"))
            elif hr.get("status") == "no_html":
                st["html"] = ("n/a", "arXiv has no HTML for it")
            else:
                st["html"] = ("missing", f"arXiv HTML {hr['status']}" if hr.get("status") else "never fetched")
        if off["facts"]:
            st["facts"] = ("off", "")
        else:
            got = facts.get(p["key"])
            st["facts"] = (("done", "") if got and (sha in got or "" in got) else
                           ("missing", "mined from another file") if got else
                           ("missing", "no row in papers/_facts.json"))
        if not models:
            st["embed"] = ("missing", "no embedding model here (no torch) and no vectors stored")
        elif not text_ok:
            st["embed"] = ("missing", "no text yet")
        else:
            text = PT.cache_file(cache, sha).read_text(encoding="utf-8")
            cur = any(PE.is_current(embed, m, sha, text) for m in models)
            st["embed"] = ("done", "") if cur else ("missing", f"no current {' / '.join(models)} vectors")
        if off["figures"]:
            st["figures"] = ("off", "")
        else:
            s, why, want, have = _figures_state(figs, sha, listed.get(sha, []))
            st["figures"] = (s, why)
            img_want, img_have = img_want + len(listed.get(sha, [])), img_have + sum(
                (figs / sha[:2] / sha / n).is_file() for n in listed.get(sha, []))
        if off["meta"]:
            st["meta"] = ("off", "")
        else:
            m = meta.get(p["key"]) or {}
            st["meta"] = (("done", "") if (m.get("measured") or {}).get("openalex") else
                          ("n/a", "no DOI to ask OpenAlex with") if not (m.get("doi") or p["work_doi"]) else
                          ("missing", "OpenAlex unmeasured (refused or not asked)") if m else
                          ("missing", "no row in papers/_meta.json"))
        out_rows.append({"key": p["key"], "sha256": sha, "file": p["file"],
                         "stages": {k: {"state": v[0], "why": v[1]} for k, v in st.items()}})
    counts = {s: {x: sum(1 for r in out_rows if r["stages"][s]["state"] == x) for x in STATES} for s in STAGES}
    complete = sum(1 for r in out_rows if all(v["state"] != "missing" for v in r["stages"].values()))
    listed_total = sum(len(v) for v in listed.values())
    return {"project": _rel(base), "short_name": cfg["atlas"]["short_name"], "pdfs": str(pdfs),
            "pdfs_found": pdfs.is_dir(), "embed_model": model, "figure_arm": arm, "papers": len(out_rows),
            "complete": complete, "counts": counts, "hashed": hashed,
            "figure_images": {"listed": listed_total, "listed_for_held": img_want, "present": img_have,
                              "off": off["figures"]},
            "notes": list(notes) + rec_notes, "rows": out_rows}


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(AP.ROOT).as_posix()
    except ValueError:
        return str(p)


def exit_code(results: Sequence[dict]) -> int:
    """2 when an atlas could not be measured, 1 when some paper misses some stage, else 0."""
    if any(r.get("error") for r in results):
        return 2
    return 1 if any(r["counts"][s]["missing"] for r in results for s in STAGES) else 0


# ------------------------------------------------------------------------------------------------- report
def _a(s) -> str:
    return str(s).encode("ascii", "backslashreplace").decode("ascii")


def report(results: Sequence[dict], say: Callable = print, missing: str = "", examples: int = 6) -> None:
    """The console report: a stage x state table per atlas, and the incomplete papers grouped by what they miss; with
    `missing`, the papers missing that stage (key, file, why) instead."""
    say(f"atlas coverage on {_a(socket.gethostname())}: per held paper, the evidence each stage's own tool leaves")
    for r in results:
        say("")
        if r.get("error"):
            say(f"{_a(r['project'])}  CANNOT MEASURE: {_a(r['error'])}")
            continue
        say(f"{_a(r['short_name'])}  ({_a(r['project'])}): {r['papers']} held papers, {r['complete']} complete here; "
            f"PDFs {_a(r['pdfs'])}{'' if r['pdfs_found'] else ' (NOT FOUND)'}")
        say(f"  embeddings {r['embed_model'] or 'none'}; figures {r['figure_arm']}"
            + (f"; hashed {r['hashed']} PDFs the text manifest did not vouch for" if r["hashed"] else ""))
        for n in r["notes"]:
            say(f"  note: {_a(n)}")
        if missing:
            rows = [x for x in r["rows"] if x["stages"][missing]["state"] == "missing"]
            say(f"  missing {missing}: {len(rows)} papers")
            for x in rows:
                say(f"    {_a(x['key']):40} {_a(x['stages'][missing]['why'])}  [{_a(x['file'][:50])}]")
            continue
        say(f"  {'stage':8} {'done':>6} {'missing':>8} {'n/a':>6} {'off':>6}")
        for s in STAGES:
            c = r["counts"][s]
            say(f"  {s:8} {c['done']:>6} {c['missing']:>8} {c['n/a']:>6} {c['off']:>6}")
        fi = r["figure_images"]
        if not fi["off"]:
            say(f"  figure images: {fi['present']} of the {fi['listed_for_held']} index/atlas_figures.js lists for held "
                f"papers are here ({fi['listed']} listed in all)")
        groups: Dict[tuple, List[str]] = {}
        for x in r["rows"]:
            miss = tuple(s for s in STAGES if x["stages"][s]["state"] == "missing")
            if miss:
                groups.setdefault(miss, []).append(x["key"])
        for miss, keys in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            more = f" (+{len(keys) - examples})" if len(keys) > examples else ""
            say(f"  missing {'+'.join(miss)}: {len(keys)} -- {', '.join(_a(k) for k in keys[:examples])}{more}")
    code = exit_code(results)
    say("")
    say({0: "coverage: nothing missing", 1: "coverage: papers are missing stages -- `--missing <stage>` lists them "
                                            "(atlas.py update --only <stages> mines them)",
         2: "coverage: some atlas could not be measured"}[code])


def as_json(results: Sequence[dict]) -> str:
    return json.dumps({"schema": SCHEMA, "host": socket.gethostname(), "exit": exit_code(results),
                       "atlases": list(results)}, indent=1, sort_keys=True, ensure_ascii=True)

