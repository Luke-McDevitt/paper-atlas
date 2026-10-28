"""report/boat_world_model/bundle_atlas.py -- ONE portable .zip: the atlas page, every paper PDF we hold, the index and
every report of the BlueBoat world-model program -- unzip anywhere, open `atlas.html`, click a held paper's PDF.

WHY (2026-10-12, owner): "generate a way to produce a .zip with the atlas and all of the referenced papers so it is more
portable". The online atlas is a private Artifact and the PDFs (~5 GB, copyrighted publisher files) live outside git in
one folder on the laptop; this packs them together for a USB stick, the lab computer or a colleague -- and is the
first half of the laptop <-> lab paper sharing the owner deferred (each PDF's sha256 is in MANIFEST.json and
papers/_held.json, so the other machine can tell which copies it already has).

    python report/boat_world_model/build_bwm.py bundle [--pdfs C:/svfr/papers/boat_wm] [--out X.zip] [--no-pdfs]

LAYOUT (one top folder, so unzipping never spills files):
    boat_wm_atlas/atlas.html          the atlas as a standalone page; a held paper links its papers/<name>.pdf
    boat_wm_atlas/atlas_citations.js  the Citations tab's data, and atlas_figures.js the Figures tab's (2026-10-15)
    boat_wm_atlas/figures/<sha[:2]>/<sha>/<image>.webp   every figure-bank image the Figures tab shows (unless
                                      --no-figures; the bank is tools/figure_bank.py's, beside the PDF folder)
    boat_wm_atlas/papers/<name>.pdf   every PDF in papers/_held.json, under its folder name (Surname_Year_Title)
    boat_wm_atlas/index/...           PAPERS.md, LEXICON.md, LABS.md, PAPER_DOWNLOAD_LIST.md + the JSON, and
                                      ATLAS.xlsx, LEXICON.docx, PAPERS.docx (the Office copies)
    boat_wm_atlas/reports/...         README.md (the hub), STATE_OF_RESEARCH.md, LIT_*.md, plans/, recon/
    boat_wm_atlas/README.txt          what is inside, counts, and what was left out
    boat_wm_atlas/MANIFEST.json       every file: path, bytes, sha256
    boat_wm_atlas/atlas_<id>.html     (atlas.toml sections_mode = "atlases") each section's own atlas page and its
                                      atlas_<id>_citations.js / _figures.js, the banner row linking them; and beside the
                                      zip, one zip PER SECTION (<short_name>_<id>_<date>.zip): its page as atlas.html,
                                      its papers' PDFs and figures only, no link to any other page (2026-10-15)

RULES: the page and index come from a FRESH build (never a stale index/); a PDF is packed only if its file is present
with the bytes `scan` recorded -- under its recorded name, or (2026-10-15) under another name with the recorded
sha256, a copy this machine names differently (else it is listed as left out, never silently dropped); PDFs are STORED (already
compressed), text DEFLATED; written to <out>.part and renamed (the .part is deleted on any failure), so an
interrupted run leaves no half zip; one fixed timestamp and header on every entry, so the same inputs give the same
bytes on one machine (another Python build may deflate the text differently -- the MANIFEST sha256s still match).
A held paper whose PDF is missing or changed is LEFT OUT and listed (exit 1); the atlas then gives it its Get-it links.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys  # noqa: E402
if str(Path(__file__).resolve().parents[2]) not in sys.path:   # run by path: the repo root, for tools/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.atomic_replace import replace_retry  # noqa: E402

TOP = "boat_wm_atlas"
REPORT_GLOBS = ("README.md", "STATE_OF_RESEARCH.md", "LIT_*.md", "plans/*.md", "recon/*.md", "figures/*.png",
                "figures/*.json")
INDEX_FILES = ("PAPERS.md", "LEXICON.md", "LABS.md", "PAPER_DOWNLOAD_LIST.md", "papers.json", "lexicon.json",
               "labs.json", "web.json", "atlas_citations.js")
SIDECARS = ("atlas_citations.js", "atlas_figures.js")      # beside atlas.html: the page loads them by <script src>
#: a section's own atlas page and its two sidecars (build_bwm.section_atlases, `atlases` mode)
SECTION_PAGE = re.compile(r"^atlas_([a-z0-9][a-z0-9_-]*)\.html$")
SECTION_SIDE = re.compile(r"^atlas_[a-z0-9][a-z0-9_-]*_(citations|figures)\.js$")


def figure_files(base: Path, store: Optional[Path]) -> Tuple[List[Tuple[Path, str]], int]:
    """[(bank image, path in the zip)] for every figure index/atlas_figures.js lists whose image is in the bank, and
    how many it lists that are not there (left out -- the card then shows its caption only)."""
    f = Path(base) / "index" / "atlas_figures.js"
    if store is None or not f.is_file() or not Path(store).is_dir():
        return [], 0
    text = f.read_text(encoding="utf-8")
    doc = json.loads(text[text.index("=") + 1:].rstrip().rstrip(";"))
    out, missing = [], 0
    for key, sha, figs in doc.get("papers") or []:
        for i, x in enumerate(figs, 1):
            name = x[3] if len(x) > 3 else f"fig{i:03d}-{x[0] or 'u'}-p{x[1]}.webp"
            src = Path(store) / sha[:2] / sha / name
            if src.is_file():
                out.append((src, f"figures/{sha[:2]}/{sha}/{name}"))
            else:
                missing += 1
    return out, missing


def safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._") or "paper"


def _sha256(f: Path) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_pdfs(base: Path, pdf_dir: Path, alias_map: Dict[str, str], rejected: Optional[Dict[str, set]] = None
              ) -> Tuple[List[Tuple[str, Path, str, dict]], List[str]]:
    """[(key, source file, path in the zip, held row)] for every held PDF present with its recorded size, and
    [reason] for each one left out. `rejected` = build_bwm.rejected_ids(...)[0] ({sha256: keys that file is NOT}):
    such a file never ships (third fix check 2026-10-15: a second, hand-written parser here dropped every rejection
    on a trailing comma or a padded sha). A held file missing under its recorded name is looked for BY CONTENT among
    the folder's unclaimed PDFs of the recorded size -- only an exact sha256 match counts -- and packed under the
    recorded name (2026-10-15: the lab names one copy `2014_Performance_...pdf` where the scan recorded
    `Miller_2014_Performance_...pdf`; the first lab bundle left it out)."""
    held = json.loads((base / "papers" / "_held.json").read_text(encoding="utf-8")) \
        if (base / "papers" / "_held.json").is_file() else {"files": []}
    keep, left, names, best = [], [], set(), {}
    rejected = rejected or {}
    for r in held.get("files") or []:              # one file per paper: a key merged into another since the scan
        key = alias_map.get(r["key"], r["key"])    # resolves to it, and the larger file wins
        if key in rejected.get(str(r.get("sha256") or "").strip().lower(), ()):
            left.append(f"{key}: {r['file']} is NOT this paper (papers/_rejected.json)")
            continue
        if key not in best or (r.get("bytes") or 0, r["file"]) > (best[key].get("bytes") or 0, best[key]["file"]):
            best[key] = r
    pool: Optional[Dict[int, List[Path]]] = None     # the folder's unclaimed PDFs by size, built on first need
    for key in sorted(best):
        r = best[key]
        src = Path(pdf_dir) / r["file"]
        if not src.is_file():
            if pool is None:
                claimed, pool = {b["file"] for b in best.values()}, {}
                for f in (sorted(Path(pdf_dir).glob("*.pdf")) if Path(pdf_dir).is_dir() else []):
                    if f.name not in claimed:
                        pool.setdefault(f.stat().st_size, []).append(f)
            want = str(r.get("sha256") or "").strip().lower()
            src = next((f for f in pool.get(r.get("bytes"), []) if want and _sha256(f) == want), None)
            if src is None:
                left.append(f"{key}: {r['file']} not in {pdf_dir}")
                continue
        if src.stat().st_size != r.get("bytes"):
            left.append(f"{key}: {r['file']} is {src.stat().st_size} bytes, the scan recorded {r.get('bytes')} "
                        f"(re-run scan)")
            continue
        name = safe_name(Path(r["file"]).stem)       # the folder's own (title) name: Surname_Year_Title
        while name.lower() in names:
            name += "_"
        names.add(name.lower())
        keep.append((key, src, f"papers/{name}.pdf", r))
    return keep, left


def _info(arc: str, when, compress, top: str = TOP) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(f"{top}/{arc}", date_time=when)
    zi.compress_type = compress
    zi.create_system = 3                                  # the same header on every host (ZipInfo says 0 on Windows)
    zi.external_attr = 0o644 << 16
    return zi


def _put(zf: zipfile.ZipFile, arc: str, data: bytes, when, compress=zipfile.ZIP_DEFLATED) -> dict:
    zf.writestr(_info(arc, when, compress, getattr(zf, "atlas_top", TOP)), data)
    return {"path": arc, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _put_file(zf: zipfile.ZipFile, arc: str, src: Path, when) -> dict:
    zi = _info(arc, when, zipfile.ZIP_STORED, getattr(zf, "atlas_top", TOP))
    h, n = hashlib.sha256(), 0
    big = src.stat().st_size >= (1 << 31) - (1 << 20)    # ZIP64 headers only where needed: some unzippers balk
    with open(src, "rb") as f, zf.open(zi, "w", force_zip64=big) as out:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            out.write(chunk)
            n += len(chunk)
    return {"path": arc, "bytes": n, "sha256": h.hexdigest()}


def bundle(base: Path, pdf_dir: Path, out: Path, build_all, build_papers, standalone_page, include_pdfs: bool = True,
           date: str = "2026-01-01", say=print, office_outputs=None, rejected_ids=None,
           figure_store: Optional[Path] = None, names: Optional[dict] = None,
           section_titles: Optional[Dict[str, str]] = None) -> dict:
    """Write the zip; -> a summary {pdfs, left_out, bytes, files, figures, sections}. The build functions come from
    build_bwm (passed in, so this module never imports it back). `figure_store`: the figure bank whose images the
    Figures tab shows (None: no images, the cards show captions only). In `atlases` mode the section pages ride in the
    zip and each section gets its own zip beside it (`sections`: their summaries)."""
    _, alias_map, _ = build_papers(base)
    rejected = rejected_ids(base, alias_map)[0] if rejected_ids else {}
    pdfs, left = plan_pdfs(base, pdf_dir, alias_map, rejected) if include_pdfs else ([], [])
    local = {key: arc for key, _, arc, _ in pdfs}
    outs = build_all(base, local_pdfs=local, all_page="atlas.html")
    office = {k: v for k, v in (office_outputs(base, outs) if office_outputs else {}).items()
              if not k.startswith("_") and v}                 # no pandoc: the .docx are simply absent
    stats = json.loads(outs["papers.json"])["stats"]
    figs, figs_missing = figure_files(base, figure_store)
    y, m, d = (int(x) for x in date.split("-"))
    when = (y, m, d, 0, 0, 0)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    try:
        total = _write(part, base, pdfs, left, outs, stats, standalone_page, include_pdfs, date, when, say, office,
                       figs, figs_missing, names or {})
    except BaseException:
        part.unlink(missing_ok=True)                     # never leave a multi-GB half zip behind (full disk, Ctrl+C)
        raise
    replace_retry(part, out)
    secs = []
    if any(SECTION_PAGE.match(n) for n in outs):       # `atlases` mode: one zip per section, with no other page
        solo = build_all(base, local_pdfs=local, all_page="", section_links=False)
        secs = section_bundles(base, pdfs, solo, figs, out, names or {}, section_titles or {}, date, when,
                               standalone_page, include_pdfs, say)
    return {"pdfs": len(pdfs), "left_out": left, "bytes": out.stat().st_size, "files": total[1],
            "uncompressed": total[0], "figures": len(figs), "figures_missing": figs_missing, "sections": secs}


def section_bundles(base: Path, pdfs, solo: Dict[str, str], figs, out: Path, names: dict, titles: Dict[str, str],
                    date: str, when, standalone_page, include_pdfs: bool, say=print) -> List[dict]:
    """`atlases` mode: per section ONE zip beside the atlas's -- <short_name>_<id>_<date>.zip: the section's page as
    atlas.html (rendered with no link to another page), its two sidecars, its papers' PDFs and figure images, a
    README and a MANIFEST. -> [{id, zip, papers, pdfs, figures}]."""
    papers = json.loads(solo["papers.json"])["papers"]
    short = names.get("short_name") or TOP
    res = []
    for name in sorted(n for n in solo if SECTION_PAGE.match(n)):
        sid = SECTION_PAGE.match(name).group(1)
        keys = {p["key"] for p in papers if sid in (p.get("sections") or ())}
        cjs, fjs = f"atlas_{sid}_citations.js", f"atlas_{sid}_figures.js"
        shas = set()
        if fjs in solo:
            text = solo[fjs]
            doc = json.loads(text[text.index("=") + 1:].rstrip().rstrip(";"))
            shas = {r[1] for r in doc.get("papers") or []}
        mine_figs = [(src, arc) for src, arc in figs if arc.split("/")[2] in shas]
        mine_pdfs = [x for x in pdfs if x[0] in keys]
        z = out.with_name(f"{short}_{sid}_{date}{'' if include_pdfs else '_no_pdfs'}.zip")
        part = z.with_name(z.name + ".part")
        manifest: List[dict] = []
        try:
            with zipfile.ZipFile(part, "w", allowZip64=True) as zf:
                zf.atlas_top = f"{short}_{sid}"
                manifest.append(_put(zf, "atlas.html", standalone_page(solo[name]).encode("utf-8"), when))
                for side in (cjs, fjs):
                    if side in solo:
                        manifest.append(_put(zf, side, solo[side].encode("utf-8"), when))
                for src, arc in mine_figs:
                    manifest.append(_put_file(zf, arc, src, when))
                for key, src, arc, row in mine_pdfs:
                    manifest.append({**_put_file(zf, arc, src, when), "key": key, "source_file": src.name})
                readme = "\n".join([
                    f"{titles.get(sid) or sid} -- one section of {names.get('title') or 'the paper atlas'}, as its own atlas",
                    f"Built {date} by build_bwm.py bundle (atlas.toml sections_mode = \"atlases\").", "",
                    "Open atlas.html in a browser (keep the two atlas_*.js files beside it). A held paper's card opens",
                    "its PDF from papers/; the Figures tab's images are in figures/.", "",
                    f"Papers in this section: {len(keys)}. PDFs in this zip: {len(mine_pdfs)}"
                    + ("" if include_pdfs else " (built with --no-pdfs)") + f". Figure images: {len(mine_figs)}.",
                    "The PDFs are publisher / author copies for the lab's internal research use: do not redistribute them.",
                    ""])
                manifest.append(_put(zf, "README.txt", readme.encode("utf-8"), when))
                _put(zf, "MANIFEST.json", (json.dumps({"schema": "svfr.bwm.bundle/1", "date": date, "section": sid,
                                                       "files": manifest}, indent=1, sort_keys=True) + "\n"
                                           ).encode("utf-8"), when)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        replace_retry(part, z)
        say(f"  section {sid}: {z.name} ({z.stat().st_size / 1e6:.1f} MB, {len(keys)} papers, {len(mine_pdfs)} PDFs, "
            f"{len(mine_figs)} figure images)")
        res.append({"id": sid, "zip": str(z), "papers": len(keys), "pdfs": len(mine_pdfs), "figures": len(mine_figs)})
    return res


def _write(part, base, pdfs, left, outs, stats, standalone_page, include_pdfs, date, when, say,
           office=None, figs=(), figs_missing=0, names: Optional[dict] = None) -> Tuple[int, int]:
    manifest: List[dict] = []
    side = {"atlas_citations.js": outs["atlas_citations.js"].encode("utf-8")}
    fjs = Path(base) / "index" / "atlas_figures.js"
    if fjs.is_file():
        side["atlas_figures.js"] = fjs.read_bytes()
    names = names or {}
    with zipfile.ZipFile(part, "w", allowZip64=True) as zf:
        zf.atlas_top = names.get("short_name") or TOP     # one top folder, named after the project (atlas.toml)
        manifest.append(_put(zf, "atlas.html", standalone_page(outs["browser.html"]).encode("utf-8"), when))
        for name in SIDECARS:                            # beside the page (it loads them) and in index/
            if name in side:
                manifest.append(_put(zf, name, side[name], when))
        for name in sorted(n for n in outs if SECTION_PAGE.match(n)):         # `atlases` mode: each section's page,
            manifest.append(_put(zf, name, standalone_page(outs[name]).encode("utf-8"), when))   # beside atlas.html
        for name in sorted(n for n in outs if SECTION_SIDE.match(n)):
            manifest.append(_put(zf, name, outs[name].encode("utf-8"), when))
        for name in INDEX_FILES:
            manifest.append(_put(zf, f"index/{name}", outs[name].encode("utf-8"), when))
        if "atlas_figures.js" in side:
            manifest.append(_put(zf, "index/atlas_figures.js", side["atlas_figures.js"], when))
        for name in sorted(office or {}):
            manifest.append(_put(zf, f"index/{name}", office[name], when))
        for i, (src, arc) in enumerate(figs, 1):         # WebP: already compressed, STORED
            manifest.append(_put_file(zf, arc, src, when))
            if i % 5000 == 0 or i == len(figs):
                say(f"  packed {i}/{len(figs)} figure images")
        seen = set()
        for pat in REPORT_GLOBS:
            for f in sorted(base.glob(pat)):
                rel = f.relative_to(base).as_posix()
                if rel not in seen:
                    seen.add(rel)
                    manifest.append(_put(zf, f"reports/{rel}", f.read_bytes(), when))
        for i, (key, src, arc, row) in enumerate(pdfs, 1):
            got = _put_file(zf, arc, src, when)
            if row.get("sha256") and got["sha256"] != row["sha256"]:
                left.append(f"{key}: {src.name} changed since the scan (sha256 differs) -- packed anyway, re-run scan")
            manifest.append({**got, "key": key, "source_file": src.name})
            if i % 100 == 0 or i == len(pdfs):
                say(f"  packed {i}/{len(pdfs)} PDFs")
        total = sum(e["bytes"] for e in manifest)
        n_lit = sum(1 for e in manifest if re.match(r"reports/LIT_.*[.]md$", e["path"]))
        n_plans = sum(1 for e in manifest if e["path"].startswith("reports/plans/PLAN_"))
        readme = "\n".join([
            names.get("bundle_title") or "BlueBoat world-model research atlas -- portable bundle",
            f"Built {date} by build_bwm.py bundle (the atlas tools: tools/atlas.py).", "",
            "Open atlas.html in a browser. Tabs: Find (type a task or a few terms: papers ranked by the lexicon, each",
            "term's full-text ranking and the papers' own descriptions, with the reason each is listed), Lexicon,",
            "Papers, Citations (which indexed papers cite which: a map, topic flows, the most-cited backbone), Figures",
            "(search every figure's caption; the images are in figures/), Labs and Trends. In Papers, a held paper has",
            "an 'Open the PDF (in this bundle)' link; every other paper has 'Get it' links (arXiv, free copies found,",
            "publisher, Google Scholar, a PDF search). Keep atlas_citations.js and atlas_figures.js beside atlas.html.",
            "Fonts load from Google Fonts when online, system fonts otherwise.", "",
            f"Papers in the index: {stats['papers']} ({stats['published']} published, {stats['preprints']} preprints).",
            f"PDFs in this bundle: {len(pdfs)}" + ("" if include_pdfs else " (built with --no-pdfs)") + ".",
            f"Figure images in this bundle: {len(figs)}" + (f" ({figs_missing} listed but not in the bank)"
                                                           if figs_missing else "") + ".",
            f"Lexicon: {stats['terms']} terms / {stats['citations']} citations. Labs: {stats['labs']} groups.",
            f"Citations among the indexed papers: {stats.get('citation_links', 'n/a')}.", "",
            "From the repository, the same lookup with the full text of every page:",
            "  python tools/paper_find.py \"<your task or terms>\"   (see report/boat_world_model/LIT_PAPER_LOOKUP.md)", "",
            "index/PAPER_DOWNLOAD_LIST.md: what is still missing, with links to hunt for each paper.",
            "index/ATLAS.xlsx (Excel: papers, lexicon, labs, code, facts) and index/LEXICON.docx, PAPERS.docx (Word,",
            "with the equations as Word equations): the same index as Office files.",
            f"reports/README.md: the program hub (decisions, timeline); reports/LIT_*.md: the {n_lit} topic reviews;",
            f"reports/plans/: the {n_plans} plans. MANIFEST.json: every file with its sha256.", "",
            "The PDFs are publisher / author copies for the lab's internal research use: do not redistribute them.",
        ] + ([f"", f"Left out ({len(left)}):"] + [f"  - {x}" for x in left] if left else []) + [""])
        manifest.append(_put(zf, "README.txt", readme.encode("utf-8"), when))
        _put(zf, "MANIFEST.json", (json.dumps({"schema": "svfr.bwm.bundle/1", "date": date, "files": manifest},
                                              indent=1, sort_keys=True) + "\n").encode("utf-8"), when)
    return total, len(manifest) + 1


def main_bundle(base: Path, pdf_dir: Path, out: Path, B, include_pdfs: bool = True, date: str = "2026-01-01",
                include_figures: bool = True, names: Optional[dict] = None) -> int:
    """`B` is the build_bwm module (it calls this; nothing here imports it back)."""
    if include_pdfs and not (base / "papers" / "_held.json").is_file():
        print("no papers/_held.json: run `build_bwm.py scan --pdfs <folder>` first (or bundle --no-pdfs)")
        return 2
    if include_pdfs and not Path(pdf_dir).is_dir():
        print(f"no PDF folder at {pdf_dir}: pass --pdfs <folder> (or bundle --no-pdfs for the atlas and reports only)")
        return 2
    if include_pdfs and B._refuse_unreadable_rejections(base):   # never ship PDFs blind (third fix check)
        return 2
    store = B._tool("figure_bank").default_store(pdf_dir) if include_figures else None
    print(f"bundle: {'PDFs from ' + str(pdf_dir) if include_pdfs else 'no PDFs'}; "
          f"{'figure images from ' + str(store) if store and Path(store).is_dir() else 'no figure images'} -> {out}")
    titles = {s["id"]: s["title"] for s in B._tool("atlas_project").sections(base)}
    res = bundle(base, pdf_dir, out, B.build_all, B.build_papers, B.standalone_page, include_pdfs=include_pdfs, date=date,
                 office_outputs=B.office_outputs, rejected_ids=B.rejected_ids, figure_store=store, names=names,
                 section_titles=titles)
    print(f"wrote {out} ({res['bytes'] / 1e9:.2f} GB, {res['files']} files, {res['pdfs']} PDFs, "
          f"{res['figures']} figure images)" + (f"; and {len(res['sections'])} section zip(s) beside it"
                                                 if res["sections"] else ""))
    for x in res["left_out"]:
        print(f"  left out: {x}")
    if res["left_out"]:
        print(f"{len(res['left_out'])} held paper(s) are NOT in the zip (listed above and in its README.txt; the atlas "
              f"gives them 'Get it' links instead)")
    return 1 if res["left_out"] else 0
