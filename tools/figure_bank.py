r"""Every figure of every held paper: found by a layout detector, captioned from the PDF's own text layer, stored as
DATA (page + box + caption) with a browsing image, and re-rendered at full resolution on demand.

    python tools/figure_bank.py detect --bwm [--arm yolo|ppdl3] [--limit N]   # GPU: layout boxes per page (cached)
    python tools/figure_bank.py build --bwm [--arm yolo|ppdl3]                  # captions + figures.json + WebP images
    python tools/figure_bank.py eval --bwm [--arm ppdl3|yolo|heuristic] [--n 60] # vs arXiv-HTML ground truth
    python tools/figure_bank.py export <key> --out <dir> [--dpi 300] [--fig 3]  # full-resolution PNGs for a slide
    python tools/figure_bank.py show <key>                                       # one paper's figures and captions

NEEDS PyMuPDF (AGPL: fine for internal use, recorded) for text blocks and rendering, and for `detect` torch +
transformers (PP-DocLayoutV3, Apache-2.0) or doclayout-yolo (DocStructBench weights, Apache-2.0; code AGPL) -- run
those with the GPU venv (report/boat_world_model/README.md). The geometry (grouping, caption matching, scoring) is
pure Python and tested in the repo's suite.

HOW. 1 Detect: each page rendered (long side 1,024 px, fp32, ONE process -- measured fastest) -> boxes labelled
figure (YOLO; image / chart for PP-DocLayoutV3) and caption / table. 2 Group: figure boxes that touch (8 pt) are one
figure (panels). 3 Caption: a TEXT-LAYER block starting "Fig. N" / "Figure N" (the PDF's own words, never OCR'd) takes
the group just ABOVE it, else the one just below (a caption printed above its figure), same page, sharing at least a
third of its width, and only groups on ONE side of it; a group with no caption is kept as `uncaptioned`.
4 Store `<pdfs>_figures/<sha[:2]>/<sha>/figures.json` + one WebP per figure (long side 1280 px).

WHY. Owner 2026-10-14: "I want to be able to extract figures from any paper and use them / keep them". The research
(LIT_PAPER_MINING.md) found no published figure-class score for the newest detectors, so the arm is chosen by `eval`
against arXiv's own HTML, which lists every figure with its number and caption for the arXiv papers -- MEASURED
2026-10-14 on 617 figures: DocLayout-YOLO recall 0.919 / precision 0.983, PP-DocLayoutV3 0.908, the caption heuristic
0.885; re-measured 2026-10-15 on 60 held papers / 575 figures (report/boat_world_model/papers/_figure_eval.json):
0.892 / 0.950, 0.887 / 0.945, 0.897 / 0.952 -- no difference significant. DEFAULT_ARM is YOLO (it also keeps figures
with no caption and needs no caption text) and the bank was built with it; a default of ppdl3 re-ran 765 papers with
the other detector when `detect --bwm` was run as the README says (caught 2026-10-14).
"""
from __future__ import annotations

import argparse
import gzip
import json
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

CAPTION = re.compile(r"^\s*(?:Fig\.?|FIG\.?|Figure|FIGURE)\s*(S?\d{1,3})(?:\s*\(?[a-z]\)?)?\s*[.:|\-–—]?")
TABLE_CAP = re.compile(r"^\s*(?:Table|TABLE|Tab\.)\s*[IVX\d]+")
FIG_LABELS = {"ppdl3": {"image", "chart"}, "yolo": {"figure"}}
CAP_LABELS = {"ppdl3": {"figure_title"}, "yolo": {"figure_caption"}}
Box = Tuple[float, float, float, float]


def default_store(pdfs: Path) -> Path:
    p = Path(pdfs)
    return p.parent / (p.name + "_figures")


# ------------------------------------------------------------------------------------------ geometry (pure)
def _touch(a: Box, b: Box, pad: float) -> bool:
    return a[0] - pad <= b[2] and b[0] - pad <= a[2] and a[1] - pad <= b[3] and b[1] - pad <= a[3]


def _union(boxes: Sequence[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def group_boxes(boxes: Sequence[Box], pad: float = 8.0) -> List[Box]:
    """Figure boxes that touch (within pad points) -> one box per connected group (a figure and its panels)."""
    groups = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _touch(tuple(groups[i]), tuple(groups[j]), pad):
                    groups[i] = list(_union([tuple(groups[i]), tuple(groups[j])]))
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(g) for g in sorted(groups, key=lambda g: (g[1], g[0]))]


def _x_overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def match_captions(groups: Sequence[Box], captions: Sequence[Tuple[Box, str]], page_h: float,
                   max_gap: float = 0.35) -> Tuple[List[dict], List[Box]]:
    """Figure groups + caption blocks (box, text) of ONE page -> ([{box, caption, number}], uncaptioned groups).
    A caption takes the group just above it (the usual layout), else the one just below (a caption printed above its
    figure); the groups a caption takes are merged (sub-figures). Gap <= max_gap * page height, x-overlap >= 1/3."""
    pairs = []
    for gi, g in enumerate(groups):
        for ci, (c, _) in enumerate(captions):
            if _x_overlap(g, c) < 1 / 3:
                continue
            if g[3] <= c[1] + 6:                      # figure above the caption
                gap, prior, side = c[1] - g[3], 0.0, "above"
            elif c[3] <= g[1] + 6:                    # caption above the figure
                gap, prior, side = g[1] - c[3], 0.25 * page_h, "below"
            else:
                continue
            if gap <= max_gap * page_h:
                pairs.append((max(0.0, gap) + prior, gi, ci, side))
    pairs.sort()
    taken_g, cap_of, cap_side = set(), {}, {}
    for _, gi, ci, side in pairs:
        if gi in taken_g or cap_side.get(ci, side) != side:   # a caption keeps to the side of its best group (review
            continue                                          # 2026-10-14: it took an uncaptioned group across itself)
        taken_g.add(gi)
        cap_side[ci] = side
        cap_of.setdefault(ci, []).append(gi)
    figs = []
    for ci, gis in sorted(cap_of.items()):
        box, text = captions[ci]
        m = CAPTION.match(text)
        figs.append({"box": _union([groups[g] for g in gis]), "caption": re.sub(r"\s+", " ", text).strip(),
                     "number": m.group(1) if m else "", "caption_box": box, "panels": len(gis)})
    return figs, [g for i, g in enumerate(groups) if i not in taken_g]


# ----------------------------------------------------------------------------------- scoring (pure)
def _tok(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def caption_dice(a: str, b: str) -> float:
    ta, tb = set(_tok(CAPTION.sub("", a or "")[:300])), set(_tok(CAPTION.sub("", b or "")[:300]))
    return 2 * len(ta & tb) / max(1, len(ta) + len(tb))


def html_truth(parsed: dict) -> List[dict]:
    """arXiv-HTML figures a reader sees as one figure: every ltx_figure with a 'Figure N' tag, unless its parent figure
    is itself numbered (then it is a panel). LaTeXML nests side-by-side figures in an untagged wrapper; requiring no
    parent at all dropped 265 numbered figures in 88 papers (review 2026-10-14)."""
    figs = parsed.get("figures") or []
    numbered = {f.get("id") for f in figs if re.match(r"(?:Figure|Fig\.?)\s*S?\d+", f.get("tag") or "")}
    out = []
    for f in figs:
        m = re.match(r"(?:Figure|Fig\.?)\s*(S?\d+)", f.get("tag") or "")
        if m and f.get("parent") not in numbered:
            out.append({"number": m.group(1), "caption": f.get("caption") or ""})
    return out


def score_paper(pred: Sequence[dict], truth: Sequence[dict], min_dice: float = 0.5) -> dict:
    """Per paper: truth figures FOUND (a predicted figure with the same number and a caption that agrees, Dice >=
    min_dice), predicted figures that are REAL (number + caption agree with a truth figure)."""
    found = sum(any(p["number"] == t["number"] and caption_dice(p["caption"], t["caption"]) >= min_dice for p in pred)
                for t in truth)
    real = sum(any(p["number"] == t["number"] and caption_dice(p["caption"], t["caption"]) >= min_dice for t in truth)
               for p in pred)
    return {"truth": len(truth), "pred": len(pred), "found": found, "real": real}


# ------------------------------------------------------------------------------------------- PDF parts
def _pm():
    try:
        import pymupdf
    except ImportError as e:                          # pragma: no cover - depends on the interpreter
        raise SystemExit(f"figure_bank needs PyMuPDF (python -m pip install pymupdf) -- {e}")
    return pymupdf


def caption_blocks(page) -> List[Tuple[Box, str]]:
    """Text-layer blocks that start like a figure caption ("Fig. 3", "Figure 3:") -- the PDF's own words."""
    out = []
    for b in page.get_text("blocks"):
        if len(b) >= 5 and isinstance(b[4], str) and CAPTION.match(b[4]) and not TABLE_CAP.match(b[4]):
            out.append(((b[0], b[1], b[2], b[3]), b[4]))
    return out


class Detector:
    """A layout detector on the GPU: page image -> [(label, score, box in image pixels)]."""

    # half=False and 1024-px renders: MEASURED 2026-10-14 on 91 pages alone -- fp32 14.9 pages/s, fp16 3.3 (slower on
    # this GPU), 1280 px 11.6; the same figures found. Three processes at once ran ~3 pages/s in TOTAL: run one.
    def __init__(self, arm: str, threshold: float = 0.4, half: bool = False):
        import torch
        self.arm, self.thr = arm, threshold
        self.half = half and torch.cuda.is_available()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if arm == "ppdl3":
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            rid = "PaddlePaddle/PP-DocLayoutV3_safetensors"
            self.proc = AutoImageProcessor.from_pretrained(rid)
            self.model = AutoModelForObjectDetection.from_pretrained(rid).to(self.dev).eval()
            self.labels = self.model.config.id2label
        elif arm == "yolo":
            from doclayout_yolo import YOLOv10
            from huggingface_hub import hf_hub_download
            w = hf_hub_download("juliozhao/DocLayout-YOLO-DocStructBench", "doclayout_yolo_docstructbench_imgsz1024.pt")
            self.model = YOLOv10(w)
        else:
            raise ValueError(arm)

    def __call__(self, images) -> List[List[Tuple[str, float, Box]]]:
        import torch
        if self.arm == "ppdl3":
            with torch.no_grad():
                inp = self.proc(images=images, return_tensors="pt").to(self.dev)
                out = self.model(**inp)
            sizes = [(im.height, im.width) for im in images]
            res = self.proc.post_process_object_detection(out, threshold=self.thr, target_sizes=sizes)
            return [[(self.labels[int(l)], float(s), tuple(float(x) for x in b))
                     for s, l, b in zip(r["scores"], r["labels"], r["boxes"])] for r in res]
        rs = self.model.predict(images, imgsz=1024, conf=self.thr, device=0 if self.dev == "cuda" else "cpu",
                                verbose=False, half=self.half)
        return [[(r.names[int(c)], float(s), tuple(float(x) for x in b))
                 for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())] for r in rs]


def detect_pdf(pdf: Path, det: Detector, long_side: int = 1024, batch: int = 8) -> dict:
    """Every page of one PDF -> {"pages": [{"n", "w", "h", "boxes": [[label, score, x0, y0, x1, y1] in POINTS]}]}."""
    from PIL import Image
    pm = _pm()
    doc = pm.open(str(pdf))
    pages, imgs, meta = [], [], []

    def flush():
        if not imgs:
            return
        for (n, w, h, k), dets in zip(meta, det(imgs)):
            pages.append({"n": n, "w": round(w, 1), "h": round(h, 1),
                          "boxes": [[lab, round(sc, 3)] + [round(v / k, 1) for v in b] for lab, sc, b in dets]})
        imgs.clear()
        meta.clear()
    for i, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        k = long_side / max(w, h)
        pix = page.get_pixmap(matrix=pm.Matrix(k, k), alpha=False)
        imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        meta.append((i + 1, w, h, k))
        if len(imgs) >= batch:
            flush()
    flush()
    doc.close()
    return {"pages": pages}


def figures_of(pdf: Path, det_doc: dict, arm: str, min_score: float = 0.5) -> List[dict]:
    """Detections + the PDF's text layer -> figures [{n, page, box, caption, number, panels}] and uncaptioned ones."""
    pm = _pm()
    doc = pm.open(str(pdf))
    out = []
    for pg in det_doc["pages"]:
        page = doc[pg["n"] - 1]
        boxes = [tuple(b[2:6]) for b in pg["boxes"] if b[0] in FIG_LABELS[arm] and b[1] >= min_score
                 and (b[4] - b[2]) * (b[5] - b[3]) >= 0.01 * pg["w"] * pg["h"]]
        figs, rest = match_captions(group_boxes(boxes), caption_blocks(page), pg["h"])
        for f in figs:
            out.append({"page": pg["n"], **f})
        for g in rest:
            out.append({"page": pg["n"], "box": g, "caption": "", "number": "", "panels": 1, "uncaptioned": True})
    doc.close()
    return out


def heuristic_figures(pdf: Path) -> List[dict]:
    """The existing caption-anchored heuristic (tools/paper_figures.py) as an arm, same output shape."""
    import paper_figures as PF
    out = []
    for f in PF.find_figures(str(pdf)):
        out.append({"page": f["page"], "box": tuple(f.get("clip_figure") or f["clip"]), "caption": f["caption"],
                    "number": str(f["fig"]), "panels": 1})
    return out


def render(pdf: Path, page: int, box: Box, out: Path, long_side: int = 1280, dpi: int = 0) -> Path:
    """One figure crop -> an image file (WebP for browsing by default; PNG at `dpi` for export)."""
    pm = _pm()
    doc = pm.open(str(pdf))
    pg = doc[page - 1]
    clip = pm.Rect(*box) & pg.rect
    if dpi:
        pix = pg.get_pixmap(clip=clip, dpi=dpi, alpha=False)
    else:
        k = long_side / max(1.0, max(clip.width, clip.height))
        pix = pg.get_pixmap(clip=clip, matrix=pm.Matrix(k, k), alpha=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".webp":
        from PIL import Image
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(out, "WEBP", quality=82, method=4)
    else:
        pix.save(str(out))
    doc.close()
    return out


# ------------------------------------------------------------------------------------------------- corpus
def _held(pdfs: Path) -> Dict[str, dict]:
    """key -> {sha, file, title} for every held PDF of the BlueBoat index."""
    out = {}
    for sha, n in PT.bwm_names().items():
        f = Path(pdfs) / n["file"] if n.get("file") else None
        if f is not None and f.is_file():
            out[n["key"]] = {"sha": sha, "file": f, "title": n.get("title", "")}
    return out


def det_path(store: Path, arm: str, sha: str) -> Path:
    return store / "det" / arm / sha[:2] / f"{sha}.json.gz"


def detect_all(pdfs: Path, store: Path, arm: str, keys: Optional[Sequence[str]] = None, limit: int = 0,
               say=print, shard: Tuple[int, int] = (0, 1)) -> dict:
    """shard (i, n): only every n-th paper from i. MEASURED 2026-10-14: ONE process (14.9 pages/s alone) beat three
    (~3 pages/s in total, fighting over the GPU) -- shards are for several machines, not one."""
    held = _held(pdfs)

    def done(v):                                      # an error record is NOT done: it is retried
        f = det_path(store, arm, v["sha"])
        if not f.is_file():
            return False
        try:
            return "error" not in json.loads(gzip.decompress(f.read_bytes()))
        except (OSError, ValueError, EOFError):
            return False
    todo = [(k, v) for j, (k, v) in enumerate(sorted(held.items())) if (keys is None or k in keys)
            and j % shard[1] == shard[0] and not done(v)]
    # papers never tried first, retries of error records last (fix check 2026-10-14: ten broken PDFs sorted ahead of
    # a new paper stopped every run at '10 failed in a row' before the new paper was reached)
    todo.sort(key=lambda kv: (det_path(store, arm, kv[1]["sha"]).is_file(), kv[0]))
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"arm": arm, "new": 0}
    if arm == "heuristic":                                    # no model, no detections to cache: `build` finds them
        return {"arm": arm, "new": 0, "note": "the heuristic arm needs no detection -- run build --arm heuristic"}
    det, t0, pages, errors, in_a_row = Detector(arm), time.time(), 0, 0, 0
    for i, (k, v) in enumerate(todo, 1):
        try:
            d = detect_pdf(v["file"], det)
            in_a_row = 0
        except Exception as e:                                # noqa: BLE001 -- recorded, retried on the next run
            if "CUDA" in str(e) or "cuda" in type(e).__name__.lower() or isinstance(e, MemoryError):
                raise                                         # a GPU failure is the run's, not this paper's
            d = {"pages": [], "error": f"{type(e).__name__}: {e}"[:200]}
            errors += 1
            in_a_row += 1
            if in_a_row >= 10:
                raise SystemExit(f"detect: 10 papers failed in a row (last: {k}: {d['error']}) -- stopping")
        out = det_path(store, arm, v["sha"])
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_bytes(gzip.compress(json.dumps(d).encode("utf-8")))
        replace_retry(tmp, out)
        pages += len(d["pages"])
        if i % 25 == 0 or i == len(todo):
            say(f"  [{i}/{len(todo)}] {pages} pages, {pages / max(1e-9, time.time() - t0):.1f} pages/s")
    return {"arm": arm, "new": len(todo), "pages": pages, "errors": errors, "seconds": round(time.time() - t0)}


def eval_arm(pdfs: Path, store: Path, arm: str, keys: Sequence[str], html_cache: Path, say=print,
             allow_partial: bool = False) -> dict:
    """Score an arm on papers whose arXiv HTML is parsed: figures found (recall) and real (precision). A sampled paper
    the arm has no detections for makes it REFUSE (arms compared on different papers are not compared -- fix check
    2026-10-14), unless allow_partial; the skipped keys are named either way."""
    import arxiv_html as AH
    held = _held(pdfs)
    P = json.loads((PT.BWM / "index" / "papers.json").read_text(encoding="utf-8"))
    ax = {p["key"]: re.sub(r"v\d+$", "", p["arxiv_id"]) for p in P["papers"] if p.get("arxiv_id")}
    tot = {"papers": 0, "truth": 0, "pred": 0, "found": 0, "real": 0, "uncaptioned": 0, "version_mismatch": 0,
           "no_detections": 0}
    man = AH._load_manifest(html_cache)
    rows, skipped, why_not = [], [], {}
    for k in keys:
        if k not in held or k not in ax:
            why_not[k] = "not held here" if k not in held else "no arXiv id in the index"   # named, never silent
            continue
        parsed = AH.load_parsed(html_cache, ax[k])
        truth = html_truth(parsed) if parsed else []
        if not truth:
            why_not[k] = "no figures in its arXiv HTML"
            continue
        if (man.get(ax[k]) or {}).get("version_match") is False:
            tot["version_mismatch"] += 1                       # HTML of another version than the held PDF: skipped
            continue
        if arm == "heuristic":
            figs = heuristic_figures(held[k]["file"])
        else:
            dp = det_path(store, arm, held[k]["sha"])
            if not dp.is_file():
                tot["no_detections"] += 1                      # reported: compare arms only on the same papers
                skipped.append(k)
                continue
            figs = figures_of(held[k]["file"], json.loads(gzip.decompress(dp.read_bytes())), arm)
        pred = [f for f in figs if not f.get("uncaptioned")]
        s = score_paper(pred, truth)
        rows.append({"key": k, **s})
        tot["papers"] += 1
        tot["uncaptioned"] += sum(1 for f in figs if f.get("uncaptioned"))
        for x in ("truth", "pred", "found", "real"):
            tot[x] += s[x]
    if skipped and not allow_partial:
        raise SystemExit(f"eval {arm}: no detections for {len(skipped)} sampled paper(s) ({', '.join(skipped[:8])}) "
                         f"-- run `detect --arm {arm}` on them, or pass --allow-partial (then arms are scored on "
                         f"different papers)")
    tot["recall"] = round(tot["found"] / max(1, tot["truth"]), 3)
    tot["precision"] = round(tot["real"] / max(1, tot["pred"]), 3)
    say(f"{arm}: {tot['papers']} papers, {tot['truth']} HTML figures -- recall {tot['recall']} "
        f"({tot['found']}/{tot['truth']}), precision {tot['precision']} ({tot['real']}/{tot['pred']}), "
        f"{tot['uncaptioned']} uncaptioned boxes; skipped: {tot['version_mismatch']} other-version HTML, "
        f"{tot['no_detections']} without detections, {len(why_not)} not scored"
        + (": " + "; ".join(f"{k} ({w})" for k, w in sorted(why_not.items())[:8]) if why_not else ""))
    return {"arm": arm, "total": tot, "papers": rows, "skipped_no_detections": skipped, "not_scored": why_not}


def _det_sig(dp: Path) -> str:
    """A detection file's identity: its size and the first 16 hex of its sha256."""
    import hashlib
    b = dp.read_bytes()
    return f"{len(b)}:{hashlib.sha256(b).hexdigest()[:16]}"


def build_all(pdfs: Path, store: Path, arm: str, keys: Optional[Sequence[str]] = None, say=print) -> dict:
    """figures.json + one WebP per figure for every paper the detector has seen (skips papers already built from these
    detections whose listed images are all on disk; one with an image missing is rebuilt -- 2026-10-27)."""
    held = _held(pdfs)
    n_p = n_f = 0
    for k, v in sorted(held.items()):
        if keys is not None and k not in keys:
            continue
        dp = det_path(store, arm, v["sha"])
        out_dir = store / v["sha"][:2] / v["sha"]
        fj = out_dir / "figures.json"
        if arm == "heuristic":                                 # no model, no detections: the caption heuristic reads
            sig = HEURISTIC_SIG                                # the PDF itself (2026-10-15: the kit's CPU path had
        elif not dp.is_file():                                 # built NO figures -- build needed detections)
            continue
        else:
            sig = _det_sig(dp)
        if fj.is_file():
            try:
                old = json.loads(fj.read_text(encoding="utf-8"))
            except ValueError:
                old = {}
            # built by this arm from these detections: the recorded content signature decides (a bank unpacked on
            # another machine has fresh file times -- fix check 2026-10-14); banks from before it, the file times
            # ... and only while every image it lists is on disk (2026-10-27: abadie2018georeferenced listed 10 images
            # with none on disk, and `atlas.py update --only figures --force` skipped it, so it was never repaired)
            if (old.get("arm") == arm and (old.get("det") == sig if old.get("det")
                                           else dp.is_file() and fj.stat().st_mtime_ns >= dp.stat().st_mtime_ns)
                    and all((out_dir / f.get("image", "-")).is_file() for f in old.get("figures") or [])):
                continue
        if fj.is_file():                                       # rebuilt: the old images go first
            try:
                for f in json.loads(fj.read_text(encoding="utf-8")).get("figures") or []:
                    (out_dir / f.get("image", "-")).unlink(missing_ok=True)
            except ValueError:
                pass
        out_dir.mkdir(parents=True, exist_ok=True)             # a paper with no figures still gets its figures.json
        figs = (heuristic_figures(v["file"]) if arm == "heuristic"
                else figures_of(v["file"], json.loads(gzip.decompress(dp.read_bytes())), arm))
        for i, f in enumerate(figs, 1):
            name = f"fig{i:03d}-{f['number'] or 'u'}-p{f['page']}.webp"      # unique even for repeats
            render(v["file"], f["page"], f["box"], out_dir / name)
            f["image"] = name
            f["box"] = [round(x, 1) for x in f["box"]]
            f.pop("caption_box", None)
        tmp = fj.with_name("figures.json.tmp")
        tmp.write_text(json.dumps({"key": k, "sha256": v["sha"], "arm": arm, "det": sig, "figures": figs},
                                  ensure_ascii=False, indent=0), encoding="utf-8")
        replace_retry(tmp, fj)
        n_p += 1
        n_f += len(figs)
        if n_p % 50 == 0:
            say(f"  built {n_p} papers, {n_f} figures")
    return {"papers": n_p, "figures": n_f}


#: the detector the bank is built with: the `eval` winner (see the module docstring)
DEFAULT_ARM = "yolo"
#: what a heuristic-arm figures.json was built from (no detections: the arm's version stands in for their signature)
HEURISTIC_SIG = "heuristic/1"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["detect", "build", "eval", "export", "show"])
    ap.add_argument("key", nargs="?", default="")
    ap.add_argument("--bwm", action="store_true")
    ap.add_argument("--arm", default=DEFAULT_ARM, choices=["ppdl3", "yolo", "heuristic"])
    ap.add_argument("--pdfs", type=Path, default=PT.DEFAULT_PDFS)
    ap.add_argument("--store", type=Path, default=None, help="figure store (default <pdfs>_figures)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys", default="", help="comma list of paper keys (detect / build / eval)")
    ap.add_argument("--n", type=int, default=60, help="eval: seeded sample of arXiv papers with parsed HTML")
    ap.add_argument("--out", type=Path, default=None, help="export: output folder; eval: result JSON")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fig", default="", help="export: one figure number")
    ap.add_argument("--shard", default="0/1", help="detect: i/n -- this process takes every n-th paper from i")
    ap.add_argument("--allow-partial", action="store_true",
                    help="eval: score although some sampled papers have no detections for this arm (recorded)")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    store = a.store or default_store(a.pdfs)
    keys = [k for k in a.keys.split(",") if k] or None
    if a.cmd == "detect":
        i, n = (int(x) for x in a.shard.split("/"))
        print(json.dumps(detect_all(a.pdfs, store, a.arm, keys, a.limit, shard=(i, n))))
        return 0
    if a.cmd == "build":
        print(json.dumps(build_all(a.pdfs, store, a.arm, keys)))
        return 0
    if a.cmd == "eval":
        import random
        import arxiv_html as AH
        hc = AH.default_cache(a.pdfs)
        man = AH._load_manifest(hc)
        if keys is None:                               # held papers whose INDEX arXiv id has HTML (review
            held_keys = {it["key"] for it in AH.held_arxiv(PT.default_cache(a.pdfs))}   # 2026-10-14: a stale row's
            ok = sorted(r["key"] for r in man.values() if r.get("status") == "ok" and r.get("key") in held_keys)
            keys = random.Random(13).sample(ok, min(a.n, len(ok)))
        res = eval_arm(a.pdfs, store, a.arm, keys, hc, allow_partial=a.allow_partial)
        if a.out:
            a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")   # LF: it is committed
        return 0
    held = _held(a.pdfs)
    if a.key not in held:
        print(f"{a.cmd}: {a.key!r} is not a held paper")
        return 1
    v = held[a.key]
    fj = store / v["sha"][:2] / v["sha"] / "figures.json"
    if not fj.is_file():
        print(f"{a.cmd}: no figures built for {a.key} (detect + build first)")
        return 1
    figs = json.loads(fj.read_text(encoding="utf-8"))["figures"]
    if a.cmd == "show":
        for f in figs:
            print(f"p.{f['page']:<3} Fig. {f['number'] or '-':<4} {f['image']}  {f['caption'][:110]}")
        return 0
    out = a.out or Path(".")
    for f in figs:
        if a.fig and f["number"] != a.fig:
            continue
        p = render(v["file"], f["page"], tuple(f["box"]), out / f"{a.key}_{Path(f['image']).stem}.png", dpi=a.dpi)
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
