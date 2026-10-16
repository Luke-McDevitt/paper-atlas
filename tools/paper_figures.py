"""paper_figures -- find the captioned figures in a paper PDF and render them to PNG, so a slide can show the paper's
own figure. Choices are stored as DATA (page + clip box), and the images are re-rendered from the local PDF each time.

    python tools/paper_figures.py list <pdf> --out-dir <dir> [--prefix key]   # every figure -> PNG, figures.json, contact.png
    python tools/paper_figures.py render <spec.json> --pdf-dir <dir> --out-dir <dir>   # the chosen ones, for a deck build

HOW A FIGURE IS FOUND. A text block that STARTS with "Fig. N" / "Figure N" is a caption; the figure is the union of the
raster images and vector drawings above it (within --max-up points, overlapping the caption's columns), extended to the
caption's bottom so the caption travels with it. Rendered with PyMuPDF at --zoom (default 2).

WHY. The owner wanted one slide per most-promising paper with the paper's own figures (2026-10-01). Screenshotting PDFs
by hand is not re-runnable; committing the publishers' images is not ours to do. A spec of {key, file, page, clip}
is small, reviewable and committed; the PNGs live next to the gitignored PDFs and are rebuilt on demand.

HONEST SCOPE: caption-anchored geometry, not layout understanding. A figure whose caption sits ABOVE it, a figure built
from text (a table, an algorithm box) or one split across columns can come out cropped or padded -- `list` renders every
candidate and a contact sheet precisely so a person picks and can hand-adjust a clip before it is used. Needs PyMuPDF
(`python -m pip install pymupdf`; AGPL -- fine for internal use, per the repo's licence policy).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPTION = re.compile(r"^(?:Fig\.?|FIG\.?|Figure|FIGURE)\s*(\d+)\s*[.:|]")


def _pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError as e:
        raise RuntimeError("PyMuPDF is not installed: python -m pip install pymupdf") from e


def figure_clip(caption, graphics, page_rect, max_up=520.0, with_caption=True):
    """Clip box (x0, y0, x1, y1) for the figure whose caption box is `caption`, from the graphic boxes on the page, or
    None when nothing sits above the caption. Boxes are (x0, y0, x1, y1) tuples in PDF points. Pure."""
    cx0, cy0, cx1, cy1 = caption
    # a graphic belongs to this caption when it sits above it and its CENTRE lies within the caption's span: an overlap
    # tolerance let the neighbouring column's figure leak in (caught by a test, 2026-10-01)
    above = [g for g in graphics
             if g[3] <= cy0 + 2 and g[1] >= cy0 - max_up and cx0 - 10 <= (g[0] + g[2]) / 2 <= cx1 + 10
             and (g[2] - g[0]) > 4 and (g[3] - g[1]) > 4]
    if not above:
        return None
    x0 = min(min(g[0] for g in above), cx0)
    y0 = min(g[1] for g in above)
    x1 = max(max(g[2] for g in above), cx1)
    px0, py0, px1, py1 = page_rect
    bottom = cy1 + 2 if with_caption else cy0 - 1          # the slide carries its own caption
    return (max(x0 - 4, px0), max(y0 - 4, py0), min(x1 + 4, px1), min(bottom, py1))


def find_figures(pdf, max_up=520.0):
    """[{page, fig, caption, clip}] for every caption-anchored figure in a PDF."""
    pm = _pymupdf()
    doc = pm.open(str(pdf))
    out = []
    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("dict")["blocks"]
        texts = [(tuple(b["bbox"]), " ".join(s["text"] for ln in b["lines"] for s in ln["spans"]).strip())
                 for b in blocks if b["type"] == 0]
        graphics = [tuple(b["bbox"]) for b in blocks if b["type"] == 1]
        try:
            graphics += [tuple(d["rect"]) for d in page.get_drawings()]
        except Exception:
            pass
        for box, txt in texts:
            m = CAPTION.match(txt)
            if not m:
                continue
            clip = figure_clip(box, graphics, tuple(page.rect), max_up)
            if clip:
                bare = figure_clip(box, graphics, tuple(page.rect), max_up, with_caption=False)
                out.append({"page": pno + 1, "fig": int(m.group(1)), "caption": re.sub(r"\s+", " ", txt)[:300],
                            "clip": [round(v, 1) for v in clip], "clip_figure": [round(v, 1) for v in bare]})
    return out


def render(pdf, page, clip, png, zoom=2.0):
    """Render one clip of one page (1-based) to PNG; returns the PNG path."""
    pm = _pymupdf()
    doc = pm.open(str(pdf))
    pix = doc[page - 1].get_pixmap(matrix=pm.Matrix(zoom, zoom), clip=pm.Rect(*clip))
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(png))
    return str(png)


def _contact(pngs, out):
    spec = importlib.util.spec_from_file_location("office_render", ROOT / "tools" / "office_render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.contact_sheet(pngs, out, cols=4, tile_w=420)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    li = sub.add_parser("list", help="render every caption-anchored figure of one PDF for review")
    li.add_argument("pdf")
    li.add_argument("--out-dir", required=True)
    li.add_argument("--prefix", default="", help="file-name prefix (e.g. the option key)")
    li.add_argument("--max-up", type=float, default=520.0)
    li.add_argument("--zoom", type=float, default=1.0, help="review renders are small (default 1x)")
    rd = sub.add_parser("render", help="render the figures a spec chose, from the local PDFs")
    rd.add_argument("spec", help="JSON list of {key, file, page, clip, ...}")
    rd.add_argument("--pdf-dir", required=True)
    rd.add_argument("--out-dir", required=True)
    rd.add_argument("--zoom", type=float, default=2.0)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "list":
            figs = find_figures(a.pdf, a.max_up)
            out = Path(a.out_dir)
            pngs = []
            for f in figs:
                f["png"] = render(a.pdf, f["page"], f["clip"],
                                  out / f"{a.prefix + '_' if a.prefix else ''}p{f['page']:02d}_fig{f['fig']}.png", a.zoom)
                pngs.append(f["png"])
            (out / f"{a.prefix or 'figures'}.json").write_text(json.dumps(figs, indent=1, ensure_ascii=False) + "\n",
                                                               encoding="utf-8", newline="\n")
            sheet = _contact(pngs, out / f"{a.prefix or 'figures'}_contact.png") if pngs else None
            print(f"{len(figs)} figure(s) found in {Path(a.pdf).name}; contact sheet: {sheet}")
        else:
            spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
            for f in spec:
                pdf = Path(a.pdf_dir) / f["file"]
                if not pdf.exists():
                    print(f"WARNING: {f['file']} not in {a.pdf_dir} -- {f['key']} renders without its figure")
                    continue
                render(pdf, f["page"], f["clip"], Path(a.out_dir) / f"{f['key']}.png", a.zoom)
            print(f"rendered {len(spec)} figure(s) -> {a.out_dir}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
