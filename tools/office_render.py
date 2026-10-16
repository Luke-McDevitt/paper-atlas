"""office_render -- render a .pptx / .docx through the INSTALLED Microsoft Office (COM) to PNGs (+ PDF) and check it.

    python tools/office_render.py deck.pptx --out-dir out/render                  # every slide -> PNG + report
    python tools/office_render.py deck.pptx --out-dir out/render --pages 1-3,9 --pdf
    python tools/office_render.py glossary.docx --out-dir out/render --pages 1-4  # Word pages -> PNG via EMF

WHY. A generated deck or document that BUILDS is not one that READS: text overflows its box, a slide has no speaker
notes, an equation did not convert, an internal link points at a missing bookmark. Looking is the check, and on this
lab computer there is no LibreOffice or pdftoppm (the skills' soffice route does not exist here) -- but PowerPoint and
Word are installed and drivable over COM. This renders what the lab will actually open, and writes `render_report.json`
so the mechanical checks are a number, not an impression. Generated decks/glossaries in this repo were QA'd with it.

CHECKS (render_report.json; also printed):
  pptx  slides, notes characters per slide (empty notes = issue), OVERFLOW: text shapes whose laid-out text is taller
        than the box by more than --overflow (default 15 %) -- pptxgenjs boxes marked "shrink" are shrunk by
        PowerPoint only when edited/re-laid-out, so an overflow here means small or clipped text on screen.
  docx  pages, equations (OMaths), internal hyperlinks whose bookmark is missing (= broken), Heading-3 paragraphs and
        how many open collapsed (only with --deep: it walks every paragraph, slow on long documents).
EXIT CODES (repo contract): 0 = rendered, no issue; 1 = rendered, issues found (listed); 2 = could not render (not
Windows, Office missing, COM error).

PowerShell variable names are CASE-INSENSITIVE: a script variable that differs from a parameter only in case
overwrites it (the first version's $pages page COUNT replaced the -Pages LIST, so a docx exported only its last
page). Keep script variables distinct from the param names.

HONEST SCOPE: Windows + Microsoft Office only (it is the renderer). The overflow check reads PowerPoint's own text
layout of each shape (TextRange.BoundHeight), not table cells. Word pages are rasterised from Word's page metafile,
which matches Word's print layout. Nothing here edits the input file: it is opened read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PS_PPTX = r"""
param([string]$Path, [string]$OutDir, [string]$Pages, [int]$Width, [int]$Pdf, [double]$Overflow)
$ErrorActionPreference = 'Stop'
$app = New-Object -ComObject PowerPoint.Application
try {
  $p = $app.Presentations.Open($Path, -1, 0, 0)
  $n = $p.Slides.Count
  $want = @{}
  if ($Pages) { foreach ($x in $Pages.Split(',')) { $want[[int]$x] = $true } }
  $h = [int]($Width * $p.PageSetup.SlideHeight / $p.PageSetup.SlideWidth)
  $slides = @()
  for ($i = 1; $i -le $n; $i++) {
    $s = $p.Slides.Item($i)
    $notes = ''
    foreach ($ph in $s.NotesPage.Shapes.Placeholders) { if ($ph.PlaceholderFormat.Type -eq 2) { $notes = $ph.TextFrame.TextRange.Text } }
    $over = @()
    foreach ($sh in $s.Shapes) {
      if ($sh.HasTextFrame -and $sh.TextFrame.HasText) {
        $bh = $sh.TextFrame.TextRange.BoundHeight
        if ($sh.Height -gt 0 -and $bh -gt $sh.Height * (1 + $Overflow)) {
          $t = $sh.TextFrame.TextRange.Text; if ($t.Length -gt 60) { $t = $t.Substring(0, 60) }
          $over += @{ shape = $sh.Name; text = $t; box_pt = [math]::Round($sh.Height, 1); text_pt = [math]::Round($bh, 1) }
        }
      }
    }
    $png = ''
    if (-not $Pages -or $want.ContainsKey($i)) {
      $png = Join-Path $OutDir ('slide_{0:D2}.png' -f $i)
      $s.Export($png, 'PNG', $Width, $h)
    }
    $slides += @{ n = $i; notes_chars = $notes.Length; overflow = $over; png = $png }
  }
  $pdfPath = ''
  if ($Pdf -eq 1) { $pdfPath = Join-Path $OutDir 'render.pdf'; $p.SaveAs($pdfPath, 32) }
  $p.Close()
  @{ kind = 'pptx'; slides = $slides; n = $n; pdf = $pdfPath } | ConvertTo-Json -Depth 6 -Compress
} finally { $app.Quit() }
"""

PS_DOCX = r"""
param([string]$Path, [string]$OutDir, [string]$Pages, [int]$Width, [int]$Pdf, [int]$Deep)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$w = New-Object -ComObject Word.Application
$w.Visible = $false
try {
  $d = $w.Documents.Open($Path, $false, $true)
  $nPages = $d.ComputeStatistics(2)   # NOT $pages: PowerShell names are case-insensitive and $Pages is the param
  $broken = @()
  foreach ($hl in $d.Hyperlinks) {
    $sub = $hl.SubAddress
    if ($sub -and -not $d.Bookmarks.Exists($sub)) { $broken += $sub }
  }
  $h3 = -1; $col = -1
  if ($Deep -eq 1) {
    $h3 = 0; $col = 0
    foreach ($para in $d.Paragraphs) { if ($para.OutlineLevel -eq 3) { $h3++; if ($para.CollapsedState) { $col++ } } }
  }
  $pngs = @()
  if ($Pages) {
    $pane = $d.ActiveWindow.Panes.Item(1)
    foreach ($x in $Pages.Split(',')) {
      $i = [int]$x
      if ($i -gt $nPages) { continue }
      $bytes = $pane.Pages.Item($i).EnhMetaFileBits
      $ms = New-Object System.IO.MemoryStream(,$bytes)
      $mf = New-Object System.Drawing.Imaging.Metafile($ms)
      $hh = [int]($Width * $mf.Height / $mf.Width)
      $bmp = New-Object System.Drawing.Bitmap($Width, $hh)
      $g = [System.Drawing.Graphics]::FromImage($bmp)
      $g.Clear([System.Drawing.Color]::White)
      $g.DrawImage($mf, 0, 0, $Width, $hh)
      $png = Join-Path $OutDir ('page_{0:D2}.png' -f $i)
      $bmp.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
      $g.Dispose(); $bmp.Dispose(); $mf.Dispose()
      $pngs += $png
    }
  }
  $pdfPath = ''
  if ($Pdf -eq 1) { $pdfPath = Join-Path $OutDir 'render.pdf'; $d.ExportAsFixedFormat($pdfPath, 17) }
  $res = @{ kind = 'docx'; pages = $nPages; omaths = $d.OMaths.Count; hyperlinks = $d.Hyperlinks.Count;
            broken_links = $broken; heading3 = $h3; heading3_collapsed = $col; pngs = $pngs; pdf = $pdfPath }
  $d.Close(0)
  $res | ConvertTo-Json -Depth 5 -Compress
} finally { $w.Quit() }
"""


def parse_pages(text, n=None):
    """'1-3,9' -> [1, 2, 3, 9] (sorted, unique, >= 1, <= n when given). '' -> []. Raises ValueError. Pure."""
    out = set()
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                raise ValueError(f"bad page range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if any(p < 1 for p in out):
        raise ValueError("pages are 1-based")
    return sorted(p for p in out if n is None or p <= n)


def issues(report):
    """The mechanical problems in a render report, as readable strings. Pure."""
    out = []
    if report.get("kind") == "pptx":
        for s in report.get("slides", []):
            if not s.get("notes_chars"):
                out.append(f"slide {s['n']}: no speaker notes")
            for o in s.get("overflow") or []:
                out.append(f"slide {s['n']}: text overflows its box ({o['text_pt']} pt of text in {o['box_pt']} pt): "
                           f"{o['text']!r}")
    elif report.get("kind") == "docx":
        for b in report.get("broken_links") or []:
            out.append(f"internal link to missing bookmark {b!r}")
        if report.get("heading3", -1) > 0 and report.get("heading3_collapsed", -1) not in (-1, report["heading3"]):
            out.append(f"{report['heading3_collapsed']} of {report['heading3']} Heading-3 open collapsed")
    return out


def contact_sheet(pngs, out_path, cols=4, tile_w=480, gap=8):
    """Tile rendered pages into one labelled PNG -- one image to look at instead of N (cheaper to review, human or
    model). Needs Pillow; returns the path, or None when there is nothing to tile."""
    from PIL import Image, ImageDraw
    pngs = [p for p in pngs if p and Path(p).exists()]
    if not pngs:
        return None
    ims = [Image.open(p).convert("RGB") for p in pngs]
    th = max(int(tile_w * im.height / im.width) for im in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tile_w + gap) + gap, rows * (th + gap + 18) + gap), "white")
    d = ImageDraw.Draw(sheet)
    for i, (p, im) in enumerate(zip(pngs, ims)):
        r, c = divmod(i, cols)
        x, y = gap + c * (tile_w + gap), gap + r * (th + gap + 18)
        sheet.paste(im.resize((tile_w, int(tile_w * im.height / im.width))), (x, y + 18))
        d.text((x, y + 2), Path(p).stem, fill=(90, 90, 90))
    sheet.save(out_path)
    return str(out_path)


def _ps():
    return shutil.which("powershell") or shutil.which("powershell.exe")


def render(path, out_dir, pages="", width=1600, pdf=False, overflow=0.15, deep=False):
    """Run the COM script; -> report dict. Raises RuntimeError when it could not render."""
    path, out_dir = Path(path).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = path.suffix.lower().lstrip(".")
    if kind not in ("pptx", "docx"):
        raise RuntimeError(f"unsupported file type {path.suffix!r} (pptx or docx)")
    if os.name != "nt" or not _ps():
        raise RuntimeError("needs Windows PowerShell + Microsoft Office (COM)")
    script = PS_PPTX if kind == "pptx" else PS_DOCX
    with tempfile.TemporaryDirectory() as td:
        ps1 = Path(td) / f"render_{kind}.ps1"
        ps1.write_text(script, encoding="ascii", newline="\r\n")
        pg = ",".join(map(str, parse_pages(pages))) if pages else ""
        args = [_ps(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
                "-Path", str(path), "-OutDir", str(out_dir), "-Pages", pg, "-Width", str(width), "-Pdf", str(int(pdf))]
        args += ["-Overflow", str(overflow)] if kind == "pptx" else ["-Deep", str(int(deep))]
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError((r.stderr or r.stdout or "no output").strip()[-600:])
    report = json.loads(r.stdout.strip().splitlines()[-1])
    report["file"] = str(path)
    report["issues"] = issues(report)
    (out_dir / "render_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", help=".pptx or .docx")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pages", default="", help="pptx: slides to export (default all); docx: pages to rasterise, e.g. 1-3,9")
    ap.add_argument("--width", type=int, default=1600, help="PNG width in pixels (default 1600)")
    ap.add_argument("--pdf", action="store_true", help="also save render.pdf")
    ap.add_argument("--overflow", type=float, default=0.15, help="pptx: flag text taller than its box by this fraction")
    ap.add_argument("--deep", action="store_true", help="docx: also count collapsed Heading-3 paragraphs (slow)")
    ap.add_argument("--contact", type=int, default=0, metavar="COLS",
                    help="also tile every exported PNG into contact.png with COLS columns (one image to review)")
    a = ap.parse_args(argv)
    try:
        if a.pages:
            parse_pages(a.pages)
        rep = render(a.file, a.out_dir, a.pages, a.width, a.pdf, a.overflow, a.deep)
    except (RuntimeError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"ERROR: could not render: {str(e)[:400]}".encode("ascii", "replace").decode())
        return 2
    if rep["kind"] == "pptx":
        print(f"{rep['n']} slides; exported {sum(bool(s['png']) for s in rep['slides'])} PNG(s) to {a.out_dir}")
    else:
        print(f"{rep['pages']} pages, {rep['omaths']} equations, {rep['hyperlinks']} hyperlinks; "
              f"{len(rep['pngs'])} PNG(s) to {a.out_dir}")
    if a.contact:
        pngs = [s["png"] for s in rep.get("slides", [])] if rep["kind"] == "pptx" else rep.get("pngs", [])
        cs = contact_sheet(pngs, Path(a.out_dir) / "contact.png", cols=a.contact)
        print(f"contact sheet: {cs}")
    for i in rep["issues"]:
        print("ISSUE: " + i.encode("ascii", "replace").decode())
    print(f"{len(rep['issues'])} issue(s); report: {Path(a.out_dir) / 'render_report.json'}")
    return 1 if rep["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
