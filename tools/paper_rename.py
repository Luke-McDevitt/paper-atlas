"""paper_rename -- rename (or copy) paper PDFs to their TITLE, the way the lab library names papers:
`<FirstAuthorSurname>_<Year>_<Title_Words>.pdf` (ASCII, no spaces), so a folder of `1-s2.0-S0029...-main.pdf`,
`s7.pdf` and `jmse-12-02173.pdf` reads -- and searches -- as what the papers are.

    python tools/paper_rename.py C:/svfr/papers/boat_wm --bwm                  # DRY RUN: old -> new, and why
    python tools/paper_rename.py C:/svfr/papers/boat_wm --bwm --apply          # rename in place (renames.tsv logged)
    python tools/paper_rename.py "C:/Users/me/Downloads/papers" --copy-to D:/sorted     # any PDFs, copies
    python tools/paper_rename.py one.pdf --pattern "{title}" --sep " " --apply           # just the title, spaces kept

WHY (2026-10-12, owner: "copy / keep research papers in easily searched and indexed directory, make a tool that
renames a paper pdf to its title"). Publishers name downloads by their own ids; a person (Explorer / Everything
search) and the lab library (`tools/library/RULES.md`: `20_papers/<collection>/Hafner_2023_DreamerV3.pdf`) both want
names that say what the paper is.

WHERE THE TITLE COMES FROM, best first (the `source` column of the dry run):
  index    `--bwm`: the BlueBoat world-model index, through `report/boat_world_model/papers/_held.json` BY CONTENT
           (sha256) -- a verified title, authors and year; or `--index <papers.json>` + `--held <held.json>`.
  metadata the PDF's own Title / Author / CreationDate, when the title is plausible (not "untitled", a file name, a
           journal or a figure name)
  page 1   the largest-font line(s) at the top of page 1 (pdfminer layout) -- a heuristic, marked as such
A PDF with no usable title is left alone and listed. Never overwrites: an occupied name gets `_2`, `_3`, ...; a file
with identical content already at the target is reported, not duplicated. `--apply` appends every rename to
`<folder>/renames.tsv` (old name, new name, date, source), as library rule 3 asks.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_project as _AP  # noqa: E402
BWM = _AP.project_dir()              # the atlas project `--bwm` means (default: the BlueBoat one)
DEFAULT_PATTERN = "{author}_{year}_{title}"
MAX_STEM = 120
_BAD_TITLE = re.compile(r"^(untitled|microsoft word|print|document\d*|title|abstract|article|paper|preprint|"
                        r"manuscript|cover|front matter|contents)\b|\.(docx?|tex|dvi|pdf|eps|indd|qxd)$|^\S+\.\w{2,4}$|"
                        r"^[\w-]*\d{3,}[\w-]*$|^\d|fig(ure)?[-_ ]?\d", re.I)
_ALL_CAPS = re.compile(r"^[A-Z0-9 _.,:&-]{3,}$")       # a journal / series header, not a title (case-SENSITIVE)
#: letters NFKD cannot decompose to ASCII: transliterate instead of dropping them (Gierłowski, Ørsted, Straße)
_TRANSLIT = {0x131: "i", 0x142: "l", 0x141: "L", 0xF8: "o", 0xD8: "O", 0xDF: "ss", 0xE6: "ae", 0xC6: "AE",
             0x153: "oe", 0x152: "OE", 0x111: "d", 0x110: "D", 0xF0: "d", 0xFE: "th", 0x2010: "-", 0x2013: "-",
             0x2014: "-"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- naming (pure)
def ascii_words(s: str) -> List[str]:
    """HTML tags and entities removed (titles from publishers carry '<scp>3D</scp>', '&amp;'), accents dropped,
    ligatures and Unicode dashes folded, then the alphanumeric words."""
    s = html.unescape(html.unescape(str(s or "")))
    s = re.sub(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>", " ", s)
    s = unicodedata.normalize("NFKD", s.translate(_TRANSLIT))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[A-Za-z0-9]+", s.encode("ascii", "ignore").decode())


def surname(authors) -> str:
    """The first author's surname from ['Brian Bingham', ...], 'Bingham, Brian; ...' or 'B. Bingham and ...'."""
    if isinstance(authors, (list, tuple)):
        first = next((a for a in authors if a and a != "et al."), "")
    else:
        first = re.split(r";| and |&|\n", str(authors or ""))[0]
    first = first.strip()
    if not first:
        return ""
    if "," in first:                                     # 'Bingham, Brian'
        name = first.split(",")[0]
    else:
        parts = [p for p in re.split(r"\s+", first) if p]
        while len(parts) > 1 and re.fullmatch(r"(?i)(jr\.?|sr\.?|ii|iii|iv|phd|dr\.?)", parts[-1]):
            parts.pop()
        name = parts[-1] if parts else ""
    words = ascii_words(name)
    return "".join(w[:1].upper() + w[1:] for w in words)[:30]


def file_name(title: str, author: str = "", year=None, key: str = "", pattern: str = DEFAULT_PATTERN, sep: str = "_",
              max_stem: int = MAX_STEM) -> str:
    """The new file name. Empty fields drop out with their separators; the title is cut at a word boundary so the
    stem stays within `max_stem` characters. Pure."""
    words = ascii_words(title)
    au = surname([author]) if author and (" " in author.strip() or "," in author) else "".join(ascii_words(author))[:30]
    fields = {"author": au[:1].upper() + au[1:], "year": str(year or "").strip()[:4], "key": "".join(ascii_words(key))}
    head = pattern.replace("{title}", "\0")
    for k, v in fields.items():
        head = head.replace("{" + k + "}", v)
    pre, _, post = head.partition("\0")
    clean = lambda s: re.sub(r"(?:[_ -])+", sep, re.sub(r"[^A-Za-z0-9_ -]+", "", s)).strip("_ -")
    pre, post = clean(pre), clean(post)
    room = max_stem - len(pre) - len(post) - 2 * len(sep)
    t = ""
    for w in words:
        nxt = (t + sep + w) if t else w
        if len(nxt) > room:
            break
        t = nxt
    stem = sep.join(x for x in (pre, t, post) if x) or "paper"
    return stem + ".pdf"


def plausible_title(t: str) -> bool:
    t = " ".join(str(t or "").split())
    return 8 <= len(t) <= 300 and len(t.split()) >= 2 and not _BAD_TITLE.search(t) and not _ALL_CAPS.match(t)


# ------------------------------------------------------------------------------------------- title sources
def bwm_lookup() -> Dict[str, dict]:
    """sha256 -> {title, authors, year, key} for every PDF the BlueBoat world-model index holds."""
    held = json.loads((BWM / "papers" / "_held.json").read_text(encoding="utf-8"))
    papers = {p["key"]: p for p in json.loads((BWM / "index" / "papers.json").read_text(encoding="utf-8"))["papers"]}
    out = {}
    for r in held.get("files") or []:
        p = papers.get(r["key"])
        if p:
            out[r["sha256"]] = {"title": p["title"], "authors": p.get("authors") or [], "year": p.get("year"),
                                "key": p["key"]}
    return out


def index_lookup(index: Path, held: Path) -> Dict[str, dict]:
    """sha256 -> paper, from any {papers: [{key, title, authors, year}]} index and a {files: [{key, sha256}]} manifest."""
    data = json.loads(Path(index).read_text(encoding="utf-8"))
    rows = data.get("papers", data) if isinstance(data, dict) else data
    papers = {r["key"]: r for r in rows if isinstance(r, dict) and r.get("key")}
    out = {}
    for r in json.loads(Path(held).read_text(encoding="utf-8")).get("files") or []:
        p = papers.get(r.get("key"))
        if p and r.get("sha256"):
            out[r["sha256"]] = {"title": p.get("title", ""), "authors": p.get("authors") or [], "year": p.get("year"),
                                "key": p["key"]}
    return out


def metadata(path) -> dict:
    """{title, author, year} from the PDF's info dictionary ('' when absent)."""
    try:
        import pypdf
        info = pypdf.PdfReader(str(path)).metadata or {}
        title = str(info.get("/Title") or "").strip()
        author = str(info.get("/Author") or "").strip()
        m = re.search(r"(19[5-9]\d|20[0-4]\d)", str(info.get("/CreationDate") or ""))
        return {"title": title, "author": author, "year": m.group(1) if m else ""}
    except Exception:                                    # noqa: BLE001 -- an unreadable PDF has no metadata
        return {"title": "", "author": "", "year": ""}


def page1_title(path) -> str:
    """The largest-font text near the top of page 1 (pdfminer layout): the title of almost any typeset paper."""
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTChar, LTTextContainer, LTTextLine
    except ImportError:
        return ""
    lines: List[Tuple[float, float, str]] = []
    try:
        for page in extract_pages(str(path), maxpages=1):
            for el in page:
                if not isinstance(el, LTTextContainer):
                    continue
                for line in el:
                    if not isinstance(line, LTTextLine):
                        continue
                    sizes = [c.size for c in line if isinstance(c, LTChar)]
                    text = " ".join(line.get_text().split())
                    if sizes and len(text) >= 3:
                        lines.append((round(max(set(sizes), key=sizes.count), 1), -line.y1, text))
    except Exception:                                    # noqa: BLE001
        return ""
    if not lines:
        return ""
    cands = [ln for ln in lines if plausible_title(ln[2]) or len(ln[2].split()) >= 2]
    if not cands:
        return ""
    biggest = max(s for s, _, _ in cands)
    top = sorted((ln for ln in lines if ln[0] >= biggest - 0.6), key=lambda ln: ln[1])
    title = " ".join(t for _, _, t in top[:4])
    return title if plausible_title(title) else ""


def identify(path, known: Optional[Dict[str, dict]] = None, use_page1: bool = True) -> dict:
    """{title, author, year, key, source, sha256} for one PDF -- from `known` (by content), its metadata, or page 1."""
    sha = sha256_file(path)
    k = (known or {}).get(sha)
    if k:
        author = (k.get("authors") or [""])[0] if isinstance(k.get("authors"), list) else k.get("authors", "")
        if not author or author == "et al.":         # the index has no authors for it: the PDF's own metadata
            author = metadata(path)["author"]
        return {"title": k["title"], "author": author, "year": k.get("year"), "key": k.get("key", ""),
                "source": "index", "sha256": sha}
    meta = metadata(path)
    if plausible_title(meta["title"]):
        return {**meta, "key": "", "source": "metadata", "sha256": sha}
    t = page1_title(path) if use_page1 else ""
    if t:
        return {"title": t, "author": meta["author"], "year": meta["year"], "key": "", "source": "page 1",
                "sha256": sha}
    return {"title": "", "author": "", "year": "", "key": "", "source": "none", "sha256": sha}


# ------------------------------------------------------------------------------------------------- planning
def plan(paths: List[Path], known=None, pattern=DEFAULT_PATTERN, sep="_", dest: Optional[Path] = None,
         use_page1=True) -> List[dict]:
    """[{src, dst, source, title, status}] without touching anything. status: rename | same (already so named) |
    duplicate (identical content already at the target) | no title."""
    out, taken = [], {}
    for p in sorted(paths, key=lambda x: x.name.lower()):
        info = identify(p, known, use_page1)
        row = {"src": str(p), "source": info["source"], "title": info["title"], "key": info.get("key", ""),
               "sha256": info["sha256"]}
        if not info["title"]:
            out.append({**row, "dst": "", "status": "no title"})
            continue
        folder = Path(dest) if dest else p.parent
        name = file_name(info["title"], info["author"], info["year"], info.get("key", ""), pattern, sep)
        stem, n = name[:-4], 1
        while True:
            target = folder / name
            same_file = target.resolve() == p.resolve() if target.exists() else False
            if same_file and target.name == p.name:
                out.append({**row, "dst": str(target), "status": "same"})
                break
            if same_file:                                 # x.pdf -> X.pdf on a case-insensitive disk
                taken[str(target).lower()] = info["sha256"]
                out.append({**row, "dst": str(target), "status": "rename"})
                break
            claimed = taken.get(str(target).lower())
            if not target.exists() and not claimed:
                taken[str(target).lower()] = info["sha256"]
                out.append({**row, "dst": str(target), "status": "rename"})
                break
            if (claimed == info["sha256"]) or (target.exists() and sha256_file(target) == info["sha256"]):
                out.append({**row, "dst": str(target), "status": "duplicate"})
                break
            n += 1
            name = f"{stem}{sep}{n}.pdf"
    return out


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def apply(rows: List[dict], copy: bool = False, log: Optional[Path] = None,
          errors: Optional[List[Tuple[str, str]]] = None) -> int:
    """Rename (or copy) the planned rows. Each rename is appended to `log` the moment it is done, so a run that stops
    half way still records what it changed; a file that cannot be moved (open in a PDF viewer) goes to `errors` and
    the others go on."""
    done = 0
    today = datetime.date.today().isoformat()
    for r in rows:
        if r["status"] != "rename":
            continue
        src, dst = Path(r["src"]), Path(r["dst"])
        case_only = not copy and dst.exists() and _same_file(src, dst) and src.name != dst.name
        if dst.exists() and not case_only:                # appeared since planning: never overwrite
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                tmp = dst.with_name(dst.name + ".part")
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)
            elif case_only:                               # via a temporary name: some disks ignore a case-only rename
                tmp = src.with_name(src.name + ".renaming")
                os.rename(src, tmp)
                os.rename(tmp, dst)
            else:
                os.rename(src, dst)
        except OSError as e:
            if errors is not None:
                errors.append((src.name, str(e)))
            continue
        done += 1
        if log and not copy:
            new = not Path(log).exists()
            with open(log, "a", encoding="utf-8", newline="\n") as fh:
                if new:
                    fh.write("old\tnew\tdate\ttitle_source\n")
                fh.write(f"{src.name}\t{dst.name}\t{today}\t{r['source']}\n")
    return done


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="PDF files and/or folders of PDFs")
    ap.add_argument("--apply", action="store_true", help="rename (default: a dry run that changes nothing)")
    ap.add_argument("--copy-to", type=Path, default=None, help="copy under the new names into this folder instead")
    ap.add_argument("--bwm", action="store_true", help="titles from the BlueBoat world-model index (by content)")
    ap.add_argument("--index", type=Path, default=None, help="titles from this {papers: [...]} index (with --held)")
    ap.add_argument("--held", type=Path, default=None, help="{files: [{key, sha256}]} manifest for --index")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="fields {author} {year} {title} {key}; default "
                                                              f"{DEFAULT_PATTERN!r} (the lab library's)")
    ap.add_argument("--sep", default="_", help="word separator (default '_'; use ' ' to keep spaces)")
    ap.add_argument("--no-page1", action="store_true", help="never guess a title from page 1's layout")
    a = ap.parse_args(argv)
    files: List[Path] = []
    for p in a.paths:
        if p.is_dir():
            files += [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
        elif p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p)
        else:
            print(f"skip: {p} is not a PDF or folder")
    if not files:
        print("no PDFs given")
        return 2
    known = bwm_lookup() if a.bwm else (index_lookup(a.index, a.held) if a.index and a.held else None)
    if a.index and not a.held:
        ap.error("--index needs --held (which file is which paper)")
    rows = plan(files, known, a.pattern, a.sep, a.copy_to, not a.no_page1)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] in ("rename", "duplicate", "no title"):
            arrow = Path(r["dst"]).name if r["dst"] else "(left alone: no usable title)"
            print(f"  [{r['source']:8s}] {Path(r['src']).name}\n      -> {arrow}" +
                  ("   (identical file already there)" if r["status"] == "duplicate" else ""))
    print(f"{len(rows)} PDFs: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if not (a.apply or a.copy_to):
        print("dry run: nothing changed -- add --apply to rename, or --copy-to <folder> to copy")
        return 0
    folders = {Path(r["src"]).parent for r in rows}
    log = None if a.copy_to or len(folders) != 1 else next(iter(folders)) / "renames.tsv"
    errors: List[Tuple[str, str]] = []
    n = apply(rows, copy=bool(a.copy_to), log=log, errors=errors)
    print(f"{'copied' if a.copy_to else 'renamed'} {n} file(s)" + (f"; logged in {log}" if log else ""))
    for name, why in errors:
        print(f"  ! {name}: not {'copied' if a.copy_to else 'renamed'} -- {why}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
