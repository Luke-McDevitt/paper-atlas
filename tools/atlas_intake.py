r"""atlas_intake -- bring papers INTO an atlas project: from a folder of PDFs, a Zotero / BibTeX export (.bib, or CSL
JSON), or a plain list of DOIs / arXiv ids / titles. Each item is identified, checked against an authoritative registry
(arXiv, Crossref, DataCite, OpenAlex -- tools/verify_refs.py's fetchers, pacing and OpenAlex key) and, for a PDF,
checked to BE that paper; then its row (papers/<topic>.json) and its API record (refs/<topic>/verified_refs.json) are
written and the PDF is staged for `build_bwm.py add`, which names and files it. `tools/atlas.py add` is the command; this
module is its library (tools/atlas_project.py says which project).

WHY (owner, 2026-10-15): "make an easy onboarding process for new papers that does the entire thing from mining to
getting figures to updating the atlas to rebuilding the zip ... a generalized version I can hand off to lab mates so
they can add their papers"; their starting points, all three: a folder of PDFs, a Zotero / BibTeX export, a paper list
with no PDFs yet.

RULES (enforced here, not trusted):
  * NOTHING ENTERS UNCHECKED. A PDF is filed under a paper only when its own first pages print that paper's title as
    whole lines of the title block AND an author's surname after it or the DOI / arXiv id
    (report/boat_world_model/paper_links.py::identity_evidence -- the rule the 2,690-PDF BlueBoat corpus was audited
    with: 0 of 545 wrong pairs passed it). A bibliography or list entry enters only when the registry returns it for its
    DOI / arXiv id with the entry's title (tools/verify_refs.py MATCH_MIN_RATIO), or a title search finds it near-exactly
    (SEARCH_MIN_RATIO). Anything else is listed under NEEDS A LOOK with what was found -- never guessed. `--as
    FILE=DOI` (or =arXiv id) decides one PDF by hand: the registry still has to know the id, and the PDF still has to
    match it unless the PDF has no text layer.
  * ONE ROW PER PAPER: an item whose DOI / arXiv id / title is already in the project is reported as present, never
    added twice (its PDF is still staged when the project holds none for it).
  * The user's files are only READ (copied to a staging folder; build_bwm.py add copies them into the project's PDF
    folder as Surname_Year_Title.pdf).
  * --offline (no network): only bibliography entries can enter, with their own metadata, marked "not verified online"
    in status_evidence and with no verified record (so build_bwm shows no API authors for them; the row's own authors
    are used instead -- build_bwm.py reads row authors only from rows this module wrote). The kit's self-test runs so.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_project as AP  # noqa: E402

PDF_EXT, BIB_EXT, CSL_EXT, LIST_EXT = {".pdf"}, {".bib", ".bibtex"}, {".json"}, {".txt", ".csv", ".tsv"}
ACCEPT = ("title", "title+doi")                   # identity_evidence verdicts that file a PDF without a person
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}|\\^`]+)", re.I)
_ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
_ARXIV_STAMP = re.compile(r"arxiv\s*:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})", re.I)
CROSSREF_STATUS = {"journal-article": "peer-reviewed-journal", "proceedings-article": "peer-reviewed-conference",
                   "book-chapter": "book", "book": "book", "monograph": "book", "edited-book": "book",
                   "reference-book": "book", "book-section": "book", "dissertation": "thesis",
                   "posted-content": "preprint", "report": "dataset-or-report", "dataset": "dataset-or-report",
                   "standard": "standard"}
DATACITE_STATUS = {"preprint": "preprint", "dataset": "dataset-or-report", "report": "dataset-or-report",
                   "dissertation": "thesis", "journalarticle": "peer-reviewed-journal",
                   "conferencepaper": "peer-reviewed-conference", "book": "book", "bookchapter": "book"}


# ------------------------------------------------------------------------------------------------------ the items
@dataclass
class Item:
    """One thing to bring in: a PDF, a bibliography entry, or a list line."""
    source: str                                   # pdf | bibtex | csl | list
    origin: str                                   # where it came from, for messages
    topic: str = "unsorted"
    pdf: Optional[Path] = None
    doi: str = ""
    arxiv_id: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    venue: str = ""
    abstract: str = ""
    kind: str = ""                                # the bibliography's entry type (@article, CSL "paper-conference")
    vouched: str = ""                             # --as: an id the user vouches for (this PDF IS that paper)


@dataclass
class Outcome:
    item: Item
    verdict: str                                  # added | present | look | unresolved | error
    key: str = ""
    record: Optional[dict] = None
    evidence: str = ""
    note: str = ""
    file_pdf: bool = False                        # stage item.pdf for build_bwm.py add under `key`


def slug(s: str, default: str = "unsorted") -> str:
    """A topic key from a folder or a --topic name: lower case ASCII letters, digits and '_'."""
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t[:48] or default


def clean_doi(d: str) -> str:
    d = re.sub(r"(?i)^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(d or "").strip())
    return d.rstrip(").,;]}>'\"").lower()


def clean_arxiv(a: str) -> str:
    s = str(a or "").strip()
    m = _ARXIV_URL.search(s)
    if m:
        s = m.group(1)
    s = re.sub(r"(?i)^arxiv\s*:\s*", "", s)
    s = re.sub(r"\.pdf$", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s if (_ARXIV_NEW.fullmatch(s) or re.fullmatch(r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}", s)) else ""


# -------------------------------------------------------------------------------------------- the parsers (pure)
def _bib_value(text: str, i: int) -> Tuple[str, int]:
    """One BibTeX field value starting at text[i] ({...} with nesting, "...", or a bare word / number); -> (value, end)."""
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    if i >= len(text):
        return "", i
    if text[i] == "{":
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j], j + 1
            elif text[j] == "\\" and j + 1 < len(text):
                j += 1
            j += 1
        return text[i + 1:], len(text)
    if text[i] == '"':
        j, depth = i + 1, 0
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            elif text[j] == '"' and depth <= 0:
                return text[i + 1:j], j + 1
            j += 1
        return text[i + 1:], len(text)
    m = re.match(r"[^,}\s]+", text[i:])
    return (m.group(0), i + m.end()) if m else ("", i)


def _bib_clean(v: str) -> str:
    """A BibTeX value as plain text: braces dropped, the common accent and dash macros folded, spaces collapsed."""
    v = re.sub(r"\\([`'^\"~=.uvHtcdb])\s*\{?([A-Za-z])\}?", r"\2", v)     # \'{e} \"o -> e o
    v = v.replace("\\&", "&").replace("--", "-").replace("~", " ")
    v = re.sub(r"\\[A-Za-z]+\s*", "", v)
    v = v.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", v).strip()


def parse_bibtex(text: str, base_dir: Optional[Path] = None) -> List[dict]:
    """BibTeX -> [{type, key, fields...}] for every @entry (not @string / @comment / @preamble); field names lower
    case, values cleaned (`_bib_clean`) except `file`. Pure."""
    out, i = [], 0
    while True:
        m = re.compile(r"@\s*([A-Za-z]+)\s*([{(])").search(text, i)
        if not m:
            break
        etype = m.group(1).lower()
        j = m.end()
        if etype in ("comment", "string", "preamble"):
            _, i = _bib_value(text, m.start(2)) if m.group(2) == "{" else ("", j)
            continue
        km = re.match(r"\s*([^,\s]*)\s*,", text[j:])
        key = km.group(1) if km else ""
        j += km.end() if km else 0
        fields: Dict[str, str] = {}
        while j < len(text):
            fm = re.match(r"\s*([A-Za-z][\w\-:.]*)\s*=\s*", text[j:])
            if not fm:
                break
            name = fm.group(1).lower()
            val, j = _bib_value(text, j + fm.end())
            fields[name] = val if name == "file" else _bib_clean(val)
            sep = re.match(r"\s*(?:#\s*[^,}]*)?\s*,?", text[j:])
            j += sep.end() if sep else 0
            if j < len(text) and text[j] in "})":
                break
        end = re.compile(r"[})]").search(text, j)
        i = end.end() if end else len(text)
        out.append({"type": etype, "key": key, **fields})
    return out


def _bib_files(value: str, base_dir: Optional[Path]) -> List[Path]:
    """The PDFs a BibTeX `file` field points at: Zotero's 'Description:path/x.pdf:application/pdf' rows (';'-separated,
    ':' inside a path escaped as '\\:') or Better BibTeX's plain paths; a relative path is against the .bib's folder."""
    out = []
    for part in (value or "").split(";"):
        part = part.strip().replace("\\:", "\x00").replace("\\\\", "\\")
        if not part:
            continue
        bits = part.split(":")
        path = bits[1] if len(bits) >= 3 and "/" in bits[-1] else part        # 'desc:path:mime'
        path = path.replace("\x00", ":").strip()
        if not path.lower().endswith(".pdf"):
            continue
        q = Path(path)
        if base_dir is not None and not q.is_absolute():
            q = Path(base_dir) / q
        out.append(q)
    return out


def _bib_authors(v: str) -> List[str]:
    """'Last, First and First Last' -> ['First Last', ...]."""
    out = []
    for a in re.split(r"\s+and\s+", v or ""):
        a = a.strip()
        if not a or a.lower() == "others":
            continue
        if "," in a:
            last, first = a.split(",", 1)
            a = f"{first.strip()} {last.strip()}".strip()
        out.append(a)
    return out


def items_from_bibtex(text: str, origin: str, base_dir: Optional[Path], topic: str) -> List[Item]:
    items = []
    for e in parse_bibtex(text):
        doi = clean_doi(e.get("doi", ""))
        ax = clean_arxiv(e.get("eprint", "")) if e.get("archiveprefix", e.get("eprinttype", "")).lower() in (
            "arxiv", "") and e.get("eprint") else ""
        ax = ax or clean_arxiv(e.get("url", "")) or (clean_arxiv(doi.split("arxiv.", 1)[1]) if "10.48550/arxiv." in doi
                                                     else "")
        if "10.48550/arxiv." in doi:
            doi = ""
        pdfs = [p for p in _bib_files(e.get("file", ""), base_dir) if p.is_file()]
        items.append(Item(source="bibtex", origin=f"{origin}:{e.get('key') or '?'}",
                          topic=topic,
                          pdf=pdfs[0] if pdfs else None, doi=doi, arxiv_id=ax, title=e.get("title", ""),
                          authors=_bib_authors(e.get("author", "")), year=re.sub(r"\D", "", e.get("year", ""))[:4],
                          venue=e.get("journal") or e.get("booktitle") or e.get("publisher") or "",
                          abstract=e.get("abstract", ""), kind=e.get("type", "")))
    return items


def items_from_csl(text: str, origin: str, topic: str) -> List[Item]:
    """Zotero's CSL JSON export ([{id, type, title, author: [{family, given}], issued: {date-parts}, DOI, URL, ...}])."""
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("items") or [data]
    items = []
    for e in data if isinstance(data, list) else []:
        if not isinstance(e, dict) or not e.get("title"):
            continue
        authors = [" ".join(x for x in (a.get("given", ""), a.get("family", "")) if x) or a.get("literal", "")
                   for a in e.get("author") or [] if isinstance(a, dict)]
        dp = ((e.get("issued") or {}).get("date-parts") or [[None]])[0]
        doi = clean_doi(e.get("DOI", ""))
        ax = clean_arxiv(e.get("URL", "")) or (clean_arxiv(doi.split("arxiv.", 1)[1]) if "10.48550/arxiv." in doi
                                               else "")
        if "10.48550/arxiv." in doi:
            doi = ""
        items.append(Item(source="csl", origin=f"{origin}:{e.get('id', '?')}", topic=topic, doi=doi, arxiv_id=ax,
                          title=re.sub(r"\s+", " ", str(e["title"])).strip(), authors=[a for a in authors if a],
                          year=str(dp[0]) if dp and dp[0] else "", venue=str(e.get("container-title") or ""),
                          abstract=str(e.get("abstract") or ""), kind=str(e.get("type") or "")))
    return items


def items_from_list(text: str, origin: str, topic: str) -> List[Item]:
    """One paper per line: a DOI, an arXiv id or URL, or a title; '#' starts a comment; an optional second column
    (a tab, or ' | ') names its topic. A .csv / .tsv with a header row may name columns doi / arxiv_id / title / topic."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    head = re.split(r"[,\t]", lines[0].lower()) if lines else []
    items = []
    if head and {"doi", "title", "arxiv_id", "arxiv"} & {h.strip() for h in head}:
        import csv
        import io
        delim = "\t" if "\t" in lines[0] else ","
        for n, r in enumerate(csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim), 2):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            items.append(Item(source="list", origin=f"{origin}:{n}", topic=slug(r.get("topic") or topic),
                              doi=clean_doi(r.get("doi", "")), arxiv_id=clean_arxiv(r.get("arxiv_id") or r.get("arxiv", "")),
                              title=r.get("title", "")))
        return items
    for n, ln in enumerate(text.splitlines(), 1):
        ln = ln.split("#", 1)[0].strip() if not re.search(r"https?://", ln) else ln.strip()
        if not ln:
            continue
        parts = re.split(r"\t|\s+\|\s+", ln, maxsplit=1)
        val, top = parts[0].strip(), (slug(parts[1]) if len(parts) > 1 else topic)
        ax, doi = clean_arxiv(val), ""
        if not ax:
            m = _DOI.search(val)
            doi = clean_doi(m.group(1)) if m else ""
            if "10.48550/arxiv." in doi:
                ax, doi = clean_arxiv(doi.split("arxiv.", 1)[1]), ""
        items.append(Item(source="list", origin=f"{origin}:{n}", topic=top, doi=doi, arxiv_id=ax,
                          title="" if (ax or doi) else val))
    return items


def collect(paths: Sequence[Path], topic: str = "", say: Callable = print) -> List[Item]:
    """Every item in the given files / folders. A PDF in a SUBFOLDER of a given folder takes the subfolder's name as
    its topic (unless --topic); a bibliography's PDFs (BibTeX `file` fields) are not collected again as loose PDFs."""
    items: List[Item] = []
    loose: List[Tuple[Path, str]] = []
    claimed: set = set()
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(p).parts
                top = topic or (slug(rel[0]) if len(rel) > 1 else "unsorted")
                _one(f, top, items, loose, say)
        elif p.is_file():
            _one(p, topic or "unsorted", items, loose, say)
        else:
            say(f"  ! {p}: no such file or folder")
    for it in items:
        if it.pdf is not None:
            claimed.add(str(it.pdf.resolve()).lower())
    for f, top in loose:
        if str(f.resolve()).lower() not in claimed:
            items.append(Item(source="pdf", origin=str(f), topic=top, pdf=f))
    return items


#: never papers: the inbox's own README, Office lock files, hidden files (2026-10-15: the first real run read
#: inbox/README.txt as a list of seven titles)
_NOT_PAPERS = re.compile(r"^(?:readme|license|licence|notes?)\b|^~\$|^\.", re.I)


def _one(f: Path, topic: str, items: List[Item], loose: List[Tuple[Path, str]], say: Callable) -> None:
    if _NOT_PAPERS.search(f.name):
        return
    ext = f.suffix.lower()
    try:
        if ext in PDF_EXT:
            loose.append((f, topic))
        elif ext in BIB_EXT:
            items += items_from_bibtex(f.read_text(encoding="utf-8", errors="replace"), f.name, f.parent, topic)
        elif ext in CSL_EXT:
            items += items_from_csl(f.read_text(encoding="utf-8", errors="replace"), f.name, topic)
        elif ext in LIST_EXT:
            items += items_from_list(f.read_text(encoding="utf-8", errors="replace"), f.name, topic)
    except (ValueError, OSError) as e:
        say(f"  ! {f.name}: cannot read it ({e})")


# --------------------------------------------------------------------------------------------- reading a PDF
def _engine(name: str):
    full = f"_atlas_intake_{name}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, AP.ENGINE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


def first_pages(pdf: Path, n: int = 3) -> List[str]:
    """The text of a PDF's first `n` pages (pypdfium2, cleaned like tools/paper_text.py; pdfminer as the fallback);
    [] when there is no text layer."""
    try:
        import pypdfium2 as pdfium
        import paper_text as PT
        doc = pdfium.PdfDocument(str(pdf))
        try:
            out = []
            for i in range(min(n, len(doc))):
                page = doc[i]
                tp = page.get_textpage()
                out.append(PT._clean_page(tp.get_text_range()))
                tp.close()
                page.close()
            return out
        finally:
            doc.close()
    except Exception:                                     # noqa: BLE001 -- the fallback reads page 1 only
        t = _engine("paper_links").page1_text(pdf)
        return [t] if t.strip() else []


def hints(pdf: Path, pages: Sequence[str]) -> dict:
    """What the PDF says about itself: arXiv id / DOI from its metadata and file name (tools/paper_pdfs.py
    pdf_identity), the arXiv stamp and DOIs printed before any references heading, and title candidates (the title
    block's first lines, alone and joined in pairs)."""
    PL = _engine("paper_links")
    try:
        import paper_pdfs as PP
        meta = PP.pdf_identity(pdf)
    except Exception:                                     # noqa: BLE001 -- an unreadable file: text only
        meta = {"title": "", "doi": "", "arxiv_id": "", "title_from": ""}
    head = "\n".join(pages[:2])
    m = PL._REFS_HEAD.search(head)
    if m:
        head = head[:m.start()]
    dois = []
    for d in [meta.get("doi") or ""] + [x.group(1) for x in _DOI.finditer(head)]:
        d = clean_doi(d)
        if d and d not in dois:
            dois.append(d)
    ax = clean_arxiv(meta.get("arxiv_id") or "")
    if not ax:
        s = _ARXIV_STAMP.search(head)
        ax = s.group(1) if s else ""
    lines = []
    blocks = PL._title_blocks(list(pages)) if pages else []
    for ln in (blocks[0] if blocks else "").splitlines():
        w = ln.strip()
        if 2 <= len(w.split()) <= 30 and not re.search(r"@|https?://|\bdoi\b|arxiv|©|copyright|journal|vol\.|issn",
                                                         w, re.I):
            lines.append(w)
        if len(lines) >= 6:
            break
    titles = []
    if meta.get("title_from") == "metadata" and len(meta.get("title", "").split()) >= 3:
        titles.append(meta["title"])
    for i, ln in enumerate(lines[:4]):
        titles.append(ln)
        if i + 1 < len(lines):
            titles.append(f"{ln} {lines[i + 1]}")
    seen, uniq = set(), []
    for t in titles:
        k = re.sub(r"[^a-z0-9]+", "", t.lower())
        if len(k) >= 12 and k not in seen:
            seen.add(k)
            uniq.append(t)
    return {"dois": dois[:3], "arxiv_id": ax, "titles": uniq[:5]}


def confirm(record: dict, pages: Sequence[str]) -> dict:
    """Is this PDF the record's paper? paper_links.identity_evidence on the first pages: verdict 'title' / 'title+doi'
    pass; 'doi only', 'weak', 'notice', 'supplement', 'sibling' need a person; 'no text' (a scan) cannot tell."""
    PL = _engine("paper_links")
    return PL.identity_evidence(record.get("title") or "", record.get("doi") or "", record.get("arxiv_id") or "",
                                list(pages), authors=(record.get("authors") or [])[:3])


# --------------------------------------------------------------------------------------------- the registry
def status_from(crossref_type: str = "", datacite_type: str = "", oa_type: str = "", oa_source_type: str = "",
                arxiv_only: bool = False) -> Tuple[str, str]:
    """(status, evidence) in build_bwm.py's vocabulary from what the registry said about the WORK. Pure."""
    if crossref_type:
        st = CROSSREF_STATUS.get(crossref_type.lower(), "")
        return st, f"Crossref type {crossref_type}"
    if datacite_type:
        st = DATACITE_STATUS.get(re.sub(r"[^a-z]", "", datacite_type.lower()), "")
        return st, f"DataCite resource type {datacite_type}"
    if oa_type:
        t, s = oa_type.lower(), (oa_source_type or "").lower()
        if t in ("preprint", "posted-content") or s == "repository":
            return "preprint", f"OpenAlex type {oa_type}, source {oa_source_type or 'none'}"
        if t == "article":
            st = {"journal": "peer-reviewed-journal", "conference": "peer-reviewed-conference"}.get(s, "")
            return st, f"OpenAlex type {oa_type}, source {oa_source_type or 'none'}"
        st = {"book-chapter": "book", "book": "book", "dissertation": "thesis", "dataset": "dataset-or-report",
              "report": "dataset-or-report", "standard": "standard"}.get(t, "")
        return st, f"OpenAlex type {oa_type}"
    if arxiv_only:
        return "preprint", "arXiv only (no published version recorded)"
    return "", "no type recorded"


class Registry:
    """The authoritative lookups (tools/verify_refs.py: its fetchers, its pacing -- arxiv.org 15 s, the export API
    3 s -- and the OpenAlex key, tools/openalex_key.py), cached for one run. Every method returns a RECORD:
    {title, authors, year, venue, doi, arxiv_id, abstract, url, status, status_evidence, source, openalex_id, oa_url}
    or None (asked: not found); a network failure raises verify_refs.FetchError."""

    def __init__(self, say: Callable = print):
        import verify_refs as VR
        self.VR, self.say, self._memo = VR, say, {}

    def _openalex_raw(self, doi: str) -> Optional[dict]:
        k = ("oa", doi)
        if k not in self._memo:
            import urllib.parse
            txt = self.VR._get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='/')}")
            self._memo[k] = json.loads(txt) if txt else None
        return self._memo[k]

    def _from_openalex(self, w: dict) -> dict:
        rec = self.VR.openalex_norm(w)
        loc = w.get("primary_location") or {}
        src = (loc.get("source") or {}) if isinstance(loc, dict) else {}
        ids = w.get("ids") or {}
        ax = ""
        for u in [ids.get("arxiv") or ""] + [(x or {}).get("landing_page_url") or "" for x in w.get("locations") or []]:
            ax = ax or clean_arxiv(u)
        rec["arxiv_id"] = ax
        rec["status"], rec["status_evidence"] = status_from(oa_type=w.get("type") or "",
                                                            oa_source_type=src.get("type") or "")
        return rec

    def by_doi(self, doi: str) -> Optional[dict]:
        doi = clean_doi(doi)
        if not doi:
            return None
        if doi.startswith("10.48550/arxiv."):
            return self.by_arxiv(doi.split("arxiv.", 1)[1])
        k = ("doi", doi)
        if k in self._memo:
            return self._memo[k]
        rec = self.VR.fetch_crossref(doi)                 # Crossref, then DataCite
        if rec is None:
            w = self._openalex_raw(doi)
            rec = self._from_openalex(w) if w else None
        else:
            src = rec.get("source", "")
            rec["status"], rec["status_evidence"] = status_from(
                crossref_type=rec.get("type", "") if src == "crossref" else "",
                datacite_type=rec.get("type", "") if src == "datacite" else "")
            rec.setdefault("arxiv_id", "")
            try:                                          # OpenAlex: the abstract Crossref often lacks, the work id
                w = self._openalex_raw(doi)
            except self.VR.FetchError:
                w = None
            if w:
                oa = self._from_openalex(w)
                rec["abstract"] = rec.get("abstract") or oa.get("abstract", "")
                rec["openalex_id"], rec["oa_url"] = oa.get("openalex_id", ""), oa.get("oa_url", "")
                rec["arxiv_id"] = rec.get("arxiv_id") or oa.get("arxiv_id", "")
                if not rec["status"]:
                    rec["status"], rec["status_evidence"] = oa["status"], oa["status_evidence"]
        self._memo[k] = rec
        return rec

    def by_arxiv(self, aid: str) -> Optional[dict]:
        aid = clean_arxiv(aid)
        if not aid:
            return None
        k = ("arxiv", aid)
        if k in self._memo:
            return self._memo[k]
        rec = self.VR.fetch_arxiv(aid)
        if rec is not None:
            rec["arxiv_id"] = aid
            pub = clean_doi(rec.get("doi") or "")
            published = None
            if pub and not pub.startswith("10.48550/"):
                try:
                    published = self.by_doi(pub)
                except self.VR.FetchError:
                    published = None
            if published and published.get("status") and published["status"] != "preprint":
                rec = {**published, "arxiv_id": aid,
                       "abstract": published.get("abstract") or rec.get("abstract", "")}
            else:
                rec["doi"] = pub if pub and not pub.startswith("10.48550/") else ""
                rec["status"], rec["status_evidence"] = status_from(arxiv_only=True)
        self._memo[k] = rec
        return rec

    def search(self, title: str, n: int = 5) -> List[dict]:
        """OpenAlex title search: the first `n` works, as records, best title match first."""
        import urllib.parse
        k = ("search", title.lower())
        if k not in self._memo:
            q = urllib.parse.quote(re.sub(r"[^\w\s-]", " ", title))
            txt = self.VR._get(f"https://api.openalex.org/works?search={q}&per-page={n}")
            res = [self._from_openalex(w) for w in (json.loads(txt).get("results") or [])] if txt else []
            res.sort(key=lambda r: -self.VR.title_ratio(title, r.get("title", "")))
            self._memo[k] = res
        return self._memo[k]


# ------------------------------------------------------------------------------------------ identifying an item
def identify_pdf(item: Item, reg: Registry, pages: Optional[List[str]] = None) -> Outcome:
    """A PDF -> the paper it is, or NEEDS A LOOK. Ids it prints (or the one the user vouched for) first, then title
    searches; a candidate is accepted only when `confirm` passes (or the user vouched and the PDF has no text)."""
    pages = first_pages(item.pdf) if pages is None else pages
    h = hints(item.pdf, pages) if pages or item.pdf else {"dois": [], "arxiv_id": "", "titles": []}
    tried: List[Tuple[dict, dict]] = []

    def check(rec: Optional[dict], how: str) -> Optional[Outcome]:
        if not rec or not rec.get("title"):
            return None
        ev = confirm(rec, pages)
        tried.append((rec, ev))
        if ev["verdict"] in ACCEPT:
            return Outcome(item, "added", record=rec, evidence=f"{how}; the PDF prints it ({ev['verdict']})",
                           file_pdf=True)
        if item.vouched and ev["verdict"] == "no text":
            return Outcome(item, "added", record=rec, evidence=f"{how}; vouched by --as (the PDF has no text layer)",
                           file_pdf=True)
        return None

    if item.vouched:
        v = item.vouched
        rec = reg.by_arxiv(v) if clean_arxiv(v) else reg.by_doi(v)
        if rec is None:
            return Outcome(item, "unresolved", note=f"--as {v}: the registry does not know this id")
        got = check(rec, f"--as {v}")
        if got:
            return got
        ev = tried[-1][1] if tried else {"verdict": "?"}
        return Outcome(item, "look", record=rec, evidence=f"--as {v}",
                       note=f"the PDF does not print this paper's title + an author or the id ({ev['verdict']})")
    if h["arxiv_id"]:
        got = check(reg.by_arxiv(h["arxiv_id"]), f"arXiv {h['arxiv_id']} printed on the PDF")
        if got:
            return got
    for d in h["dois"]:
        got = check(reg.by_doi(d), f"DOI {d} printed on the PDF")
        if got:
            return got
    for t in h["titles"]:
        for rec in reg.search(t)[:3]:
            if reg.VR.title_ratio(t, rec.get("title", "")) < 0.6:
                continue
            got = check(rec, f"OpenAlex title search '{t[:60]}'")
            if got:
                return got
    if not pages:
        return Outcome(item, "look", note="no text layer (a scan?): OCR it, or name it with --as FILE=DOI")
    if tried:
        rec, ev = max(tried, key=lambda x: x[1].get("sim", 0))
        return Outcome(item, "look", record=rec,
                       note=f"closest: '{rec.get('title', '')[:80]}' ({ev['verdict']}, title similarity "
                            f"{ev.get('sim', 0):.2f}) -- check it, then `--as {item.pdf.name}="
                            f"{rec.get('doi') or rec.get('arxiv_id') or 'DOI'}`")
    return Outcome(item, "unresolved", note="no DOI, arXiv id or findable title on its first pages")


def resolve_entry(item: Item, reg: Optional[Registry]) -> Outcome:
    """A bibliography or list entry -> its registry record (title-checked), or NEEDS A LOOK. Offline (reg None): the
    entry's own metadata, marked not verified."""
    if reg is None:
        if not item.title:
            return Outcome(item, "unresolved", note="offline, and the entry has no title")
        rec = {"title": item.title, "authors": item.authors, "year": item.year, "venue": item.venue,
               "doi": item.doi, "arxiv_id": item.arxiv_id, "abstract": item.abstract, "url": "", "source": item.source}
        rec["status"], ev = _entry_status(item)
        rec["status_evidence"] = f"{ev}; from your bibliography, NOT verified online (--offline)"
        return Outcome(item, "added", record=rec, evidence="offline: the bibliography's own metadata",
                       file_pdf=item.pdf is not None)
    VR = reg.VR
    rec = None
    how = ""
    if item.arxiv_id:
        rec, how = reg.by_arxiv(item.arxiv_id), f"arXiv {item.arxiv_id}"
    if rec is None and item.doi:
        rec, how = reg.by_doi(item.doi), f"DOI {item.doi}"
    if rec is not None and item.title and VR.title_ratio(item.title, rec.get("title", "")) < VR.MATCH_MIN_RATIO:
        return Outcome(item, "look", record=rec, evidence=how,
                       note=f"the registry's title for {how} is '{rec.get('title', '')[:80]}', the entry says "
                            f"'{item.title[:80]}'")
    if rec is None and (item.arxiv_id or item.doi):
        return Outcome(item, "unresolved", note=f"{how or 'its id'}: not found in the registry")
    if rec is None:
        if not item.title:
            return Outcome(item, "unresolved", note="no id and no title")
        hits = reg.search(item.title)
        best = hits[0] if hits else None
        r = VR.title_ratio(item.title, best.get("title", "")) if best else 0.0
        if best is None or r < VR.SEARCH_MIN_RATIO:
            near = best is not None and r >= 0.6         # a far-off hit is noise: never shown as a candidate
            return Outcome(item, "look", record=best if near else None,
                           note=(f"title search: closest '{best.get('title', '')[:80]}' ({r:.2f} < "
                                 f"{VR.SEARCH_MIN_RATIO}) -- check it" if near else
                                 f"no near match on OpenAlex (best title similarity {r:.2f})"))
        rec, how = best, f"OpenAlex title search ({r:.2f})"
    out = Outcome(item, "added", record=rec, evidence=f"{how}: the registry's title matches")
    if item.pdf is not None:
        pages = first_pages(item.pdf)
        ev = confirm(rec, pages)
        if ev["verdict"] in ACCEPT:
            out.file_pdf, out.evidence = True, out.evidence + f"; its PDF prints it ({ev['verdict']})"
        else:
            out.note = f"its attached PDF {item.pdf.name} does not look like this paper ({ev['verdict']}): not filed"
    return out


def _entry_status(item: Item) -> Tuple[str, str]:
    k = (item.kind or "").lower()
    bib = {"article": "peer-reviewed-journal", "inproceedings": "peer-reviewed-conference",
           "conference": "peer-reviewed-conference", "incollection": "book", "book": "book", "inbook": "book",
           "phdthesis": "thesis", "mastersthesis": "thesis", "techreport": "dataset-or-report",
           "article-journal": "peer-reviewed-journal", "paper-conference": "peer-reviewed-conference",
           "chapter": "book", "thesis": "thesis", "report": "dataset-or-report", "dataset": "dataset-or-report"}
    if item.arxiv_id and not item.venue:
        return "preprint", "arXiv id, no venue"
    st = bib.get(k, "")
    return st, f"entry type {k or 'unknown'}"


# ------------------------------------------------------------------------------------------- writing the rows
def existing(project: Path) -> Tuple[Dict[str, str], set]:
    """({handle: key} for every row already in the project, every key and alias in use)."""
    B = _engine("build_bwm")
    handles: Dict[str, str] = {}
    keys: set = set()
    for _, r in B.topic_rows(project):
        keys.add(r["key"])
        for h in B._handles(r.get("arxiv_id"), r.get("doi"), r.get("title")):
            handles.setdefault(h, r["key"])
    canon = B._load(project / "papers" / "_canonical.json", {})
    keys |= set((canon.get("aliases") or {}).keys()) | set((canon.get("aliases") or {}).values())
    return handles, keys


def handles_of(rec: dict) -> List[str]:
    return _engine("build_bwm")._handles(rec.get("arxiv_id"), rec.get("doi"), rec.get("title"))


def row_for(key: str, rec: dict, item: Item, evidence: str, today: str, held: bool) -> dict:
    """A papers/<topic>.json row (build_bwm.py's fields) for a paper this module brought in."""
    try:
        year = int(str(rec.get("year") or "")[:4])
    except ValueError:
        year = None
    row = {"key": key, "title": rec.get("title") or item.title, "year": year, "venue": rec.get("venue") or "",
           "status": rec.get("status") or "", "status_evidence": rec.get("status_evidence") or "",
           "arxiv_id": rec.get("arxiv_id") or None, "doi": rec.get("doi") or None, "relevance": "unrated",
           "tags": [], "summary": re.sub(r"\s+", " ", rec.get("abstract") or "").strip(), "why_for_us": "",
           "pdf_held": bool(held), "authors": list(rec.get("authors") or [])[:30],
           "intake": {"source": item.source, "origin": item.origin, "added": today, "evidence": evidence}}
    if rec.get("oa_url"):
        row["oa_pdf_url"] = rec["oa_url"]
    return row


def ref_for(rec: dict, item: Item, evidence: str, verified: bool) -> dict:
    """A refs/<topic>/verified_refs.json row (tools/verify_refs.py's shape) for the paper's API record."""
    return {"input": {"title": item.title or rec.get("title", ""), "doi": item.doi, "arxiv_id": item.arxiv_id,
                      "source": item.source, "origin": item.origin},
            "status": "VERIFIED" if verified else "UNVERIFIED_LOCAL", "reason": evidence,
            "arxiv_id": rec.get("arxiv_id") or "", "record": {k: v for k, v in rec.items() if k != "status_evidence"}}


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def write(project: Path, outcomes: List[Outcome], today: str, verified: bool, say: Callable = print) -> dict:
    """Give every 'added' outcome a key (or mark it 'present' when the project already holds the paper), append the
    rows and records to papers/<topic>.json and refs/<topic>/verified_refs.json. -> {topic: rows added}."""
    import verify_refs as VR
    handles, keys = existing(project)
    added: Dict[str, List[dict]] = {}
    refs: Dict[str, List[dict]] = {}
    for o in outcomes:
        if o.verdict != "added" or not o.record:
            continue
        hs = handles_of(o.record)
        hit = next((handles[h] for h in hs if h in handles), None)
        if hit:
            o.verdict, o.key = "present", hit
            continue
        o.key = VR.make_key(o.record, keys)
        for h in hs:
            handles[h] = o.key
        added.setdefault(o.item.topic, []).append(row_for(o.key, o.record, o.item, o.evidence, today, o.file_pdf))
        refs.setdefault(o.item.topic, []).append(ref_for(o.record, o.item, o.evidence, verified))
    for topic, rows in sorted(added.items()):
        f = project / "papers" / f"{topic}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else []
        f.write_text(_dump(list(old) + rows), encoding="utf-8", newline="\n")
        r = project / "refs" / topic / "verified_refs.json"
        r.parent.mkdir(parents=True, exist_ok=True)
        old_r = json.loads(r.read_text(encoding="utf-8")) if r.is_file() else []
        r.write_text(_dump(list(old_r) + refs[topic]), encoding="utf-8", newline="\n")
        say(f"  papers/{topic}.json: +{len(rows)} paper(s)")
    return {t: len(r) for t, r in added.items()}


def stage(outcomes: List[Outcome], folder: Path, held_keys: Iterable[str] = ()) -> List[str]:
    """Copy every PDF to file into `folder` (a fresh staging folder) -> build_bwm.py add's --as NAME=KEY pairs. A paper
    the project already holds a PDF for is not staged again."""
    held = set(held_keys)
    folder.mkdir(parents=True, exist_ok=True)
    pairs = []
    for n, o in enumerate(outcomes, 1):
        if not (o.file_pdf and o.key and o.item.pdf is not None and o.verdict in ("added", "present")):
            continue
        if o.key in held:
            continue
        name = f"{n:04d}_{re.sub(r'[^A-Za-z0-9._-]+', '_', o.item.pdf.stem)[:80]}.pdf"
        shutil.copy2(o.item.pdf, folder / name)
        pairs.append(f"{name}={o.key}")
    return pairs


def fetch_open(outcomes: List[Outcome], folder: Path, say: Callable = print) -> int:
    """For accepted entries with NO PDF: download an open copy (arXiv, one request per 15 s as arxiv.org asks; else
    the OpenAlex open-access locations -- tools/paper_pdfs.py) into `folder`, and file it only when it passes
    `confirm`. -> how many were found."""
    import paper_pdfs as PP
    folder.mkdir(parents=True, exist_ok=True)
    got = 0
    for n, o in enumerate(outcomes, 1):
        if o.verdict != "added" or o.item.pdf is not None or not o.record:
            continue
        rec = o.record
        dest = folder / f"fetched_{n:04d}.pdf"
        ok = False
        if rec.get("arxiv_id"):
            ok, _ = PP.fetch_arxiv_pdf(rec["arxiv_id"], dest)
        if not ok and rec.get("doi"):
            try:
                urls = PP.open_pdf_urls([rec["doi"]]).get(clean_doi(rec["doi"]), [])
            except Exception:                             # noqa: BLE001 -- no copy found is reported, never fatal
                urls = []
            for u in urls[:4]:
                try:
                    data = PP.http_get(u)
                except Exception:                         # noqa: BLE001
                    continue
                if data.startswith(b"%PDF-") and PP._ends(data):
                    dest.write_bytes(data)
                    ok = True
                    break
        if not ok:
            o.note = (o.note + "; " if o.note else "") + "no open copy found (the atlas lists where to get it)"
            continue
        ev = confirm(rec, first_pages(dest))
        if ev["verdict"] in ACCEPT:
            o.item.pdf, o.file_pdf = dest, True
            o.evidence += f"; open copy downloaded and checked ({ev['verdict']})"
            got += 1
        else:
            o.note = (o.note + "; " if o.note else "") + f"the open copy found does not look like it ({ev['verdict']})"
            dest.unlink(missing_ok=True)
        say(f"  open copy {'filed' if o.file_pdf else 'NOT filed'}: {rec.get('title', '')[:70]}")
    return got


def report_md(outcomes: List[Outcome], today: str) -> str:
    """The intake report: what came in, what was already there, what needs a person."""
    by = {v: [o for o in outcomes if o.verdict == v] for v in ("added", "present", "look", "unresolved", "error")}
    out = [f"# Intake {today}", "",
           f"{len(by['added'])} added, {len(by['present'])} already in the atlas, {len(by['look'])} NEED A LOOK, "
           f"{len(by['unresolved'])} not identified, {len(by['error'])} errors.", ""]
    for v, head in (("added", "Added"), ("present", "Already in the atlas"), ("look", "NEEDS A LOOK"),
                    ("unresolved", "Not identified"), ("error", "Errors")):
        if not by[v]:
            continue
        out += [f"## {head}", ""]
        for o in by[v]:
            t = (o.record or {}).get("title") or o.item.title or (o.item.pdf.name if o.item.pdf else o.item.origin)
            pdf = " [PDF]" if o.file_pdf else ""
            out.append(f"- `{o.key or '-'}` {t[:110]}{pdf} -- {o.item.origin} ({o.item.topic})"
                       + (f"; {o.evidence}" if o.evidence and v in ("added", "present") else "")
                       + (f"; **{o.note}**" if o.note else ""))
        out.append("")
    return "\n".join(out)


def today_str() -> str:
    return datetime.date.today().isoformat()
