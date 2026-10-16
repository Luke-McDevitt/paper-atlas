"""verify_refs -- check every reference in a list against arXiv / Crossref / OpenAlex before it is cited.

WHAT IT IS. A reusable CLI for CLAUDE.md ground rule 2 ("no unverified references, ever"): give it a JSON
list of candidate references (an arXiv id and/or a DOI plus the title you BELIEVE the paper has), and it
fetches each record from the authoritative API, compares the returned title against the claimed one, and
writes the API's own metadata (title, authors, year, venue, DOI, citation count, abstract) as JSON, a Markdown
verification log, BibTeX and a linked reading list. Nothing it writes was typed by a person or a model.

    python tools/verify_refs.py refs.json --out-dir out/refs            # verify, write 4 artifacts
    python tools/verify_refs.py refs.json --out-dir out/refs --no-openalex
    python tools/verify_refs.py workflow_result.json --out-dir out/refs  # {"papers": [...]} works too
    python tools/verify_refs.py out/refs/verified_refs.json --out-dir out/refs --from-json --group-by category

Input rows need `title` plus at least one of `arxiv_id` / `doi`; `key` is optional (derived from the API's
first author + year + first title word when absent). A row with neither id is resolved by an OpenAlex TITLE
search and accepted only on a near-exact title match (see `SEARCH_MIN_RATIO`), never on a loose one.

WHY IT EXISTS, measured. `references/build_seed_bib.py` does this for one hard-coded list and cannot be
reused; and on 2026-09-23 its arXiv call returned HTTP 406. A literature sweep by several agents produces
dozens of ids that a model may have half-remembered; the cheap, decisive check is to ask the registry and
compare titles, so that is what this does.

THE ARXIV EXPORT API IS NOT RELIABLE ENOUGH TO BE THE ONLY SOURCE (measured 2026-09-23): a 74-id run got
HTTP 429 on 39 ids and HTTP 406 on 6, and afterwards the SAME single-id request answered 200 for one id and
406 for another a few seconds apart; comma-separated `id_list` batches were refused outright. So an arXiv id
is resolved by a CHAIN, each link authoritative: the export API, then DataCite (arXiv registers every paper
as DOI `10.48550/arXiv.<id>` -- the registration record itself), then the `citation_title` meta tag of the
arxiv.org abstract page. DOIs go Crossref -> DataCite (Zenodo and arXiv DOIs are DataCite, not Crossref)
-> OpenAlex. The record's `source` says which link answered.

EXIT CODES are the repo contract (CONTRIBUTING sec 2b): 0 = every row MEASURED as verified; 1 = at least one
row measured FALSE (title mismatch, id not found, arXiv/DOI disagree); 2 = no row failed but at least one
could NOT be measured (network error, rate limit). A row that could not be checked never reads as verified.

HONEST SCOPE. A title match proves the identifier points at a paper with that title -- not that the paper
says what a summary claims it says; read the paper for that. The arXiv `published` year is the v1 date and
can precede the venue year by one. OpenAlex citation counts are a snapshot and differ from Google Scholar.
Title matching reuses `scripts/litreview_pipeline/litreview_core.py::norm` / `::norm_doi` so the pipeline
and this tool agree on what "the same title" means.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import openalex_key as _OK  # noqa: E402
import atlas_project as _AP  # noqa: E402
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_litreview_core():
    spec = importlib.util.spec_from_file_location(
        "litreview_core", ROOT / "scripts" / "litreview_pipeline" / "litreview_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LRC = _load_litreview_core()
norm, norm_doi = _LRC.norm, _LRC.norm_doi

#: arXiv's export API answered HTTP 406 to a request with no Accept header (measured 2026-09-23).
HEADERS = {"User-Agent": _AP.user_agent("verify_refs"),     # no owner address in an exported kit (atlas_project)
           "Accept": "application/atom+xml, application/json;q=0.9, */*;q=0.8"}
DATACITE_HEADERS = dict(HEADERS, Accept="application/vnd.api+json")
ATOM = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}

#: normalised-title similarity needed to call an id-resolved record the claimed paper.
MATCH_MIN_RATIO = 0.85
#: a title SEARCH (no id) must be near-exact: a loose search hit is exactly the half-remembered case.
SEARCH_MIN_RATIO = 0.93

VERIFIED, VERIFIED_SEARCH = "VERIFIED", "VERIFIED_BY_TITLE_SEARCH"
MISMATCH, NOT_FOUND, ID_CONFLICT, UNRESOLVED = "TITLE_MISMATCH", "NOT_FOUND", "ARXIV_DOI_CONFLICT", "UNRESOLVED"
UNMEASURED = "COULD_NOT_MEASURE"
OK_STATES = {VERIFIED, VERIFIED_SEARCH}
FALSE_STATES = {MISMATCH, NOT_FOUND, ID_CONFLICT, UNRESOLVED}


class FetchError(Exception):
    """The API could not be asked (network, rate limit, 5xx). Distinct from 'asked and not found'."""


# ------------------------------------------------------------------------------------------ pure parts ----

def clean_arxiv_id(aid):
    """'arXiv:2301.04104v2' / 'https://arxiv.org/abs/2301.04104' -> '2301.04104'. '' for empty."""
    if not aid:
        return ""
    s = str(aid).strip()
    s = re.sub(r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s)
    return re.sub(r"v\d+$", "", s)


def title_ratio(a, b):
    """Similarity of two titles after `litreview_core.norm` (alphanumerics only, lowercase)."""
    na, nb = norm(a or ""), norm(b or "")
    if not na or not nb:
        return 0.0
    if len(na) >= 20 and len(nb) >= 20 and (na in nb or nb in na):
        return max(0.95, difflib.SequenceMatcher(None, na, nb).ratio())
    return difflib.SequenceMatcher(None, na, nb).ratio()


def parse_arxiv_atom(xml_text):
    """arXiv Atom feed -> list of records. An id that does not exist comes back as an entry titled 'Error'."""
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", ATOM):
        t = e.find("a:title", ATOM)
        title = re.sub(r"\s+", " ", t.text if t is not None and t.text else "").strip()
        authors = [a.findtext("a:name", default="", namespaces=ATOM) for a in e.findall("a:author", ATOM)]
        if (title == "Error" or not title) and not authors:
            continue
        eid = e.findtext("a:id", default="", namespaces=ATOM)
        out.append({
            "source": "arxiv",
            "title": title,
            "authors": authors,
            "year": (e.findtext("a:published", default="", namespaces=ATOM) or "")[:4],
            "doi": e.findtext("ar:doi", default="", namespaces=ATOM) or "",
            "venue": e.findtext("ar:journal_ref", default="", namespaces=ATOM) or "",
            "arxiv_id": clean_arxiv_id(eid.rsplit("/abs/", 1)[-1]) if eid else "",
            "url": eid.replace("http://", "https://") if eid else "",
            "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=ATOM) or "").strip(),
        })
    return out


def _strip_tags(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def openalex_abstract(inv):
    """OpenAlex `abstract_inverted_index` ({word: [positions]}) -> plain text. '' when absent."""
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def crossref_norm(msg):
    """Crossref /works message -> record."""
    title = re.sub(r"\s+", " ", ((msg.get("title") or [""])[0] or "")).strip()
    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() or a.get("name", "")
               for a in msg.get("author", [])]
    parts = None
    for k in ("published-print", "published-online", "issued", "created"):
        dp = (msg.get(k) or {}).get("date-parts") or [[None]]
        if dp and dp[0] and dp[0][0]:
            parts = dp[0][0]
            break
    venue = (msg.get("container-title") or [""])
    doi = msg.get("DOI") or ""
    return {"source": "crossref", "title": title, "authors": authors,
            "year": str(parts) if parts else "", "doi": doi, "venue": venue[0] if venue else "",
            "type": msg.get("type", ""), "url": f"https://doi.org/{doi}" if doi else "",
            "abstract": _strip_tags(msg.get("abstract", ""))}


def openalex_norm(w):
    """OpenAlex work -> record (adds cited_by_count, open-access URL)."""
    doi = norm_doi(w.get("doi") or "")
    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}) if isinstance(loc, dict) else {}
    oa = (w.get("open_access") or {}).get("oa_url") or ""
    return {"source": "openalex", "title": re.sub(r"\s+", " ", w.get("title") or w.get("display_name") or "").strip(),
            "authors": [((a.get("author") or {}).get("display_name") or "") for a in w.get("authorships", [])],
            "year": str(w.get("publication_year") or ""), "doi": doi, "venue": src.get("display_name") or "",
            "cited_by_count": w.get("cited_by_count"), "oa_url": oa, "openalex_id": w.get("id", ""),
            "url": f"https://doi.org/{doi}" if doi else (w.get("id") or ""),
            "abstract": openalex_abstract(w.get("abstract_inverted_index"))}


def datacite_norm(data):
    """DataCite /dois/<doi> JSON (`{"data": {"attributes": ...}}`) -> record."""
    a = (data.get("data") or {}).get("attributes") or {}
    titles = a.get("titles") or [{}]
    authors = []
    for c in a.get("creators") or []:
        g, f = c.get("givenName"), c.get("familyName")
        authors.append(f"{g} {f}".strip() if (g or f) else (c.get("name") or ""))
    doi = a.get("doi") or ""
    cont = (a.get("container") or {}).get("title") or ""
    pub = a.get("publisher") or ""
    pub = pub.get("name", "") if isinstance(pub, dict) else pub
    abstract = next((d.get("description") for d in a.get("descriptions") or []
                     if d.get("descriptionType") == "Abstract" and d.get("description")), "")
    return {"source": "datacite", "title": re.sub(r"\s+", " ", titles[0].get("title") or "").strip(),
            "authors": authors, "year": str(a.get("publicationYear") or ""), "doi": doi,
            "venue": cont or pub, "type": (a.get("types") or {}).get("resourceTypeGeneral", ""),
            "url": f"https://doi.org/{doi}" if doi else "", "abstract": _strip_tags(abstract)}


_META_RE = re.compile(r'<meta\s+name="(citation_title|citation_author|citation_date|citation_doi)"\s+content="([^"]*)"')


def parse_arxiv_abs_html(html_text, aid):
    """arxiv.org/abs/<id> page -> record from its Highwire `citation_*` meta tags. None when there is no title."""
    import html as _html
    tags = {}
    for k, v in _META_RE.findall(html_text or ""):
        tags.setdefault(k, []).append(_html.unescape(v))
    if not tags.get("citation_title"):
        return None
    return {"source": "arxiv_abs_page", "title": re.sub(r"\s+", " ", tags["citation_title"][0]).strip(),
            "authors": tags.get("citation_author", []), "year": (tags.get("citation_date", [""])[0] or "")[:4],
            "doi": (tags.get("citation_doi") or [""])[0], "venue": "", "arxiv_id": aid,
            "url": f"https://arxiv.org/abs/{aid}"}


def arxiv_doi(aid):
    """arXiv's own DataCite DOI for an id (what OpenAlex indexes preprints under)."""
    return f"10.48550/arxiv.{aid}".lower()


def classify(claimed_title, arxiv_rec, doi_rec, *, arxiv_asked, doi_asked, arxiv_err=None, doi_err=None):
    """Decide one row's status from what the APIs returned. Pure: every branch is unit-tested.

    Returns (status, reason, best_record)."""
    reasons, best = [], None
    missing = mismatched = False
    for label, asked, rec, err in (("arXiv", arxiv_asked, arxiv_rec, arxiv_err),
                                   ("DOI", doi_asked, doi_rec, doi_err)):
        if not asked:
            continue
        if err is not None:
            reasons.append(f"{label}: could not query ({err})")
            continue
        if rec is None:
            reasons.append(f"{label}: id not found")
            missing = True
            continue
        r = title_ratio(claimed_title, rec["title"])
        if r < MATCH_MIN_RATIO:
            reasons.append(f"{label}: title mismatch ratio {r:.2f} -> API says '{rec['title'][:90]}'")
            mismatched = True
        else:
            reasons.append(f"{label}: title match {r:.2f}")
            best = best or rec
    if arxiv_rec is not None and doi_rec is not None:
        cross = title_ratio(arxiv_rec["title"], doi_rec["title"])
        if cross < MATCH_MIN_RATIO:
            return ID_CONFLICT, "; ".join(reasons + [f"arXiv and DOI records disagree ({cross:.2f})"]), best
    if mismatched:
        return MISMATCH, "; ".join(reasons), best
    if missing:
        return NOT_FOUND, "; ".join(reasons), best
    if best is not None:
        return VERIFIED, "; ".join(reasons), best
    if any("could not query" in x for x in reasons):
        return UNMEASURED, "; ".join(reasons), None
    return UNRESOLVED, "; ".join(reasons) or "no arxiv_id or doi given", None


def exit_code(statuses):
    """0 all verified; 1 any measured false; 2 none false but some unmeasured."""
    if any(s in FALSE_STATES for s in statuses):
        return 1
    if any(s == UNMEASURED for s in statuses):
        return 2
    return 0


def bib_escape(s):
    return str(s).replace("\\", "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def make_key(rec, used):
    last = (rec.get("authors") or ["Unknown"])[0].split()[-1] if rec.get("authors") else "Unknown"
    word = next((w for w in re.findall(r"[A-Za-z0-9]+", rec.get("title", "")) if len(w) > 3), "ref")
    base = re.sub(r"[^a-z0-9]", "", f"{last}{rec.get('year', '')}{word}".lower()) or "ref"
    key, n = base, 1
    while key in used:
        n += 1
        key = f"{base}{chr(ord('a') + n - 2)}" if n <= 27 else f"{base}{n}"
    used.add(key)
    return key


def to_bibtex(key, rec, arxiv_id=""):
    fields = {"title": "{" + bib_escape(rec["title"]) + "}",
              "author": bib_escape(" and ".join(a for a in rec.get("authors", []) if a) or "Unknown"),
              "year": rec.get("year", "")}
    venue = rec.get("venue", "")
    doi = rec.get("doi", "")
    if arxiv_id:
        fields["eprint"], fields["archiveprefix"] = arxiv_id, "arXiv"
    if doi and not norm_doi(doi).startswith("10.48550/"):
        fields["doi"] = doi
    fields["url"] = rec.get("url", "") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
    kind = rec.get("type", "")
    if kind == "journal-article" and venue:
        entry, fields["journal"] = "article", bib_escape(venue)
    elif kind in ("proceedings-article", "book-chapter") and venue:
        entry, fields["booktitle"] = "inproceedings", bib_escape(venue)
    else:
        entry = "misc"
        if venue:
            fields["howpublished"] = bib_escape(venue)
    body = ",\n".join(f"  {k} = {v}" if k == "title" else f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@{entry}{{{key},\n{body}\n}}\n"


def load_rows(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("papers") or data.get("refs") or data.get("references")
        if rows is None:
            raise ValueError("input JSON object has no 'papers' / 'refs' / 'references' list")
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError("input must be a JSON list of reference objects")
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not r.get("title"):
            raise ValueError(f"row {i} has no 'title'")
    return rows


# ---------------------------------------------------------------------------------------------- I/O ----

def _load_arxiv_pace(tools_dir: Path):
    """tools/arxiv_pace.py, ONE instance per process (sys.modules), loaded by path: every arXiv request waits on it."""
    import importlib.util
    mod = sys.modules.get("arxiv_pace")
    if mod is None:
        spec = importlib.util.spec_from_file_location("arxiv_pace", Path(tools_dir) / "arxiv_pace.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["arxiv_pace"] = mod
        spec.loader.exec_module(mod)
    return mod


def _get(url, retries=3, pause=3.0, headers=None):
    last = None
    for i in range(retries):
        _load_arxiv_pace(Path(__file__).resolve().parent).wait(url)    # arxiv.org 15 s, the API 3 s (2026-10-14)
        try:
            with _OK.urlopen(url, headers or HEADERS, timeout=40) as r:   # OpenAlex: the key (openalex_key.py)
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(pause * (i + 1))
    raise FetchError(str(last))


def fetch_datacite(doi):
    txt = _get(f"https://api.datacite.org/dois/{urllib.parse.quote(norm_doi(doi), safe='/')}", headers=DATACITE_HEADERS)
    if not txt:
        return None
    rec = datacite_norm(json.loads(txt))
    return rec if rec["title"] else None


def fetch_arxiv(aid):
    """Resolve an arXiv id through the chain export API -> DataCite -> abstract page (module docstring).
    Returns None only when a link that ANSWERED says the id does not exist; raises FetchError when no link
    could be asked."""
    errors = []
    try:
        txt = _get(f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid)}&max_results=1", retries=1)
        recs = parse_arxiv_atom(txt) if txt else []
        if recs:
            return recs[0]
    except FetchError as e:
        errors.append(f"export API: {e}")
    try:
        rec = fetch_datacite(arxiv_doi(aid))
        if rec is not None:
            rec.update(source="datacite(arXiv DOI)", arxiv_id=aid, url=f"https://arxiv.org/abs/{aid}", doi="")
            return rec
        return None                                     # the registry answered: no such arXiv DOI
    except FetchError as e:
        errors.append(f"DataCite: {e}")
    try:
        txt = _get(f"https://arxiv.org/abs/{urllib.parse.quote(aid)}", retries=2,
                   headers={"User-Agent": HEADERS["User-Agent"], "Accept": "text/html"})
        return parse_arxiv_abs_html(txt, aid) if txt else None
    except FetchError as e:
        errors.append(f"abs page: {e}")
    raise FetchError("; ".join(errors))


def fetch_crossref(doi):
    """DOI -> record via Crossref, then DataCite (Zenodo / arXiv / many datasets register there, not Crossref)."""
    txt = _get(f"https://api.crossref.org/works/{urllib.parse.quote(norm_doi(doi), safe='/')}")
    if txt:
        return crossref_norm(json.loads(txt)["message"])
    return fetch_datacite(doi)


def fetch_openalex_doi(doi):
    txt = _get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(norm_doi(doi), safe='/')}")
    return openalex_norm(json.loads(txt)) if txt else None


def search_openalex(title):
    q = urllib.parse.quote(re.sub(r"[^\w\s-]", " ", title))
    txt = _get(f"https://api.openalex.org/works?search={q}&per-page=5")
    if not txt:
        return None, 0.0
    best, br = None, 0.0
    for w in json.loads(txt).get("results", []):
        rec = openalex_norm(w)
        r = title_ratio(title, rec["title"])
        if r > br:
            best, br = rec, r
    return best, br


def verify_row(row, *, openalex=True, sleep=0.5):
    """Verify one input row against the APIs. Returns the output record (never raises on network)."""
    claimed = row["title"]
    aid, doi = clean_arxiv_id(row.get("arxiv_id")), norm_doi(row.get("doi"))
    if doi.startswith("10.48550/arxiv."):
        aid, doi = aid or clean_arxiv_id(doi.split("arxiv.", 1)[1]), ""
    arec = drec = None
    aerr = derr = None
    if aid:
        try:
            arec = fetch_arxiv(aid)
        except FetchError as e:
            aerr = str(e)
        time.sleep(sleep)
    if doi:
        try:
            drec = fetch_crossref(doi)
            if drec is None and openalex:
                drec = fetch_openalex_doi(doi)
        except FetchError as e:
            derr = str(e)
        time.sleep(sleep)
    status, reason, best = classify(claimed, arec, drec, arxiv_asked=bool(aid), doi_asked=bool(doi),
                                    arxiv_err=aerr, doi_err=derr)
    if status == UNRESOLVED and not aid and not doi:
        try:
            srec, r = search_openalex(claimed)
            if srec is not None and r >= SEARCH_MIN_RATIO:
                status, reason, best = VERIFIED_SEARCH, f"OpenAlex title search match {r:.2f}", srec
            else:
                reason = f"no id given; best OpenAlex title match only {r:.2f}"
        except FetchError as e:
            status, reason = UNMEASURED, f"no id given; OpenAlex search failed ({e})"
    out = {"input": row, "status": status, "reason": reason, "arxiv_id": aid,
           "record": best, "arxiv_record": arec, "doi_record": drec, "openalex": None}
    if openalex and status in OK_STATES:
        lookup = (best or {}).get("doi") or doi or (arxiv_doi(aid) if aid else "")
        if lookup:
            try:
                out["openalex"] = fetch_openalex_doi(lookup)
            except FetchError:
                out["openalex"] = None
            time.sleep(sleep)
    return out


def _links(res):
    rec = res.get("record") or {}
    out = []
    if res.get("arxiv_id"):
        out.append(f"[arXiv:{res['arxiv_id']}](https://arxiv.org/abs/{res['arxiv_id']})")
    doi = ""
    for cand in (rec.get("doi"), (res.get("doi_record") or {}).get("doi"), res["input"].get("doi")):
        d = norm_doi(cand)
        if d and not d.startswith("10.48550/"):
            doi = d
            break
    if doi:
        out.append(f"[doi:{doi}](https://doi.org/{doi})")
    return " / ".join(out) or (rec.get("url") or "")


def reading_list_md(results, group_by=None):
    """VERIFIED rows as a Markdown reading list (title, first author, year, venue, download links, citations),
    optionally grouped by an input field. Only API metadata is printed -- never the claimed title."""
    groups = {}
    for res in results:
        if res["status"] not in OK_STATES or not res.get("record"):
            continue
        g = str(res["input"].get(group_by, "")) if group_by else ""
        groups.setdefault(g, []).append(res)
    lines = []
    for g in sorted(groups):
        if group_by:
            lines += [f"**{g or '(ungrouped)'}**", ""]
        lines += ["| # | title | first author | year | venue | link | cited by |", "|---|---|---|---|---|---|---|"]
        rows = sorted(groups[g], key=lambda r: (-(int(r["record"].get("year") or 0)), r["record"]["title"].lower()))
        for i, res in enumerate(rows, 1):
            rec = res["record"]
            oa = res.get("openalex") or {}
            venue = (oa.get("venue") or rec.get("venue") or "").replace("|", "/")
            first = (rec.get("authors") or [""])[0].replace("|", "/")
            etal = " et al." if len(rec.get("authors") or []) > 1 else ""
            cites = oa.get("cited_by_count")
            lines.append(f"| {i} | {rec['title'].replace('|', '/')} | {first}{etal} | {rec.get('year', '')} | {venue} | "
                         f"{_links(res)} | {'' if cites is None else cites} |")
        lines.append("")
    return "\n".join(lines)


def write_outputs(results, out_dir, group_by=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used, bib, log = set(), [], []
    for res in results:
        rec = res["record"]
        key = res["input"].get("key") or (make_key(rec, used) if rec else "")
        if res["input"].get("key"):
            used.add(key)
        res["key"] = key
        oa = res.get("openalex") or {}
        cites = oa.get("cited_by_count")
        if rec and res["status"] in OK_STATES:
            bib.append(to_bibtex(key, rec, res["arxiv_id"]))
        log.append("| {k} | {s} | {t} | {y} | {c} | {why} |".format(
            k=key or "-", s=res["status"],
            t=(rec["title"] if rec else res["input"]["title"]).replace("|", "/")[:110],
            y=(rec or {}).get("year", ""), c="" if cites is None else cites,
            why=res.get("reason", "").replace("|", "/")[:160]))
    (out_dir / "verified_refs.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8", newline="\n")
    (out_dir / "verified_refs.bib").write_text(
        "% Generated by tools/verify_refs.py from arXiv / Crossref / OpenAlex API records. Do not hand-edit.\n\n"
        + "\n".join(bib), encoding="utf-8", newline="\n")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    head = ("# Reference verification log\n\nGenerated by `tools/verify_refs.py`. Each row's metadata is what the "
            "API returned; `VERIFIED` means the identifier resolves to a record whose title matches the claimed title "
            f"(normalised similarity >= {MATCH_MIN_RATIO}). Citation counts are an OpenAlex snapshot.\n\n"
            + "Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "\n\n"
            "| key | status | title (API) | year | cited_by (OpenAlex) | reason |\n|---|---|---|---|---|---|\n")
    (out_dir / "verification_log.md").write_text(head + "\n".join(log) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "reading_list.md").write_text(
        "<!-- generated by tools/verify_refs.py from API records; do not hand-edit -->\n\n"
        + reading_list_md(results, group_by), encoding="utf-8", newline="\n")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", help="JSON list of {title, arxiv_id?, doi?, key?} (or an object with a 'papers' list)")
    ap.add_argument("--out-dir", required=True, help="where verified_refs.json / .bib / verification_log.md go")
    ap.add_argument("--no-openalex", action="store_true", help="skip OpenAlex (citation counts, DOI fallback)")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between API calls (be polite)")
    ap.add_argument("--group-by", default=None, help="input field to group reading_list.md by (e.g. 'category')")
    ap.add_argument("--from-json", action="store_true",
                    help="re-render the outputs from an EXISTING verified_refs.json (no network) -- e.g. to regroup")
    a = ap.parse_args(argv)
    if a.from_json:
        results = json.loads(Path(a.refs).read_text(encoding="utf-8"))
        counts = write_outputs(results, a.out_dir, a.group_by)
        print("re-rendered: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return exit_code([r["status"] for r in results])
    try:
        rows = load_rows(a.refs)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}")
        return 2
    results = []
    for i, row in enumerate(rows, 1):
        res = verify_row(row, openalex=not a.no_openalex, sleep=a.sleep)
        results.append(res)
        print(f"[{i}/{len(rows)}] {res['status']:<26} {row['title'][:70].encode('ascii', 'replace').decode()}")
    counts = write_outputs(results, a.out_dir, a.group_by)
    print("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    rc = exit_code([r["status"] for r in results])
    print({0: "ALL VERIFIED", 1: "SOME REFERENCES FAILED VERIFICATION", 2: "SOME COULD NOT BE MEASURED"}[rc])
    return rc


if __name__ == "__main__":
    sys.exit(main())
