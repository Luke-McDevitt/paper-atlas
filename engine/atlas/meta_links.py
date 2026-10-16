"""The facts ABOUT a paper, for the BlueBoat index (owner 2026-10-14: "which journal the paper is published in, when that
was, how many papers cite it, which papers cite it, what papers it cites, what is the impact factor of the journal, how
many stars its github repo has").

SOURCES, each value dated by the run that measured it (counts change):
  OpenAlex works      publication date, citation count (+ by year), the works it cites, its venue (open, no key; the
                      polite rate, never the owner's e-mail)
  OpenAlex sources    the venue's 2-YEAR MEAN CITEDNESS -- an open analogue of the Journal Impact Factor (citations in a
                      year to the venue's works of the two years before, per work) -- plus h-index and i10-index. The
                      Impact Factor itself is Clarivate's licensed data: NOT fetched, never scraped.
  OpenAlex `cites:`   the most-cited works citing a paper (the full list is a link, not data we keep)
  Semantic Scholar    citation and INFLUENTIAL-citation counts (a cross-check), and the publication date and venue where
                      OpenAlex has none
  GitHub (via `gh`)   stars, forks, last push, archived -- GraphQL, 50 repositories a query, the owner's own login
The citation graph INSIDE the corpus (which index papers cite which) is built from OpenAlex's reference lists.
A request that got no answer is NOT MEASURED (`failed` / unchecked), never 'absent'.
Network access is injected (`get` / `post` / `run`), so every function here is testable offline.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

OA = "https://api.openalex.org"
WORK_FIELDS = ("id,doi,ids,title,publication_date,cited_by_count,referenced_works,primary_location,locations,type,"
               "counts_by_year")
#: venue types whose citedness says something about the venue's papers; a REPOSITORY (arXiv, Zenodo, SSRN) is not a
#: venue, and arXiv's 2-year citedness is not the paper's journal metric (review 2026-10-14)
VENUE_TYPES = ("journal", "conference", "book series", "ebook platform")
SOURCE_FIELDS = "id,display_name,issn_l,type,host_organization_name,summary_stats,works_count,cited_by_count"
S2_FIELDS = "citationCount,influentialCitationCount,publicationDate,venue,externalIds"
#: a work whose title shares at least this much with the index title (title_sim) is the paper; below it, of SEVERAL
#: works on one DOI none is chosen (review 2026-10-14: 10.3390/jmse7010016 answers the CADDY dataset paper AND an
#: unrelated, more-cited arXiv paper -- their titles share 0.32)
TITLE_MATCH = 0.5
_STOP = frozenset("a an and are as at by for from in into is of on or the to toward towards using via with".split())


def short_id(url: Optional[str]) -> str:
    """'https://openalex.org/W123' -> 'W123'."""
    return (url or "").rstrip("/").rsplit("/", 1)[-1]


def _doi(x) -> str:
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(x or "").strip().lower())
    return d if d.startswith("10.") else ""


def work_doi(p: dict) -> str:
    """The DOI OpenAlex knows a paper by: its own DOI, else its PUBLISHED version's (review 2026-10-14: 257 published
    papers were measured as their arXiv preprint -- 'arXiv' as the venue, a fraction of the citations), else arXiv's
    (10.48550/arxiv.<id>, version dropped), else ''."""
    d = _doi(p.get("doi"))
    if d:
        return d
    pv = p.get("published_version")
    d = _doi(pv.get("doi") if isinstance(pv, dict) else pv)
    if d:
        return d
    ax = re.sub(r"v\d+$", "", (p.get("arxiv_id") or "").strip())
    return f"10.48550/arxiv.{ax.lower()}" if ax else ""


def covered_dois(w: dict) -> set:
    """Every DOI an OpenAlex work answers for: its own, its locations' DOIs, and the arXiv DOI of an arXiv location
    (OpenAlex MERGES a preprint into its published work: asked by the arXiv DOI, it answers with the journal DOI --
    review 2026-10-14: 205 papers were dropped because the answer was filed under the DOI it carried)."""
    out = {_doi(w.get("doi"))}
    for loc in w.get("locations") or []:
        for u in (loc.get("landing_page_url"), loc.get("pdf_url")):
            out.add(_doi(u))
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})", u or "", re.I)
            if m:
                out.add("10.48550/arxiv." + m.group(1).lower())
    out.discard("")
    return out


def parse_work(w: dict) -> dict:
    """An OpenAlex work -> the fields kept (venue from its primary location)."""
    src = ((w.get("primary_location") or {}).get("source")) or {}
    return {"openalex": short_id(w.get("id")), "doi": re.sub(r"^https://doi.org/", "", (w.get("doi") or "").lower()),
            "title": w.get("title") or "", "publication_date": w.get("publication_date") or "",
            "type": w.get("type") or "", "cited_by": w.get("cited_by_count"),
            "cited_by_by_year": {str(c["year"]): c["cited_by_count"] for c in w.get("counts_by_year") or []
                                 if c.get("year") is not None},
            "refs": [short_id(r) for r in w.get("referenced_works") or []],
            "source": short_id(src.get("id")), "source_name": src.get("display_name") or "",
            "source_type": src.get("type") or ""}


def _words(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) > 1 and w not in _STOP}


def title_sim(a: str, b: str) -> float:
    """How much two titles share: the Jaccard overlap of their words (lower case, stop words dropped), 0..1."""
    wa, wb = _words(a), _words(b)
    return round(len(wa & wb) / len(wa | wb), 3) if wa and wb else 0.0


def _venue_rank(w: dict) -> int:
    t = w.get("source_type") or ""
    return 2 if t in VENUE_TYPES else 0 if t == "repository" else 1


def choose_work(cands: Sequence[dict], title: str = "") -> Tuple[Optional[dict], str]:
    """The OpenAlex works answering for one DOI -> (the one that IS the paper, how). One work: it ('only'). Several
    ('chosen'): among those titled like the index title (title_sim >= TITLE_MATCH; all of them when no title is given),
    a journal / conference record over a repository copy, then the most cited, then the most references (review
    2026-10-14: 'the last' and then 'the most cited' kept a repository copy with no references, or ANOTHER paper).
    Several and none titled like it: (None, 'ambiguous') -- left blank, never guessed; none: (None, 'absent').
    The chosen work carries `copies` (the other works titled like it: a citation to one counts) and `title_sim`."""
    uniq = list({w["openalex"]: w for w in cands if w.get("openalex")}.values())
    if not uniq:
        return None, "absent"
    sims = {w["openalex"]: title_sim(title, w.get("title", "")) for w in uniq} if title else {}
    pool = [w for w in uniq if sims[w["openalex"]] >= TITLE_MATCH] if title and len(uniq) > 1 else uniq
    if not pool:
        return None, "ambiguous"
    best = max(pool, key=lambda w: (_venue_rank(w), w.get("cited_by") or 0, len(w.get("refs") or []), w["openalex"]))
    out = {**best, "copies": sorted(w["openalex"] for w in pool if w["openalex"] != best["openalex"])}
    if title:
        out["title_sim"] = sims[best["openalex"]]
    return out, "only" if len(uniq) == 1 else "chosen"


def openalex_candidates(dois: Iterable[str], get: Callable, batch: int = 50, per_page: int = 200
                        ) -> Tuple[Dict[str, List[dict]], List[str]]:
    """DOIs -> ({asked doi: [every parsed work answering for it]; [] = asked and not in OpenAlex}, [dois NOT looked up:
    a failed request -- unchecked, never 'not in OpenAlex']). Each work is filed under every ASKED DOI it covers
    (covered_dois); a batch is paged until OpenAlex's count is read (review 2026-10-14: per-page = the batch size cut a
    batch with a duplicate short)."""
    dois = sorted({d for d in dois if d})
    found: Dict[str, List[dict]] = {}
    unchecked: List[str] = []
    for i in range(0, len(dois), batch):
        chunk = dois[i:i + batch]
        asked, works, page, total, ok = set(chunk), [], 1, None, True
        while total is None or len(works) < total:
            q = urllib.parse.urlencode({"filter": "doi:" + "|".join(chunk), "per-page": per_page, "page": page,
                                        "select": WORK_FIELDS})
            res = _json(get, f"{OA}/works?{q}")
            if not isinstance(res, dict):
                ok = False
                break
            got = res.get("results") or []
            works += got
            total = int((res.get("meta") or {}).get("count") or len(works))
            if not got:
                break
            page += 1
        if not ok:
            unchecked += chunk
            continue
        for d in chunk:
            found[d] = []
        for w in works:
            pw = parse_work(w)
            for d in sorted(covered_dois(w) & asked):
                found[d].append(pw)
    return found, unchecked


def openalex_works(dois: Iterable[str], get: Callable, titles: Optional[Dict[str, str]] = None, batch: int = 50,
                   per_page: int = 200) -> Tuple[Dict[str, dict], List[str], Dict[str, List[str]]]:
    """DOIs -> ({asked doi: the work that is the paper (choose_work, against `titles`: {doi: index title})},
    [unchecked dois], {doi: [work ids]: several works, none titled like the paper -- ambiguous, left blank})."""
    cands, unchecked = openalex_candidates(dois, get, batch, per_page)
    found: Dict[str, dict] = {}
    ambiguous: Dict[str, List[str]] = {}
    for d, cs in cands.items():
        w, how = choose_work(cs, (titles or {}).get(d, ""))
        if w:
            found[d] = w
        elif how == "ambiguous":
            ambiguous[d] = sorted({c["openalex"] for c in cs})
    return found, unchecked, ambiguous


def openalex_sources(ids: Iterable[str], get: Callable, batch: int = 50, failed: Optional[List[str]] = None
                     ) -> Dict[str, dict]:
    """Venue ids (S...) -> {S: name, issn_l, type, publisher, mean_citedness_2yr, h_index, i10_index, works, cited_by}.
    The ids of a batch that got no answer go into `failed` (not measured)."""
    ids = sorted({i for i in ids if i and i.startswith("S")})
    out = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        q = urllib.parse.urlencode({"filter": "openalex_id:" + "|".join(chunk), "per-page": batch,
                                    "select": SOURCE_FIELDS})
        res = _json(get, f"{OA}/sources?{q}")
        if not isinstance(res, dict):
            if failed is not None:
                failed += chunk
            continue
        for s in res.get("results") or []:
            st = s.get("summary_stats") or {}
            out[short_id(s.get("id"))] = {
                "name": s.get("display_name") or "", "issn_l": s.get("issn_l") or "", "type": s.get("type") or "",
                "publisher": s.get("host_organization_name") or "",
                "mean_citedness_2yr": _round(st.get("2yr_mean_citedness"), 3), "h_index": st.get("h_index"),
                "i10_index": st.get("i10_index"), "works": s.get("works_count"), "cited_by": s.get("cited_by_count")}
    return out


def top_citing(work: str, get: Callable, n: int = 5) -> Optional[List[dict]]:
    """The n most-cited works citing `work` (W...) -> [{title, year, cited_by, doi, openalex}]; None if not asked."""
    q = urllib.parse.urlencode({"filter": f"cites:{work}", "sort": "cited_by_count:desc", "per-page": n,
                                "select": "id,doi,title,publication_year,cited_by_count"})
    res = _json(get, f"{OA}/works?{q}")
    if not isinstance(res, dict):
        return None
    return [{"title": w.get("title") or "", "year": w.get("publication_year"), "cited_by": w.get("cited_by_count"),
             "doi": re.sub(r"^https://doi.org/", "", (w.get("doi") or "").lower()), "openalex": short_id(w.get("id"))}
            for w in res.get("results") or []]


def corpus_graph(works: Dict[str, dict]) -> Dict[str, dict]:
    """{key: {openalex, copies, refs}} -> {key: {refs_in_corpus: [keys it cites], cited_by_in_corpus: [keys citing
    it]}}. A citation to ANY of a work's ids (its `copies` too) counts. Keys on one work (an alias not yet folded) are
    one alias set: each gets the citation, and none of them cites another (review 2026-10-14: one key won, and only in
    one direction). Pure."""
    ids = {k: {i for i in [w.get("openalex"), *(w.get("copies") or [])] if i} for k, w in works.items()}
    by_id: Dict[str, set] = {}
    for k, s in ids.items():
        for i in s:
            by_id.setdefault(i, set()).add(k)
    out = {k: {"refs_in_corpus": [], "cited_by_in_corpus": []} for k in works}
    for k, w in works.items():
        same = {x for i in ids[k] for x in by_id[i]} | {k}
        cited = sorted({x for r in w.get("refs") or [] for x in by_id.get(r, ())} - same)
        out[k]["refs_in_corpus"] = cited
        for c in cited:
            out[c]["cited_by_in_corpus"].append(k)
    for v in out.values():
        v["cited_by_in_corpus"].sort()
    return out


def s2_batch(ids: Sequence[str], post: Callable, batch: int = 500, failed: Optional[List[str]] = None
             ) -> Dict[str, dict]:
    """Semantic Scholar paper ids -> {id: citations, influential, publication_date, venue}. The ids of a batch that got
    no answer go into `failed` (not measured -- never 'not in Semantic Scholar')."""
    out = {}
    ids = [i for i in dict.fromkeys(ids) if i]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        res = post(f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={S2_FIELDS}", {"ids": chunk})
        if not isinstance(res, list):
            if failed is not None:
                failed += chunk
            continue
        for pid, r in zip(chunk, res):
            if isinstance(r, dict):
                out[pid] = {"citations": r.get("citationCount"), "influential": r.get("influentialCitationCount"),
                            "publication_date": r.get("publicationDate") or "", "venue": r.get("venue") or ""}
    return out


def s2_id(url: str) -> str:
    """'https://www.semanticscholar.org/paper/<40 hex>' -> the id, else ''."""
    m = re.search(r"semanticscholar\.org/paper/(?:[^/]+/)?([0-9a-f]{40})", url or "")
    return m.group(1) if m else ""


def github_repo(url: Optional[str], repo_of: Optional[Callable] = None) -> Optional[Tuple[str, str]]:
    """A GitHub repository URL -> (owner, name); anything else -> None. With `repo_of` (code_links.repo_of) the SAME
    normalisation as the rest of the index (review 2026-10-14: a trailing '.' or ')' made the two disagree)."""
    if repo_of is not None:
        r = repo_of(url or "")
        if not r or not r.lower().startswith("github.com/"):
            return None
        owner, name = r.split("/")[1:3]
        return owner, name
    m = re.match(r"(?i)^https?://(?:www\.)?github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git)?(?:[/#?].*)?$",
                 (url or "").strip())
    if not m or m.group(1).lower() in ("orgs", "topics", "sponsors", "features"):
        return None
    return m.group(1), m.group(2).rstrip(".")


def github_stars(repos: Iterable[Tuple[str, str]], run: Callable, batch: int = 50,
                 failed: Optional[List[str]] = None) -> Dict[str, dict]:
    """(owner, name) pairs -> {'owner/name': stars, forks, pushed, archived} through ONE GraphQL query per batch
    (`run(query)` -> the parsed JSON answer, e.g. `gh api graphql`). A repo GitHub says does not exist (a NOT_FOUND
    error on its alias) is absent; one it gave no answer for (a failed query, a rate limit, any other error) goes into
    `failed` as 'owner/name' -- not measured."""
    repos = sorted({(o, n) for o, n in repos})
    out = {}
    for i in range(0, len(repos), batch):
        part = repos[i:i + batch]
        q = "query {" + " ".join(
            f'r{j}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) '
            '{ nameWithOwner stargazerCount forkCount pushedAt isArchived }' for j, (o, n) in enumerate(part)) + "}"
        res = run(q)
        res = res if isinstance(res, dict) else {}
        data = res.get("data") or {}
        gone = {str((e.get("path") or [""])[0]) for e in res.get("errors") or []
                if isinstance(e, dict) and e.get("type") == "NOT_FOUND"}
        for j, (o, n) in enumerate(part):
            r = data.get(f"r{j}")
            if isinstance(r, dict):
                out[f"{o}/{n}".lower()] = {"stars": r.get("stargazerCount"), "forks": r.get("forkCount"),
                                           "pushed": (r.get("pushedAt") or "")[:10], "archived": r.get("isArchived")}
            elif f"r{j}" not in gone and failed is not None:
                failed.append(f"{o}/{n}".lower())
    return out


def _json(get: Callable, url: str):
    try:
        body = get(url)
    except Exception:                                    # noqa: BLE001 -- recorded as unchecked by the caller
        return None
    if isinstance(body, (dict, list)) or body is None:
        return body
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return None


def _round(x, n):
    return round(x, n) if isinstance(x, (int, float)) else None
