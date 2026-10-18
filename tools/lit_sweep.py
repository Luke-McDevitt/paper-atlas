"""lit_sweep -- a scripted, cached, MEASURED literature sweep: keyword groups -> OpenAlex + arXiv + Semantic Scholar
-> one deduplicated candidate list, each row marked NEW or KNOWN against the lists you already verified.

WHAT IT IS. A sweep spec (JSON, committed under `tools/lit_sweeps/`) names one or more QUERIES, each an AND of
OR-groups of terms. The same boolean is sent to three registries, every hit is re-checked LOCALLY against the
title + abstract (OpenAlex stems, arXiv tokenises, Semantic Scholar ranks -- the local check makes them agree on what
"matched" means), the hits are merged across sources by arXiv id / DOI / normalised title, and each merged record is
marked KNOWN when it is already in one of your `--known` lists (any `tools/verify_refs.py` output works), else NEW.

    python tools/lit_sweep.py tools/lit_sweeps/world_models_marine.json --since 2026-09-01 \
        --known report/world_models/verified_refs.json --out-dir out/sweep
    python tools/lit_sweep.py SPEC --since 2018-01-01 --known KNOWN.json --out-dir out/recall --min-recall 0.6
    python tools/lit_sweep.py SPEC --since 2026-09-01 --sources openalex,arxiv --out-dir out/sweep   # skip S2

WHY IT EXISTS. The 2026-09-23 world-models review (`report/WORLD_MODELS_MARINE_REVIEW.md`) was a three-agent web
sweep; refreshing it a week later should cost one command and a read of a short table, not another agent sweep. And
a sweep with no recall number cannot be trusted: `--known` turns the papers already verified into a YARDSTICK. The
known papers that are IN SCOPE (their own title + abstract satisfies a query, inside the date window) and that the
sweep re-finds, per source, is the recall block -- the misses name the vocabulary the spec lacks.

OUTPUTS (in --out-dir): `candidates.json` (every merged record: ids, title, date, venue, abstract, sources, matched
queries and terms, score, status NEW|KNOWN), `candidates.md` (NEW first, compact -- the triage sheet),
`sweep_log.json` (per query x source: the URL asked, hits, cost, error; plus the recall block), and `refs_new.json`
(the NEW rows in `tools/verify_refs.py` input shape, ready to verify). Raw API answers are cached under
`<out-dir>/.cache` (or --cache-dir), so a re-run, a re-scoring or a recall check costs no network.

EXIT CODES (repo contract, CONTRIBUTING sec 2b): 0 = every source answered; 1 = `--min-recall` was given and the
measured recall fell below it; 2 = at least one source could not be asked (its rows are missing; the log says which).

HONEST SCOPE (do NOT overclaim):
  - Abstract-level keyword matching. A paper that never uses any of the terms is invisible, and a match is not
    relevance: triage is still a read of candidates.md. A record with no abstract is matched on its title alone.
  - Recall is measured against what you already KNOW, so it is an upper bound on recall against what exists.
    A known row with no date is placed by its arXiv id's month (YYMM), else its year, and compared at that
    precision -- so a month-long window does not count every paper of the year as in scope.
  - Dates differ by source for the same paper (OpenAlex publication_date, arXiv v1 submission, S2
    publicationDate); a preprint and its venue version can be a year apart.
  - OpenAlex charges credits per call (it reported cost_usd 0.001 per query on 2026-10-01); `$OPENALEX_API_KEY`
    is sent when set -- an unredirected header, never in a URL (tools/openalex_key.py). Semantic Scholar's keyless
    pool is shared and 429s often; a 429 that survives the retries
    is COULD_NOT_MEASURE for that query, never "no hits" (`$S2_API_KEY` is sent when set). Failed calls are not
    cached, so simply re-running the same command re-asks only what failed.
  - OpenAlex is paged by page NUMBER (`page=1, 2, ...`, 200 a page), not by cursor. MEASURED 2026-10-16 on the lab
    computer (no key): a cursor past the first page answers HTTP 401 "API key required for cursor paging", which
    lost EVERY query with more than 200 hits (2 of the 4 in tools/lit_sweeps/aqua2.json), while `page=2` answers
    keyless. Page-number paging reaches the first 10,000 results; a larger --max-per-source stops there and the
    log row carries `truncated_at`.
Network code reuses `tools/verify_refs.py` (`_get`, headers, OpenAlex record normalisation, `litreview_core.norm`).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_verify_refs():
    spec = importlib.util.spec_from_file_location("verify_refs", ROOT / "tools" / "verify_refs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_refs", mod)
    spec.loader.exec_module(mod)
    return mod


VR = _load_verify_refs()
norm, norm_doi, clean_arxiv_id, FetchError = VR.norm, VR.norm_doi, VR.clean_arxiv_id, VR.FetchError

SOURCES = ("openalex", "arxiv", "s2")
NEW, KNOWN = "NEW", "KNOWN"
UNMEASURED = "COULD_NOT_MEASURE"
OPENALEX_SELECT = ("id,doi,title,display_name,publication_year,publication_date,primary_location,authorships,"
                   "cited_by_count,open_access,abstract_inverted_index,type")
OPENALEX_PER_PAGE = 200
#: how far OpenAlex's page-number paging reaches; cursor paging goes further but needs an API key (docstring)
OPENALEX_PAGE_LIMIT = 10_000
S2_FIELDS = "title,abstract,year,publicationDate,externalIds,venue,url,citationCount,authors"
ATOM = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom",
        "os": "http://a9.com/-/spec/opensearch/1.1/"}
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)


# ------------------------------------------------------------------------------------------- pure parts ----

def load_spec(path):
    """Read + validate a sweep spec. Raises ValueError naming the problem."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = spec.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("spec needs a non-empty 'queries' list")
    seen = set()
    for q in queries:
        qid = q.get("id")
        if not qid or qid in seen:
            raise ValueError(f"query ids must be present and unique (bad: {qid!r})")
        seen.add(qid)
        groups = q.get("groups")
        if (not isinstance(groups, list) or not groups
                or not all(isinstance(g, list) and g and all(isinstance(t, str) and t.strip() for t in g)
                           for g in groups)):
            raise ValueError(f"query {qid!r}: 'groups' must be a list of non-empty lists of terms")
    spec.setdefault("exclude", [])
    spec.setdefault("name", Path(path).stem)
    return spec


def _fold(s):
    """Lower-case, hyphens / slashes / underscores -> spaces, whitespace collapsed: 'World-Model' == 'world model'."""
    return re.sub(r"\s+", " ", re.sub(r"[-_/]+", " ", (s or "").lower())).strip()


def term_hits(text, term):
    """Does `term` occur in `text` as whole words (an optional plural s/es allowed)? Pure."""
    t = _fold(term)
    if not t:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(t)}(?:s|es)?(?![a-z0-9])", _fold(text)) is not None


def match_query(text, groups):
    """AND of OR-groups. Returns the matched terms per group, or None when any group has no hit. Pure."""
    out = []
    for g in groups:
        hit = [t for t in g if term_hits(text, t)]
        if not hit:
            return None
        out.append(hit)
    return out


def is_excluded(text, exclude):
    return any(term_hits(text, t) for t in exclude or [])


def _q(term, quote_all=False):
    term = term.strip()
    return f'"{term}"' if (quote_all or re.search(r"[\s-]", term)) else term


def openalex_query(groups):
    """'("world model" OR jepa) AND (underwater OR sonar)' -- OpenAlex boolean search syntax. Pure."""
    return " AND ".join("(" + " OR ".join(_q(t) for t in g) + ")" for g in groups)


def arxiv_query(groups, since=None, until=None):
    """arXiv API search_query over title AND abstract, with an optional submittedDate window. Pure."""
    parts = ["(" + " OR ".join(f"ti:{_q(t, True)} OR abs:{_q(t, True)}" for t in g) + ")" for g in groups]
    if since or until:
        lo = (since or "1991-01-01").replace("-", "") + "0000"
        hi = (until or date.today().isoformat()).replace("-", "") + "2359"
        parts.append(f"submittedDate:[{lo} TO {hi}]")
    return " AND ".join(parts)


def s2_query(groups):
    """Semantic Scholar bulk-search syntax: '("world model" | jepa) + (underwater | sonar)'. Pure."""
    return " + ".join("(" + " | ".join(_q(t) for t in g) + ")" for g in groups)


def doi_to_arxiv(doi):
    """'10.48550/arXiv.2609.30214' -> '2609.30214'; '' for any other DOI. Pure."""
    m = _ARXIV_DOI.match(norm_doi(doi or ""))
    return clean_arxiv_id(m.group(1)) if m else ""


def record_keys(rec):
    """Every identity key a record can be matched on: arXiv id, non-arXiv DOI, normalised title. Pure."""
    keys = []
    aid = clean_arxiv_id(rec.get("arxiv_id") or "") or doi_to_arxiv(rec.get("doi") or "")
    if aid:
        keys.append("arxiv:" + aid.lower())
    doi = norm_doi(rec.get("doi") or "")
    if doi and not doi_to_arxiv(doi):
        keys.append("doi:" + doi)
    t = norm(rec.get("title") or "")
    if len(t) >= 12:
        keys.append("title:" + t)
    return keys


def merge_records(records):
    """Merge per-source records that share ANY key (arXiv id / DOI / title). Fields fill from the first record that
    has them; `sources`, `queries` and `matched` union. Order: first appearance. Pure."""
    merged, index = [], {}
    for rec in records:
        keys = record_keys(rec)
        slot = next((index[k] for k in keys if k in index), None)
        if slot is None:
            slot = len(merged)
            merged.append(dict(rec, sources=sorted(set(rec.get("sources") or [])),
                               queries=sorted(set(rec.get("queries") or [])),
                               matched=sorted(set(rec.get("matched") or []))))
        else:
            m = merged[slot]
            for k, v in rec.items():
                if k in ("sources", "queries", "matched"):
                    m[k] = sorted(set(m.get(k) or []) | set(v or []))
                elif v not in (None, "", [], {}) and m.get(k) in (None, "", [], {}):
                    m[k] = v
        for k in record_keys(merged[slot]):
            index.setdefault(k, slot)
    for m in merged:
        m["arxiv_id"] = clean_arxiv_id(m.get("arxiv_id") or "") or doi_to_arxiv(m.get("doi") or "")
    return merged


def known_rows(data):
    """A known list (verify_refs output rows, {'papers': [...]}, or plain {title, arxiv_id?, doi?} rows) ->
    normalised {title, arxiv_id, doi, abstract, year, date}. verify_refs rows prefer the REGISTRY record. Pure."""
    rows = data.get("papers", data) if isinstance(data, dict) else data
    out = []
    for r in rows or []:
        rec = r.get("record") if isinstance(r.get("record"), dict) else {}
        inp = r.get("input") if isinstance(r.get("input"), dict) else {}
        title = rec.get("title") or r.get("title") or inp.get("title") or ""
        out.append({"title": title,
                    "arxiv_id": clean_arxiv_id(rec.get("arxiv_id") or r.get("arxiv_id") or inp.get("arxiv_id") or ""),
                    "doi": norm_doi(rec.get("doi") or r.get("doi") or inp.get("doi") or ""),
                    "abstract": rec.get("abstract") or r.get("abstract") or "",
                    "year": str(rec.get("year") or r.get("year") or ""),
                    "date": rec.get("date") or r.get("date") or ""})
    return [k for k in out if k["title"]]


def known_index(rows):
    idx = {}
    for i, r in enumerate(rows):
        for k in record_keys(r):
            idx.setdefault(k, i)
    return idx


def mark_known(candidates, kidx):
    """Set status NEW / KNOWN (and `known_as`, the matching key) on each merged candidate in place."""
    for c in candidates:
        hit = next((k for k in record_keys(c) if k in kidx), None)
        c["status"] = KNOWN if hit else NEW
        c["known_as"] = hit or ""
    return candidates


def score(c, spec):
    """Triage order, not relevance: distinct matched terms + 2 when the FIRST group's term is in the title + one per
    source that found it. Pure."""
    title_bonus = 0
    for q in spec["queries"]:
        if q["id"] in (c.get("queries") or []) and match_query(c.get("title", ""), q["groups"][:1]):
            title_bonus = 2
            break
    return len(set(c.get("matched") or [])) + title_bonus + len(c.get("sources") or [])


def approx_date(rec):
    """A record's date: its own `date`, else the YYYY-MM a new-style arXiv id encodes ('2607.29393' -> '2026-07'),
    else ''. Pure."""
    d = (rec.get("date") or "")[:10]
    if d:
        return d
    m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", clean_arxiv_id(rec.get("arxiv_id") or ""))
    return f"20{m.group(1)}-{m.group(2)}" if m else ""


def in_window(rec, since, until=None):
    """Is a record's date (else its arXiv-id month, else its year) inside [since, until], compared at the precision the
    record has? Undated -> True (cannot exclude what we cannot date). Pure."""
    d = approx_date(rec)
    if d:
        return (not since or d >= since[:len(d)]) and (not until or d <= until[:len(d)])
    y = (rec.get("year") or "")[:4]
    if y.isdigit():
        return (not since or y >= since[:4]) and (not until or y <= until[:4])
    return True


def recall_block(known, candidates, spec, since, until=None):
    """The yardstick: known rows IN SCOPE (title + abstract satisfy some query, not excluded, in window) and how many
    the sweep re-found, overall and per source. Pure."""
    by_key = {}
    for c in candidates:
        for k in record_keys(c):
            by_key.setdefault(k, c)
    in_scope, found, per_src, missed = 0, 0, {s: 0 for s in SOURCES}, []
    for r in known:
        text = f"{r['title']} {r.get('abstract', '')}"
        if is_excluded(text, spec["exclude"]) or not in_window(r, since, until):
            continue
        if not any(match_query(text, q["groups"]) for q in spec["queries"]):
            continue
        in_scope += 1
        c = next((by_key[k] for k in record_keys(r) if k in by_key), None)
        if c is None:
            missed.append(r["title"])
            continue
        found += 1
        for s in c.get("sources") or []:
            per_src[s] = per_src.get(s, 0) + 1
    return {"known_total": len(known), "in_scope": in_scope, "refound": found,
            "recall": round(found / in_scope, 3) if in_scope else None,
            "recall_by_source": {s: (round(n / in_scope, 3) if in_scope else None) for s, n in per_src.items()},
            "missed": missed}


def exit_code(log, recall=None, min_recall=None):
    if min_recall is not None and recall is not None and recall.get("recall") is not None \
            and recall["recall"] < min_recall:
        return 1
    if any(e.get("error") for e in log):
        return 2
    return 0


def _md_cell(s, n=None):
    s = re.sub(r"\s+", " ", str(s or "")).replace("|", "/").strip()
    return s if n is None or len(s) <= n else s[: n - 3].rstrip() + "..."


def candidates_md(candidates, spec, meta):
    lines = [f"# Sweep `{spec['name']}` -- {meta['since']} .. {meta['until'] or 'today'}", "",
             f"*Generated by `tools/lit_sweep.py` on {meta['run_date']}; {len(candidates)} merged records "
             f"({sum(c['status'] == NEW for c in candidates)} NEW). Matching is abstract-level keywords: triage before "
             "citing, then verify with `tools/verify_refs.py` (`refs_new.json`).*", "",
             "| # | status | score | date | title | venue | ids | sources | queries | matched |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(candidates, 1):
        ids = " ".join(x for x in (c.get("arxiv_id") and f"arXiv:{c['arxiv_id']}",
                                   (c.get("doi") and not doi_to_arxiv(c["doi"])) and f"doi:{c['doi']}") if x)
        url = c.get("url") or (f"https://arxiv.org/abs/{c['arxiv_id']}" if c.get("arxiv_id") else "")
        title = f"[{_md_cell(c.get('title'), 110)}]({url})" if url else _md_cell(c.get("title"), 110)
        lines.append(f"| {i} | {c['status']} | {c['score']} | {c.get('date') or c.get('year') or ''} | {title} | "
                     f"{_md_cell(c.get('venue'), 40)} | {ids} | {','.join(c.get('sources') or [])} | "
                     f"{','.join(c.get('queries') or [])} | {_md_cell(', '.join(c.get('matched') or []), 80)} |")
    lines += ["", "## Abstract snippets (NEW only)", ""]
    for i, c in enumerate(candidates, 1):
        if c["status"] == NEW:
            lines.append(f"{i}. **{_md_cell(c.get('title'))}** -- {_md_cell(c.get('abstract') or '(no abstract)', 420)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------- parsers ----

def parse_openalex(txt):
    d = json.loads(txt)
    recs = []
    for w in d.get("results", []):
        r = VR.openalex_norm(w)
        r["date"] = w.get("publication_date") or ""
        r["arxiv_id"] = doi_to_arxiv(r.get("doi") or "")
        recs.append(r)
    meta = d.get("meta") or {}
    return recs, meta.get("next_cursor"), meta.get("count"), meta.get("cost_usd")


def parse_arxiv_feed(xml_text):
    root = ET.fromstring(xml_text)
    total = root.findtext("os:totalResults", default="0", namespaces=ATOM)
    recs = []
    for e in root.findall("a:entry", ATOM):
        title = re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=ATOM)).strip()
        eid = e.findtext("a:id", default="", namespaces=ATOM)
        if not title or title == "Error" or "/abs/" not in eid:
            continue
        aid = clean_arxiv_id(eid.rsplit("/abs/", 1)[-1])
        recs.append({"source": "arxiv", "title": title, "arxiv_id": aid, "url": f"https://arxiv.org/abs/{aid}",
                     "authors": [a.findtext("a:name", default="", namespaces=ATOM)
                                 for a in e.findall("a:author", ATOM)],
                     "date": (e.findtext("a:published", default="", namespaces=ATOM) or "")[:10],
                     "year": (e.findtext("a:published", default="", namespaces=ATOM) or "")[:4],
                     "doi": e.findtext("ar:doi", default="", namespaces=ATOM) or "",
                     "venue": e.findtext("ar:journal_ref", default="", namespaces=ATOM) or "",
                     "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=ATOM)).strip()})
    return recs, int(total or 0)


def parse_s2(txt):
    d = json.loads(txt)
    recs = []
    for p in d.get("data") or []:
        ext = p.get("externalIds") or {}
        aid = clean_arxiv_id(ext.get("ArXiv") or "")
        recs.append({"source": "s2", "title": re.sub(r"\s+", " ", p.get("title") or "").strip(), "arxiv_id": aid,
                     "doi": norm_doi(ext.get("DOI") or ""), "date": p.get("publicationDate") or "",
                     "year": str(p.get("year") or ""), "venue": p.get("venue") or "",
                     "url": (f"https://arxiv.org/abs/{aid}" if aid else p.get("url") or ""),
                     "authors": [a.get("name", "") for a in p.get("authors") or []],
                     "abstract": re.sub(r"\s+", " ", p.get("abstract") or "").strip(),
                     "cited_by_count": p.get("citationCount"), "s2_id": p.get("paperId", "")})
    return recs, d.get("token"), d.get("total")


# ---------------------------------------------------------------------------------------------- I/O ----

def cached_get(url, cache_dir, headers=None, refresh=False):
    """GET through an on-disk cache keyed by the URL. Returns text, None on 404; raises FetchError."""
    cache_dir = Path(cache_dir)
    path = cache_dir / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".txt")
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    txt = VR._get(url, retries=4, pause=4.0, headers=headers or VR.HEADERS)
    if txt is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(txt, encoding="utf-8", newline="\n")
    return txt


def run_openalex(q, since, until, cap, cache, refresh):
    filt = f"title_and_abstract.search:{openalex_query(q['groups'])}"
    if since:
        filt += f",from_publication_date:{since}"
    if until:
        filt += f",to_publication_date:{until}"
    base = (f"https://api.openalex.org/works?filter={urllib.parse.quote(filt, safe=':,')}"
            f"&per-page={OPENALEX_PER_PAGE}&select={OPENALEX_SELECT}")  # the key rides a header, never a URL
    # page NUMBERS, not cursors: a keyless cursor past page 1 answers 401 (MEASURED 2026-10-16, docstring)
    recs, cost, count, page = [], 0.0, None, 1
    while len(recs) < min(cap, OPENALEX_PAGE_LIMIT):
        txt = cached_get(base + f"&page={page}", cache, refresh=refresh)
        rows, _cursor, count, c = parse_openalex(txt or '{"results": []}')
        cost += float(c or 0)
        recs += rows
        if not rows or (count is not None and page * OPENALEX_PER_PAGE >= count):
            break
        page += 1
    info = {"url": base, "total": count, "cost_usd": round(cost, 4)}
    if cap > OPENALEX_PAGE_LIMIT and (count or 0) > OPENALEX_PAGE_LIMIT:
        info["truncated_at"] = OPENALEX_PAGE_LIMIT             # page-number paging stops here; say so in the log
    return recs[:cap], info


def run_arxiv(q, since, until, cap, cache, refresh, pause=3.0):
    sq = urllib.parse.quote(arxiv_query(q["groups"], since, until))
    base = f"https://export.arxiv.org/api/query?search_query={sq}&sortBy=submittedDate&sortOrder=descending"
    recs, start, total = [], 0, None
    while len(recs) < cap:
        url = base + f"&start={start}&max_results={min(200, cap - len(recs))}"
        fresh = refresh or not (Path(cache) / (hashlib.sha1(url.encode()).hexdigest() + ".txt")).exists()
        txt = cached_get(url, cache, refresh=refresh)
        page, total = parse_arxiv_feed(txt or "<feed xmlns='http://www.w3.org/2005/Atom'/>")
        recs += page
        start += len(page)
        if fresh:
            time.sleep(pause)                         # arXiv API etiquette: one call every ~3 s
        if not page or start >= total:
            break
    return recs[:cap], {"url": base, "total": total}


def run_s2(q, since, until, cap, cache, refresh, pause=1.5):
    window = f"{since or ''}:{until or ''}"
    base = (f"https://api.semanticscholar.org/graph/v1/paper/search/bulk?query={urllib.parse.quote(s2_query(q['groups']))}"
            f"&fields={S2_FIELDS}" + (f"&publicationDateOrYear={window}" if window != ":" else ""))
    key = os.environ.get("S2_API_KEY", "")
    headers = dict(VR.HEADERS, **({"x-api-key": key} if key else {}))
    recs, token, total = [], None, None
    while len(recs) < cap:
        url = base + (f"&token={urllib.parse.quote(token)}" if token else "")
        fresh = refresh or not (Path(cache) / (hashlib.sha1(url.encode()).hexdigest() + ".txt")).exists()
        txt = cached_get(url, cache, headers=headers, refresh=refresh)
        page, token, total = parse_s2(txt or '{"data": []}')
        recs += page
        if fresh:
            time.sleep(pause)
        if not token or not page:
            break
    return recs[:cap], {"url": base, "total": total}


RUNNERS = {"openalex": run_openalex, "arxiv": run_arxiv, "s2": run_s2}


def sweep(spec, since, until, sources, cap, cache, refresh=False, say=print):
    """Run every query on every source. Returns (raw per-source records with local match info, log rows)."""
    raw, log = [], []
    for q in spec["queries"]:
        for src in sources:
            entry = {"query": q["id"], "source": src, "hits": 0, "kept": 0, "error": ""}
            try:
                recs, info = RUNNERS[src](q, since, until, cap, cache, refresh)
                entry.update(info, hits=len(recs))
            except (FetchError, ET.ParseError, json.JSONDecodeError, ValueError) as e:
                entry["error"] = f"{UNMEASURED}: {str(e)[:200]}"
                recs = []
            for r in recs:
                text = f"{r.get('title', '')} {r.get('abstract', '')}"
                hit = match_query(text, q["groups"])
                if hit is None or is_excluded(text, spec["exclude"]) or not in_window(r, since, until):
                    continue
                entry["kept"] += 1
                raw.append(dict(r, sources=[src], queries=[q["id"]], matched=sorted({t for g in hit for t in g})))
            log.append(entry)
            say(f"  {q['id']:<24} {src:<9} hits={entry['hits']:<5} kept={entry['kept']:<5}"
                + (f" {entry['error'][:70]}" if entry["error"] else ""))
    return raw, log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", help="sweep spec JSON (see tools/lit_sweeps/)")
    ap.add_argument("--since", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--until", default="", help="YYYY-MM-DD, inclusive (default: today)")
    ap.add_argument("--known", nargs="*", default=[], help="JSON lists already verified (verify_refs output works)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sources", default=",".join(SOURCES), help=f"comma list from {','.join(SOURCES)}")
    ap.add_argument("--max-per-source", type=int, default=400, help="cap per query x source (default 400)")
    ap.add_argument("--cache-dir", default="", help="raw-response cache (default <out-dir>/.cache)")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and ask the APIs again")
    ap.add_argument("--min-recall", type=float, default=None, help="exit 1 when the in-scope known recall is lower")
    ap.add_argument("--queries", default="", help="comma list of query ids to run (default: all in the spec)")
    a = ap.parse_args(argv)

    spec = load_spec(a.spec)
    if a.queries:
        want = [q.strip() for q in a.queries.split(",") if q.strip()]
        unknown = sorted(set(want) - {q["id"] for q in spec["queries"]})
        if unknown:
            ap.error(f"unknown query id(s): {unknown}")
        spec["queries"] = [q for q in spec["queries"] if q["id"] in want]
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        ap.error(f"unknown source(s): {bad}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir) if a.cache_dir else out / ".cache"
    print(f"sweep {spec['name']}: {len(spec['queries'])} queries x {len(sources)} sources, {a.since}..{a.until or 'today'}")
    raw, log = sweep(spec, a.since, a.until or None, sources, a.max_per_source, cache, a.refresh)

    known = []
    for k in a.known:
        known += known_rows(json.loads(Path(k).read_text(encoding="utf-8")))
    cands = mark_known(merge_records(raw), known_index(known))
    for c in cands:
        c["score"] = score(c, spec)
    cands.sort(key=lambda c: c.get("date") or c.get("year") or "", reverse=True)       # newest first ...
    cands.sort(key=lambda c: (c["status"] != NEW, -c["score"]))                        # ... within NEW/score
    rec = recall_block(known, cands, spec, a.since, a.until or None) if known else None

    meta = {"spec": spec["name"], "since": a.since, "until": a.until, "run_date": date.today().isoformat(),
            "sources": sources, "max_per_source": a.max_per_source}
    (out / "candidates.json").write_text(json.dumps({"meta": meta, "candidates": cands}, indent=1,
                                                    ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out / "candidates.md").write_text(candidates_md(cands, spec, meta), encoding="utf-8", newline="\n")
    new_rows = [{"title": c["title"], "arxiv_id": c.get("arxiv_id") or "",
                 "doi": "" if doi_to_arxiv(c.get("doi") or "") else (c.get("doi") or ""),
                 "sweep": f"{spec['name']}:{','.join(c.get('queries') or [])}"} for c in cands if c["status"] == NEW]
    (out / "refs_new.json").write_text(json.dumps(new_rows, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8", newline="\n")
    (out / "sweep_log.json").write_text(json.dumps({"meta": meta, "log": log, "recall": rec}, indent=1,
                                                   ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    n_new = sum(c["status"] == NEW for c in cands)
    print(f"merged {len(cands)} records: {n_new} NEW, {len(cands) - n_new} KNOWN -> {out / 'candidates.md'}")
    cost = sum(float(e.get("cost_usd") or 0) for e in log)
    if cost:
        print(f"openalex cost reported: ${cost:.3f}")
    if rec:
        print(f"recall vs known (in scope {rec['in_scope']} of {rec['known_total']}): {rec['recall']} "
              f"by source {rec['recall_by_source']}; missed {len(rec['missed'])}")
    errs = [e for e in log if e["error"]]
    for e in errs:
        print(f"WARNING: {e['query']} / {e['source']}: {e['error'][:100]}")
    return exit_code(log, rec, a.min_recall)


if __name__ == "__main__":
    sys.exit(main())
