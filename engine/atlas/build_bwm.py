"""report/boat_world_model/build_bwm.py -- merge the BlueBoat world-model literature base into ONE master paper index,
ONE lexicon, ONE web of labs, ONE paper download list and ONE self-contained browsable page.

WHY. The 2026-10-12 literature program (owner: world-model navigation for the lab's BlueBoat, trained offline on the
Hollywood reef sites 3 and 4) was written by many research agents, one topic each: per-topic files under this folder.
Round 1 alone left 1,908 paper rows for about 1,650 papers -- 256 titles in more than one topic, 26 of them disagreeing
on status or year -- and 666 lexicon terms spread over 12 files. The owner asked for a browsable lexicon with direct
text citations, a web of labs, published vs pre-published marked, and a list of papers to download. This turns the
silos into those deliverables, the same way every time.

    python report/boat_world_model/build_bwm.py build    # write index/*
    python report/boat_world_model/build_bwm.py check    # exit 1 when index/* is stale
    python report/boat_world_model/build_bwm.py add DOWNLOADS_DIR [--as FILE=KEY]   # new PDFs in, scan, build
    python report/boat_world_model/build_bwm.py scan  [--pdfs DIR]    # which PDFs we hold -> papers/_held.json
    python report/boat_world_model/build_bwm.py links [--source s2|openalex|all]   # where to get the rest (network)
    python report/boat_world_model/build_bwm.py bundle [--pdfs DIR] [--out X.zip] [--no-pdfs]   # one portable .zip

INPUTS (this folder): papers/<topic>.json (one row per verified paper, written by the topic agents),
papers/_canonical.json (the adjudicator's aliases and re-derived status / year / venue, reconciled lexicon definitions;
optional), papers/_held.json + papers/_links.json (scan / links, paper_links.py), refs/<topic>/verified_refs.json
(tools/verify_refs.py output: the API record -- authors), lexicon/<topic>.json, labs/<topic>.json + labs/edges.json +
labs/trends.json, web/<topic>.json, LIT_*.md.
OUTPUTS (index/, GENERATED -- never hand-edit): papers.json, lexicon.json, labs.json, web.json, PAPERS.md, LEXICON.md,
LABS.md, PAPER_DOWNLOAD_LIST.md (every missing paper with where-to-get-it links), and browser.html (Artifact-ready: no
doctype / head, like sim/worldgen/webviewer.py; publish it with the Artifact tool; `bundle` wraps it as a standalone
atlas.html beside the PDFs, bundle_atlas.py). The Labs tab draws positions FIXED here by labs_layout.py -- 2026-10-12,
the first page's live force simulation never settled ("freaks out badly and is unusable"); the page never simulates.

RULES, enforced here rather than trusted:
  * ONE ROW PER WORK: rows are merged when they share an arXiv id, a DOI, a near-identical title, or an adjudicated alias;
    the canonical key is the adjudicator's, else the key used most, else the smallest.
  * NO UNVERIFIED REFERENCES (CLAUDE.md rule 2): a lexicon citation whose key resolves to no paper in the index is
    DROPPED and counted, and a term left with no citation is dropped too (stats.lexicon_*_dropped).
  * A status disagreement the adjudicator did not settle is KEPT VISIBLE (status_conflict) instead of silently resolved.
  * Deterministic: sorted keys and rows, no timestamps -- `check` compares bytes.
Abstracts are not copied into the index (they stay in refs/); the index carries the agents' own-words summaries.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
if str(Path(__file__).resolve().parents[2]) not in sys.path:   # run by path: the repo root, for tools/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.atomic_replace import replace_retry  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_NAMES = ("papers.json", "lexicon.json", "labs.json", "web.json", "PAPERS.md", "LEXICON.md", "LABS.md",
             "PAPER_DOWNLOAD_LIST.md", "browser.html", "atlas_citations.js")
FIGURES_JS = "atlas_figures.js"          # index/atlas_figures.js: written by `figures` (needs the figure bank), not `build`
RELEVANCE = {"core": 0, "high": 1, "medium": 2, "low": 3}
PUBLISHED = ("peer-reviewed-journal", "peer-reviewed-conference", "workshop", "book", "thesis", "standard",
             "dataset-or-report")
LAB_FILES_SKIP = {"edges.json", "trends.json"}


def _sibling(name: str):
    """This folder is not a package: load a sibling module (labs_layout, paper_links, bundle_atlas) once, privately."""
    full = f"_bwm_{name}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


# ------------------------------------------------------------------------------------------------- normalising
def norm_title(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def norm_arxiv(a: Any) -> Optional[str]:
    if not a:
        return None
    a = re.sub(r"(?i)^arxiv:", "", str(a).strip())
    a = re.sub(r"v\d+$", "", a)
    return a.lower() or None


def norm_doi(d: Any) -> Optional[str]:
    if not d:
        return None
    d = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", str(d).strip())
    return d.lower() or None


def norm_term(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def norm_name(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def is_published(status: Any) -> bool:
    return status in PUBLISHED


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class _UF:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ------------------------------------------------------------------------------------------------------ papers
def topic_rows(base: Path) -> List[Tuple[str, dict]]:
    rows = []
    for f in sorted((base / "papers").glob("*.json")):
        if f.name.startswith("_"):
            continue
        data = _load(f, [])
        if isinstance(data, list):
            rows += [(f.stem, r) for r in data if isinstance(r, dict) and r.get("key")]
    return rows


def verified_records(base: Path) -> Dict[str, dict]:
    """API records from every refs/*/verified_refs.json, reachable by arxiv:/doi:/title: handles."""
    out: Dict[str, dict] = {}
    for f in sorted((base / "refs").glob("*/verified_refs.json")):
        data = _load(f, [])
        for r in data if isinstance(data, list) else []:
            if not isinstance(r, dict) or str(r.get("status", "")).upper() != "VERIFIED":
                continue
            rec = r.get("record") or {}
            for h in _handles(rec.get("arxiv_id") or r.get("arxiv_id"), rec.get("doi"), rec.get("title")):
                out.setdefault(h, rec)
    return out


def _handles(arxiv, doi, title) -> List[str]:
    hs = []
    if norm_arxiv(arxiv):
        hs.append("arxiv:" + norm_arxiv(arxiv))
    if norm_doi(doi):
        hs.append("doi:" + norm_doi(doi))
    t = norm_title(title)
    if len(t) >= 24:                              # short titles collide ("World Models"): never merge on them alone
        hs.append("title:" + t)
    return hs


def _best(values: Iterable[Any]):
    """The most common non-empty value (ties -> the smallest by its JSON form); works for dicts and lists too."""
    seen: Dict[str, Any] = {}
    c: collections.Counter = collections.Counter()
    for v in values:
        if v in (None, "", [], {}):
            continue
        k = json.dumps(v, sort_keys=True)
        seen.setdefault(k, v)
        c[k] += 1
    if not c:
        return None
    top = max(c.values())
    return seen[sorted(k for k, n in c.items() if n == top)[0]]


def rejected_ids(base: Path, alias_map: Optional[Dict[str, str]] = None
                 ) -> Tuple[Dict[str, set], Dict[str, set], List[str]]:
    """papers/_rejected.json -> ({sha256: keys that FILE is not}, {key: arXiv ids that are ANOTHER paper}, problems).
    A wrong PDF found by `confirm` (and read) is recorded here, so no later scan / add / links / build brings it back
    under that key (review 2026-10-14: its old file names and Semantic Scholar's merged record both did). Keys go
    through `alias_map` -- both the alias and the paper it was folded into are refused (fix check 2026-10-14: a fold
    undid the rejection). A row that cannot work is a PROBLEM (`check` fails): a sha that is not 64 hex characters
    (upper case is folded), an arXiv id that does not parse, a key the index does not know."""
    alias_map = alias_map or {}
    files: Dict[str, set] = {}
    ax: Dict[str, set] = {}
    problems: List[str] = []
    path = base / "papers" / "_rejected.json"
    doc: dict = {}
    if path.is_file():                                 # a file that cannot be read rejects NOTHING: say so (fix check
        try:                                           # 2026-10-14: a BOM or a trailing comma disabled every row)
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError as e:
            problems.append(f"{UNREADABLE_REJECTED}: {e}")
            doc = {}
        if not isinstance(doc, dict):
            problems.append(f"{UNREADABLE_REJECTED}: the top level is not an object")
            doc = {}
        extra = sorted(set(doc) - {"schema", "about", "files", "arxiv"})
        if extra:
            problems.append(f"_rejected.json has unknown sections {extra} (a typo of 'files' / 'arxiv'?)")
        for sec in ("files", "arxiv"):
            if sec in doc and not isinstance(doc[sec], list):
                problems.append(f"_rejected.json '{sec}' must be a list")
                doc[sec] = []
    for sec, shape in (("files", '{"sha256": ..., "not": <key>}'), ("arxiv", '{"key": ..., "arxiv_id": ...}')):
        bad = [r for r in doc.get(sec) or [] if not isinstance(r, dict)]   # '["<sha>"]' rejects nothing (third fix
        if bad:                                                            # check 2026-10-15: it was a traceback)
            problems.append(f"_rejected.json '{sec}' rows must be objects {shape}, not {bad[0]!r}")
            doc[sec] = [r for r in doc.get(sec) or [] if isinstance(r, dict)]
    for r in doc.get("files") or []:
        sha, key = str(r.get("sha256") or "").strip().lower(), str(r.get("not") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or not key:
            problems.append(f"_rejected.json files: {r!r} needs a 64-hex sha256 and the key it is 'not'")
            continue
        files.setdefault(sha, set()).update({key, alias_map.get(key, key)})
    for r in doc.get("arxiv") or []:
        key = str(r.get("key") or "").strip()
        m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?|([a-z\-]+(?:\.[A-Z]{2})?/\d{7})", str(r.get("arxiv_id") or ""))
        aid = norm_arxiv((m.group(1) or m.group(2)) if m else "")
        if not key or not aid:
            problems.append(f"_rejected.json arxiv: {r!r} needs a key and an arXiv id (2208.11229, or its URL)")
            continue
        for k in {key, alias_map.get(key, key)}:
            ax.setdefault(k, set()).add(aid)
    return files, ax, problems


UNREADABLE_REJECTED = "_rejected.json cannot be read -- NO rejection applies"


def rejected_problems(base: Path, keys: Iterable[str], alias_map: Dict[str, str]) -> List[str]:
    """Every _rejected.json row that would silently do nothing, and every _held.json row a rejection says is the
    wrong file (held until the next `scan`), for `check` / `build` to report."""
    files, ax, problems = rejected_ids(base, alias_map)
    keys = set(keys)
    named = {k for ks in files.values() for k in ks} | set(ax)
    problems += [f"_rejected.json names {k!r}, which is no key of the index (a typo?)" for k in sorted(named)
                 if alias_map.get(k, k) not in keys]
    for r in (_load(base / "papers" / "_held.json", {}).get("files") or []):
        k = alias_map.get(r.get("key"), r.get("key"))
        if k in files.get(str(r.get("sha256") or "").strip().lower(), ()):
            problems.append(f"_held.json still holds {r.get('file')} as {k}, which _rejected.json rejects -- run scan")
    for r in (_load(base / "papers" / "_facts.json", {}).get("papers") or []):
        k = alias_map.get(r.get("key"), r.get("key")) if isinstance(r, dict) else None
        if k and k in files.get(str(r.get("sha256") or "").strip().lower(), ()):
            problems.append(f"_facts.json holds facts mined from a file _rejected.json says is not {k} -- run scan, "
                            f"then tools/paper_facts.py --bwm")
    return problems


def unreadable_rejections(base: Path) -> List[str]:
    """The problems that switch papers/_rejected.json (or part of it) OFF: an unreadable file, an unknown section, a
    section or row of the wrong shape. Every tool that holds, ships or files a PDF refuses on them."""
    return [p for p in rejected_ids(base)[2] if p.startswith(UNREADABLE_REJECTED) or "unknown sections" in p
            or "must be a list" in p or "must be objects" in p]


def _refuse_unreadable_rejections(base: Path) -> bool:
    """True (and says why) when papers/_rejected.json cannot be read: scan / add / audit / links / bundle must not run
    blind."""
    bad = unreadable_rejections(base)
    for p in bad:
        print("REFUSED: " + p + " -- fix papers/_rejected.json first")
    return bool(bad)


def _cites_arxiv(url: Any, ids: Iterable[str]) -> bool:
    """Does this string name one of these arXiv ids -- arxiv.org/abs|pdf/<id> or its DOI 10.48550/arxiv.<id>?"""
    return any(re.search(r"(?:arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv\.)" + re.escape(a) + r"(?:v\d+)?(?!\d)",
                         str(url or "").lower()) for a in ids)


def build_papers(base: Path) -> Tuple[List[dict], Dict[str, str], dict]:
    rows = topic_rows(base)
    _, rej_ax, _ = rejected_ids(base, (_load(base / "papers" / "_canonical.json", {}).get("aliases") or {}))
    canon = _load(base / "papers" / "_canonical.json", {})
    aliases = {k: v for k, v in (canon.get("aliases") or {}).items() if k and v}
    overrides = canon.get("papers") or {}
    recs = verified_records(base)
    uf = _UF()
    is_ax = lambda h: h.startswith("arxiv:") or (h.startswith("doi:") and "10.48550/arxiv." in h)
    by_handle: Dict[str, str] = {}
    for _, r in rows:                                  # (1) DOI / title handles and the aliases: one work's keys
        k = r["key"]
        uf.find(k)
        for h in _handles(r.get("arxiv_id"), r.get("doi"), r.get("title")):
            if is_ax(h):
                continue
            if h in by_handle:
                uf.union(k, by_handle[h])
            else:
                by_handle[h] = k
    for a, c in aliases.items():
        uf.union(a, c)
    # (2) arXiv handles join rows -- never one a key of the row's FINAL group rejects (third fix check 2026-10-15: a
    # stale topic row folded the 2013 paper into the 2022 one that owns the id; fix checks of that: under a second key
    # of the same paper too, whatever the keys' spelling or the rows' order). A union cannot be undone, so the joins are
    # REDONE from the DOI / title groups, each time without every (row, id) a final group was found to reject; the
    # excluded set only grows, so this ends (once, with no rejections: the same groups as joining on every handle).
    base_p, excl = dict(uf.p), set()
    while True:
        uf.p = dict(base_p)
        by_ax: Dict[str, str] = {}
        for i, (_, r) in enumerate(rows):
            for h in _handles(r.get("arxiv_id"), r.get("doi"), r.get("title")):
                if not is_ax(h) or (i, h) in excl:
                    continue
                if h in by_ax:
                    uf.union(r["key"], by_ax[h])
                else:
                    by_ax[h] = r["key"]
        bad: Dict[str, set] = collections.defaultdict(set)
        for _, r in rows:
            k = r["key"]
            bad[uf.find(k)] |= rej_ax.get(k, set()) | rej_ax.get(aliases.get(k, k), set())
        new = {(i, h) for i, (_, r) in enumerate(rows) for no in [bad[uf.find(r["key"])]] if no
               for h in _handles(r.get("arxiv_id"), r.get("doi"), r.get("title"))
               if is_ax(h) and (i, h) not in excl and ((h.startswith("arxiv:") and h[6:] in no) or _cites_arxiv(h, no))}
        if not new:
            break
        excl |= new
    groups: Dict[str, List[Tuple[str, dict]]] = collections.defaultdict(list)
    for t, r in rows:
        groups[uf.find(r["key"])].append((t, r))
    targets = set(aliases.values()) | set(overrides)
    papers, alias_map = [], {}
    conflicts = 0
    for members in groups.values():
        keys = [r["key"] for _, r in members]
        named = sorted({k for k in keys if k in targets})
        if named:
            ck = named[0]
        else:
            cnt = collections.Counter(keys)
            ck = sorted(cnt, key=lambda k: (-cnt[k], k))[0]
        for k in set(keys) | {a for a, c in aliases.items() if c == ck}:
            alias_map[k] = ck
        ov = overrides.get(ck) or {}
        rs = [r for _, r in members]
        bad_ax = set().union(*[rej_ax.get(k, set()) for k in set(keys) | {ck}])   # papers/_rejected.json
        arx = (_best(a for a in (norm_arxiv(r.get("arxiv_id")) for r in rs) if a not in bad_ax)
               or next((a for a in [norm_arxiv(ov.get("arxiv_id"))] if a and a not in bad_ax), None))
        doi = (_best(d for d in (norm_doi(r.get("doi")) for r in rs) if not _cites_arxiv(d, bad_ax))   # never
               or next((d for d in [norm_doi(ov.get("doi"))] if d and not _cites_arxiv(d, bad_ax)), None))  # 10.48550/arxiv.<rejected>
        rec = next((recs[h] for h in _handles(arx, doi, ov.get("title") or rs[0].get("title")) if h in recs), {})
        statuses = sorted({r.get("status") for r in rs if r.get("status")})
        if ov.get("status"):
            status, years, venue, evidence = ov["status"], [ov.get("year")], ov.get("venue"), ov.get("status_evidence")
            conflict = False
        else:
            status = _best(r.get("status") for r in rs)
            pick = [r for r in rs if r.get("status") == status]
            years, venue = [r.get("year") for r in pick], _best(r.get("venue") for r in pick)
            evidence = _best(r.get("status_evidence") for r in pick)
            conflict = len(statuses) > 1
        conflicts += conflict
        year = _best(years) if years else None
        try:
            year = int(year) if year not in (None, "") else None
        except (TypeError, ValueError):
            year = None
        rel = min((RELEVANCE.get(r.get("relevance"), 4) for r in rs), default=4)
        topics = sorted({t for t, _ in members})
        authors = (rec.get("authors") or _best(r.get("authors") for r in rs if r.get("intake"))  # tools/atlas_intake.py
                   or [])                                     # rows only: the agents' rows never fill authors here
        papers.append({
            "key": ck,
            "title": ov.get("title") or rec.get("title") or _best(r.get("title") for r in rs),
            "authors": authors[:12] + (["et al."] if len(authors) > 12 else []),
            "year": year,
            "venue": venue or "",
            "status": status or "",
            "published": is_published(status),
            "status_evidence": evidence or "",
            "status_conflict": statuses if conflict else [],
            "published_version": ov.get("published_version") or _best(r.get("published_version") for r in rs),
            "arxiv_id": arx,
            "doi": doi,
            "url": (f"https://arxiv.org/abs/{arx}" if arx else (f"https://doi.org/{doi}" if doi else _best(
                u for u in (r.get("oa_pdf_url") for r in rs) if not _cites_arxiv(u, bad_ax)))),
            "oa_pdf_url": _best(u for u in (r.get("oa_pdf_url") for r in rs) if not _cites_arxiv(u, bad_ax)),
            "code_url": _best(r.get("code_url") for r in rs),
            "relevance": next((k for k, v in RELEVANCE.items() if v == rel), "unrated"),
            "topics": topics,
            "tags": sorted({str(x) for r in rs for x in (r.get("tags") or [])}),
            "labs": sorted({str(r.get("lab")) for r in rs if r.get("lab")}),
            "summary": next((r.get("summary") for t, r in sorted(members, key=lambda m: m[0]) if r.get("summary")), ""),
            "why_for_us": {t: r.get("why_for_us") for t, r in sorted(members, key=lambda m: m[0]) if r.get("why_for_us")},
            "pdf_held": any(bool(r.get("pdf_held")) for r in rs),
            "aliases": sorted(set(keys) - {ck}),
        })
    papers.sort(key=lambda p: p["key"])
    return papers, alias_map, {"rows": len(rows), "papers": len(papers), "status_conflicts_unresolved": conflicts,
                               "aliases": sum(1 for k, v in alias_map.items() if k != v)}


# ----------------------------------------------------------------------------------------------------- lexicon
def build_lexicon(base: Path, alias_map: Dict[str, str], index: Dict[str, dict]) -> Tuple[List[dict], dict]:
    canon_lex = (_load(base / "papers" / "_canonical.json", {}).get("lexicon") or {})
    entries = []
    for f in sorted((base / "lexicon").glob("*.json")):
        if f.name.startswith("_"):                       # _math.json: equations keyed by term id, merged below
            continue
        data = _load(f, [])
        for e in data if isinstance(data, list) else []:
            if isinstance(e, dict) and str(e.get("term", "")).strip():
                entries.append((e.get("topic") or f.stem, e))
    uf = _UF()
    name_of: Dict[str, str] = {}
    for _, e in entries:
        uf.find(norm_term(e["term"]))
        name_of.setdefault(norm_term(e["term"]), norm_term(e["term"]))
    for _, e in entries:
        for a in e.get("aliases") or []:
            if norm_term(a) in name_of:
                uf.union(norm_term(e["term"]), norm_term(a))
    groups: Dict[str, List[Tuple[str, dict]]] = collections.defaultdict(list)
    for t, e in entries:
        groups[uf.find(norm_term(e["term"]))].append((t, e))
    out, dropped_cites, dropped_terms = [], 0, 0
    for members in groups.values():
        names = collections.Counter(e["term"].strip() for _, e in members)
        term = sorted(names, key=lambda n: (-names[n], n.lower(), n))[0]
        cites, seen = [], set()
        for t, e in sorted(members, key=lambda m: m[0]):
            for c in e.get("citations") or []:
                k = alias_map.get(c.get("key"), c.get("key"))
                q = str(c.get("quote") or "").strip()
                if not q or k not in index:
                    dropped_cites += 1
                    continue
                if (k, q) in seen:
                    continue
                seen.add((k, q))
                p = index[k]
                cites.append({"key": k, "quote": q, "where": c.get("where") or "", "source": c.get("source") or "",
                              "title": p["title"], "year": p["year"], "status": p["status"], "url": p["url"]})
        if not cites:
            dropped_terms += 1
            continue
        defs = [{"topic": t, "text": str(e.get("definition")).strip()}
                for t, e in sorted(members, key=lambda m: m[0]) if str(e.get("definition") or "").strip()]
        rec = next((canon_lex[n] for n in sorted({norm_term(e["term"]) for _, e in members}) if n in canon_lex), None)
        aliases = sorted({str(a).strip() for _, e in members for a in (e.get("aliases") or [])} |
                         set(names) - {term}, key=lambda a: (a.lower(), a))
        out.append({
            "id": re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-") or "term",
            "term": term,
            "aliases": [a for a in aliases if a and norm_term(a) != norm_term(term)],
            "categories": sorted({str(e.get("category")) for _, e in members if e.get("category")}),
            "topics": sorted({t for t, _ in members}),
            "definition": (rec or {}).get("definition") or (defs[0]["text"] if defs else ""),
            "senses": (rec or {}).get("senses") or [],
            "reconciled": rec is not None,
            "definitions": defs,
            "citations": sorted(cites, key=lambda c: (c["year"] or 0, c["key"], c["quote"])),
            "related": sorted({str(r).strip() for _, e in members for r in (e.get("related") or []) if str(r).strip()},
                              key=lambda r: (r.lower(), r)),
        })
    ids = collections.Counter(x["id"] for x in out)
    for x in out:
        if ids[x["id"]] > 1:
            x["id"] = f'{x["id"]}-{norm_name(x["topics"][0])}'
    out.sort(key=lambda x: (x["term"].lower(), x["id"]))
    known = {norm_term(x["term"]): x["id"] for x in out}
    for x in out:
        for a in x["aliases"]:
            known.setdefault(norm_term(a), x["id"])
    for x in out:
        x["related_ids"] = sorted({known[norm_term(r)] for r in x["related"] if norm_term(r) in known} - {x["id"]})
    mrows = {r["id"]: r for r in (_load(base / "lexicon" / "_math.json", {}).get("terms") or [])
             if isinstance(r, dict) and r.get("id")}
    math = {k: r.get("math") or [] for k, r in mrows.items()}
    IM = _sibling("inline_math")                    # plain-text symbols -> inline LaTeX, deterministically
    for x in out:
        x["math"] = [{k: m.get(k, "") for k in ("latex", "symbols", "kind", "source_key")} for m in math.get(x["id"], [])]
        for m in x["math"]:
            m["symbols_tex"] = IM.convert(m["symbols"], m["latex"])
        tex = (mrows.get(x["id"]) or {}).get("definition_tex") or IM.convert(x["definition"], "", definitions=True)
        if tex != x["definition"]:                   # the definition with its symbols as inline LaTeX
            x["definition_tex"] = tex
    ids_now = {x["id"] for x in out}
    return out, {"terms": len(out), "citations": sum(len(x["citations"]) for x in out),
                 "math_terms": sum(1 for x in out if x["math"]), "math_equations": sum(len(x["math"]) for x in out),
                 "math_orphans": sorted(k for k in math if k not in ids_now),
                 "lexicon_citations_dropped": dropped_cites, "lexicon_terms_dropped": dropped_terms,
                 "fulltext_citations": sum(1 for x in out for c in x["citations"] if c["source"] == "fulltext")}


# -------------------------------------------------------------------------------------------------------- labs
def build_labs(base: Path, alias_map: Dict[str, str], index: Dict[str, dict]) -> Tuple[dict, dict]:
    entries = []
    for f in sorted((base / "labs").glob("*.json")):
        if f.name in LAB_FILES_SKIP:
            continue
        data = _load(f, [])
        entries += [(f.stem, e) for e in (data if isinstance(data, list) else []) if isinstance(e, dict) and e.get("lab")]
    uf = _UF()
    for i, (_, e) in enumerate(entries):
        uf.find(f"n:{norm_name(e['lab'])}")
        pi, inst = norm_name(e.get("pi")), norm_name(e.get("institution"))
        if pi and inst:
            uf.union(f"n:{norm_name(e['lab'])}", f"p:{pi}|{inst}")
    groups: Dict[str, List[Tuple[str, dict]]] = collections.defaultdict(list)
    for t, e in entries:
        groups[uf.find(f"n:{norm_name(e['lab'])}")].append((t, e))
    nodes, lookup = [], {}
    for members in groups.values():
        es = [e for _, e in members]
        name = _best(e["lab"] for e in es)
        nid = norm_name(name) or "lab"
        papers = sorted({alias_map.get(k, k) for e in es for k in (e.get("paper_keys") or [])} & set(index))
        nodes.append({"id": nid, "lab": name, "pi": _best(e.get("pi") for e in es) or "",
                      "institution": _best(e.get("institution") for e in es) or "",
                      "country": _best(e.get("country") for e in es) or "",
                      "url": _best(e.get("url") for e in es) or "",
                      "focus": sorted({str(e.get("focus")).strip() for e in es if e.get("focus")})[:4],
                      "topics": sorted({t for t, _ in members}), "papers": papers,
                      "code": sorted({str(x) for e in es for x in (e.get("sims_datasets_code") or [])})[:12]})
        for e in es:
            lookup.setdefault(norm_name(e["lab"]), nid)
            if e.get("institution"):
                lookup.setdefault(norm_name(e["institution"]), nid)
    nodes.sort(key=lambda n: (-len(n["papers"]), n["id"]))
    edges: Dict[Tuple[str, str], set] = {}

    def _add(a, b, ps):
        if a and b and a != b:
            e = tuple(sorted((a, b)))
            edges.setdefault(e, set()).update(ps)

    hay = [(n["id"], norm_name(n["lab"]) + "|" + norm_name(n["institution"]) + "|" + norm_name(n["pi"]),
            len(n["papers"])) for n in nodes]

    def resolve(name) -> Optional[str]:
        """A link names a group loosely ("NTNU", "Oklahoma State (San)"): exact lab / institution first, else the
        group whose name, institution or PI contains it (>= 4 characters; the one with most papers on a tie)."""
        k = norm_name(re.sub(r"\(.*?\)", "", str(name or ""))) or norm_name(name)
        if not k:
            return None
        if k in lookup:
            return lookup[k]
        if len(k) < 4:
            return None
        hits = sorted((-n, i) for i, h, n in hay if k in h)
        return hits[0][1] if hits else None

    added = 0

    def resolve_or_add(name, papers) -> Optional[str]:
        """An edge endpoint no topic listed as a group (often an institution OpenAlex named) becomes a light
        `institution` node, so the co-authorship web keeps every link instead of silently losing it."""
        nonlocal added
        nid = resolve(name)
        if nid or not str(name or "").strip():
            return nid
        nid = "inst-" + norm_name(name)
        nodes.append({"id": nid, "lab": str(name).strip(), "pi": "", "institution": str(name).strip(),
                      "country": "", "url": "", "focus": [], "topics": [], "papers": sorted(papers),
                      "code": [], "kind": "institution"})
        hay.append((nid, norm_name(name), len(papers)))
        lookup[norm_name(name)] = nid
        added += 1
        return nid

    raw_edges = _load(base / "labs" / "edges.json", [])
    for e in raw_edges if isinstance(raw_edges, list) else []:
        ps = {alias_map.get(k, k) for k in (e.get("papers") or [])} & set(index)
        _add(resolve_or_add(e.get("a"), ps), resolve_or_add(e.get("b"), ps), ps)
    for _, e in entries:
        for c in e.get("collaborates_with") or []:
            _add(lookup.get(norm_name(e["lab"])), resolve(c), set())
    for n in nodes:
        n.setdefault("kind", "group")
    nodes.sort(key=lambda n: (n["kind"] != "group", -len(n["papers"]), n["id"]))
    elist = [{"a": a, "b": b, "papers": sorted(ps)} for (a, b), ps in sorted(edges.items())]
    lay = _sibling("labs_layout").layout(nodes, elist)            # FIXED positions: the page draws, never simulates
    for n in nodes:
        n["xy"], n["r"] = lay["xy"][n["id"]], lay["r"][n["id"]]
    box = {k: lay[k] for k in ("w", "h", "linked_h", "sections")}
    return {"layout": box, "nodes": nodes, "edges": elist}, {"labs": sum(n["kind"] == "group" for n in nodes),
                                                             "lab_edges": len(elist), "institution_nodes": added}


def build_web(base: Path) -> List[dict]:
    out, seen = [], set()
    for f in sorted((base / "web").glob("*.json")):
        data = _load(f, [])
        for w in data if isinstance(data, list) else []:
            if isinstance(w, dict) and w.get("url") and w["url"] not in seen:
                seen.add(w["url"])
                out.append({"title": w.get("title") or w["url"], "publisher": w.get("publisher") or "",
                            "url": w["url"], "accessed": w.get("accessed") or "",
                            "supports": w.get("what_it_supports") or "", "topic": f.stem})
    return sorted(out, key=lambda w: (w["topic"], w["title"].lower()))


def topic_docs(base: Path) -> List[dict]:
    docs = []
    for f in sorted(base.glob("LIT_*.md")):
        h1 = next((ln[2:].strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.startswith("# ")), f.stem)
        docs.append({"file": f.name, "title": h1})
    return docs


def meta_rows(base: Path, alias_map: Dict[str, str]) -> Dict[str, dict]:
    """papers/_meta.json by index key, aliases folded (the in-corpus key lists too; a canonical row wins over an alias
    row; review 2026-10-14)."""
    out: Dict[str, dict] = {}
    doc = _load(base / "papers" / "_meta.json", {})
    if doc.get("schema") != "svfr.bwm.meta/2":         # another layout (the first, flawed run's) is not read
        return out
    for r in sorted((r for r in (doc.get("papers") or [])
                     if isinstance(r, dict) and r.get("key")), key=lambda r: (r["key"] in alias_map, r["key"])):
        k = alias_map.get(r["key"], r["key"])
        if k in out:
            continue
        out[k] = {**r, "key": k, **{f: sorted({alias_map.get(x, x) for x in r.get(f) or []} - {k})
                                    for f in ("refs_in_corpus", "cited_by_in_corpus")}}
    return out


def links_rows(base: Path, alias_map: Dict[str, str]) -> Dict[str, dict]:
    """papers/_links.json by index key, each row SCRUBBED of arXiv ids _rejected.json says are another paper -- the one
    reader for the build AND the Office files (fix check 2026-10-14: ATLAS.xlsx / PAPERS.docx read the raw file and
    published the rejected id as an 'open PDF')."""
    PL = _sibling("paper_links")
    _, rej_ax, _ = rejected_ids(base, alias_map)
    links: Dict[str, dict] = {}
    for r in (_load(base / "papers" / "_links.json", {}).get("papers") or []):
        if isinstance(r, dict) and r.get("key"):
            k = alias_map.get(r["key"], r["key"])
            links[k] = PL.scrub_link(r, rej_ax.get(k, set()) | rej_ax.get(r["key"], set()))
    return links


def facts_rows(base: Path, alias_map: Dict[str, str]) -> Dict[str, dict]:
    """papers/_facts.json by index key -- only the rows a HELD file that _rejected.json does not reject backs: the one
    reader for the atlas, the Office files and `code` (third fix check 2026-10-15: a rejected PDF's facts stayed
    published after `scan` dropped it). A row records the sha256 it was mined from (`paper_facts --bwm`); one mined
    from another file than the key holds now is dropped too."""
    rej_files = rejected_ids(base, alias_map)[0]
    held: Dict[str, set] = {}
    for r in (_load(base / "papers" / "_held.json", {}).get("files") or []):
        if isinstance(r, dict) and r.get("key"):
            k, sha = alias_map.get(r["key"], r["key"]), str(r.get("sha256") or "").strip().lower()
            if k not in rej_files.get(sha, ()):
                held.setdefault(k, set()).add(sha)
    out: Dict[str, dict] = {}
    for r in (_load(base / "papers" / "_facts.json", {}).get("papers") or []):
        if not (isinstance(r, dict) and r.get("key")):
            continue
        k, sha = alias_map.get(r["key"], r["key"]), str(r.get("sha256") or "").strip().lower()
        if held and (k not in held or (sha and sha not in held[k])):
            continue
        out[k] = r.get("facts") or {}
    return out


def apply_holdings(base: Path, papers: List[dict], alias_map: Dict[str, str]) -> Tuple[Dict[str, dict], dict]:
    """papers/_held.json (`scan` of the PDF folder) decides "held" when it lists any file -- else the topic agents'
    pdf_held flags; papers/_links.json (`links`) says where the rest can be had. Keys merged since resolve through the
    aliases. Sets availability (held / open / library), open_url, arxiv_version, s2_url on each paper."""
    PL = _sibling("paper_links")
    held_doc = _load(base / "papers" / "_held.json", {})
    rej_files = rejected_ids(base, alias_map)[0]
    files = [r for r in (held_doc.get("files") or []) if isinstance(r, dict) and r.get("key")
             and alias_map.get(r["key"], r["key"]) not in rej_files.get(str(r.get("sha256") or "").lower(), ())]
    held = {alias_map.get(r["key"], r["key"]) for r in files}
    links = links_rows(base, alias_map)
    for r in (_load(base / "papers" / "_status.json", {}).get("papers") or []):   # `status`: found by a script
        p = next((x for x in papers if x["key"] == alias_map.get(r.get("key"), r.get("key"))), None)
        m = re.match(r"arXiv (?:abstract page|comment) (\S+)", str(r.get("source", "")))
        if p and m and norm_arxiv(m.group(1)) != norm_arxiv(p.get("arxiv_id")):
            continue                                   # found through an id that is not (any more) this paper's
        if p and str(r.get("source", "")) == "arXiv comment" and not p.get("arxiv_id"):
            continue                                   # (an older comment row names no id: no arXiv id, no comment)
        if p and r.get("status") in ("withdrawn", "retracted") and p["status"] != r["status"]:
            p["status"], p["published"] = r["status"], False
            p["status_evidence"] = f"{r.get('source', '')}: {r.get('evidence', '')}. Before: {p.get('status_evidence', '')}"
    obtained = {alias_map.get(r["key"], r["key"]): r for r in (_load(base / "papers" / "_obtained.json", {}).get("items")
                                                               or []) if isinstance(r, dict) and r.get("key")}
    for p in papers:
        if files:
            p["pdf_held"] = p["key"] in held
        ln = links.get(p["key"])
        ob = obtained.get(p["key"])
        p["obtained"] = (ob.get("kind") or "data") if ob and not p["pdf_held"] else ""
        p["availability"] = PL.availability(p, p["pdf_held"], ln, obtained=bool(p["obtained"]))
        p["open_url"], p["open_label"] = PL.open_copy(p, ln)
        p["arxiv_version"] = PL.arxiv_version(p, ln)
        p["s2_url"] = ((ln or {}).get("s2") or {}).get("url") or ""
    CL = _sibling("code_links")
    rows = [r for r in (_load(base / "papers" / "_code.json", {}).get("papers") or []) if isinstance(r, dict)]
    code = {alias_map.get(r["key"], r["key"]): r for r in rows if r.get("status") in CL.COUNTED}
    dead = {alias_map.get(r["key"], r["key"]): {t["repo"] for t in r.get("tried") or [] if t.get("status") == "REPO_NOT_FOUND"}
            for r in rows}
    for p in papers:                                   # a CHECKED repo replaces the agents' guess (code_links.py)
        c = code.get(p["key"])
        p["code_status"] = c["status"] if c else ""
        p["code_stars"] = c.get("stars") if c else None
        if c:
            p["code_url"] = c["url"]
        elif p.get("code_url") and CL.repo_of(p["code_url"]) in dead.get(p["key"], ()):
            p["code_url"] = ""                         # GitHub says that repo does not exist: no link
    meta, venue_types = meta_rows(base, alias_map), _sibling("meta_links").VENUE_TYPES
    for p in papers:                                   # `meta`: the facts ABOUT the paper, dated (meta_links.py);
        m = meta.get(p["key"]) or {}                   # COUNTS only here (papers.json stays under 5 MB), the lists
        oa, s2 = m.get("openalex") or {}, m.get("s2") or {}            # live in papers/_meta.json + the workbook
        v = oa.get("venue") or {}
        got = {"published_date": oa.get("publication_date") or s2.get("publication_date") or "",
               "cited_by": oa.get("cited_by"),                     # a key that is ABSENT was not measured: never
               "cited_by_s2": s2.get("citations"), "influential_citations": s2.get("influential"),   # shown as 0
               "cited_here": len(m.get("cited_by_in_corpus") or []) if oa else None,
               "cites_here": len(m.get("refs_in_corpus") or []) if oa else None,
               "venue_citedness_2yr": v.get("mean_citedness_2yr"), "venue_h_index": v.get("h_index"),
               "meta_measured_on": (m.get("measured") or {}).get("openalex", "") if oa else ""}
        p.update({f: x for f, x in got.items() if x not in (None, "")})    # only what was measured (5 MB rule)
        fill = (v.get("name") if v.get("type") in venue_types else "") or s2.get("venue")
        if fill and not p.get("venue"):                # the index names none: OpenAlex's venue (not a repository),
            p["venue"] = fill                          # else Semantic Scholar's
        gh = m.get("github") or {}
        if gh.get("stars") is not None and \
                (CL.repo_of(p.get("code_url") or "") or "").lower() == "github.com/" + gh["repo"].lower():
            p["code_stars"] = gh["stars"]               # the dated count replaces the verify run's
    av = collections.Counter(p["availability"] for p in papers)
    return links, {"pdfs_held": av["held"], "held_from": "scan" if files else "topic flags", "obtained": av["obtained"],
                   "papers_with_checked_code": len(code),
                   "to_download_open": av["open"], "to_download_library": av["library"]}


# ------------------------------------------------------------------------------------------------------ render
def _md_escape(s: Any) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ")


def render_papers_md(papers, stats, web, doc: Optional[dict] = None) -> str:
    doc = doc or doc_text(HERE)                        # what the project calls itself, its own commands (doc_text)
    L = [f"# {doc['papers']}", "",
         f"*GENERATED by `{doc['build']}` -- do not hand-edit. One row per work, merged "
         "across the topic reviews; every row was verified against an API record (tools/verify_refs.py).*", "",
         f"{stats['papers']} papers ({sum(p['published'] for p in papers)} published, "
         f"{sum(p['status'] == 'preprint' for p in papers)} preprints) from {stats['rows']} topic rows; "
         f"{stats['aliases']} duplicate keys folded; {stats['status_conflicts_unresolved']} status disagreements left "
         "visible (status column shows the majority, the conflict in brackets).", "",
         "| key | year | status | title | topics | relevance | code |", "|---|---|---|---|---|---|---|"]
    for p in sorted(papers, key=lambda p: (-(p["year"] or 0), p["key"])):
        st = p["status"] + (f" [{'/'.join(p['status_conflict'])}]" if p["status_conflict"] else "")
        title = f"[{_md_escape(p['title'])}]({p['url']})" if p["url"] else _md_escape(p["title"])
        code = f"[code]({p['code_url']})" if p["code_url"] else ""
        L.append(f"| `{p['key']}` | {p['year'] or ''} | {st} | {title} | {', '.join(p['topics'])} | {p['relevance']} | {code} |")
    if web:
        L += ["", "## Web sources (not papers)", "", "| topic | source | publisher | accessed | supports |", "|---|---|---|---|---|"]
        for w in web:
            L.append(f"| {w['topic']} | [{_md_escape(w['title'])}]({w['url']}) | {_md_escape(w['publisher'])} | "
                     f"{w['accessed']} | {_md_escape(w['supports'])} |")
    return "\n".join(L) + "\n"


def render_lexicon_md(lex, stats, doc: Optional[dict] = None) -> str:
    doc = doc or doc_text(HERE)
    L = [f"# {doc['lexicon']}", "",
         f"*GENERATED by `{doc['build']}` -- do not hand-edit; add terms in "
         "`lexicon/<topic>.json`. Each quote is verbatim (<= 15 words) from the cited, API-verified paper.*", "",
         f"{stats['terms']} terms ({stats['math_equations']} equations on {stats['math_terms']} of them, LaTeX); "
         f"{stats['citations']} citations ({stats['fulltext_citations']} from full texts); "
         f"{stats['lexicon_citations_dropped']} citations and {stats['lexicon_terms_dropped']} terms dropped for "
         "pointing at no verified paper.", ""]
    letter = None
    for x in lex:
        first = x["term"][:1].upper()
        if first != letter:
            letter = first
            L += [f"## {letter if letter.isalpha() else '#'}", ""]
        L.append(f"### {x['term']}")
        meta = [", ".join(x["categories"])] if x["categories"] else []
        if x["aliases"]:
            meta.append("also: " + ", ".join(x["aliases"]))
        if meta:
            L.append("*" + " -- ".join(meta) + "*")
        L += ["", x.get("definition_tex") or x["definition"], ""]
        for m in x.get("math") or []:                   # GitHub renders $$ ... $$ blocks as display math
            L += ["$$", m["latex"], "$$", ""]
            if m["symbols"]:
                L += [f"*where* {m.get('symbols_tex') or m['symbols']}", ""]
        for s in x["senses"]:
            L.append(f"- sense: {s}")
        for d in x["definitions"][1:] if not x["reconciled"] else x["definitions"]:
            if d["text"] != x["definition"]:
                L.append(f"- ({d['topic']}) {d['text']}")
        for c in x["citations"]:
            L.append(f"- \"{c['quote']}\" -- [{c['key']}]({c['url']}) ({c['year'] or 'n.d.'}, {c['status']}; {c['where']})")
        if x["related"]:
            L.append("- related: " + ", ".join(x["related"]))
        L.append("")
    return "\n".join(L) + "\n"


def render_labs_md(labs, stats, doc: Optional[dict] = None) -> str:
    doc = doc or doc_text(HERE)
    L = [f"# {doc['labs']}", "",
         f"*GENERATED by `{doc['build']}` -- do not hand-edit; groups come from "
         "`labs/<topic>.json`, links from `labs/edges.json` and each group's collaborators.*", "",
         f"{stats['labs']} groups, {stats['lab_edges']} links.", "",
         "| group | PI | institution | country | papers here | topics |", "|---|---|---|---|---|---|"]
    for n in labs["nodes"]:
        g = f"[{_md_escape(n['lab'])}]({n['url']})" if n["url"] else _md_escape(n["lab"])
        L.append(f"| {g} | {_md_escape(n['pi'])} | {_md_escape(n['institution'])} | {_md_escape(n['country'])} | "
                 f"{len(n['papers'])} | {', '.join(n['topics'])} |")
    return "\n".join(L) + "\n"


def _md_url(u: str) -> str:
    """A URL safe inside a markdown table link: angle-bracketed (parentheses in DOIs); no raw space, pipe, angle
    bracket, backslash (it would escape the closing >) or line break -- each percent-encoded (an existing %xx is kept)."""
    return "<" + (str(u).replace(" ", "%20").replace("|", "%7C").replace("<", "%3C")
                  .replace(">", "%3E").replace("\\", "%5C").replace("\n", "%0A").replace("\r", "")) + ">"


def _get_md(p: dict, link: Optional[dict]) -> str:
    return " \u00b7 ".join(f"[{lab}]({_md_url(u)})" for lab, u in _sibling("paper_links").hunt_links(p, link))


def render_download_md(papers, links=None, doc: Optional[dict] = None) -> str:
    links = links or {}
    doc = doc or doc_text(HERE)
    rank = lambda p: (RELEVANCE.get(p["relevance"], 4), -len(p["topics"]), -(p["year"] or 0), p["key"])
    got = lambda p: p["pdf_held"] or p.get("obtained")
    todo = [p for p in papers if not got(p) and not p["arxiv_id"]]
    pri = sorted([p for p in todo if p["relevance"] in ("core", "high")], key=rank)
    later = sorted([p for p in todo if p["relevance"] not in ("core", "high")], key=rank)
    auto = sorted([p for p in papers if not got(p) and p["arxiv_id"]], key=rank)
    held = sum(p["pdf_held"] for p in papers)
    obtained = sorted((p for p in papers if p.get("obtained")), key=lambda p: p["key"])
    title = lambda p: f"{_md_escape(p['title'])} `{p['key']}`"
    L = ["# Papers to download", "",
         f"*GENERATED by `{doc['build']}` -- do not hand-edit. Where-to-get-it links "
         "come from `papers/_links.json` (`build_bwm.py links`: Semantic Scholar + OpenAlex open-access lookups); "
         "\"held\" comes from `papers/_held.json` (`build_bwm.py scan --pdfs <folder>`).*", "",
         f"**{held} of {len(papers)} papers held as PDFs**" + (f" (+ {len(obtained)} obtained as data, not a PDF)"
                                                                 if obtained else "") + f". Not held: {len(auto)} on arXiv (fetched by script), "
         f"{sum(p['availability'] == 'open' for p in todo)} with a free copy found, "
         f"{sum(p['availability'] == 'library' for p in todo)} that need the library.", "",
         "**How to hunt.** Try the links left to right: *open PDF* / *free at publisher* / *arXiv version* are free "
         "copies; *publisher* opens the full text on campus or on the UF VPN; *Scholar* searches the exact title (its "
         "\"All N versions\" link finds author and repository copies); *PDF search* looks for the title as a PDF anywhere. "
         f"Download them into any folder under any name, then `{doc['add']}`: it identifies each PDF (arXiv id, key, "
         "the DOI / title in its metadata or printed on page 1), "
         "copies it into the papers folder as `Surname_Year_Title.pdf` (the download folder is left alone; a broken or cut-off PDF "
         "is refused), rescans and rebuilds -- and they leave this list.", "",
         f"Open arXiv papers: `{doc['fetch']}` downloads every one still missing (one request every 15 s: arxiv.org's "
         "robots.txt crawl delay)." + doc["pdf_note"], ""]

    def table(rows, numbered, venue):
        out = [("| # " if numbered else "") + "| paper | year | " + ("venue | " if venue else "") + "get it |",
               ("|---" if numbered else "") + "|---|---|" + ("---|" if venue else "") + "---|"]
        for i, p in enumerate(rows, 1):
            out.append((f"| {i} " if numbered else "") + f"| {title(p)} | {p['year'] or ''} | "
                       + (f"{_md_escape(p['venue'])} | " if venue else "") + f"{_get_md(p, links.get(p['key']))} |")
        return out

    for head, rows, numbered in ((f"## Priority: core / high relevance, not open on arXiv ({len(pri)})", pri, True),
                                 (f"## Later: medium / low relevance, not open on arXiv ({len(later)})", later, False)):
        L += [head, ""]
        for sub, want in (("Free copy found -- download directly", "open"), ("Needs the library", "library")):
            rows_ = [p for p in rows if p["availability"] == want]
            L += [f"### {sub} ({len(rows_)})", ""] + (table(rows_, numbered, numbered) if rows_ else ["none"]) + [""]
    if obtained:
        L += [f"## Obtained as data, not a PDF ({len(obtained)}) -- `build_bwm.py obtain`; not on this list", ""]
        L += [f"- {title(p)}: {p['obtained']}" for p in obtained] + [""]
    L += [f"## Open arXiv papers not fetched yet ({len(auto)}) -- `--fetch-arxiv` gets these", ""]
    L.append(" \u00b7 ".join(f"[`{p['key']}`](https://arxiv.org/pdf/{p['arxiv_id']})" for p in auto) if auto else "none")
    return "\n".join(L) + "\n"


def _index_trends(papers) -> dict:
    by = collections.defaultdict(lambda: {"published": 0, "preprint": 0})
    for p in papers:
        if p["year"]:
            by[str(p["year"])]["published" if p["published"] else "preprint"] += 1
    return dict(sorted(by.items()))


def find_data(base: Path, papers: List[dict], alias_map: Dict[str, str]) -> dict:
    """The Find tab's data: each lexicon term's and each program task's precomputed full-text ranking
    (papers/_lookup.json, `tools/paper_find.py precompute`) as paper INDICES into the page's paper list (top
    paper_find.ATLAS_TABLE_DEPTH), the measured fusion weights and the tokeniser's stop words -- one source of truth:
    tools/paper_find.py."""
    PF = _tool("paper_find")
    idx = {p["key"]: i for i, p in enumerate(papers)}
    look = _load(base / "papers" / PF.LOOKUP_FILE, {})
    canon = lambda ks: [idx[alias_map.get(k, k)] for k in ks if alias_map.get(k, k) in idx][:PF.ATLAS_TABLE_DEPTH]  # noqa
    return {"terms": {t: canon(v) for t, v in sorted((look.get("terms") or {}).items())},
            "tasks": [{"id": t["id"], "title": t["title"], "query": t["query"], "p": canon(t["papers"])}
                      for t in look.get("tasks") or []],
            "weights": PF.ATLAS_WEIGHTS, "coverage": PF.TERM_COVERAGE, "rrf": PF.RRF_K,
            "stop": sorted(PF.STOP), "ranker": look.get("ranker", ""), "made": look.get("made", ""),
            "relevance": PF.RELEVANCE}


def render_browser(papers, lex, labs, web, docs, trends_file, stats, local_pdfs=None, facts_base=None,
                   find=None, title: str = "Marine World-Model Atlas", sections=None, smode: str = "banners",
                   here: str = "", pages=None, cites_js: str = "atlas_citations.js",
                   figs_js: str = FIGURES_JS, all_n: int = 0, examples: Optional[dict] = None) -> str:
    """The atlas page. `local_pdfs` {key: relative path} is set only by `bundle` (a PDF link per held paper); `find` is
    the Find tab's data (find_data). `sections` (section_list): the banner row -- in `banners` mode it picks the section
    every tab shows; in `atlases` mode it links the pages in `pages` {id or 'all': file} and `here` is this page's
    section ('' = all papers). No sections: no banner row, and the page's data as before. `examples` {"find",
    "figures"}: the search boxes' "(e.g. ...)" (atlas.toml [atlas] find_example / figures_example; none = no example)."""
    sidx = {s["id"]: i for i, s in enumerate(sections or [])}

    def row(p):
        r = {"k": p["key"], "t": p["title"], "y": p["year"], "v": p["venue"], "s": p["status"],
             "pub": p["published"], "c": p["status_conflict"],
             "a": (", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")),
             "u": p["url"], "code": p["code_url"], "top": p["topics"], "r": p["relevance"], "g": p["tags"][:8],
             "sum": p["summary"], "why": next(iter(p["why_for_us"].values()), ""), "h": p["pdf_held"], "ob": p.get("obtained") or "",
             "av": p["availability"], "d": p["doi"] or "", "ax": p["arxiv_id"] or "", "oa": p["open_url"],
             "oal": p["open_label"], "oav": p["arxiv_version"], "s2": p["s2_url"]}
        if local_pdfs and p["key"] in local_pdfs:
            r["lp"] = local_pdfs[p["key"]]
        if p.get("code_status"):
            r["cs"], r["cst"] = p["code_status"], p.get("code_stars")
        if p.get("cited_by") is not None:              # `meta`: dated counts (meta_links.py)
            r["cb"], r["cbi"], r["md"] = p["cited_by"], p.get("cited_here") or 0, p.get("meta_measured_on")
            if p.get("venue_citedness_2yr") is not None:
                r["vm"] = p["venue_citedness_2yr"]
        chips = _tool("paper_find").fact_chips(facts.get(p["key"]) or {})   # one rule for the page and the lookup
        if chips:
            r["fx"] = chips
        if sidx:
            r["sc"] = [sidx[x] for x in p.get("sections") or () if x in sidx]
        return r

    facts = {f["key"]: f["facts"] for f in (_load(HERE / "papers" / "_facts.json", {}).get("papers") or [])} \
        if facts_base is None else facts_base

    data = {
        "stats": stats, "docs": docs,
        "papers": [row(p) for p in papers],
        "lex": [{"id": x["id"], "t": x["term"], "al": x["aliases"], "cat": x["categories"], "top": x["topics"],
                 "d": x["definition"], "sn": x["senses"], "ds": [d for d in x["definitions"] if d["text"] != x["definition"]],
                 "ci": [[c["key"], c["quote"], c["where"], c["year"], c["status"], c["url"], c["source"]]
                        for c in x["citations"]], "rel": x["related_ids"],
                 "mx": [[m["latex"], m.get("symbols_tex") or m["symbols"]] for m in x.get("math") or []],
                 **({"dt": x["definition_tex"]} if x.get("definition_tex") else {})} for x in lex],
        "labs": labs, "web": web,
        "trends": {"index": _index_trends(papers),
                   "arxiv": {k: v.get("by_year", {}) for k, v in (trends_file.get("arxiv_series") or {}).items()},
                   "meta": (trends_file.get("meta") or {}).get("caveats", [])[:3]},
        "find": find or {"terms": {}, "tasks": [], "weights": {}, "coverage": 0.6, "rrf": 60, "stop": [],
                         "relevance": {}},
    }
    if sections:
        tix = {t["id"]: i for i, t in enumerate(data["find"].get("tasks") or [])}
        pages = pages or {}
        data["sec"] = {"mode": smode, "here": here, "all": pages.get("all", ""), "total": all_n or len(papers),
                       "list": [{"id": s["id"], "t": s["title"], "st": s["subtitle"], "c": s["color"], "top": s["topics"],
                                 "tk": [tix[t] for t in s["tasks"] if t in tix], "n": s["n"],
                                 "href": pages.get(s["id"], "")} for s in sections]}
    blob = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    import html as _html                              # the title first (the data never holds the placeholder); ASCII
    esc = lambda s: _html.escape(str(s)).encode("ascii", "xmlcharrefreplace").decode("ascii")    # noqa: E731
    eg = lambda k: f" (e.g. {esc(examples[k])})" if (examples or {}).get(k) else ""            # noqa: E731
    return (BROWSER_TEMPLATE.replace("__ATLAS_TITLE__", esc(title)).replace("__FIND_EG__", eg("find"))
            .replace("__FIGURES_EG__", eg("figures")).replace("__CITES_JS__", cites_js)
            .replace("__FIGS_JS__", figs_js).replace("__DATA__", blob))


def _dumps_rows(obj: dict, rows_key: str) -> str:
    """Compact JSON with ONE ROW PER LINE under `rows_key` (sorted keys): small enough for the repo's 5 MB rule, and a
    changed paper / term / group is a one-line diff."""
    one = lambda o: json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    head = one({k: v for k, v in obj.items() if k != rows_key})
    rows = ",\n".join(one(r) for r in obj[rows_key])
    return head[:-1] + ("," if len(head) > 2 else "") + json.dumps(rows_key) + ":[\n" + rows + "\n]}\n"


def section_list(base: Path, papers: List[dict], alias_map: Dict[str, str]) -> List[dict]:
    """atlas.toml [[sections]] with each one's paper count; every paper gets `sections` (the ids it sits in: a section
    naming one of its topics, or its key). No [[sections]]: [] and the papers untouched (outputs as before)."""
    AP = _tool("atlas_project")
    secs = AP.sections(base)
    if not secs:
        return []
    for p in papers:
        p["sections"] = AP.section_ids(p["key"], p["topics"], secs, alias_map)
    for s in secs:
        s["n"] = sum(s["id"] in p["sections"] for p in papers)
    return secs


def section_page_name(sid: str) -> str:
    """A section's own atlas page in `atlases` mode (its sidecars: atlas_<id>_citations.js / _figures.js)."""
    return f"atlas_{sid}.html"


SECTION_FILE = re.compile(r"^atlas_[a-z0-9][a-z0-9_-]*(\.html|_citations\.js|_figures\.js)$")


def _figures_doc(base: Path) -> Optional[dict]:
    """index/atlas_figures.js as data (None when absent or unreadable): what a section page's figure sidecar is cut
    from."""
    f = Path(base) / "index" / FIGURES_JS
    try:
        text = f.read_text(encoding="utf-8")
    except OSError:
        return None
    head = "window.__MWA_FIGS="
    if not text.startswith(head):
        return None
    try:
        return json.loads(text[len(head):].rstrip().rstrip(";"))
    except ValueError:
        return None


def section_atlases(base: Path, secs: List[dict], papers: List[dict], lex: List[dict], labs: dict, web: List[dict],
                    docs: List[dict], trends_file: dict, stats: dict, local_pdfs, facts, alias_map: Dict[str, str],
                    pages: Dict[str, str], links: bool = True, examples: Optional[dict] = None) -> Dict[str, str]:
    """`atlases` mode: per section its own atlas -- the page cut to the section's papers (terms with a citation among
    them, their citations only; the groups that wrote them; its citation network and figures; its own tasks), with
    the banner row linking every page (`links` False: no links -- a section's own zip holds no other page).
    {file name: text} for index/."""
    out: Dict[str, str] = {}
    meta = meta_rows(base, alias_map)
    figs = _figures_doc(base)
    for s in secs:
        if s["id"] not in pages:                       # no papers yet: no page
            continue
        keys = {p["key"] for p in papers if s["id"] in p["sections"]}
        sub = [p for p in papers if p["key"] in keys]
        lex_sub = []
        for x in lex:
            ci = [c for c in x["citations"] if c["key"] in keys]
            if ci:
                lex_sub.append({**x, "citations": ci})
        nodes = [{**n, "papers": [k for k in n["papers"] if k in keys]} for n in labs["nodes"] if keys & set(n["papers"])]
        ids = {n["id"] for n in nodes}
        labs_sub = {"layout": labs["layout"], "nodes": nodes,
                    "edges": [e for e in labs["edges"] if e["a"] in ids and e["b"] in ids]}
        cites = _sibling("citation_map").atlas_data({p["key"]: p for p in sub}, meta, base, [p["key"] for p in sub])
        st = {**stats, "papers": len(sub), "published": sum(p["published"] for p in sub),
              "preprints": sum(p["status"] == "preprint" for p in sub),
              "withdrawn": sum(p["status"] == "withdrawn" for p in sub), "terms": len(lex_sub),
              "citations": sum(len(x["citations"]) for x in lex_sub), "labs": sum(n["kind"] == "group" for n in nodes),
              "topics": sorted({t for p in sub for t in p["topics"]}), "citation_links": cites["stats"]["links"],
              "citation_linked_papers": cites["stats"]["linked_papers"]}
        find = find_data(base, sub, alias_map)
        find["tasks"] = [t for t in find["tasks"] if t["id"] in s["tasks"]]
        name = pages[s["id"]]
        cjs, fjs = name[:-len(".html")] + "_citations.js", name[:-len(".html")] + "_figures.js"
        out[name] = render_browser(sub, lex_sub, labs_sub, web, docs, trends_file, st, local_pdfs, facts, find,
                                   title=s["title"], sections=secs, smode="atlases", here=s["id"],
                                   pages=pages if links else {}, cites_js=cjs, figs_js=fjs, all_n=len(papers),
                                   examples=examples)
        out[cjs] = sidecar_js("__MWA_CITES", {"schema": "svfr.bwm.cites/1", **cites})
        if figs is not None:
            rows = [r for r in figs.get("papers") or [] if r[0] in keys]
            out[fjs] = sidecar_js("__MWA_FIGS", {**figs, "papers": rows, "figures": sum(len(r[2]) for r in rows),
                                                 "captioned": sum(1 for r in rows for f in r[2] if f[2])})
    return out


def build_all(base: Path = HERE, local_pdfs: Optional[Dict[str, str]] = None,
              all_page: str = "browser.html", section_links: bool = True) -> Dict[str, str]:
    """Every generated index file {name: text}. `all_page`: what a section page's 'All papers' banner links to
    (browser.html here; atlas.html in the bundle); `section_links` False: section pages link no other page (each
    section's own zip)."""
    papers, alias_map, pstats = build_papers(base)
    links, hstats = apply_holdings(base, papers, alias_map)
    secs = section_list(base, papers, alias_map)
    smode = _tool("atlas_project").sections_mode(base) if secs else "banners"
    pages = ({"all": all_page, **{s["id"]: section_page_name(s["id"]) for s in secs if s["n"]}}
             if smode == "atlases" else {})                # an empty section gets no page (its banner: no link)
    index = {p["key"]: p for p in papers}
    lex, lstats = build_lexicon(base, alias_map, index)
    labs, bstats = build_labs(base, alias_map, index)
    web = build_web(base)
    stats = {**pstats, **hstats, **lstats, **bstats, "web_sources": len(web),
             "published": sum(p["published"] for p in papers),
             "preprints": sum(p["status"] == "preprint" for p in papers),
             "withdrawn": sum(p["status"] == "withdrawn" for p in papers),
             "topics": sorted({t for p in papers for t in p["topics"]})}
    trends_file = _load(base / "labs" / "trends.json", {})
    cites = _sibling("citation_map").atlas_data(index, meta_rows(base, alias_map), base, [p["key"] for p in papers])
    stats["citation_links"], stats["citation_linked_papers"] = cites["stats"]["links"], cites["stats"]["linked_papers"]
    if secs:
        stats["sections"] = {s["id"]: s["n"] for s in secs}
    docs, facts = topic_docs(base), facts_rows(base, alias_map)
    doc, names = doc_text(base), project_names(base)
    examples = {"find": names["find_example"], "figures": names["figures_example"]}
    outs = {
        "papers.json": _dumps_rows({"schema": "svfr.bwm.papers/1", "stats": stats,
                                    "aliases": dict(sorted(alias_map.items())), "papers": papers}, "papers"),
        "lexicon.json": _dumps_rows({"schema": "svfr.bwm.lexicon/1", "terms": lex}, "terms"),
        "labs.json": _dumps_rows({"schema": "svfr.bwm.labs/2", **labs}, "nodes"),
        "web.json": _dumps_rows({"schema": "svfr.bwm.web/1", "sources": web}, "sources"),
        "PAPERS.md": render_papers_md(papers, stats, web, doc),
        "LEXICON.md": render_lexicon_md(lex, stats, doc),
        "LABS.md": render_labs_md(labs, stats, doc),
        "PAPER_DOWNLOAD_LIST.md": render_download_md(papers, links, doc),
        "browser.html": render_browser(papers, lex, labs, web, docs, trends_file, stats, local_pdfs, facts,
                                       find_data(base, papers, alias_map), title=names["title"],
                                       sections=secs, smode=smode, pages=pages, examples=examples),
        "atlas_citations.js": sidecar_js("__MWA_CITES", {"schema": "svfr.bwm.cites/1", **cites}),
    }
    if pages:                                          # `atlases` mode: one atlas page per section besides
        outs.update(section_atlases(base, secs, papers, lex, labs, web, docs, trends_file, stats, local_pdfs, facts,
                                    alias_map, pages, links=section_links, examples=examples))
    return outs


def sidecar_js(var: str, obj: dict) -> str:
    """A data file the atlas loads with <script src> (works from file://, where fetch() of a JSON does not): one
    global, compact JSON, no raw '</'."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    return f"window.{var}={blob};\n"


OFFICE_NAMES = ("ATLAS.xlsx", "LEXICON.docx", "PAPERS.docx")


def office_outputs(base: Path, outs: Dict[str, str]) -> Dict[str, object]:
    """The Office files (office.py) from a build's outputs: {name: bytes or None (no pandoc)}, plus "_workbook" and
    "_stamps" ({Word file: the source stamp of its Markdown}, made with or without pandoc)."""
    O, PL, CL = _sibling("office"), _sibling("paper_links"), _sibling("code_links")
    P = json.loads(outs["papers.json"])
    papers, stats, alias = P["papers"], P["stats"], P["aliases"]
    lex = json.loads(outs["lexicon.json"])["terms"]
    labs = json.loads(outs["labs.json"])
    lstats = {**stats, **{k: v for k, v in stats.items()}}
    links = links_rows(base, alias)
    rej_files = rejected_ids(base, alias)[0]
    held = {alias.get(r["key"], r["key"]): r["file"] for r in (_load(base / "papers" / "_held.json", {}).get("files")
                                                              or [])
            if alias.get(r["key"], r["key"]) not in rej_files.get(str(r.get("sha256") or "").lower(), ())}
    for x in papers:
        x["file"] = held.get(x["key"], "")
    facts = facts_rows(base, alias)
    code = {r["key"]: {**r, "counted": r["status"] in CL.COUNTED}
            for r in (_load(base / "papers" / "_code.json", {}).get("papers") or [])}
    hunt = lambda x: PL.hunt_links(x, links.get(x["key"]))
    doc = doc_text(base)                               # the project's own titles (atlas.toml [documents])
    wb = O.workbook(papers, lex, labs, facts, code, hunt, meta=meta_rows(base, alias), doc=doc)
    md = {"LEXICON.docx": O.lexicon_markdown(lex, lstats, doc=doc),
          "PAPERS.docx": O.papers_markdown(papers, stats, code, hunt, doc=doc)}
    return {"_workbook": wb, "ATLAS.xlsx": O.xlsx_bytes(wb), **{n: O.docx_bytes(t) for n, t in md.items()},
            "_stamps": {n: O.source_stamp(t) for n, t in md.items()}}


WORD_NAMES = ("LEXICON.docx", "PAPERS.docx")


def office_stale(base: Path, outs: Dict[str, str], out_dir: Path) -> List[str]:
    """Office files whose CONTENT differs from a fresh build (bytes differ between machines: zlib, pandoc), and Word
    files that do not carry the source stamp of the Markdown a fresh build makes (office.source_stamp) -- the one
    comparison a machine WITHOUT pandoc can make, so there it is the whole check of a Word file.
    A stamp that differs is STALE on every machine, pandoc or not -- a refusal, not a note: a note is what let bed3765b
    (papers.json rebuilt where no pandoc was) push a PAPERS.docx of the old list, and with no CI on a push the pre-push
    hook is the only gate, so a warning turned origin/main red for every machine with pandoc instead of stopping the one
    push that caused it. The stamp is the sha256 of the very Markdown pandoc reads, so no pandoc version or machine
    raises it falsely. A Word file not in the index at all is a note on every machine (a machine without pandoc writes
    none, and bundle_atlas ships an index without them), so no machine fails what another passes."""
    O = _sibling("office")
    fresh = office_outputs(base, outs)
    stale = []
    if not (out_dir / "ATLAS.xlsx").is_file() or O.xlsx_content(out_dir / "ATLAS.xlsx") != O.xlsx_content(fresh["_workbook"]):
        stale.append("ATLAS.xlsx")
    lack = O.word_missing()                            # "pandoc", "python-docx" (the house styles) or None
    here = f"no {lack} here" if lack else "pandoc or python-docx failed here"
    for name in WORD_NAMES:
        path = out_dir / name
        if not path.is_file():
            print(f"NOTE: {name} not in the index (`build` writes it where pandoc and python-docx are installed)")
            continue
        have = O.docx_stamp(path)
        why = ("" if have == fresh["_stamps"][name] else
               "carries no source stamp" if have is None else "built from other Markdown")
        if fresh[name] is None:                        # no pandoc or python-docx: the stamp is all there is to compare
            if why:
                stale.append(f"{name} ({why}; {here} -- `build` it where pandoc and python-docx are installed)")
            else:
                print(f"NOTE: {name} compared by its source stamp only ({here}): built from this Markdown")
        elif why or O.docx_content(path) != O.docx_content(fresh[name]):
            stale.append(name)
    return stale


def same_zip_payload(path: Path, data: bytes) -> bool:
    """Does the Office file at `path` already hold exactly what `data` holds -- the same entries in the same order,
    byte for byte once inflated? Then `build` leaves it alone: the same parts compress to different bytes under
    another zlib (zlib-ng vs zlib, a Python update on one machine), and rewriting a committed file on every build
    would make two machines re-commit it in turn. For the same reason a Word file's word/document.xml is compared
    once what pandoc versions write differently is folded (office.docx_part_canonical: the lab's 3.6 and the laptop's
    3.6.2 write \\perp and the section properties differently); every other part byte for byte. Stricter than
    `office_stale` (the cells and text `check` reads): a change only to styles or formatting is still written. So is
    any change to a Word file's Markdown, though pandoc may write the same document.xml from it: docProps/custom.xml
    carries the Markdown's stamp (office.source_stamp), which must name the Markdown a fresh build reads or a machine
    without pandoc calls the file STALE. False for a missing or unreadable file (then it is written)."""
    import io
    import zipfile
    O = _sibling("office")
    same = lambda n, a, b: a == b or O.docx_part_canonical(n, a) == O.docx_part_canonical(n, b)
    try:
        with zipfile.ZipFile(path) as old, zipfile.ZipFile(io.BytesIO(data)) as new:
            names = [i.filename for i in old.infolist()]
            return names == [i.filename for i in new.infolist()] and all(same(n, old.read(n), new.read(n))
                                                                          for n in names)
    except Exception:                                  # missing, not a zip, a bad CRC or stream: write it afresh
        return False


def standalone_page(html: str) -> str:
    """The Artifact-ready page (no doctype / head: the publisher adds them) as a document a browser opens from disk."""
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n' + html +
            "\n</body>\n</html>\n")


def _paper_pdfs_module():
    spec = importlib.util.spec_from_file_location("_bwm_paper_pdfs", HERE.parents[1] / "tools" / "paper_pdfs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def project_pdfs(base: Path) -> Path:
    """The PDF folder of the project at `base` (tools/atlas_project.py: $ATLAS_PAPERS, its atlas.toml [paths] papers;
    the BlueBoat atlas: $SVFR_BWM_PAPERS, else the copy tools/paper_text.py finds)."""
    return _tool("atlas_project").papers_dir(base, legacy=lambda: _tool("paper_text").DEFAULT_PDFS)


def project_names(base: Path) -> dict:
    """The project's page title, zip name and bundle README title (its atlas.toml [atlas])."""
    return _tool("atlas_project").names(base)


def _shell_arg(s: str) -> str:
    return f'"{s}"' if " " in s else s


def doc_text(base: Path, root: Optional[Path] = None) -> dict:
    """What the generated documents call the project at `base` and the commands they hand a reader (2026-10-16:
    every project's PAPERS.md was titled "BlueBoat world-model literature" and its download list pointed
    `paper_pdfs.py --refs` at the BlueBoat index). Titles: tools/atlas_project.py documents (atlas.toml [documents],
    else made from the title). Paths: the project's own, relative to `root` (the repository, or the kit, the commands
    run from); a project outside it is `<project>`, so the bytes do not depend on where a machine keeps it.
    The builder's own project (the BlueBoat atlas in this repository) needs no --base / --pdfs: its commands read
    as they always did."""
    AP = _tool("atlas_project")
    b = Path(base).resolve()
    own = b == HERE.resolve()
    try:
        proj = b.relative_to(Path(root or HERE.parents[1]).resolve()).as_posix()
    except ValueError:
        proj = "<project>"
    engine = HERE.relative_to(HERE.parents[1]).as_posix()        # report/boat_world_model; engine/atlas in a kit
    builder = f"python {engine}/build_bwm.py"
    base_arg = "" if own else f" --base {_shell_arg(proj)}"
    # --dir names the folder that HOLDS the papers: an empty one makes --fetch-arxiv download all of them again
    # (pointed at one, it once re-fetched 57 PDFs a project already held)
    note = "" if own else (" `<PDF folder>` is the folder holding this project's PDFs -- `$ATLAS_PAPERS` where a "
                           "machine sets it, else atlas.toml `[paths] papers` -- never an empty one: `--fetch-arxiv` "
                           "would download every arXiv paper again."
                           + (" `<project>` is this project's folder." if proj == "<project>" else ""))
    return {**AP.documents(base), "title": AP.names(base)["title"], "own": own, "project": proj, "engine": engine,
            "pdf_note": note,
            "repo": "sonar-video-fusion-review" if AP.BLUEBOAT.is_dir() else "paper-atlas",
            "build": f"{builder} build{base_arg}",
            "add": f"{builder} add <that folder>{base_arg}" + ("" if own else " --pdfs <PDF folder>"),
            "add_here": f'{builder} add "<download folder>"{base_arg}' + ("" if own else ' --pdfs "<this folder>"'),
            "fetch": f"python tools/paper_pdfs.py --refs {_shell_arg(proj + '/index/papers.json')} --dir "
                     f"{'<folder>' if own else '<PDF folder>'} --fetch-arxiv"}


def _tool(name: str):
    """A module from the repo's tools/ folder (paper_rename, paper_text), loaded once under a private name."""
    full = f"_bwm_tool_{name}"
    if full not in sys.modules:
        tools = str(HERE.parents[1] / "tools")        # a tool's own sibling imports (paper_text -> data_roots)
        if tools not in sys.path:
            sys.path.append(tools)
        spec = importlib.util.spec_from_file_location(full, HERE.parents[1] / "tools" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


def paper_file_name(paper: dict) -> str:
    """The library-style file name of an index paper: `Surname_Year_Title_Words.pdf` (tools/paper_rename.py)."""
    authors = paper.get("authors") or []
    return _tool("paper_rename").file_name(paper.get("title") or paper["key"], authors[0] if authors else "",
                                           paper.get("year"), paper["key"])


def cmd_scan(base: Path, pdf_dir: Path, prune: bool = False, extra: Optional[List[dict]] = None) -> int:
    """Rewrite papers/_held.json from `pdf_dir`. `extra` = rows `add` just copied (their identity, by content)."""
    if not pdf_dir.is_dir():
        print(f"no PDF folder at {pdf_dir} (pass --pdfs)")
        return 2
    if _refuse_unreadable_rejections(base):
        return 2
    papers, alias_map, _ = build_papers(base)
    PL = _sibling("paper_links")
    path = base / "papers" / "_held.json"
    previous = _load(path, {}) if path.is_file() else {}
    if extra:
        previous = {**previous, "files": list(previous.get("files") or []) + list(extra)}
    doc = PL.scan(pdf_dir, papers, alias_map, identity=_paper_pdfs_module().pdf_identity, page_text=PL.page1_text,
                  previous=previous, prune=prune, rejected=rejected_ids(base, alias_map)[0])
    # not_in_folder describes THIS machine's folder, and the file is shared by machines holding different parts of
    # the papers (2026-10-14: a lab scan of 624 of 1,796 wrote ~1,170 names a laptop scan would erase again), so it is
    # printed below, never committed.
    path.write_text(_dumps_rows({k: v for k, v in doc.items() if k not in ("not_in_folder", "rejected")}, "files"),
                    encoding="utf-8", newline="\n")    # (this machine's file names: never the shared file)
    gb = sum(r["bytes"] for r in doc["files"]) / 1e9
    print(f"scan: {len(doc['files'])} papers held ({gb:.2f} GB); {len(doc['duplicates'])} duplicate file(s), "
          f"{len(doc['unmatched'])} PDF(s) in {pdf_dir} matched nothing -> papers/_held.json. Now run: build")
    if doc["not_in_folder"]:
        print(f"  kept {len(doc['not_in_folder'])} held paper(s) whose file is not in this folder (held on another "
              f"machine or drive); `scan --prune` drops them")
    if doc.get("pruned"):
        print(f"  --prune: dropped {len(doc['pruned'])} held paper(s) whose file is gone")
    for f in doc["unmatched"][:20]:
        print(f"  unmatched: {f}  (rename it <key>.pdf, or put the arXiv id / DOI in its name)")
    for f in doc.get("rejected") or []:
        print(f"  REJECTED: {f} is NOT the index paper it matches (papers/_rejected.json) -- move it out of {pdf_dir}")
    if doc.get("rejected"):
        print("  Then run: tools/paper_facts.py --bwm (a rejected file's facts go), then build")
    write_catalog(base, pdf_dir, doc, papers)
    return 0


def write_catalog(base: Path, pdf_dir: Path, held: dict, papers: List[dict], doc: Optional[dict] = None) -> None:
    """The papers folder's own README.md (the lab library's front-matter, so `python -m tools.library ingest <folder>
    20_papers/boat-world-model` files it as-is) and papers.tsv: one grep-able line per PDF in the folder. The README
    names the project's own title, index and commands (doc_text)."""
    doc = doc or doc_text(base)
    by_key = {p["key"]: p for p in papers}
    here = {r["file"]: r for r in held.get("files") or [] if (pdf_dir / r["file"]).is_file()}
    cols = ["file", "key", "year", "first_author", "title", "venue", "doi", "arxiv_id", "status", "relevance",
            "topics", "sha256_16"]
    clean = lambda v: re.sub(r"[\t\r\n]+", " ", str(v if v is not None else "")).strip()
    lines = ["\t".join(cols)]
    for name in sorted(here, key=str.lower):
        r, p = here[name], by_key.get(here[name]["key"], {})
        lines.append("\t".join(clean(x) for x in (
            name, r["key"], p.get("year"), (p.get("authors") or [""])[0], p.get("title"), p.get("venue"),
            p.get("doi"), p.get("arxiv_id"), p.get("status"), p.get("relevance"), ",".join(p.get("topics") or []),
            r.get("sha256", "")[:16])))
    (pdf_dir / "papers.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    gb = sum(r["bytes"] for r in here.values()) / 1e9
    p = doc["project"]
    bwm = "" if doc["own"] else (f"`--bwm` below means this atlas when `ATLAS_PROJECT={p}` and "
                                 "`ATLAS_PAPERS=<this folder>` are set. ")
    (pdf_dir / "README.md").write_text("\n".join([
        "---",
        f"what: {doc['pdfs_what']} (one file per paper, named Surname_Year_Title)",
        f"when: {datetime.date.today().isoformat()}",
        "where: n/a (literature)",
        "who: the owner's downloads (publishers, UF library) and arXiv, matched by the repo tooling",
        f"source: {doc['engine']}/build_bwm.py add / scan ({doc['repo']})",
        f"commit: {commit}",
        "status: derived",
        "---", "",
        f"# {doc['pdfs']}", "",
        f"{len(here)} PDFs ({gb:.2f} GB), each matched to a paper of the program's index "
        f"(`{p}/index/papers.json`); `papers.tsv` lists them, one line each: file, key, year, first "
        "author, title, venue, DOI, arXiv id, status, relevance, topics. GENERATED by `build_bwm.py scan` -- do not "
        "edit by hand.", "",
        f"Find a paper: grep `papers.tsv`, or open the atlas (`{p}/index/browser.html`). {bwm}Search the "
        "FULL TEXT of every paper: `python tools/paper_text.py grep \"<regex>\" --bwm` (page-numbered hits). "
        f"New downloads: `{doc['add_here']}`. "
        "Rename to titles: `python tools/paper_rename.py <folder> --bwm --apply` (logged in renames.tsv).", "",
    ]), encoding="utf-8", newline="\n")


def data_root(pdf_dir: Path) -> Path:
    """Where `obtain` keeps datasets (not PDFs): beside the papers folder, `<pdfs>_data/<key>/`."""
    return Path(pdf_dir).parent / (Path(pdf_dir).name + "_data")


def cmd_meta(base: Path, cache: Path, citing: int = 5, stars: bool = True, s2: bool = True, today: str = "",
             get=None, post=None, run=None) -> int:
    """The facts ABOUT every index paper (meta_links.py): OpenAlex date, citation count (+ by year), references and the
    venue's 2-year mean citedness (an open Impact-Factor analogue, for journals / conferences only -- the IF itself is
    licensed and not fetched); Semantic Scholar citation + influential-citation counts; the citation graph INSIDE the
    corpus; the `citing` most-cited works citing each HELD paper; GitHub stars through the owner's `gh` login.
    MERGED, never rewritten: a source this run could not ask keeps its previous answer and that answer's own date, so a
    run cut short by a quota never erases a measurement (review 2026-10-14). OpenAlex's free quota is 1,000 requests a
    day (MEASURED: then 429 with Retry-After ~5 h): the citing lists go to papers WITHOUT one first, and after the first
    refusal none is asked. Answers are cached for the day (`<cache>/meta-<date>`). -> committed papers/_meta.json."""
    ML, CL, PL = _sibling("meta_links"), _sibling("code_links"), _sibling("paper_links")
    import subprocess
    import time
    import urllib.error
    import urllib.request
    day = cache / f"meta-{today}"

    def _once(url):                                    # one try: a 429 here means the day's quota is spent
        try:                                           # the key: tools/openalex_key.py (unredirected; a refused key
            with PL.openalex_urlopen(url, {"User-Agent": "svfr-bwm/1"}, timeout=60) as r:   # -> keyless)
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            raise CL.Transient(url)
        except Exception:                              # noqa: BLE001 -- not cached; this run marks it unmeasured
            raise CL.Transient(url)
    net = CL.Net(day, get=_once, pause=0.12)
    get = get or net.get

    def _post(url, body):                              # Semantic Scholar's batch endpoint, cached for the day
        import hashlib
        f = day / ("s2-" + hashlib.sha1((url + json.dumps(body, sort_keys=True)).encode()).hexdigest() + ".json")
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                                  "User-Agent": "svfr-bwm/1"})
        res = None
        for wait in (0, 20, 60):                       # the unauthenticated pool is shared and answers 429 often
            time.sleep(wait)                           # (MEASURED 2026-10-14: 1 of 6 batches on the first try)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    res = json.loads(r.read().decode("utf-8"))
                break
            except Exception:                          # noqa: BLE001 -- retried, then that batch stays unmeasured
                res = None
        if res is None:
            return None
        day.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res), encoding="utf-8")
        time.sleep(3.0)
        return res

    def _gh(query):                                    # the owner's own `gh` login: read-only GraphQL
        r = subprocess.run(["gh", "api", "graphql", "-f", f"query={query}"], capture_output=True, text=True,
                           timeout=120)
        try:
            return json.loads(r.stdout or "null")      # GitHub answers partial data + errors for unknown repos
        except ValueError:
            return None

    papers = json.loads((base / "index" / "papers.json").read_text(encoding="utf-8"))["papers"]
    held = {p["key"] for p in papers if p.get("pdf_held")}
    prev = {r["key"]: r for r in (_load(base / "papers" / "_meta.json", {}).get("papers") or [])
            if isinstance(r, dict) and r.get("key")}
    doi_of = {p["key"]: ML.work_doi(p) for p in papers}
    dfile = day / "openalex-dois.json"                 # EACH DOI's answer, so a re-run asks only the DOIs not answered
    cands = _load(dfile, {})                           # today (review 2026-10-14: one added paper changed every batch)
    new, unchecked = ML.openalex_candidates(sorted({d for d in doi_of.values() if d and d not in cands}), get)
    if new:
        cands.update(new)
        day.mkdir(parents=True, exist_ok=True)
        dfile.write_text(json.dumps(cands), encoding="utf-8")
    unchecked = set(unchecked)
    works, ambiguous = {}, {}
    for p in papers:                                   # the work that IS the paper: by its index title (choose_work)
        d = doi_of[p["key"]]
        if d in cands:
            w, how = ML.choose_work(cands[d], p.get("title") or "")
            if w:
                works[p["key"]] = w
            elif how == "ambiguous":
                ambiguous[p["key"]] = sorted({c["openalex"] for c in cands[d]})
    src_failed: List[str] = []
    sources = ML.openalex_sources({w["source"] for w in works.values()}, get, failed=src_failed)
    src_failed = set(src_failed)
    s2ids = {p["key"]: ML.s2_id(p.get("s2_url") or "") for p in papers}
    s2_failed: List[str] = []
    s2v = ML.s2_batch([i for i in s2ids.values() if i], post or _post, failed=s2_failed) if s2 else {}
    s2_failed = set(s2_failed)
    repos = {p["key"]: ML.github_repo(p.get("code_url"), CL.repo_of) for p in papers}
    gh_failed: List[str] = []
    gh = ML.github_stars([r for r in repos.values() if r], run or _gh, failed=gh_failed) if stars else {}
    gh_failed = set(gh_failed)
    rows: Dict[str, dict] = {}
    for p in papers:                                   # every date is set here, from today or from the KEPT answer
        k, w, old = p["key"], works.get(p["key"]), prev.get(p["key"]) or {}
        d, om = doi_of[k], old.get("measured") or {}
        row = {"key": k, "doi": d, "measured": {}}
        M = row["measured"]
        if w:
            src, ov = sources.get(w["source"]) or {}, (old.get("openalex") or {}).get("venue") or {}
            venue_type = w["source_type"] or src.get("type", "")
            venue = {"name": w["source_name"] or src.get("name", ""), "openalex": w["source"], "type": venue_type,
                     "issn_l": src.get("issn_l", ""), "publisher": src.get("publisher", "")}
            if venue_type in ML.VENUE_TYPES and w["source"] in sources:
                venue.update(mean_citedness_2yr=src.get("mean_citedness_2yr"), h_index=src.get("h_index"))
                M["venue"] = today
            elif venue_type in ML.VENUE_TYPES and w["source"] in src_failed:
                if ov.get("openalex") == w["source"] and "mean_citedness_2yr" in ov:   # the earlier answer + its date
                    venue.update({f: ov.get(f) for f in ("issn_l", "publisher", "mean_citedness_2yr", "h_index")})
                    M["venue"] = om.get("venue", "")
                else:
                    row["venue_unmeasured"] = True
            row["openalex"] = {"id": w["openalex"], "publication_date": w["publication_date"], "type": w["type"],
                               "cited_by": w["cited_by"], "cited_by_by_year": w["cited_by_by_year"], "refs": w["refs"],
                               "venue": venue}
            if w.get("copies"):
                row["openalex"]["copies"] = w["copies"]   # other records of this paper: a citation to one counts
            if w.get("title_sim", 1.0) < ML.TITLE_MATCH:
                row["openalex"]["title_sim"] = w["title_sim"]   # the one record answering, titled otherwise: check
            M["openalex"] = today
        elif k in ambiguous:
            row["openalex_ambiguous"] = ambiguous[k]      # several records, none titled like the paper: left blank
            M["openalex"] = today
        elif d and d in unchecked:                    # not asked this run: the earlier answer, with its own date
            if old.get("openalex") or old.get("openalex_ambiguous"):
                for f in ("openalex", "openalex_ambiguous", "venue_unmeasured"):
                    if old.get(f):
                        row[f] = old[f]
                M.update({f: om[f] for f in ("openalex", "venue") if om.get(f)})
            else:
                row["openalex_unmeasured"] = True
        elif d:
            M["openalex"] = today                     # asked and not found: a dated 'not in OpenAlex'
        sid = s2ids.get(k) or ""
        if sid and sid in s2v:
            row["s2"] = {"id": sid, **s2v[sid]}
            M["s2"] = today
        elif sid and (not s2 or sid in s2_failed):
            if old.get("s2") and old["s2"].get("id", sid) == sid:
                row["s2"], M["s2"] = old["s2"], om.get("s2", "")
            elif s2:
                row["s2_unmeasured"] = True
        elif sid:
            M["s2"] = today                           # asked and not found: a dated 'not in Semantic Scholar'
        r = repos.get(k)
        rk = f"{r[0]}/{r[1]}".lower() if r else ""
        if rk and rk in gh:
            row["github"] = {"repo": f"{r[0]}/{r[1]}", **gh[rk]}
            M["github"] = today
        elif rk and (not stars or rk in gh_failed):
            if (old.get("github") or {}).get("repo", "").lower() == rk:
                row["github"], M["github"] = old["github"], om.get("github", "")
            elif stars:
                row["github_unmeasured"] = True
        elif rk:
            M["github"] = today                       # GitHub: no such repository (NOT_FOUND), dated
        old_id = (old.get("openalex") or {}).get("id")
        if old.get("top_citing") is not None and old_id in (None, (row.get("openalex") or {}).get("id")):
            row["top_citing"] = old["top_citing"]      # the same work's list, with its own date
            if om.get("top_citing"):
                M["top_citing"] = om["top_citing"]
        rows[k] = row
    # the citing lists: papers WITHOUT one first (no date), then the oldest; after the first refusal none is asked
    want = [k for k in rows if citing and k in held and ((rows[k].get("openalex") or {}).get("cited_by") or 0) > 0]
    want.sort(key=lambda k: (rows[k]["measured"].get("top_citing", ""), k))
    open_ = bool(citing)
    for k in want:
        top = ML.top_citing(rows[k]["openalex"]["id"], get, n=citing) if open_ else None
        if top is None:
            open_ = False
            if rows[k].get("top_citing") is None:
                rows[k]["top_citing_unmeasured"] = True
            continue
        rows[k]["top_citing"] = top
        rows[k]["measured"]["top_citing"] = today
    graph = ML.corpus_graph({k: {"openalex": r["openalex"]["id"], "copies": r["openalex"].get("copies") or [],
                                 "refs": r["openalex"].get("refs") or []} for k, r in rows.items() if r.get("openalex")})
    for k, g in graph.items():
        rows[k]["refs_in_corpus"], rows[k]["cited_by_in_corpus"] = g["refs_in_corpus"], g["cited_by_in_corpus"]
    shared: Dict[str, List[str]] = {}                  # two index keys on one OpenAlex work: an alias not yet folded
    for k, r in sorted(rows.items()):
        if r.get("openalex"):
            shared.setdefault(r["openalex"]["id"], []).append(k)
    shared = {i: ks for i, ks in shared.items() if len(ks) > 1}
    out = sorted(rows.values(), key=lambda r: r["key"])
    flag = lambda f: sum(1 for r in out if r.get(f))
    summary = {"papers": len(out), "openalex": flag("openalex"),
               "openalex_measured_today": sum(1 for r in out if r["measured"].get("openalex") == today),
               "openalex_unmeasured": flag("openalex_unmeasured"), "openalex_ambiguous": flag("openalex_ambiguous"),
               "title_differs": sum(1 for r in out if "title_sim" in (r.get("openalex") or {})),
               "keys_sharing_a_work": sum(len(ks) for ks in shared.values()),
               "s2": flag("s2"), "s2_unmeasured": flag("s2_unmeasured"), "venues_with_metrics": sum(
                   1 for r in out if "mean_citedness_2yr" in ((r.get("openalex") or {}).get("venue") or {})),
               "venue_unmeasured": flag("venue_unmeasured"),
               "github_repos": flag("github"), "github_unmeasured": flag("github_unmeasured"),
               "in_corpus_citation_links": sum(len(g["refs_in_corpus"]) for g in graph.values()),
               "with_top_citing": sum(1 for r in out if r.get("top_citing") is not None),
               "top_citing_unmeasured": flag("top_citing_unmeasured")}
    doc = {"schema": "svfr.bwm.meta/2", "last_run": today, "summary": summary,
           "sources": "OpenAlex works + sources (2yr_mean_citedness: an open Impact-Factor analogue, journals / "
                      "conferences only; the JIF is Clarivate-licensed and not fetched), Semantic Scholar (citations, "
                      "influential), GitHub GraphQL; every block dated in `measured`",
           "papers": out}
    (base / "papers" / "_meta.json").write_text(_dumps_rows(doc, "papers"), encoding="utf-8", newline="\n")
    print(f"meta: {summary} -> papers/_meta.json (run {today}). Now run: build")
    for k in sorted(ambiguous):
        print(f"  AMBIGUOUS {k}: OpenAlex answers {', '.join(ambiguous[k])} for {doi_of[k]}, none titled like the paper")
    for i, ks in sorted(shared.items()):
        print(f"  ONE WORK {i}: index keys {', '.join(ks)} -- an alias to fold?")
    return 0


STATUS_SCHEMA = "svfr.bwm.status/2"
STATUS_SOURCES = ("arxiv_comments", "arxiv_abstract_pages", "openalex_dois")   # in the order a run applies them: the
                                                                                # later source's row wins a paper


def status_source(source: str) -> Tuple[Optional[str], str]:
    """A papers/_status.json row's `source` -> (the STATUS_SOURCES entry that found it, the arXiv id it names -- '' for
    an older 'arXiv comment' row and for OpenAlex); (None, '') for a source `status` never reads (added by hand)."""
    m = re.match(r"arXiv (comment|abstract page)(?:\s+(\S+))?\s*$", source)
    if m:
        return ("arxiv_comments" if m.group(1) == "comment" else "arxiv_abstract_pages"), norm_arxiv(m.group(2)) or ""
    return ("openalex_dois", "") if source.startswith("OpenAlex") else (None, "")


def status_checked(doc: dict) -> Dict[str, dict]:
    """papers/_status.json `checked` -> {source: {"n": papers read, "on": that run's date}} for every STATUS_SOURCES
    entry. Schema /1 wrote one `checked_on` for all three counts, and 0 for a source that run did not read: a /1 zero
    is no record ({"n": 0, "on": ""})."""
    got = doc.get("checked") if isinstance(doc.get("checked"), dict) else {}
    out = {}
    for s in STATUS_SOURCES:
        v = got.get(s)
        n = v.get("n") if isinstance(v, dict) else v
        if isinstance(n, int) and not isinstance(n, bool) and n > 0:
            out[s] = {"n": n, "on": str((v.get("on") if isinstance(v, dict) else doc.get("checked_on")) or "")}
        else:
            out[s] = {"n": 0, "on": ""}
    return out


def cmd_status(base: Path, cache: Path, abs_pages: bool = False, today: str = "") -> int:
    """Withdrawn / retracted papers, found by a script (owner 2026-10-13: we found 4 withdrawn arXiv papers by hand).
    arXiv: the authors' comments in the arXiv API answers `code --stage collect` cached (free; caught 2 of 4), and with
    --abs every arXiv abstract page (one request each, 15 s apart: arxiv.org/robots.txt sets Crawl-delay 15; cached;
    the notice caught 4 of 4). Journal papers:
    OpenAlex is_retracted for every DOI (~40 requests). -> committed papers/_status.json; `build` applies it.

    A re-run that can see LESS than the run that wrote the file never erases what that run found (2026-10-15: a code
    cache without a paper's arXiv answer dropped its comment-found withdrawal, and each source a run did not re-read
    was written as 0 -- 1410 abstract pages -> 0 in 972c0f7f). An earlier row is dropped only when its paper left the
    index, the arXiv id / DOI it was found through is not the paper's any more, or THIS run read the same evidence
    (that paper's arXiv entry, its abstract page, OpenAlex's answer for its DOI) and found nothing there; a row from a
    source `status` never reads (added by hand) stays. A paper both runs have evidence for keeps the later source's row
    (STATUS_SOURCES), so a run without --abs never swaps a page row for a comment row. `checked` holds per source the
    papers the LAST run that read it read, with its date; a source this run read nothing of keeps both. A file that
    does not parse is REFUSED (exit 2): read as empty, it was overwritten. The file stays BYTE-IDENTICAL when nothing
    but a date would change: a re-check that finds the same is not news, and on the paper-drop path every rewrite is a
    commit (c6d688f4's status change was a new date and two zeroed counts).
    HONEST SCOPE: a date is the last run that read its source AND wrote the file -- it can be OLDER than the latest
    check, never newer; a partial re-read (a cache holding fewer answers) writes its own smaller count: the rows found
    before stay, the count is what THIS run read."""
    import glob as _glob
    PL, CL = _sibling("paper_links"), _sibling("code_links")
    path = base / "papers" / "_status.json"
    try:                                               # read FIRST: never ask anything before knowing what to keep
        old = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    except (OSError, ValueError) as e:
        old = e
    if not isinstance(old, dict) or not isinstance(old.get("papers") or [], list):
        why_not = old if isinstance(old, Exception) else "not an object with a list of papers"
        print(f"status: REFUSED -- papers/_status.json cannot be read ({why_not}); running over it would erase what "
              f"it holds. Fix it or restore it (git checkout), then run again.")
        return 2
    papers, alias_map, _ = build_papers(base)
    ax = {re.sub(r"v\d+$", "", p["arxiv_id"]): p["key"] for p in papers if p.get("arxiv_id")}
    rows: Dict[str, dict] = {}
    read: Dict[str, set] = {s: set() for s in STATUS_SOURCES}   # the arXiv ids / DOIs whose evidence THIS run read
    for f in _glob.glob(str(Path(cache) / "http" / "*.json")):
        d = _load(Path(f), {})
        if "export.arxiv.org" not in (d.get("url") or "") or not d.get("body"):
            continue
        for e in re.findall(r"<entry>(.*?)</entry>", d["body"], re.S):
            m = re.search(r"<id>https?://arxiv\.org/abs/([^<]+?)(v\d+)?</id>", e)
            if not m or m.group(1) not in ax:
                continue
            read["arxiv_comments"].add(m.group(1))   # a paper once: the lab's cache holds 2 answers for each (10-15)
            c = PL.withdrawn_comment(e)
            if c:
                rows[ax[m.group(1)]] = {"status": "withdrawn", "source": f"arXiv comment {m.group(1)}",
                                        "evidence": c[:200]}   # the id: `build` checks it is still the paper's
    if abs_pages:
        net = CL.Net(cache, pause=15.0)              # arxiv.org/robots.txt: Crawl-delay 15
        for aid, key in sorted(ax.items()):
            body = net.get(f"https://arxiv.org/abs/{aid}")
            if body is None:
                continue
            read["arxiv_abstract_pages"].add(aid)
            n = PL.withdrawn_notice(body)
            if n:
                rows[key] = {"status": "withdrawn", "source": f"arXiv abstract page {aid}", "evidence": n[:200]}
    dois = {p["doi"].lower(): p["key"] for p in papers if p.get("doi") and not p["doi"].lower().startswith("10.48550")}
    flagged, looked, why = PL.openalex_retracted(dois)
    read["openalex_dois"].update(looked)
    for d, title in flagged.items():
        if d in dois:
            rows[dois[d]] = {"status": "retracted", "source": "OpenAlex is_retracted (Retraction Watch)",
                             "evidence": f"doi:{d} {title[:120]}"}
    keys, own_ax, own_doi = {p["key"] for p in papers}, {k: i for i, k in ax.items()}, {k: d for d, k in dois.items()}
    rank = lambda src: STATUS_SOURCES.index(src) if src else -1
    for r in old.get("papers") or []:                  # MERGE: an earlier row stays unless this run could judge it
        if not isinstance(r, dict) or not r.get("key"):
            continue
        k = alias_map.get(r["key"], r["key"])
        src, rid = status_source(str(r.get("source", "")))
        if k not in keys:
            continue                                   # the paper left the index
        if src in ("arxiv_comments", "arxiv_abstract_pages"):
            owner = ax.get(rid) if rid else (k if k in own_ax else None)   # an older comment row names no id
            if owner != k:
                continue                               # found through an id that is not THIS paper's (any more):
            if own_ax[k] in read[src]:                 # rejected, another's (third fix check 2026-10-15), or none
                continue                               # this run READ that entry / page: what it found decides
        elif src == "openalex_dois":
            m = re.match(r"doi:(\S+)", str(r.get("evidence", "")))   # the DOI it was found through
            if m and dois.get(m.group(1).lower()) != k:
                continue                               # not THIS paper's DOI (any more)
            if own_doi.get(k) in read[src]:
                continue                               # OpenAlex answered for the paper's DOI this run
        if k not in rows or rank(src) > rank(status_source(str(rows[k].get("source", "")))[0]):
            rows[k] = {x: r[x] for x in ("status", "source", "evidence") if x in r}
    was = status_checked(old)
    checked = {s: {"n": len(read[s]), "on": today} if read[s] else was[s] for s in STATUS_SOURCES}
    doc = {"schema": STATUS_SCHEMA, "checked": checked, "papers": [{"key": k, **rows[k]} for k in sorted(rows)]}
    undated = dict(doc, checked={s: dict(checked[s], on=was[s]["on"]) for s in STATUS_SOURCES})
    same = path.is_file() and path.read_bytes() == _dumps_rows(undated, "papers").encode("utf-8")
    if not same:
        path.write_text(_dumps_rows(doc, "papers"), encoding="utf-8", newline="\n")
    print(f"status: {sum(r.get('status') == 'withdrawn' for r in rows.values())} withdrawn, "
          f"{sum(r.get('status') == 'retracted' for r in rows.values())} retracted -> papers/_status.json "
          + ("UNCHANGED (only a date would move: left as it was)" if same else "written")
          + "; read this run: " + ", ".join(f"{s} {len(read[s])}" for s in STATUS_SOURCES)
          + ("" if abs_pages else " (add --abs to read every arXiv abstract page: the authoritative withdrawal notice)"))
    for s in STATUS_SOURCES:
        if not read[s] and was[s]["n"]:
            print(f"  {s}: nothing read here -- kept {was[s]['n']} from {was[s]['on'] or 'an undated run'}")
    for k in sorted(rows):
        print(f"  {k}: {rows[k].get('status')} -- {rows[k].get('source')}: {str(rows[k].get('evidence', ''))[:110]}")
    if why:
        print("  NOTE: " + why)
    return 1 if why else 0


def cmd_confirm(base: Path, pdf_dir: Path, keys: Optional[List[str]] = None, report: Optional[Path] = None) -> int:
    """Is every held PDF the paper the index says? (owner 2026-10-14: three ingests, each checked by a throwaway
    script -- now a tool.) Reads each held file's first pages from the text cache (`tools/paper_text.py build`, OCR
    pages included) and asks paper_links.identity_evidence: the index title as whole lines of the TITLE BLOCK, the DOI
    / arXiv id. Prints the files to READ ('doi only', 'weak', 'notice', 'sibling', 'supplement', 'no text', not in
    the text cache) and every --keys entry that is not held; exit 1 while any is left."""
    PL, PT = _sibling("paper_links"), _tool("paper_text")
    cache = PT.default_cache(pdf_dir)
    man = PT.read_manifest(cache)
    papers = {p["key"]: p for p in _load(base / "index" / "papers.json", {}).get("papers") or []}
    held = [r for r in (_load(base / "papers" / "_held.json", {}).get("files") or []) if isinstance(r, dict)]
    missing = []
    if keys:
        missing = sorted(set(keys) - {r.get("key") for r in held})   # a typo or a paper not held: never 'nothing to read'
        held = [r for r in held if r.get("key") in set(keys)]
    words = {k: " " + " ".join(PL._id_streams(PL._clean_title(p.get("title", "")))[0]) + " "
             for k, p in papers.items()}
    longer = {k: [(o, papers[o].get("title", "")) for o, ow in words.items()   # other index titles HOLDING this one
                  if o != k and len(ow) > len(w) > 2 and w in ow] for k, w in words.items()}
    surnames = frozenset(w[-1] for p in papers.values() for a in p.get("authors") or []   # 'Chen'+'g' = 'Cheng'
                         for w in [PL._id_streams(str(a))[0]] if w)
    rows = []
    for r in sorted(held, key=lambda r: r["key"]):
        p = papers.get(r["key"]) or {}
        cf = PT.cache_file(cache, r.get("sha256", ""))
        if r.get("sha256") not in man or not cf.is_file():
            ev = {"verdict": "not in text cache", "how": "", "sim": 0.0, "n": 0, "doi": False, "arxiv": False}
        else:
            ev = PL.identity_evidence(p.get("title", ""), p.get("doi") or "", p.get("arxiv_id") or "",
                                      cf.read_text(encoding="utf-8").split(PT.FF), authors=p.get("authors") or [],
                                      longer=longer.get(r["key"], ()), surnames=surnames)
        rows.append({"key": r["key"], "file": r.get("file", ""), **ev, "title": p.get("title", "")})
    counts = collections.Counter(x["verdict"] for x in rows)
    print(f"confirm: {len(rows)} held PDF(s) -- " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    read = [x for x in rows if not x["verdict"].startswith("title")]
    for k in missing:
        print(f"  [not held] {k}: no held PDF under this key (not in the index, or not held here) -- nothing checked")
    for x in read:
        print(f"  [{x['verdict']}] {x['key']}: {x['file']} -- title-block similarity {x['sim']:.2f} ({x['n']} title "
              f"words){', DOI printed' if x['doi'] else ''}{', arXiv id printed' if x['arxiv'] else ''}"
              + (f"; the longer title of {x['sibling']} matches too" if x.get("sibling") else ""))
    if report:
        cols = ["key", "verdict", "how", "sim", "n", "doi", "arxiv", "file", "title"]
        Path(report).write_text("\t".join(cols) + "\n" + "".join(
            "\t".join(str(x[c]).replace("\t", " ") for c in cols) + "\n" for x in rows), encoding="utf-8")
    return 1 if read or missing else 0


def cmd_audit(base: Path, folder: Optional[Path], pdf_dir: Path, report: Optional[Path] = None) -> int:
    """Is every file in a download folder saved elsewhere (papers folder by content, or `obtain`ed data)? Read-only;
    exit 1 while anything is NOT saved, so "can I delete my downloads?" has a yes/no answer."""
    if folder is None or not Path(folder).is_dir():
        print(f"audit: give the download folder to check, e.g. audit \"C:/Users/<you>/Downloads/more papers\" "
              f"(got {folder})")
        return 2
    if _refuse_unreadable_rejections(base):
        return 2
    PL = _sibling("paper_links")
    papers, alias_map, _ = build_papers(base)
    held = (_load(base / "papers" / "_held.json", {}).get("files") or [])
    ob = (_load(base / "papers" / "_obtained.json", {}).get("items") or [])
    rows = PL.audit_folder(folder, pdf_dir, papers, alias_map, held, identity=_paper_pdfs_module().pdf_identity,
                           page_text=PL.page1_text, data_dir=data_root(pdf_dir), obtained=ob,
                           rejected=rejected_ids(base, alias_map)[0])
    counts = collections.Counter(r["status"] for r in rows)
    print(f"audit of {folder}: {len(rows)} file(s) -- " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    for r in rows:
        if r["status"] != "saved":
            print(f"  [{r['status']}] {r['file']}" + (f" -> {r['where']}" if r.get("where") else "")
                  + (f": {r['why']}" if r.get("why") else "") + (f" (closest: {', '.join(r['hints'])})"
                                                                 if r.get("hints") else ""))
    lost = [r for r in rows if r["status"] == "NOT saved"]
    if report:
        Path(report).write_text("file\tstatus\twhere\tkey\tbytes\twhy\n" + "".join(
            f"{r['file']}\t{r['status']}\t{r.get('where', '')}\t{r.get('key', '')}\t{r['bytes']}\t{r.get('why', '')}\n"
            for r in rows), encoding="utf-8", newline="\n")
        print(f"  every file's row: {report}")
    if lost:
        print(f"{len(lost)} file(s), {sum(r['bytes'] for r in lost) / 1e9:.2f} GB, exist ONLY in {folder} -- keep it, "
              f"or `add` / `obtain` them first")
        return 1
    print(f"every file in {folder} is saved elsewhere (the papers folder {pdf_dir} or {data_root(pdf_dir)}); "
          f"deleting it loses nothing the index holds")
    return 0


def cmd_obtain(base: Path, src: Optional[Path], pdf_dir: Path, key: str, only: str = "*", kind: str = "dataset",
               note: str = "", today: str = "") -> int:
    """Keep a NON-PDF download that belongs to an indexed work (a data release, a dataset archive) beside the papers
    folder, `<pdfs>_data/<key>/`, verified by sha256 and recorded in committed papers/_obtained.json -- the work then
    reads 'obtained' (atlas: '<kind> held') and leaves the download list. The source is only read."""
    import fnmatch
    if src is None or not Path(src).exists():
        print(f"obtain: give the file or folder to keep, with --key KEY (got {src})")
        return 2
    papers, alias_map, _ = build_papers(base)
    k = alias_map.get(key or "", key or "")
    if k not in {p["key"] for p in papers}:
        print(f"obtain: --key {key!r} is no paper in the index -- nothing was copied")
        return 2
    src = Path(src)
    files = [src] if src.is_file() else sorted((f for f in src.iterdir() if f.is_file() and fnmatch.fnmatch(f.name, only)),
                                               key=lambda f: f.name.lower())
    if not files:
        print(f"obtain: no file in {src} matches {only!r}")
        return 2
    PL = _sibling("paper_links")
    dest = data_root(pdf_dir) / k
    need = sum(f.stat().st_size for f in files)
    free = shutil.disk_usage(Path(pdf_dir).parent).free
    if free < need + 1e9:
        print(f"obtain: {need / 1e9:.2f} GB to copy and only {free / 1e9:.2f} GB free (1 GB kept) -- nothing was copied")
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    rows, refused = [], []
    for f in files:
        sha = PL._sha256(f)
        target = dest / f.name
        if target.exists() and PL._sha256(target) != sha:
            refused.append(f"{f.name}: {target} exists with different bytes -- not overwritten")
            continue
        if not target.exists():
            tmp = target.with_name(target.name + ".part")
            shutil.copyfile(f, tmp)
            if PL._sha256(tmp) != sha:
                tmp.unlink(missing_ok=True)
                refused.append(f"{f.name}: the copy did not match the source -- not kept")
                continue
            replace_retry(tmp, target)
        rows.append({"name": f.name, "bytes": f.stat().st_size, "sha256": sha})
    path = base / "papers" / "_obtained.json"
    doc = _load(path, {}) if path.is_file() else {}
    doc = {"schema": "svfr.bwm.obtained/1", **doc}
    items = {r["key"]: r for r in doc.get("items") or [] if isinstance(r, dict) and r.get("key")}
    item = items.get(k) or {"key": k, "kind": kind, "files": [], "date": today, "note": note}
    have = {r["sha256"] for r in item["files"]}
    item["files"] = sorted(item["files"] + [r for r in rows if r["sha256"] not in have], key=lambda r: r["name"])
    item["kind"], item["note"] = kind or item.get("kind", "dataset"), note or item.get("note", "")
    items[k] = item
    doc["items"] = [items[x] for x in sorted(items)]
    path.write_text(_dumps_rows(doc, "items"), encoding="utf-8", newline="\n")
    print(f"obtain: {len(rows)} file(s), {sum(r['bytes'] for r in rows) / 1e9:.2f} GB kept in {dest} for {k} "
          f"({item['kind']}) -> papers/_obtained.json. Now run: build")
    for r in refused:
        print(f"  ! {r}")
    return 1 if refused else 0


def cmd_add(base: Path, inbox: Optional[Path], pdf_dir: Path, as_: List[str], rebuild: bool = True) -> int:
    """New PDFs from a download folder -> the papers folder as Surname_Year_Title.pdf, then scan + build: the owner's
    whole "I downloaded some papers" step. The download folder is only read."""
    if inbox is None or not inbox.is_dir():
        print(f"add: give the folder the new PDFs are in, e.g. add \"C:/Users/<you>/Downloads/more papers\" "
              f"(got {inbox})")
        return 2
    if pdf_dir.exists() and inbox.resolve() == pdf_dir.resolve():
        print(f"add: {inbox} IS the papers folder (--pdfs) -- give the folder the new downloads are in; to re-match "
              f"the papers folder itself, run scan")
        return 2
    if _refuse_unreadable_rejections(base):
        return 2
    overrides = {}
    for spec in as_:
        name, sep, key = spec.rpartition("=")
        if not sep or not name or not key:
            print(f"add: --as wants FILE=KEY (got {spec!r})")
            return 2
        overrides[name] = key
    PL = _sibling("paper_links")
    papers, alias_map, _ = build_papers(base)
    held_doc = _load(base / "papers" / "_held.json", {})
    held_here = {alias_map.get(r["key"], r["key"]) for r in (held_doc.get("files") or [])
                 if isinstance(r, dict) and r.get("key") and r.get("file") and (pdf_dir / r["file"]).is_file()}
    by_key = {p["key"]: p for p in papers}
    held_shas = {r["sha256"]: alias_map.get(r["key"], r["key"]) for r in (held_doc.get("files") or [])
                 if isinstance(r, dict) and r.get("sha256") and r.get("key")}
    rep = PL.add_pdfs(inbox, pdf_dir, papers, alias_map, held_here, identity=_paper_pdfs_module().pdf_identity,
                      overrides=overrides, page_text=PL.page1_text, name_for=lambda k: paper_file_name(by_key[k]),
                      held_shas=held_shas, rejected=rejected_ids(base, alias_map)[0],
                      recorded=held_doc.get("files") or [])   # bytes the record holds keep its row (paper_links)
    if rep["bad_overrides"]:
        for name in rep["bad_overrides"]:
            print(f"add: --as names {name!r}, which is no PDF in {inbox} -- nothing was copied")
        return 2
    changed = bool(rep["added"]) and rebuild
    if changed:                                      # the record first: a listing problem can never leave it stale
        cmd_scan(base, pdf_dir, extra=rep.get("rows"))
        main(["build", "--base", str(base)])
    print(f"add: {len(rep['added'])} copied into {pdf_dir}, {len(rep['already_held'])} already held, "
          f"{len(rep['unmatched'])} not identified, {len(rep['refused'])} refused (the download folder is untouched)")
    for name, key, how in rep["added"]:
        print(f"  + {rep.get('names', {}).get(key, key + '.pdf')}  <- {name}  ({how})")
    for name, key in rep["already_held"]:
        print(f"  = {name}: {key} is already in {pdf_dir}")
    for name, hints in rep["unmatched"]:
        hint = ", ".join(f"{k} ({r})" for r, k in hints) or "no close title in the index"
        print(f"  ? {name}: not identified -- best title matches: {hint}. If you know it: --as \"{name}=<key>\"")
    for name, why in rep["refused"]:
        print(f"  ! {name}: {why}")
    if rep["added"] and not rebuild:                 # the identities are RECORDED even so: the later `scan` knows
        path = base / "papers" / "_held.json"        # these files by sha256 (an --as name or a title match is not
        doc = _load(path, {}) if path.is_file() else {"files": []}   # recoverable from a title-named file alone)
        new = list(rep.get("rows") or [])
        shas, keys = {r["sha256"] for r in new}, {r["key"] for r in new}   # one row per paper: the copy made here
        doc["files"] = sorted([r for r in doc.get("files") or [] if r.get("sha256") not in shas
                               and r.get("key") not in keys] + new, key=lambda r: (r["key"], r["file"]))
        path.write_text(_dumps_rows(doc, "files"), encoding="utf-8", newline="\n")
        print("  (--no-build: recorded in papers/_held.json; run `scan` and `build` before relying on the download "
              "list)")
    return 0 if not (rep["unmatched"] or rep["refused"]) else 1


def _code_doc(base: Path) -> Dict[str, dict]:
    return {r["key"]: r for r in (_load(base / "papers" / "_code.json", {}).get("papers") or [])}


def _write_code(base: Path, rows: Dict[str, dict]) -> dict:
    CL = _sibling("code_links")
    recs = CL.record(rows) if rows and "facts" in next(iter(rows.values()), {}) else [rows[k] for k in sorted(rows)]
    summary = {"papers_with_code": sum(r["status"] in CL.COUNTED for r in recs),
               "by_status": dict(sorted(collections.Counter(r["status"] for r in recs).items())),
               "by_source": dict(sorted(collections.Counter(r["source"] for r in recs
                                                            if r["status"] in CL.COUNTED).items()))}
    (base / "papers" / "_code.json").write_text(_dumps_rows({"schema": "svfr.bwm.code/1", "summary": summary,
                                                              "papers": recs}, "papers"), encoding="utf-8", newline="\n")
    return summary


def merge_verified(have: Dict[str, dict], checked: Dict[str, dict], counted) -> Dict[str, dict]:
    """A verify run's rows over the previous _code.json: earlier SEARCH finds are kept (verify does not redo them),
    and a counted search find is never replaced by an uncounted stated link (dead / README silent). Pure."""
    rows = {k: r for k, r in have.items() if r.get("status") == "CANDIDATE_BY_SEARCH"}
    for k, new in checked.items():
        if k in rows and new.get("status") not in counted:
            continue
        rows[k] = new
    return rows


def cmd_code(base: Path, stage: str, cache: Path, relevance: List[str], strategies: str, limit) -> int:
    """Find each paper's code (code_links.py): collect what the authors state, check it, search for the rest."""
    CL = _sibling("code_links")
    VR = _tool("verify_repos")
    VR.CACHE["dir"] = str(Path(cache) / "gh")
    papers, alias_map, _ = build_papers(base)
    facts = facts_rows(base, alias_map)
    cand_file = Path(cache) / "candidates.json"
    if stage == "collect":
        Path(cache).mkdir(parents=True, exist_ok=True)
        cands = CL.collect(papers, facts, CL.Net(cache, pause=0.3))
        cand_file.write_text(json.dumps(cands, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
        by = collections.Counter(c["source"] for cs in cands.values() for c in cs)
        print(f"code collect: {len(cands)} papers with a stated link ({dict(by)}) -> {cand_file}. Next: --stage verify")
        return 0
    if stage == "verify":
        if not cand_file.is_file():
            print(f"no {cand_file}: run `code --stage collect` first")
            return 2
        cands = json.loads(cand_file.read_text(encoding="utf-8"))
        checked = CL.verify(papers, cands, VR)
        rows = merge_verified(_code_doc(base), {k: CL.record({k: v})[0] for k, v in checked.items()}, CL.COUNTED)
        summary = _write_code(base, rows)
        print(f"code verify: {summary['papers_with_code']} papers with code -> papers/_code.json; {summary}")
        return 0
    have = _code_doc(base)
    VR_strats = VR.parse_strategies(strategies)

    def save(found):                                     # checkpoint: a stopped run keeps what it found
        rows = dict(have)
        rows.update({k: CL.record({k: v})[0] for k, v in found.items()})
        return _write_code(base, rows)

    found = CL.search(papers, have, VR, relevance=tuple(relevance), strategies=VR_strats, limit=limit, save=save)
    summary = save(found)
    print(f"code search ({','.join(relevance)}; {','.join(VR_strats)}): {len(found)} more found; "
          f"{summary['papers_with_code']} papers with code; {summary}")
    return 0


FIG_CAPTION_CHARS = 160                  # a caption in index/atlas_figures.js: cut at a word boundary (the file < 5 MB)
#: how each tools/figure_bank.py arm finds a figure: the sidecar's note names the arm(s) its bank was built with
#: (2026-10-15: a kit project's heuristic bank was described as DocLayout-YOLO boxes)
FIG_ARM_NOTE = {"yolo": "DocLayout-YOLO boxes", "heuristic": "caption-anchored boxes, no model"}


def caption_digest(text: str, limit: int = FIG_CAPTION_CHARS) -> str:
    """A figure caption for the atlas: whitespace collapsed, cut at the last word boundary before `limit` (+ '...')."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(t) <= limit:
        return t
    cut = t.rfind(" ", 0, limit)
    return t[:cut if cut > limit // 2 else limit].rstrip(" ,;:") + "..."


def figure_image_name(i: int, number: str, page: int) -> str:
    """The figure bank's image name for the i-th (1-based) figure of a paper (tools/figure_bank.py build_all)."""
    return f"fig{i:03d}-{number or 'u'}-p{page}.webp"


def figures_digest(base: Path, store: Path) -> dict:
    """The figure bank (tools/figure_bank.py, `<pdfs>_figures/<sha[:2]>/<sha>/figures.json`) reduced to what the atlas's
    Figures tab searches: per HELD, not-rejected paper its key, its PDF's sha256 (the image folder) and per figure
    [number, page, caption digest] (+ the image name when it is not the bank's usual one)."""
    papers, alias_map, _ = build_papers(base)
    rej = rejected_ids(base, alias_map)[0]
    held: Dict[str, str] = {}
    for r in (_load(base / "papers" / "_held.json", {}).get("files") or []):
        if not (isinstance(r, dict) and r.get("key") and r.get("sha256")):
            continue
        k, sha = alias_map.get(r["key"], r["key"]), str(r["sha256"]).strip().lower()
        if k not in rej.get(sha, ()):
            held[sha] = k
    rows, n, cap, no_bank, arms = [], 0, 0, 0, set()
    for sha, key in sorted(held.items(), key=lambda kv: (kv[1], kv[0])):
        f = Path(store) / sha[:2] / sha / "figures.json"
        if not f.is_file():
            no_bank += 1
            continue
        rec = _load(f, {})
        arms.add(str(rec.get("arm") or "yolo"))               # banks from before the arm was recorded: YOLO
        figs = []
        for i, x in enumerate(rec.get("figures") or [], 1):
            num, page = str(x.get("number") or ""), int(x.get("page") or 0)
            row = [num, page, caption_digest(x.get("caption") or "")]
            if x.get("image") and x["image"] != figure_image_name(i, num, page):
                row.append(x["image"])
            figs.append(row)
            n += 1
            cap += bool(row[2])
        if figs:
            rows.append([key, sha, figs])
    how = " + ".join(FIG_ARM_NOTE.get(a, a) for a in sorted(arms)) or FIG_ARM_NOTE["yolo"]
    return {"schema": "svfr.bwm.figures/1", "base": "figures/", "papers": rows, "figures": n, "captioned": cap,
            "papers_without_bank": no_bank,
            "note": f"tools/figure_bank.py ({how}, captions from the PDF's text layer); images are in the "
                    "portable bundle (build_bwm.py bundle), not in git"}


def cmd_figures(base: Path, pdf_dir: Path) -> int:
    """`figures`: write index/atlas_figures.js from the figure bank beside the PDF folder (run where the bank is)."""
    store = _tool("figure_bank").default_store(pdf_dir)
    if not store.is_dir():
        print(f"no figure bank at {store} (tools/figure_bank.py build --bwm makes it; pass --pdfs <papers folder>)")
        return 2
    doc = figures_digest(base, store)
    text = sidecar_js("__MWA_FIGS", doc)
    out = base / "index" / FIGURES_JS
    out.parent.mkdir(parents=True, exist_ok=True)             # a base no `build` has run in yet
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}: {doc['figures']} figures ({doc['captioned']} captioned) of {len(doc['papers'])} papers, "
          f"{len(text) / 1e6:.2f} MB; {doc['papers_without_bank']} held papers have no bank entry")
    return 0 if len(text) < 5_000_000 else 1


def cmd_links(base: Path, source: str, today: str) -> int:
    if _refuse_unreadable_rejections(base):
        return 2
    PL = _sibling("paper_links")
    papers, alias_map, _ = build_papers(base)
    apply_holdings(base, papers, alias_map)
    todo = PL.to_look_up(papers, [p["key"] for p in papers if p["pdf_held"]])
    path = base / "papers" / "_links.json"
    doc = {"schema": "svfr.bwm.links/1", "sources": {}, "papers": []}
    if path.is_file():                               # strict: a merge-conflicted / truncated file is never overwritten
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"{path} is not valid JSON ({e}): fix or delete it first -- nothing was looked up")
            return 2
    print(f"links: {len(todo)} papers not held, not on arXiv, with a DOI")
    notes = []
    _, rej_ax, _ = rejected_ids(base, alias_map)    # rows stored before an id was rejected are cleaned too
    doc["papers"] = [PL.scrub_link(r, rej_ax.get(r.get("key"), set())) for r in doc.get("papers") or []]
    for src in (["s2", "openalex"] if source == "all" else [source]):
        found, looked, why = (PL.s2_links(todo) if src == "s2" else PL.openalex_links(todo))
        found = {k: (PL.scrub_link({src: v}, rej_ax.get(k, set())) or {}).get(src, v) for k, v in found.items()}
        if looked:                                   # a source refused outright leaves the file as it was
            doc = PL.merge_links(doc, src, found, looked, today)
            path.write_text(_dumps_rows(doc, "papers"), encoding="utf-8", newline="\n")   # saved per source
        if why:
            notes.append(why)
    for n in notes:
        print("  NOTE: " + n)
    print(f"-> {path.relative_to(base)} ({len(doc.get('papers') or [])} papers with at least one answer). "
          f"Now run: build")
    return 1 if notes else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "check", "scan", "links", "bundle", "add", "code", "audit", "obtain", "meta",
                                    "status", "confirm", "citemap", "figures"])
    ap.add_argument("--iters", type=int, default=700, help="citemap: layout iterations")
    ap.add_argument("--if-changed", action="store_true",
                    help="citemap: keep the committed layout and figures when the citation graph is the one it lays out")
    ap.add_argument("--abs", action="store_true", help="status: read every arXiv abstract page (cached, 15 s apart)")
    ap.add_argument("--citing", type=int, default=5, help="meta: the N most-cited works citing each held paper (0: none)")
    ap.add_argument("--no-stars", action="store_true", help="meta: skip GitHub (it needs a `gh` login)")
    ap.add_argument("--no-s2", action="store_true", help="meta: skip Semantic Scholar")
    ap.add_argument("inbox", nargs="?", type=Path, default=None,
                    help="add / audit: the download folder; obtain: the file or folder to keep")
    ap.add_argument("--as", dest="as_", action="append", default=[], metavar="FILE=KEY",
                    help="add: identify a file the matcher cannot (repeatable)")
    ap.add_argument("--no-build", action="store_true", help="add: copy only; skip the scan + build")
    ap.add_argument("--prune", action="store_true", help="scan: drop held papers whose file is not in --pdfs")
    ap.add_argument("--stage", choices=["collect", "verify", "search"], default="verify",
                    help="code: collect stated links / check them / GitHub-search the rest")
    ap.add_argument("--relevance", default="core,high", help="code search: which relevance levels (comma list)")
    ap.add_argument("--strategies", default="title,id,doi",
                    help="code search: verify_repos strategies (comma list / all); title finds the most (measured)")
    ap.add_argument("--limit", type=int, default=0, help="code search: at most this many papers this run")
    ap.add_argument("--code-cache", type=Path, default=None,
                    help="code: where network answers are cached (re-runs ask nothing twice)")
    ap.add_argument("--base", type=Path, default=HERE)
    ap.add_argument("--pdfs", type=Path, default=None,
                    help="scan / bundle / add: the folder of paper PDFs (default: the project's, tools/atlas_project.py)")
    ap.add_argument("--source", choices=["s2", "openalex", "all"], default="all", help="links: which index to ask")
    ap.add_argument("--out", type=Path, default=None, help="bundle: the .zip to write (default: next to --pdfs)")
    ap.add_argument("--no-pdfs", action="store_true", help="bundle: the atlas and reports only (a few MB)")
    ap.add_argument("--no-figures", action="store_true",
                    help="bundle: leave out the figure images (the Figures tab then shows captions only)")
    ap.add_argument("--report", type=Path, default=None, help="audit / confirm: write every file's row to this TSV")
    ap.add_argument("--keys", default="", help="confirm: only these index keys (comma list)")
    ap.add_argument("--key", default="", help="obtain: the indexed work the files belong to")
    ap.add_argument("--only", default="*", help="obtain: from a folder, only the files matching this glob")
    ap.add_argument("--kind", default="dataset", help="obtain: what the files are (shown as '<kind> held')")
    ap.add_argument("--note", default="", help="obtain: a note kept with the record")
    a = ap.parse_args(argv)
    if a.inbox is not None and a.cmd not in ("add", "audit", "obtain"):
        ap.error(f"only add / audit / obtain take a path argument (got {str(a.inbox)!r} with {a.cmd}); scan / bundle "
                 f"use --pdfs")
    for stream in (sys.stdout, sys.stderr):      # a piped cp1252 console must not crash on a Unicode file name
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    today = datetime.date.today().isoformat()
    if a.pdfs is None:                                 # the project's own folder (atlas.toml; BlueBoat: as before)
        a.pdfs = project_pdfs(a.base)
    pdfs_here = a.pdfs
    if a.code_cache is None:                           # the network-answer cache sits beside the project's PDFs
        a.code_cache = _tool("atlas_project").beside(project_pdfs(a.base), "code_cache")   # (BlueBoat laptop: as before)
    if a.cmd == "citemap":                             # layout (an INPUT: papers/_citemap.json), then the figures
        CM = _sibling("citation_map")
        if a.if_changed:                               # tools/atlas.py update: a rerun must not churn the layout
            papers_cm, meta_cm = CM.inputs(a.base)     # (floats differ between machines -- the module docstring)
            graph = CM.signature(CM.edges_of(meta_cm, list(papers_cm)))
            have = CM.load_citemap(a.base)
            if have.get("graph") == graph and (a.base / "figures" / CM.METRICS).is_file():
                print(f"citemap: the citation graph is the one papers/{CM.CITEMAP} lays out ({graph}, made "
                      f"{have.get('made') or '?'}) -- layout and figures kept")
                return 0
        CM.write_citemap(a.base, today, a.iters)
        print(f"wrote {CM.render(HERE.parents[1], a.base)} (+ topic flows, backbone, ages; figures/{CM.METRICS})")
        print("next: build_bwm.py build (the atlas's Citations tab reads the new layout)")
        return 0
    if a.cmd == "figures":
        return cmd_figures(a.base, pdfs_here)
    if a.cmd == "scan":
        return cmd_scan(a.base, a.pdfs, prune=a.prune)
    if a.cmd == "code":
        return cmd_code(a.base, a.stage, a.code_cache, [x for x in a.relevance.split(",") if x],
                        a.strategies, a.limit or None)
    if a.cmd == "add":
        return cmd_add(a.base, a.inbox, a.pdfs, a.as_, rebuild=not a.no_build)
    if a.cmd == "status":
        return cmd_status(a.base, a.code_cache, a.abs, today)
    if a.cmd == "meta":
        return cmd_meta(a.base, a.code_cache, a.citing, not a.no_stars, not a.no_s2, today)
    if a.cmd == "audit":
        return cmd_audit(a.base, a.inbox, a.pdfs, a.report)
    if a.cmd == "confirm":
        return cmd_confirm(a.base, a.pdfs, [k for k in a.keys.split(",") if k], a.report)
    if a.cmd == "obtain":
        return cmd_obtain(a.base, a.inbox, a.pdfs, a.key, a.only, a.kind, a.note, today)
    if a.cmd == "links":
        return cmd_links(a.base, a.source, today)
    if a.cmd == "bundle":                               # the PDF folder: --pdfs, else where paper_text finds it
        names = project_names(a.base)
        out = a.out or pdfs_here.parent / f"{names['short_name']}_{today}{'_no_pdfs' if a.no_pdfs else ''}.zip"
        return _sibling("bundle_atlas").main_bundle(a.base, pdfs_here, out, sys.modules[__name__],
                                                     include_pdfs=not a.no_pdfs, date=today,
                                                     include_figures=not a.no_figures, names=names)
    outs = build_all(a.base)
    out_dir = a.base / "index"
    # a section page (`atlases` mode) of a section since removed, emptied or switched to `banners`
    gone = sorted(f.name for f in out_dir.glob("atlas_*") if SECTION_FILE.match(f.name) and f.name not in outs
                  and f.name not in ("atlas_citations.js", FIGURES_JS)) if out_dir.is_dir() else []
    if a.cmd == "build":
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, text in outs.items():
            (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
        for name in gone:
            (out_dir / name).unlink()
            print(f"removed index/{name} (no such section page any more)")
        office = office_outputs(a.base, outs)
        lack = _sibling("office").word_missing()       # no python-docx: no Word file in pandoc's stock styles
        why = {"pandoc": "no pandoc here", "python-docx": "no python-docx in this Python -- `pip install python-docx` "
               "(requirements.txt); without it pandoc's stock styles would replace the house styles"}.get(
            lack, "pandoc or python-docx failed")
        for name in OFFICE_NAMES:
            if office[name] is None:
                print(f"NOTE: {name} not written ({why})")
                if (out_dir / name).is_file() and \
                        _sibling("office").docx_stamp(out_dir / name) != office["_stamps"][name]:
                    print(f"PROBLEM: index/{name} was not built from this Markdown and nothing here can rebuild it: "
                          "`check` (so the pre-push hook) calls it STALE until `build` runs where pandoc and "
                          "python-docx are installed")
            elif not same_zip_payload(out_dir / name, office[name]):    # only a changed part rewrites the file
                (out_dir / name).write_bytes(office[name])
        st = json.loads(outs["papers.json"])["stats"]
        print(f"wrote {len(outs)} files: {st['papers']} papers ({st['published']} published, {st['preprints']} "
              f"preprints; {st['pdfs_held']} PDFs held, {st['to_download_open']} more free, "
              f"{st['to_download_library']} need the library), {st['terms']} terms / {st['citations']} citations, "
              f"{st['labs']} labs, {st['web_sources']} web sources")
        P = json.loads(outs["papers.json"])
        for b in rejected_problems(a.base, [p["key"] for p in P["papers"]], P["aliases"]):
            print("PROBLEM: " + b)
        for n in build_notes(a.base, outs):
            print("NOTE: " + n)
        return 0
    stale = [n for n, t in outs.items()
             if not (out_dir / n).is_file() or (out_dir / n).read_text(encoding="utf-8") != t]
    stale += office_stale(a.base, outs, out_dir) + [f"{n} (a section page no build makes now)" for n in gone]
    print("STALE: " + ", ".join(stale) if stale else "OK: index current")
    P = json.loads(outs["papers.json"])
    bad = rejected_problems(a.base, [p["key"] for p in P["papers"]], P["aliases"])
    for b in bad:
        print("PROBLEM: " + b)
    for n in build_notes(a.base, outs):
        print("NOTE: " + n)
    return 1 if stale or bad else 0


def build_notes(base: Path, outs: Dict[str, str]) -> List[str]:
    """What a build used that is older than the index (never a failure -- each is made by a command that needs the
    corpus, the GPU or the figure bank): papers linked after the last `citemap`, lexicon terms with no precomputed
    full-text ranking, atlas_figures.js papers the index no longer holds."""
    out = []
    cm = json.loads(outs["atlas_citations.js"][len("window.__MWA_CITES="):].rstrip().rstrip(";"))
    if cm.get("unplaced"):
        out.append(f"{cm['unplaced']} linked papers have no position in papers/_citemap.json -- they sit on an outer "
                   "ring in the Citations tab (run: build_bwm.py citemap)")
    lex = json.loads(outs["lexicon.json"])["terms"]
    look = _load(base / "papers" / _tool("paper_find").LOOKUP_FILE, {}).get("terms") or {}
    miss = [t["id"] for t in lex if t["id"] not in look]
    if miss:
        out.append(f"{len(miss)} lexicon terms have no precomputed full-text ranking ({', '.join(miss[:3])}...) -- "
                   "the Find tab uses their citations and the index words only (run: tools/paper_find.py precompute)")
    figs = base / "index" / FIGURES_JS
    if figs.is_file():
        keys = {p["key"] for p in json.loads(outs["papers.json"])["papers"]}
        try:
            doc = json.loads(figs.read_text(encoding="utf-8")[len("window.__MWA_FIGS="):].rstrip().rstrip(";"))
            gone = [r[0] for r in doc.get("papers") or [] if r[0] not in keys]
        except ValueError:
            gone = ["(unreadable)"]
        if gone:
            out.append(f"index/{FIGURES_JS} names {len(gone)} papers the index does not hold (run: build_bwm.py figures)")
    return out


BROWSER_TEMPLATE = r"""<title>__ATLAS_TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap">
<script>
// MathJax 3 (SVG output: no web fonts or stylesheets to load) typesets the lexicon's equations; offline, the LaTeX
// source stays readable as text.
window.MathJax = {tex: {inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]]}, svg: {fontCache: "global"},
  startup: {typeset: false, ready: function(){ MathJax.startup.defaultReady();
    MathJax.startup.promise.then(function(){ if (window.__mwaTypeset) window.__mwaTypeset(); }); }}};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg-full.min.js" async></script>
<style>
/* Layout: a chart-room reference shelf -- one sticky header with tabs, one searchable column per tab, data in mono. */
:root{
  --bg:#eef3f3; --panel:#f8fbfb; --ink:#13262f; --muted:#52676f; --rule:#cbd8db;
  --depth:#2b6a86; --mag:#9b2a64; --pub:#2f6b4a; --pre:#9a6a12; --chip:#e1eaec;
  --display:"Newsreader", Georgia, "Times New Roman", serif;
  --body:"IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0d181d; --panel:#132229; --ink:#dbe7ea; --muted:#93a9b0; --rule:#26404a;
  --depth:#7bb8d4; --mag:#e48bba; --pub:#7fc79e; --pre:#e0b45a; --chip:#1b2f37; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0d181d; --panel:#132229; --ink:#dbe7ea; --muted:#93a9b0; --rule:#26404a;
  --depth:#7bb8d4; --mag:#e48bba; --pub:#7fc79e; --pre:#e0b45a; --chip:#1b2f37; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);margin:0}
.wrap{max-width:1080px;margin:0 auto;padding-inline:16px;padding-block:0 48px}
header{position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);border-bottom:1px solid var(--rule);z-index:5}
.head{max-width:1080px;margin:0 auto;padding-inline:16px;padding-block:14px 0;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline}
h1{font:600 26px/1.15 var(--display);margin:0;text-wrap:balance}
.sub{color:var(--muted);font-size:13px}
nav{max-width:1080px;margin:0 auto;padding-inline:16px;display:flex;gap:4px;flex-wrap:wrap}
nav button{font:500 14px var(--body);background:none;border:0;border-bottom:2px solid transparent;color:var(--muted);padding:10px 12px;cursor:pointer}
nav button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--mag)}
nav button:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid var(--depth);outline-offset:2px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-block:18px 10px}
.bar input[type=search]{flex:1 1 260px;min-width:0;font:15px var(--body);padding:9px 12px;border:1px solid var(--rule);border-radius:6px;background:var(--panel);color:var(--ink)}
.bar select{font:14px var(--body);padding:8px;border:1px solid var(--rule);border-radius:6px;background:var(--panel);color:var(--ink);max-width:100%}
.count{color:var(--muted);font:13px var(--mono);font-variant-numeric:tabular-nums}
.letters{display:flex;flex-wrap:wrap;gap:2px;margin-bottom:8px}
.letters button{font:500 12px var(--mono);border:1px solid var(--rule);background:var(--panel);color:var(--depth);border-radius:4px;padding:2px 7px;cursor:pointer}
.list{display:grid;gap:0}
.term{border-top:1px solid var(--rule);padding-block:14px;min-width:0}
.term h3{font:600 20px/1.25 var(--display);margin:0;display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}
.alias{color:var(--muted);font:13px var(--body)}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-block:6px}
.chip{font:12px var(--mono);background:var(--chip);color:var(--muted);border-radius:3px;padding:1px 6px}
.chip.cat{color:var(--depth)}
.chip.fx{color:var(--pub)}
.def{max-width:72ch;margin:6px 0}
.alt{max-width:72ch;color:var(--muted);font-size:14px;margin:4px 0}
.eq{max-width:100%;overflow-x:auto;overflow-y:hidden;margin:8px 0 2px;padding:6px 10px;background:var(--panel);border-left:2px solid var(--depth);font:13px var(--mono);color:var(--ink)}
.eq mjx-container{margin:0 !important}
.sym{max-width:76ch;color:var(--muted);font-size:13.5px;margin:2px 0 6px}
details{margin-top:6px}
summary{cursor:pointer;color:var(--depth);font-size:14px}
.q{margin:8px 0 0;padding-left:12px;border-left:2px solid var(--mag);max-width:76ch}
.q blockquote{margin:0;font:italic 15px/1.5 var(--display)}
.q .src{font:12.5px var(--mono);color:var(--muted);overflow-wrap:anywhere}
a{color:var(--depth)}
.rel button{font:13px var(--body);background:none;border:0;color:var(--depth);text-decoration:underline;cursor:pointer;padding:0 6px 0 0}
.paper{border-top:1px solid var(--rule);padding-block:12px;min-width:0}
.paper .t{font:600 16.5px/1.35 var(--display)}
.meta{font:12.5px var(--mono);color:var(--muted);display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:3px}
.pill{font:600 11px var(--mono);letter-spacing:.04em;text-transform:uppercase;padding:1px 6px;border-radius:3px;border:1px solid currentColor}
.pill.pub{color:var(--pub)} .pill.pre{color:var(--pre)} .pill.cf{color:var(--mag)}
.why{font-size:14px;max-width:76ch;margin:4px 0 0}
.more{margin-top:14px;font:14px var(--body);padding:8px 14px;border:1px solid var(--rule);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer}
.graph{position:relative;border:1px solid var(--rule);border-radius:8px;background:var(--panel);overflow:hidden}
canvas{display:block;width:100%;height:520px;touch-action:none;cursor:grab}
canvas:active{cursor:grabbing}
.gtools{position:absolute;top:8px;right:8px;display:flex;gap:4px}
.gtools button{font:600 14px var(--mono);min-width:34px;height:32px;border:1px solid var(--rule);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer}
.lk{font:inherit;background:none;border:0;color:var(--depth);text-decoration:underline;cursor:pointer;padding:0;text-align:left}
.get{font-size:13.5px;margin-top:4px;overflow-wrap:anywhere}
.get a{white-space:nowrap}
.pill.held{color:var(--pub)} .pill.open{color:var(--depth)} .pill.lib{color:var(--mag)}
@media (max-width:640px){canvas{height:min(62vh,112vw);min-height:300px}}
.labinfo{min-height:72px;margin-top:10px;max-width:80ch}
.tablewrap{overflow-x:auto;margin-top:16px}
table{border-collapse:collapse;font-size:14px;min-width:640px;width:100%}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font:600 12px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
td.n{font-variant-numeric:tabular-nums;font-family:var(--mono)}
.chart{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--panel);padding:12px;margin-top:12px}
.chart svg{display:block;min-width:600px;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font:12.5px var(--mono);color:var(--muted);margin-top:8px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.note{color:var(--muted);font-size:13px;max-width:80ch}
.docs{columns:2 300px;font-size:14px}
/* Find / Citations / Figures (2026-10-15). Graph roles use the dataviz reference palette's first three slots, validated
   all-pairs on these surfaces (light aqua < 3:1, so every highlighted paper is also listed by name). */
:root{--c-sel:#eb6834;--c-out:#2a78d6;--c-in:#1baf7a;--c-node:#b9c3c6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--c-sel:#d95926;--c-out:#3987e5;--c-in:#199e70;--c-node:#3c5560}}
:root[data-theme="dark"]{--c-sel:#d95926;--c-out:#3987e5;--c-in:#199e70;--c-node:#3c5560}
.tasks{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}
.tasks[hidden]{display:none}
.tasks button,.tbtn{font:13px var(--body);border:1px solid var(--rule);background:var(--panel);color:var(--ink);border-radius:14px;padding:4px 10px;cursor:pointer}
.tasks button[aria-pressed="true"]{border-color:var(--mag);color:var(--mag)}
.acts{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.acts button{font:12.5px var(--body);border:1px solid var(--rule);background:var(--panel);color:var(--depth);border-radius:5px;padding:2px 8px;cursor:pointer}
.whyl{font-size:13.5px;color:var(--muted);max-width:80ch;margin:3px 0 0;overflow-wrap:anywhere}
.whyl q{font-style:italic;color:var(--ink)}
.sug{display:flex;flex-wrap:wrap;gap:4px;margin:-4px 0 8px}
.sw2{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.cols3{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:4px 18px;margin-top:6px}
.cols3 h4{font:600 13px var(--body);margin:6px 0 2px}
.cols3 ol{margin:0;padding-left:20px;font-size:13.5px}
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin-top:6px}
.fcard{border:1px solid var(--rule);border-radius:8px;background:var(--panel);padding:8px;min-width:0;cursor:pointer;display:flex;flex-direction:column;gap:6px}
.fcard .im{aspect-ratio:4/3;background:var(--chip);border-radius:5px;display:flex;align-items:center;justify-content:center;overflow:hidden;color:var(--muted);font:12px var(--mono);text-align:center;padding:4px}
.fcard img{max-width:100%;max-height:100%;object-fit:contain;display:block;background:#fff}
.fcard .cap{font-size:12.5px;line-height:1.4;max-height:5.6em;overflow:hidden;overflow-wrap:anywhere}
.fcard .src{font:11.5px var(--mono);color:var(--muted);overflow-wrap:anywhere}
.lb{position:fixed;inset:0;background:rgba(5,12,15,.82);z-index:20;display:flex;align-items:center;justify-content:center;padding:16px}
.lb[hidden]{display:none}
.lb .box{background:var(--panel);color:var(--ink);border-radius:10px;max-width:min(1100px,100%);max-height:100%;overflow:auto;padding:14px}
.lb img{max-width:100%;max-height:70vh;display:block;margin:0 auto 10px;background:#fff}
.lb .x{float:right;font:600 16px var(--body);border:1px solid var(--rule);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer;padding:2px 10px}
.flowsvg text{font:10.5px var(--body)}
#cinfo{max-width:none}
/* Sections (atlas.toml [[sections]], 2026-10-15): a banner row above the tabs; the header's top band takes the shown
   section's colour. The colour marks a section; its name and count are always written beside it. */
.secs{max-width:1080px;margin:0 auto;padding-inline:16px;padding-block:8px 0;display:flex;flex-wrap:wrap;gap:6px}
.secs[hidden]{display:none}
.sb{font:500 13px/1.2 var(--body);display:inline-flex;align-items:center;gap:6px;border:1px solid var(--rule);background:var(--panel);color:var(--ink);border-radius:6px;padding:5px 10px;cursor:pointer;text-decoration:none;max-width:100%}
.sb .dot{width:10px;height:10px;border-radius:2px;flex:none;background:var(--sc)}
.sb .n{font:12px var(--mono);color:var(--muted)}
.sb[aria-pressed="true"],.sb[aria-current="page"]{border-color:var(--ink);font-weight:600;box-shadow:inset 0 -3px 0 var(--sc)}
.sb:disabled{opacity:.55;cursor:default}
.chip.sec{color:var(--ink);background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--sc)}
header.insec{border-top:4px solid var(--sc)}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>
<header>
  <div class="head"><h1>__ATLAS_TITLE__</h1><span class="sub" id="sub"></span></div>
  <div class="secs" id="secs" role="group" aria-label="Atlas sections" hidden></div>
  <nav role="tablist" aria-label="Views">
    <button role="tab" id="tab-find" data-tab="find" aria-selected="true">Find</button>
    <button role="tab" id="tab-lexicon" data-tab="lexicon" aria-selected="false">Lexicon</button>
    <button role="tab" id="tab-papers" data-tab="papers" aria-selected="false">Papers</button>
    <button role="tab" id="tab-cites" data-tab="cites" aria-selected="false">Citations</button>
    <button role="tab" id="tab-figs" data-tab="figs" aria-selected="false">Figures</button>
    <button role="tab" id="tab-labs" data-tab="labs" aria-selected="false">Labs</button>
    <button role="tab" id="tab-trends" data-tab="trends" aria-selected="false">Trends</button>
  </nav>
</header>
<main class="wrap">
<section id="p-find">
  <div class="bar"><input type="search" id="fq" placeholder="Describe a task, or type a few terms__FIND_EG__" aria-label="Find papers for a task or a few terms">
    <select id="ftop" aria-label="Topic"></select>
    <select id="fheld" aria-label="Availability"><option value="">Held or not</option><option value="held">PDF held</option></select>
    <span class="count" id="fcount"></span></div>
  <div class="tasks" id="ftasks" aria-label="The program's tasks"></div>
  <div class="chips" id="fterms"></div>
  <p class="note" id="fnote"></p>
  <div class="list" id="flist"></div><button class="more" id="fmore" hidden>Show more</button>
</section>
<section id="p-lexicon" hidden>
  <div class="bar"><input type="search" id="lq" placeholder="Search terms, aliases, definitions" aria-label="Search the lexicon">
    <select id="lcat" aria-label="Category"></select><select id="ltop" aria-label="Topic"></select><span class="count" id="lcount"></span></div>
  <div class="letters" id="letters"></div>
  <div class="list" id="llist"></div><button class="more" id="lmore" hidden>Show more</button>
</section>
<section id="p-papers" hidden>
  <div class="bar"><input type="search" id="pq" placeholder="Search titles, authors, keys, tags, simulators, methods" aria-label="Search papers">
    <select id="ptop" aria-label="Topic"></select>
    <select id="pst" aria-label="Status"><option value="">Published and preprint</option><option value="pub">Published (peer-reviewed etc.)</option><option value="pre">Preprints only</option></select>
    <select id="prel" aria-label="Relevance"><option value="">Any relevance</option><option value="core">Core</option><option value="high">High or core</option></select>
    <select id="pav" aria-label="Availability"><option value="">Held or not</option><option value="held">PDF held</option><option value="get">To download (all)</option><option value="open">To download: free copy</option><option value="library">To download: needs library</option></select>
    <select id="psort" aria-label="Sort"><option value="y">Newest first</option><option value="r">Most relevant first</option></select>
    <span class="count" id="pcount"></span></div>
  <div class="list" id="plist"></div><button class="more" id="pmore" hidden>Show more</button>
</section>
<section id="p-cites" hidden>
  <p class="note" id="cnote"></p>
  <div class="bar"><input type="search" id="cq" placeholder="Find a paper in the network: key or title words" aria-label="Find a paper in the citation network">
    <select id="cgrp" aria-label="Highlight a topic group"></select><span class="count" id="ccount"></span></div>
  <div class="sug" id="csug"></div>
  <div class="graph"><canvas id="ccv" role="img" aria-label="Citation network of the indexed papers; the lists below name every highlighted paper"></canvas>
    <div class="gtools"><button id="cin" aria-label="Zoom in">+</button><button id="cout" aria-label="Zoom out">&minus;</button><button id="cfit" aria-label="Fit the whole network">Fit</button></div></div>
  <div class="legend" id="clegend"></div>
  <div class="labinfo" id="cinfo"></div>
  <h2 style="font:600 20px var(--display);margin-block:24px 4px">Which topics build on which</h2>
  <p class="note">Each row is a review topic; a cell is the share of that topic's in-index citations that go to papers of the column's topic (a citation between papers of several topics is shared out evenly). Hover a cell for the count.</p>
  <div class="chart" id="cflows"></div>
  <h2 style="font:600 20px var(--display);margin-block:24px 4px">The backbone</h2>
  <p class="note">The papers cited most often by other papers in this index.</p>
  <ol id="cback" class="cols3" style="display:block;columns:2 320px"></ol>
</section>
<section id="p-figs" hidden>
  <p class="note" id="gnote"></p>
  <div class="bar"><input type="search" id="gq" placeholder="Search figure captions and papers__FIGURES_EG__" aria-label="Search figures">
    <select id="gtop" aria-label="Topic"></select>
    <select id="gcap" aria-label="Which figures"><option value="cap">Captioned figures</option><option value="">All figures (incl. uncaptioned)</option></select>
    <span class="count" id="gcount"></span></div>
  <div class="fgrid" id="glist"></div><button class="more" id="gmore" hidden>Show more</button>
  <div class="lb" id="glb" hidden role="dialog" aria-modal="true" aria-label="Figure"><div class="box" id="glbbox"></div></div>
</section>
<section id="p-labs" hidden>
  <p class="note">Each circle is a research group (size: papers in this base; grey: an institution named only in a link). Lines join groups that co-authored a paper here or list each other as collaborators. Drag to pan; zoom with the buttons, a double-click or Ctrl + scroll (pinch on a phone); tap a group for details, drag it to move it. The positions are fixed when the atlas is built.</p>
  <div class="graph"><canvas id="cv" role="img" aria-label="Network of research groups; the table below lists every group"></canvas>
    <div class="gtools"><button id="gin" aria-label="Zoom in">+</button><button id="gout" aria-label="Zoom out">&minus;</button><button id="gfit" aria-label="Fit the whole web">Fit</button></div></div>
  <div class="labinfo" id="labinfo"></div>
  <div class="bar"><input type="search" id="bq" placeholder="Filter groups by name, PI, institution, country" aria-label="Filter groups"><span class="count" id="bcount"></span></div>
  <div class="tablewrap"><table><thead><tr><th>Group</th><th>PI</th><th>Institution</th><th>Country</th><th>Papers</th></tr></thead><tbody id="btab"></tbody></table></div>
</section>
<section id="p-trends" hidden>
  <h2 style="font:600 20px var(--display);margin-block:20px 4px">Papers in this base by year</h2>
  <p class="note">Every verified paper in the base, by year, split into published (peer-reviewed journals, conferences, workshops, books, theses, reports) and preprints. The base is a curated selection, so read the shape, not the absolute numbers.</p>
  <div class="chart" id="c1"></div>
  <h2 style="font:600 20px var(--display);margin-block:24px 4px">Complete arXiv result sets, by year</h2>
  <p class="note" id="c2note">Query result sets that were retrieved in full, so their per-year counts are unbiased; the current year is partial.</p>
  <div class="chart" id="c2"></div>
  <h2 style="font:600 20px var(--display);margin-block:24px 4px">Topic reviews</h2>
  <ul class="docs" id="docs"></ul>
</section>
</main>
<script src="__CITES_JS__"></script>
<script src="__FIGS_JS__"></script>
<script type="application/json" id="data">__DATA__</script>
<script>
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const web = u => /^https?:[/][/]/i.test(String(u || ''));
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const S = D.stats;
$('sub').textContent = S.terms + ' terms \u00b7 ' + S.papers + ' papers (' + S.published + ' published, ' + S.preprints + ' preprints' + (S.withdrawn ? ', ' + S.withdrawn + ' withdrawn' : '') + ') \u00b7 ' + (S.citation_links ? S.citation_links + ' citations among them \u00b7 ' : '') + S.labs + ' groups';
const tabs = ['find','lexicon','papers','cites','figs','labs','trends'];
function show(t){ if(tabs.indexOf(t) < 0) t = 'find';
  tabs.forEach(x => { $('p-'+x).hidden = x !== t; $('tab-'+x).setAttribute('aria-selected', x === t ? 'true' : 'false'); });
  if(t === 'labs') startGraph(); if(t === 'trends') drawTrends(); if(t === 'cites') startCites(); if(t === 'figs') startFigs();
  try{ localStorage.setItem('mwa-tab', t); }catch(e){} }
function go(t){ history.replaceState(null, '', '#' + hashFor(t)); show(t); window.scrollTo(0, 0); }
D.papers.forEach((p, i) => { p._i = i; });
const byKey = {}; D.papers.forEach((p, i) => { byKey[p.k] = i; });
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => { history.replaceState(null,'','#'+hashFor(b.dataset.tab)); show(b.dataset.tab); }));
const topicsAll = S.topics || [];
// ---------------- sections (atlas.toml [[sections]]): the banner row. `banners` mode: it picks the section every tab
// shows, in this page (#in=<id>&<tab> links to one); `atlases` mode: each section is its own page and the row links them.
const SEC = D.sec || null, SL = SEC ? SEC.list : [];
const SPAL = ['#2a78d6', '#eb6834', '#1baf7a', '#9b2a64', '#9a6a12', '#5b5bd6', '#0f766e', '#c2410c', '#6b7280', '#b5179e', '#3a86ff', '#588157'];
let sec = SEC && SEC.here ? SL.findIndex(s => s.id === SEC.here) : -1;
const inSec = p => sec < 0 || (p.sc || []).indexOf(sec) >= 0;
const keyIn = k => sec < 0 || (k in byKey && inSec(D.papers[byKey[k]]));
const lexIn = x => sec < 0 || x.ci.some(c => keyIn(c[0]));
const secColor = i => (SL[i] && SL[i].c) || SPAL[i % SPAL.length];
function scopeTopics(){ return sec < 0 ? topicsAll : [...new Set(D.papers.filter(inSec).flatMap(p => p.top || []))].sort(); }
function hashFor(rest){ return (SEC && SEC.mode === 'banners' && sec >= 0 ? 'in=' + SL[sec].id + (rest ? '&' : '') : '') + rest; }
function fillSel(sel, first, vals){ sel.innerHTML = '<option value="">'+first+'</option>' + vals.map(v => '<option value="'+esc(v)+'">'+esc(v.replace(/_/g,' '))+'</option>').join(''); }
// ---------------- find: tools/paper_find.py's ATLAS mode, line for line (the same tokens, term matcher, BM25 over the
// index's words and weighted reciprocal-rank fusion; report/boat_world_model/test_build_bwm.py runs both and compares)
const F = D.find, FSTOP = new Set(F.stop || []), SEP = '\u0001';
function stemW(w){ const n = w.length;
  if(n > 4 && w.endsWith('ies') && !w.endsWith('eies') && !w.endsWith('aies')) return w.slice(0, -3) + 'y';
  if(n > 3 && w.endsWith('es') && !w.endsWith('aes') && !w.endsWith('ees') && !w.endsWith('oes')) return w.slice(0, -1);
  if(n > 3 && w.endsWith('s') && !w.endsWith('us') && !w.endsWith('ss')) return w.slice(0, -1);
  return w; }
function toks(s){ return (String(s || '').toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => !FSTOP.has(w)).map(stemW); }
function metaText(p){ return [p.t || '', p.t || '', p.sum || '', p.why || '', (p.g || []).join(' '), (p.fx || []).join(' '),
  (p.top || []).map(t => String(t).replace(/_/g, ' ')).join(' '), p.v || ''].join(' '); }
function MetaIndex(rows){             // rows: [[key, text]] -- the same Okapi BM25 (k1 1.2, b 0.75) as paper_find.MetaBM25
  const K1 = 1.2, B = 0.75, tf = new Map(), dl = new Map(), df = new Map(), post = new Map();
  const ks = rows.map(r => r[0]).sort(); const text = new Map(rows);
  for(const k of ks){ const c = new Map(); const ts = toks(text.get(k));
    for(const t of ts) c.set(t, (c.get(t) || 0) + 1);
    tf.set(k, c); dl.set(k, ts.length); for(const t of c.keys()) df.set(t, (df.get(t) || 0) + 1); }
  const N = ks.length; let tot = 0; for(const k of ks) tot += dl.get(k); const avg = N ? tot / N : 1;
  const idf = new Map(); for(const [t, d] of df) idf.set(t, Math.log(1 + (N - d + 0.5) / (d + 0.5)));
  for(const k of ks) for(const t of tf.get(k).keys()){ if(!post.has(t)) post.set(t, []); post.get(t).push(k); }
  this.scores = function(q){ const sc = new Map(); const seen = new Set();
    for(const t of toks(q)){ if(seen.has(t)) continue; seen.add(t); const w = idf.get(t); if(w === undefined) continue;
      for(const k of post.get(t)){ const f = tf.get(k).get(t);
        sc.set(k, (sc.get(k) || 0) + w * f * (K1 + 1) / (f + K1 * (1 - B + B * dl.get(k) / avg))); } }
    return sc; };
  this.rank = function(q){ return [...this.scores(q)].sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)).map(x => x[0]); };
}
function caseExact(n){ const a = (String(n).match(/[A-Za-z0-9]+/g) || []).join(''); return a.length <= 6 && (!/[a-z]/.test(a) || (a.match(/[A-Z]/g) || []).length >= 2); }
function nameForms(name){             // paper_find.name_forms: the name, without its parenthetical, and its abbreviations
  const n = String(name || '').trim(), out = [n];
  const core = n.replace(/\s*\([^()]*\)\s*/g, ' ').replace(/\s+/g, ' ').trim(); if(core && core !== n) out.push(core);
  (n.match(/\(([^()]+)\)/g) || []).forEach(par => par.slice(1, -1).split('/').forEach(p => { p = p.trim(); if(p && caseExact(p)) out.push(p); }));
  return [...new Set(out)]; }
function TermFinder(lex){             // paper_find.Terms: longest match, acronyms in exact case, other names S-stemmed
  const EX = new Map(), FO = new Map(); let maxlen = 1; const byId = {};
  [...lex].sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0).forEach(t => { byId[t.id] = t;
    [t.t].concat(t.al || []).flatMap(nameForms).forEach(n => { const raw = String(n).match(/[A-Za-z0-9]+/g) || [];
      if(raw.join('').length < 3) return; const acro = caseExact(n);
      const key = acro ? raw.join(SEP) : raw.map(w => stemW(w.toLowerCase())).join(SEP);
      const M = acro ? EX : FO; if(!M.has(key)) M.set(key, []); if(M.get(key).indexOf(t.id) < 0) M.get(key).push(t.id);
      maxlen = Math.max(maxlen, raw.length); }); });
  this.find = function(text){ const T = []; const re = /[A-Za-z0-9]+/g; let m; text = String(text || '');
    while((m = re.exec(text))) T.push([m[0], m.index]);
    const out = []; let i = 0;
    while(i < T.length){ let hit = false;
      for(let n = Math.min(maxlen, T.length - i); n >= 1; n--){ const raw = T.slice(i, i + n).map(x => x[0]);
        const ids = EX.get(raw.join(SEP)) || FO.get(raw.map(w => stemW(w.toLowerCase())).join(SEP));
        if(ids && ids.length){ const end = T[i + n - 1][1] + T[i + n - 1][0].length;
          ids.forEach(id => out.push({id: id, term: byId[id].t, start: T[i][1], matched: text.slice(T[i][1], end)}));
          i += n; hit = true; break; } }
      if(!hit) i++; }
    return out; };
}
function fuseW(lists){                // weighted reciprocal-rank fusion, lists in signal-name order, ties by key
  const sc = new Map(); for(const [w, r] of lists){ if(!w) continue;
    r.slice(0, 1000).forEach((p, i) => sc.set(p, (sc.get(p) || 0) + w / (F.rrf + i + 1))); }
  return [...sc].sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)); }
function regimeOf(q, ms){ const qt = toks(q); if(!qt.length || !ms.length) return 'task';
  const sp = new Map(); ms.forEach(m => sp.set(m.start, m.matched));
  return toks([...sp.values()].join(' ')).length / qt.length >= F.coverage ? 'term' : 'task'; }
let MI = null, TF = null; const lexById = {}; D.lex.forEach(x => { lexById[x.id] = x; });
function findInit(){ if(MI) return; MI = new MetaIndex(D.papers.map(p => [p.k, metaText(p)])); TF = new TermFinder(D.lex); }
function citedByTerms(ids){ const out = new Map();
  [...new Set(ids)].forEach(id => ((lexById[id] || {}).ci || []).forEach(c => { if(!(c[0] in byKey)) return;
    if(!out.has(c[0])) out.set(c[0], []); if(out.get(c[0]).indexOf(id) < 0) out.get(c[0]).push(id); }));
  return out; }
function findRank(q){                 // -> {order: [[key, score]], ids, matches, regime, cb}
  findInit(); const ms = TF.find(q), ids = [...new Set(ms.map(m => m.id))], reg = regimeOf(q, ms), cb = citedByTerms(ids);
  const rel = k => { const r = F.relevance[D.papers[byKey[k]].r]; return r === undefined ? 4 : r; };
  const lexicon = [...cb.keys()].sort((a, b) => (cb.get(b).length - cb.get(a).length) || (rel(a) - rel(b)) ||
    ((D.papers[byKey[b]].y || 0) - (D.papers[byKey[a]].y || 0)) || (a < b ? -1 : a > b ? 1 : 0));
  const table = fuseW(ids.map(id => [1, (F.terms[id] || []).map(i => D.papers[i].k)]).filter(x => x[1].length)).map(x => x[0]);
  const meta = MI.rank(q), W = F.weights[reg] || {};
  const sig = {lexicon: lexicon, meta: meta, table: table};
  return {order: fuseW(Object.keys(sig).sort().map(n => [W[n] || 0, sig[n]])), ids: ids, matches: ms, regime: reg, cb: cb,
          pos: {meta: new Map(meta.slice(0, 200).map((k, i) => [k, i + 1])), table: new Map(table.slice(0, 200).map((k, i) => [k, i + 1]))}}; }
window.__mwaFind = q => findRank(q).order.map(x => x[0]);       // for the equivalence test (paper_find --mode atlas)
let fShown = 25, fTask = null, fLast = null;
fillSel($('ftop'), 'All topics', topicsAll);
function visTasks(){ return sec < 0 ? (F.tasks || []).map((t, i) => i) : (SL[sec].tk || []); }   // a section: its own
function renderTasks(){ const vis = visTasks(); if(fTask !== null && vis.indexOf(fTask) < 0) fTask = null;
  $('ftasks').hidden = !vis.length;
  $('ftasks').innerHTML = '<span class="note" style="margin-right:4px">Program tasks:</span>' + vis.map(i => { const t = F.tasks[i];
    return '<button data-task="' + i + '" aria-pressed="false" title="' + esc(t.query) + '">' + esc(t.title) + '</button>'; }).join(''); }
function findCard(p, why){
  const st = p.pub ? '<span class="pill pub">' + esc(p.s.replace('peer-reviewed-', 'peer-reviewed ')) + '</span>' : '<span class="pill pre">' + esc(p.s || 'unknown') + '</span>';
  const av = p.h ? '<span class="pill held">PDF held</span>' : (p.av === 'open' ? '<span class="pill open">free copy</span>' : '<span class="pill lib">needs library</span>');
  return '<article class="paper" data-k="' + esc(p.k) + '"><div class="t">' + (web(p.u) ? '<a href="' + esc(p.u) + '" target="_blank" rel="noopener">' + esc(p.t) + '</a>' : esc(p.t)) + '</div>' +
    '<div class="meta"><span>' + (p.y || 'n.d.') + '</span>' + st + av + '<span>' + esc(p.v || '') + '</span><span>' + esc(p.k) + '</span><span>' + esc(p.r) + '</span>' +
    (p.lp ? '<a href="' + esc(p.lp) + '" target="_blank" rel="noopener">PDF</a>' : '') + '</div>' +
    why.map(w => '<p class="whyl">' + w + '</p>').join('') + acts(p) + '</article>'; }
function acts(p){ const n = citeNode(p._i), f = figCount(p._i);
  return '<div class="acts"><button data-act="paper" data-i="' + p._i + '">Paper card</button>' +
    (n >= 0 ? '<button data-act="cites" data-i="' + p._i + '">In the citation map</button>' : '') +
    (f ? '<button data-act="figs" data-i="' + p._i + '">Figures (' + f + ')</button>' : '') + '</div>'; }
function renderFind(){
  const q = $('fq').value.trim(), top = $('ftop').value, held = $('fheld').value;
  const keep = p => inSec(p) && (!top || p.top.indexOf(top) >= 0) && (!held || p.h);
  document.querySelectorAll('#ftasks button').forEach(b => b.setAttribute('aria-pressed', fTask !== null && +b.dataset.task === fTask ? 'true' : 'false'));
  let rows = [], note = '';
  if(fTask !== null){ const t = F.tasks[fTask];
    rows = t.p.map(i => ({p: D.papers[i], why: ['full-text rank ' + (t.p.indexOf(i) + 1) + ' for this task (BM25 + dense, ranked when the atlas was built)']}));
    note = 'Task: \u201c' + esc(t.query) + '\u201d \u2014 the ' + t.p.length + ' papers the full-text ranker put first' + (F.made ? ' (' + esc(F.made) + ')' : '') + '. Edit the box to search your own words.';
    $('fterms').innerHTML = '';
  } else if(q){ const R = fLast && fLast.q === q ? fLast.r : findRank(q); fLast = {q: q, r: R};
    rows = R.order.map(x => { const p = D.papers[byKey[x[0]]], why = [];
      (R.cb.get(p.k) || []).forEach(id => { const t = lexById[id], c = (t.ci || []).find(c => c[0] === p.k);
        why.push('term <strong>' + esc(t.t) + '</strong>' + (c ? ': <q>' + esc(c[1]) + '</q> (' + esc(c[2] || '') + ')' : '')); });
      if(R.pos.table.has(p.k)) why.push('full-text rank ' + R.pos.table.get(p.k) + ' for the matched term' + (R.ids.length > 1 ? 's' : ''));
      if(R.pos.meta.has(p.k)) why.push('index words rank ' + R.pos.meta.get(p.k));
      return {p: p, why: why}; });
    const seen = new Set(); $('fterms').innerHTML = R.matches.filter(m => !seen.has(m.id) && seen.add(m.id)).map(m =>
      '<button class="tbtn" data-term="' + esc(m.id) + '" title="open in the lexicon">' + esc(m.term) + '</button>').join('');
    note = (R.ids.length ? 'Lexicon terms found: ' + R.ids.length + '. ' : 'No lexicon term found: ranked by the index\u2019s own words. ') +
      'Read as a ' + R.regime + ' query. In the browser the atlas ranks by the lexicon, each term\u2019s precomputed full-text ranking and the index\u2019s words; ' +
      'for the full text of every page run <code>python tools/paper_find.py "' + esc(q) + '"</code>.';
  } else {
    $('fterms').innerHTML = '';
    note = 'Type a task in your own words, or a few terms. The atlas maps them onto the lexicon (' + D.lex.filter(lexIn).length + ' terms with verified citations), ' +
      'each term\u2019s full-text ranking and the papers\u2019 own descriptions, and says why each paper is listed. Or pick one of the program\u2019s tasks above.';
  }
  rows = rows.filter(r => keep(r.p));
  $('fnote').innerHTML = note;
  $('fcount').textContent = (q || fTask !== null) ? rows.length + ' papers' : '';
  $('flist').innerHTML = rows.slice(0, fShown).map(r => findCard(r.p, r.why)).join('');
  $('fmore').hidden = rows.length <= fShown;
}
let fTimer = 0;
$('fq').addEventListener('input', () => { fTask = null; fShown = 25; clearTimeout(fTimer); fTimer = setTimeout(() => {
  try{ history.replaceState(null, '', '#' + hashFor('find=' + encodeURIComponent($('fq').value.trim()))); }catch(e){} renderFind(); }, 160); });
['ftop', 'fheld'].forEach(id => $(id).addEventListener('input', () => { fShown = 25; renderFind(); }));
$('fmore').addEventListener('click', () => { fShown += 50; renderFind(); });
$('ftasks').addEventListener('click', e => { const b = e.target.closest('button[data-task]'); if(!b) return;
  fTask = fTask === +b.dataset.task ? null : +b.dataset.task; $('fq').value = ''; fShown = 25; renderFind(); });
$('fterms').addEventListener('click', e => { const b = e.target.closest('button[data-term]'); if(b) openTerm(b.dataset.term); });
document.addEventListener('click', e => { const b = e.target.closest('button[data-act]'); if(!b) return; const i = +b.dataset.i;
  if(b.dataset.act === 'paper') openPaper(i); else if(b.dataset.act === 'cites') openCites(i); else if(b.dataset.act === 'figs') openFigs(i); });
function openPaper(i){ const p = D.papers[i]; if(!inSec(p)) setSection(-1, true);   // a paper outside the section shown
  $('pq').value = p.k; ['ptop', 'pst', 'prel', 'pav'].forEach(id => { $(id).value = ''; }); pShown = 80; renderPapers(); go('papers'); }
function openTerm(id){ const x = lexById[id]; if(!x) return; $('lq').value = x.t; $('lcat').value = ''; $('ltop').value = ''; lShown = 120; renderLex(); go('lexicon'); }
// ---------------- lexicon
const cats = [...new Set(D.lex.flatMap(x => x.cat))].sort();
fillSel($('lcat'), 'All categories', cats);
$('lcat').insertAdjacentHTML('beforeend', '<option value="__math">With equations</option>'); fillSel($('ltop'), 'All topics', topicsAll);
const byId = {}; D.lex.forEach(x => byId[x.id] = x);
let lShown = 120, lRes = [];
function termHtml(x){
  const ci = sec < 0 ? x.ci : x.ci.filter(c => keyIn(c[0]));      // in a section: its papers' quotes only
  const cites = ci.map(c => '<div class="q"><blockquote>\u201c'+esc(c[1])+'\u201d</blockquote><div class="src">'+(web(c[5])?'<a href="'+esc(c[5])+'" target="_blank" rel="noopener">'+esc(c[0])+'</a>':esc(c[0]))+' \u00b7 '+(c[3]||'n.d.')+' \u00b7 '+esc(c[4])+' \u00b7 '+esc(c[2])+(c[6]==='fulltext'?' \u00b7 full text':'')+'</div></div>').join('');
  const alts = x.ds.map(d => '<p class="alt">('+esc(d.topic.replace(/_/g,' '))+') '+esc(d.text)+'</p>').join('');
  const senses = x.sn.map(s => '<p class="alt">Sense: '+esc(s)+'</p>').join('');
  const rel = x.rel.length ? '<div class="rel">Related: '+x.rel.map(r => '<button data-term="'+esc(r)+'">'+esc(byId[r] ? byId[r].t : r)+'</button>').join('')+'</div>' : '';
  return '<article class="term" id="t-'+esc(x.id)+'"><h3>'+esc(x.t)+(x.al.length?'<span class="alias">also '+esc(x.al.slice(0,5).join(', '))+'</span>':'')+'</h3>'+
    '<div class="chips">'+x.cat.map(c=>'<span class="chip cat">'+esc(c)+'</span>').join('')+x.top.map(c=>'<span class="chip">'+esc(c.replace(/_/g,' '))+'</span>').join('')+'</div>'+
    '<p class="def">'+(x.dt?mathText(x.dt):esc(x.d))+'</p>'+x.mx.map(m => '<div class="eq">\\['+esc(m[0])+'\\]</div>'+(m[1]?'<p class="sym"><em>where</em> '+mathText(m[1])+'</p>':'')).join('')+senses+(alts?'<details><summary>Other definitions ('+x.ds.length+')</summary>'+alts+'</details>':'')+
    '<details'+(ci.length<=2?' open':'')+'><summary>Citations ('+ci.length+')</summary>'+cites+'</details>'+rel+'</article>';
}
function renderLex(){
  const q = $('lq').value.trim().toLowerCase(), cat = $('lcat').value, top = $('ltop').value;
  lRes = D.lex.filter(x => lexIn(x) && (!cat || (cat === '__math' ? x.mx.length > 0 : x.cat.indexOf(cat) >= 0)) && (!top || x.top.indexOf(top) >= 0) &&
    (!q || x.t.toLowerCase().indexOf(q) >= 0 || x.al.some(a => a.toLowerCase().indexOf(q) >= 0) || x.d.toLowerCase().indexOf(q) >= 0));
  if(q) lRes.sort((a,b) => (a.t.toLowerCase().indexOf(q) === 0 ? 0 : 1) - (b.t.toLowerCase().indexOf(q) === 0 ? 0 : 1));
  $('lcount').textContent = lRes.length + ' of ' + (sec < 0 ? D.lex.length : D.lex.filter(lexIn).length) + ' terms';
  $('llist').innerHTML = lRes.slice(0, lShown).map(termHtml).join('');
  $('lmore').hidden = lRes.length <= lShown;
  typesetLex();
}
// "$...$" spans in a symbols explanation -> MathJax inline math; the prose between them is escaped as usual
function mathText(s){ return String(s).split(/(?<!\\)(\$[^$]+?(?<!\\)\$)/).map(function(part){
  return (part.length > 2 && part[0] === '$' && part[part.length-1] === '$') ? '\\(' + esc(part.slice(1, -1)) + '\\)'
    : esc(part.replace(/\\\$/g, '$')); }).join(''); }
function typesetLex(){ try{ if(window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([$('llist')]).catch(function(){}); }catch(e){} }
window.__mwaTypeset = typesetLex;
['lq','lcat','ltop'].forEach(id => $(id).addEventListener('input', () => { lShown = 120; renderLex(); }));
$('lmore').addEventListener('click', () => { lShown += 200; renderLex(); });
$('llist').addEventListener('click', e => { const b = e.target.closest('button[data-term]'); if(!b) return;
  const x = byId[b.dataset.term]; if(x){ $('lq').value = x.t; $('lcat').value=''; $('ltop').value=''; lShown=120; renderLex(); window.scrollTo(0,0); } });
const letters = [...new Set(D.lex.map(x => /[a-z]/i.test(x.t[0]) ? x.t[0].toUpperCase() : '#'))].sort();
$('letters').innerHTML = letters.map(l => '<button data-l="'+l+'">'+l+'</button>').join('');
$('letters').addEventListener('click', e => { const b = e.target.closest('button'); if(!b) return;
  $('lq').value=''; $('lcat').value=''; $('ltop').value=''; lShown = D.lex.length; renderLex();
  const x = D.lex.find(t => (/[a-z]/i.test(t.t[0]) ? t.t[0].toUpperCase() : '#') === b.dataset.l);
  if(x){ const el = $('t-'+x.id); if(el) el.scrollIntoView(); } });
// ---------------- papers
fillSel($('ptop'), 'All topics', topicsAll);
let pShown = 80;
function paperHtml(p){
  const st = p.pub ? '<span class="pill pub">'+esc(p.s.replace('peer-reviewed-','peer-reviewed '))+'</span>' : '<span class="pill pre">'+esc(p.s||'unknown')+'</span>';
  const cf = p.c.length ? ' <span class="pill cf" title="topics disagreed: '+esc(p.c.join(' / '))+'">status disputed</span>' : '';
  const av = p.h ? '<span class="pill held">PDF held</span>' : (p.ob ? '<span class="pill held">'+esc(p.ob)+' held</span>' : (p.av === 'open' ? '<span class="pill open">free copy</span>' : '<span class="pill lib">needs library</span>'));
  const get = p.lp ? '<div class="get"><a href="'+esc(p.lp)+'" target="_blank" rel="noopener">Open the PDF (in this bundle)</a></div>'
    : '<div class="get">Get it: '+huntLinks(p).map(l => '<a href="'+esc(l[1])+'" target="_blank" rel="noopener">'+esc(l[0])+'</a>').join(' \u00b7 ')+'</div>';
  return '<article class="paper"><div class="t">'+(web(p.u)?'<a href="'+esc(p.u)+'" target="_blank" rel="noopener">'+esc(p.t)+'</a>':esc(p.t))+'</div>'+
   '<div class="meta"><span>'+(p.y||'n.d.')+'</span>'+st+cf+av+'<span>'+esc(p.v||'')+'</span>'+(p.cb!=null?'<span title="'+esc('citations: OpenAlex'+(p.md?', measured '+p.md:'')+(p.cbi?'; '+p.cbi+' from papers in this index':'')+(p.vm!=null?'; venue 2-year mean citedness '+p.vm+' (an open Impact-Factor analogue)':''))+'">cited '+p.cb+(p.cbi?' ('+p.cbi+' here)':'')+'</span>':'')+'<span>'+esc(p.a)+'</span><span>'+esc(p.k)+'</span>'+(web(p.code)?'<a href="'+esc(p.code)+'" target="_blank" rel="noopener" title="'+(p.cs?esc(codeWhy[p.cs]||p.cs):'listed by the literature agents, not checked')+'">code'+(p.cst?' \u2605'+p.cst:'')+(p.cs==='CANDIDATE_BY_SEARCH'?' (search)':'')+'</a>':'')+'</div>'+get+
   '<div class="chips"><span class="chip cat">'+esc(p.r)+'</span>'+(p.sc||[]).map(i=>'<span class="chip sec" style="--sc:'+secColor(i)+'" title="in the section">'+esc(SL[i].t)+'</span>').join('')+p.top.map(t=>'<span class="chip">'+esc(t.replace(/_/g,' '))+'</span>').join('')+(p.fx||[]).map(t=>'<span class="chip fx" title="named in the full text">'+esc(t)+'</span>').join('')+'</div>'+
   (p.sum?'<p class="why">'+esc(p.sum)+'</p>':'')+(p.why?'<p class="why"><strong>For us:</strong> '+esc(p.why)+'</p>':'')+acts(p).replace('<button data-act="paper" data-i="'+p._i+'">Paper card</button>','')+'</article>';
}
// the same links, in the same order, as paper_links.hunt_links (minus OpenAlex's extra copies); one per URL
function huntLinks(p){ const L = [];
  if(p.ax) L.push(['arXiv PDF', 'https://arxiv.org/pdf/'+p.ax]);
  if(web(p.oa)) L.push([p.oal || 'open copy', p.oa]);
  if(p.oav) L.push(['arXiv version', 'https://arxiv.org/abs/'+p.oav]);
  if(p.d) L.push(['publisher', 'https://doi.org/'+p.d]);
  else if(!p.ax && web(p.u)) L.push(['landing page', p.u]);
  if(p.t){ L.push(['Scholar', 'https://scholar.google.com/scholar?q='+encodeURIComponent('"'+p.t+'"')]);
    L.push(['PDF search', 'https://www.google.com/search?q='+encodeURIComponent('"'+p.t+'" filetype:pdf')]); }
  if(web(p.s2)) L.push(['Semantic Scholar', p.s2]);
  const seen = new Set(); return L.filter(l => !seen.has(l[1]) && seen.add(l[1])); }
const relRank = {core:0, high:1, medium:2, low:3};
const codeWhy = {VERIFIED_LINKED: 'checked: the repo exists and its README cites the paper',
  STATED: 'checked: the repo exists and the paper or its arXiv record names it',
  CANDIDATE_BY_SEARCH: 'found by GitHub search; its README cites the paper (officialness unverified)'};
function renderPapers(){
  const q = $('pq').value.trim().toLowerCase(), top = $('ptop').value, st = $('pst').value, rel = $('prel').value, av = $('pav').value;
  let r = D.papers.filter(p => inSec(p) && (!top || p.top.indexOf(top) >= 0) && (!st || (st === 'pub' ? p.pub : !p.pub)) &&
    (!av || (av === 'held' ? (p.h || p.ob) : (av === 'get' ? !(p.h || p.ob) : (!p.h && !p.ob && p.av === av)))) &&
    (!rel || (rel === 'core' ? p.r === 'core' : (p.r === 'core' || p.r === 'high'))) &&
    (!q || (p.t + ' ' + p.a + ' ' + p.k + ' ' + p.g.join(' ') + ' ' + (p.fx || []).join(' ')).toLowerCase().indexOf(q) >= 0));
  if($('psort').value === 'r') r.sort((a,b) => ((relRank[a.r]??4) - (relRank[b.r]??4)) || ((b.y||0) - (a.y||0)));
  else r.sort((a,b) => ((b.y||0) - (a.y||0)) || ((relRank[a.r]??4) - (relRank[b.r]??4)));
  $('pcount').textContent = r.length + ' of ' + (sec < 0 ? D.papers.length : D.papers.filter(inSec).length) + ' papers';
  $('plist').innerHTML = r.slice(0, pShown).map(paperHtml).join('');
  $('pmore').hidden = r.length <= pShown;
}
['pq','ptop','pst','prel','psort','pav'].forEach(id => $(id).addEventListener('input', () => { pShown = 80; renderPapers(); }));
$('pmore').addEventListener('click', () => { pShown += 150; renderPapers(); });
// ---------------- labs: table + the web. Positions are FIXED at build time (labs_layout.py, tested there): this code
// only draws them, pans and zooms, and draws again on demand -- nothing here simulates or animates on its own.
const LN = D.labs.nodes, LL = D.labs.layout;
const labIn = i => sec < 0 || LN[i].papers.some(keyIn);             // a group with a paper in the section shown
const labPapers = n => sec < 0 ? n.papers : n.papers.filter(keyIn);
let bMatch = null, G = null;
function renderLabTable(){
  const q = $('bq').value.trim().toLowerCase(), r = [];
  bMatch = q ? new Set() : null;
  LN.forEach((n,i) => { if(labIn(i) && (!q || (n.lab+' '+n.pi+' '+n.institution+' '+n.country).toLowerCase().indexOf(q) >= 0)){ r.push(i); if(q) bMatch.add(i); } });
  $('bcount').textContent = r.length + ' of ' + (sec < 0 ? LN.length : LN.filter((n, i) => labIn(i)).length) + ' groups';
  $('btab').innerHTML = r.map(i => { const n = LN[i]; return '<tr><td>'+(n.kind==='institution'?'<span class="chip">institution</span> ':'')+'<button class="lk" data-i="'+i+'" title="Show in the web">'+esc(n.lab)+'</button>'+(web(n.url)?' <a href="'+esc(n.url)+'" target="_blank" rel="noopener">site</a>':'')+'</td><td>'+esc(n.pi)+'</td><td>'+esc(n.institution)+'</td><td>'+esc(n.country)+'</td><td class="n">'+labPapers(n).length+'</td></tr>'; }).join('');
  if(G) draw();
}
$('bq').addEventListener('input', renderLabTable);
$('btab').addEventListener('click', e => { const b = e.target.closest('button[data-i]'); if(!b || !G) return; locate(+b.dataset.i); });
function graphInit(){
  const cv = $('cv'), idx = {};
  const P = LN.map((n,i) => { idx[n.id] = i; return {x: n.xy[0], y: n.xy[1], r: n.r}; });
  const E = D.labs.edges.filter(e => e.a in idx && e.b in idx).map(e => [idx[e.a], idx[e.b]]);
  const nb = LN.map(() => []); E.forEach(([a,b]) => { nb[a].push(b); nb[b].push(a); });
  G = {cv, P, E, nb, sel: -1, k: 1, k0: 1, tx: 0, ty: 0, W: 0, H: 0, dpr: 1, user: false, draws: 0, raf: 0,
       ptr: new Map(), mode: '', drag: -1, moved: 0, slop: 3, last: [0,0], pinch: null};
  const at = ev => { const b = cv.getBoundingClientRect(); return [ev.clientX - b.left, ev.clientY - b.top]; };
  const slop = ev => ev.pointerType === 'mouse' ? 3 : 10;
  cv.addEventListener('pointerdown', ev => { if(ev.pointerType === 'mouse' && ev.button !== 0) return;
    const [sx, sy] = at(ev); G.ptr.set(ev.pointerId, [sx, sy]); G.slop = slop(ev);
    try{ cv.setPointerCapture(ev.pointerId); }catch(e){}
    if(G.ptr.size === 2){ const [a, b] = [...G.ptr.values()];
      G.pinch = {d: Math.max(1, Math.hypot(a[0]-b[0], a[1]-b[1])), k: G.k, tx: G.tx, ty: G.ty, mx: (a[0]+b[0])/2, my: (a[1]+b[1])/2};
      G.mode = 'pinch'; return; }
    if(G.ptr.size > 2) return;
    const h = hit(sx, sy); G.mode = h >= 0 ? 'node' : 'pan'; G.drag = h; G.last = [sx, sy]; G.moved = 0;
    if(h >= 0) select(h); });
  cv.addEventListener('pointermove', ev => { if(!G.ptr.has(ev.pointerId)) return;
    if(ev.pointerType === 'mouse' && !ev.buttons) return up(ev);         // released outside the window
    const [sx, sy] = at(ev); G.ptr.set(ev.pointerId, [sx, sy]);
    if(G.mode === 'pinch'){ if(G.ptr.size < 2) return; const [a, b] = [...G.ptr.values()], q = G.pinch;
      const k = clampK(q.k * Math.hypot(a[0]-b[0], a[1]-b[1]) / q.d), mx = (a[0]+b[0])/2, my = (a[1]+b[1])/2;
      G.tx = mx - (q.mx - q.tx) * k / q.k; G.ty = my - (q.my - q.ty) * k / q.k; G.k = k; G.user = true; later(); return; }
    const dx = sx - G.last[0], dy = sy - G.last[1]; G.last = [sx, sy]; G.moved += Math.abs(dx) + Math.abs(dy);
    if(G.mode === 'node' && G.moved > G.slop){ const p = G.P[G.drag]; p.x += dx / G.k; p.y += dy / G.k; later(); }
    else if(G.mode === 'pan'){ G.tx += dx; G.ty += dy; if(G.moved > G.slop) G.user = true; later(); } });
  function up(ev){ if(!G.ptr.has(ev.pointerId)) return; G.ptr.delete(ev.pointerId);
    if(G.mode === 'pan' && G.moved <= G.slop && ev.type === 'pointerup') select(-1);
    G.mode = G.ptr.size ? 'idle' : ''; }
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(t => cv.addEventListener(t, up));
  cv.addEventListener('wheel', ev => { if(!(ev.ctrlKey || ev.metaKey)) return; ev.preventDefault(); const [sx, sy] = at(ev);
    const dy = ev.deltaY * (ev.deltaMode === 1 ? 33 : ev.deltaMode === 2 ? 400 : 1);
    zoomAt(Math.exp(-Math.max(-300, Math.min(300, dy)) * 0.004), sx, sy); }, {passive: false});
  cv.addEventListener('dblclick', ev => { const [sx, sy] = at(ev); zoomAt(1.8, sx, sy); });
  $('gin').addEventListener('click', () => zoomAt(1.5, G.W/2, G.H/2));
  $('gout').addEventListener('click', () => zoomAt(1/1.5, G.W/2, G.H/2));
  $('gfit').addEventListener('click', () => { fit(); draw(); });
  if('ResizeObserver' in window) new ResizeObserver(() => draw()).observe(cv); else window.addEventListener('resize', () => draw());
  try{ matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => draw()); }catch(e){}
  new MutationObserver(() => draw()).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
}
function startGraph(){ if(!G) graphInit(); renderLabTable(); draw(); }
function clampK(k){ return Math.max(G.k0 * 0.6, Math.min(G.k0 * 20, k)); }
function fit(){ const m = 10; G.k = G.k0 = Math.max(1e-3, Math.min((G.W - 2*m) / LL.w, (G.H - 2*m) / LL.h));
  G.tx = (G.W - LL.w * G.k) / 2; G.ty = (G.H - LL.h * G.k) / 2; G.user = false; }
function zoomAt(f, sx, sy){ const k = clampK(G.k * f); G.tx = sx - (sx - G.tx) * k / G.k; G.ty = sy - (sy - G.ty) * k / G.k; G.k = k; G.user = true; later(); }
function locate(i){ select(i); const p = G.P[i]; G.k = clampK(Math.max(G.k, G.k0 * 3)); G.tx = G.W/2 - p.x * G.k; G.ty = G.H/2 - p.y * G.k; G.user = true; draw();
  try{ $('cv').scrollIntoView({block: 'center', behavior: 'smooth'}); }catch(e){ $('cv').scrollIntoView(); } }
function hit(sx, sy){ let best = -1, bd = Infinity;
  G.P.forEach((p,i) => { const dx = p.x * G.k + G.tx - sx, dy = p.y * G.k + G.ty - sy, d = dx*dx + dy*dy, rr = Math.max(p.r * G.k, 6) + 3; if(d <= rr*rr && d < bd){ bd = d; best = i; } });
  return best; }
function select(i){ G.sel = i; info(); draw(); }
function later(){ if(!G.raf) G.raf = requestAnimationFrame(() => { G.raf = 0; draw(); }); }
function size(){ const cv = G.cv, W = cv.clientWidth, H = cv.clientHeight; if(!W || !H) return false;
  const dpr = window.devicePixelRatio || 1;
  if(W !== G.W || H !== G.H || dpr !== G.dpr){ const first = !G.W; G.W = W; G.H = H; G.dpr = dpr; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); if(first || !G.user) fit(); }
  return true; }
function draw(){
  if(!G || $('p-labs').hidden || !size()) return;
  const c = G.cv.getContext('2d'), cs = getComputedStyle(document.documentElement), col = v => cs.getPropertyValue(v).trim();
  const C = {rule: col('--rule'), ink: col('--ink'), muted: col('--muted'), depth: col('--depth'), mag: col('--mag'), pre: col('--pre'), panel: col('--panel'), font: col('--body')};
  c.setTransform(G.dpr, 0, 0, G.dpr, 0, 0); c.clearRect(0, 0, G.W, G.H);
  const k = G.k, X = p => p.x * k + G.tx, Y = p => p.y * k + G.ty;
  const near = G.sel >= 0 ? new Set(G.nb[G.sel]) : null;
  const dim = i => !labIn(i) || (bMatch && !bMatch.has(i)) || (near && i !== G.sel && !near.has(i));
  c.lineWidth = 1;
  G.E.forEach(([a,b]) => { const hot = G.sel >= 0 && (a === G.sel || b === G.sel);
    c.strokeStyle = hot ? C.mag : C.rule; c.globalAlpha = !hot && (dim(a) || dim(b)) ? 0.3 : 1;
    c.beginPath(); c.moveTo(X(G.P[a]), Y(G.P[a])); c.lineTo(X(G.P[b]), Y(G.P[b])); c.stroke(); });
  let inview = 0;
  G.P.forEach((p,i) => { const x = X(p), y = Y(p), r = Math.max(1.5, p.r * k);
    if(x - r >= -0.5 && y - r >= -0.5 && x + r <= G.W + 0.5 && y + r <= G.H + 0.5) inview++;
    c.globalAlpha = dim(i) ? 0.18 : 1;
    c.fillStyle = i === G.sel ? C.mag : (near && near.has(i) ? C.pre : (LN[i].kind === 'institution' ? C.muted : C.depth));
    c.beginPath(); c.arc(x, y, r, 0, 6.2832); c.fill(); });
  c.globalAlpha = 1; c.fillStyle = C.muted; c.font = '12px ' + C.font;
  (LL.sections || []).forEach(sec => c.fillText(sec.label, 10 * k + G.tx, sec.y * k + G.ty));
  // labels: the selected group and its links first, then a short search result, then the biggest circles on screen;
  // one that would collide with a label already drawn is skipped
  const order = [];
  if(G.sel >= 0){ order.push(G.sel); G.nb[G.sel].forEach(j => order.push(j)); }
  if(bMatch && bMatch.size <= 40) bMatch.forEach(i => order.push(i));
  // (a group with no link sits in a tight row where any label covers its neighbours: labelled only when picked)
  G.P.map((p,i) => i).filter(i => !dim(i) && G.nb[i].length && (LN[i].papers.length >= 6 || G.P[i].r * k >= 9)).sort((a,b) => G.P[b].r - G.P[a].r || a - b).forEach(i => order.push(i));
  const boxes = [[G.W - 132, 0, G.W, 46]], done = new Set(); c.font = '11px ' + C.font; c.lineJoin = 'round';   // [0]: the zoom buttons
  order.forEach(i => { if(done.has(i)) return; done.add(i);
    const p = G.P[i], rr = Math.max(1.5, p.r * k), y = Y(p) + 4;
    const t = LN[i].lab.length > 30 ? LN[i].lab.slice(0, 29) + '\u2026' : LN[i].lab, w = c.measureText(t).width;
    let x = X(p) + rr + 3;
    if(x + w > G.W - 4) x = X(p) - rr - 3 - w;                  // no room on the right: the label goes on the left
    if(x < 0 || x + w > G.W || y < 10 || y > G.H) return;
    const b = [x - 2, y - 11, x + w + 2, y + 3];
    if(i !== G.sel && boxes.some(o => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1])) return;
    boxes.push(b); c.strokeStyle = C.panel; c.lineWidth = 3; c.strokeText(t, x, y); c.fillStyle = C.ink; c.fillText(t, x, y); });
  G.draws++;
  const d = G.cv.dataset; d.draws = G.draws; d.nodes = G.P.length; d.inview = inview; d.px = G.cv.width + 'x' + G.cv.height;
}
function info(){ const i = G.sel; if(i < 0){ $('labinfo').innerHTML = ''; return; } const n = LN[i];
  const links = G.nb[i].map(j => '<button class="lk" data-i="'+j+'">'+esc(LN[j].lab)+'</button>').join(', ');
  $('labinfo').innerHTML = '<strong>'+esc(n.lab)+'</strong>'+(n.pi?' \u00b7 '+esc(n.pi):'')+(n.institution?' \u00b7 '+esc(n.institution):'')+(n.country?' \u00b7 '+esc(n.country):'')+(web(n.url)?' \u00b7 <a href="'+esc(n.url)+'" target="_blank" rel="noopener">site</a>':'')+
  (n.focus.length?'<br><span class="note">'+esc(n.focus.join(' '))+'</span>':'')+'<br><span class="meta">'+labPapers(n).length+' papers: '+esc(labPapers(n).slice(0,14).join(', '))+(labPapers(n).length>14?' \u2026':'')+'</span>'+
  (links?'<div class="rel">Linked to: '+links+'</div>':'<div class="note">No recorded link to another group.</div>'); }
$('labinfo').addEventListener('click', e => { const b = e.target.closest('button[data-i]'); if(b && G) locate(+b.dataset.i); });
// ---------------- citations: index/atlas_citations.js (positions FIXED by citation_map.py -- this only draws them)
const CD = window.__MWA_CITES || null;
let CG = null, cSel = -1, cGrp = -1;
const cNodeOf = new Map(); if(CD) CD.nodes.forEach((n, j) => cNodeOf.set(n[0], j));
function citeNode(pi){ return cNodeOf.has(pi) ? cNodeOf.get(pi) : -1; }
const cVis = j => sec < 0 || inSec(D.papers[CD.nodes[j][0]]);     // drawn: a node whose paper is in the section shown
function citesNote(){ const st = CD.stats;
  if(sec < 0){ $('cnote').innerHTML = esc(st.links.toLocaleString()) + ' citations among ' + esc(st.linked_papers.toLocaleString()) + ' papers of this index (OpenAlex reference lists; ' +
    esc(st.with_reference_list) + ' papers have one). ' + esc(st.isolated_papers) + ' papers have no citation link inside the index and are not drawn. ' +
    Math.round(st.within_topic_share * 100) + '% of citations stay inside one review topic; median citation age ' + esc(st.age_median) + ' years. ' +
    'Click a paper to see what it cites and what cites it; drag to pan, the buttons or Ctrl + scroll to zoom. Positions fixed ' + esc(CD.laid_out || '') + ' (citation_map.py).'; return; }
  let nl = 0; const lp = new Set(); CD.links.forEach(([a, b]) => { if(cVis(a) && cVis(b)){ nl++; lp.add(a); lp.add(b); } });
  $('cnote').innerHTML = esc(nl.toLocaleString()) + ' citations among ' + esc(lp.size.toLocaleString()) + ' papers of this section (OpenAlex reference lists; the whole atlas: ' +
    esc(st.links.toLocaleString()) + '). Papers outside the section are hidden; the positions are the whole atlas\u2019s map. ' +
    'Click a paper to see what it cites and what cites it; drag to pan, the buttons or Ctrl + scroll to zoom.'; }
function citesInit(){
  if(!CD){ $('cnote').textContent = 'The citation data (atlas_citations.js) is not beside this page: open the atlas from its folder, or rebuild it (build_bwm.py build).'; return false; }
  const N = CD.nodes.length, out = CD.nodes.map(() => []), inn = CD.nodes.map(() => []);
  CD.links.forEach(([a, b]) => { out[a].push(b); inn[b].push(a); });
  CG = {cv: $('ccv'), P: CD.nodes.map(n => ({x: n[1], y: n[2], pi: n[0], g: n[3]})), out: out, inn: inn, k: 1, k0: 1, tx: 0, ty: 0,
        W: 0, H: 0, dpr: 1, user: false, raf: 0, ptr: new Map(), last: [0, 0], moved: 0, mode: ''};
  CG.r = inn.map(l => 1.3 + 0.42 * Math.sqrt(l.length));
  citesNote();
  $('cgrp').innerHTML = '<option value="-1">All topic groups</option>' + CD.groups.map((g, i) => '<option value="' + i + '">Highlight: ' + esc(g) + '</option>').join('');
  $('clegend').innerHTML = '<span><span class="sw2" style="background:var(--c-sel)"></span>selected paper</span><span><span class="sw2" style="background:var(--c-out)"></span>papers it cites / the highlighted group</span>' +
    '<span><span class="sw2" style="background:var(--c-in)"></span>papers that cite it</span><span><span class="sw2" style="background:var(--c-node)"></span>other papers (size: citations received here)</span>';
  const cv = CG.cv, at = ev => { const b = cv.getBoundingClientRect(); return [ev.clientX - b.left, ev.clientY - b.top]; };
  cv.addEventListener('pointerdown', ev => { if(ev.pointerType === 'mouse' && ev.button !== 0) return; const p = at(ev);
    CG.ptr.set(ev.pointerId, p); CG.last = p; CG.moved = 0; CG.mode = 'pan'; try{ cv.setPointerCapture(ev.pointerId); }catch(e){} });
  cv.addEventListener('pointermove', ev => { if(!CG.ptr.has(ev.pointerId)) return; const p = at(ev);
    const dx = p[0] - CG.last[0], dy = p[1] - CG.last[1]; CG.last = p; CG.moved += Math.abs(dx) + Math.abs(dy);
    if(CG.moved > 3){ CG.tx += dx; CG.ty += dy; CG.user = true; cLater(); } });
  const up = ev => { if(!CG.ptr.has(ev.pointerId)) return; CG.ptr.delete(ev.pointerId);
    if(CG.moved <= 3 && ev.type === 'pointerup'){ const p = at(ev); cSelect(cHit(p[0], p[1])); } };
  ['pointerup', 'pointercancel'].forEach(t => cv.addEventListener(t, up));
  cv.addEventListener('wheel', ev => { if(!(ev.ctrlKey || ev.metaKey)) return; ev.preventDefault(); const p = at(ev);
    cZoom(Math.exp(-Math.max(-300, Math.min(300, ev.deltaY)) * 0.004), p[0], p[1]); }, {passive: false});
  cv.addEventListener('dblclick', ev => { const p = at(ev); cZoom(1.8, p[0], p[1]); });
  $('cin').addEventListener('click', () => cZoom(1.5, CG.W / 2, CG.H / 2));
  $('cout').addEventListener('click', () => cZoom(1 / 1.5, CG.W / 2, CG.H / 2));
  $('cfit').addEventListener('click', () => { cFit(); cDraw(); });
  if('ResizeObserver' in window) new ResizeObserver(() => cDraw()).observe(cv);
  try{ matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => cDraw()); }catch(e){}
  new MutationObserver(() => cDraw()).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  drawFlows(); drawBackbone(); return true;
}
let citesReady = false;
function startCites(){ if(!citesReady) citesReady = citesInit(); if(CG) cDraw(); }
function cFit(){ let bw = CD.box[0], bh = CD.box[1], ox = 0, oy = 0;
  if(sec >= 0){ const V = CG.P.filter((p, j) => cVis(j));          // a section: fit the nodes it shows
    if(V.length){ const xs = V.map(p => p.x), ys = V.map(p => p.y); ox = Math.min(...xs) - 20; oy = Math.min(...ys) - 20;
      bw = Math.max(...xs) - ox + 20; bh = Math.max(...ys) - oy + 20; } }
  CG.k = CG.k0 = Math.max(1e-3, Math.min((CG.W - 20) / bw, (CG.H - 20) / bh));
  CG.tx = (CG.W - bw * CG.k) / 2 - ox * CG.k; CG.ty = (CG.H - bh * CG.k) / 2 - oy * CG.k; CG.user = false; }
function cZoom(f, sx, sy){ const k = Math.max(CG.k0 * 0.6, Math.min(CG.k0 * 30, CG.k * f));
  CG.tx = sx - (sx - CG.tx) * k / CG.k; CG.ty = sy - (sy - CG.ty) * k / CG.k; CG.k = k; CG.user = true; cLater(); }
function cLater(){ if(!CG.raf) CG.raf = requestAnimationFrame(() => { CG.raf = 0; cDraw(); }); }
function cHit(sx, sy){ let best = -1, bd = 1e18; CG.P.forEach((p, i) => { if(!cVis(i)) return; const dx = p.x * CG.k + CG.tx - sx, dy = p.y * CG.k + CG.ty - sy, d = dx * dx + dy * dy;
    const rr = Math.max(CG.r[i] * Math.sqrt(CG.k / CG.k0), 5) + 3; if(d <= rr * rr && d < bd){ bd = d; best = i; } }); return best; }
function cSelect(j){ cSel = j; cInfo(); cDraw(); }
function cDraw(){
  if(!CG || $('p-cites').hidden) return; const cv = CG.cv, W = cv.clientWidth, H = cv.clientHeight; if(!W || !H) return;
  const dpr = window.devicePixelRatio || 1;
  if(W !== CG.W || H !== CG.H || dpr !== CG.dpr){ const first = !CG.W; CG.W = W; CG.H = H; CG.dpr = dpr; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); if(first || !CG.user) cFit(); }
  const c = cv.getContext('2d'), cs = getComputedStyle(document.documentElement), col = v => cs.getPropertyValue(v).trim();
  const C = {rule: col('--rule'), node: col('--c-node'), sel: col('--c-sel'), out: col('--c-out'), inn: col('--c-in'), ink: col('--ink'), panel: col('--panel'), font: col('--body')};
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const k = CG.k, X = p => p.x * k + CG.tx, Y = p => p.y * k + CG.ty, s = Math.sqrt(k / CG.k0);
  const outS = cSel >= 0 ? new Set(CG.out[cSel]) : null, inS = cSel >= 0 ? new Set(CG.inn[cSel]) : null;
  c.lineWidth = 1; c.strokeStyle = C.rule; c.globalAlpha = cSel >= 0 || cGrp >= 0 ? 0.25 : 0.6; c.beginPath();
  CD.links.forEach(([a, b]) => { if(!cVis(a) || !cVis(b)) return; c.moveTo(X(CG.P[a]), Y(CG.P[a])); c.lineTo(X(CG.P[b]), Y(CG.P[b])); }); c.stroke();
  if(cSel >= 0){ c.globalAlpha = 0.9; c.lineWidth = 1.5;
    [[CG.out[cSel], C.out], [CG.inn[cSel], C.inn]].forEach(([l, color]) => { c.strokeStyle = color; c.beginPath();
      l.forEach(j => { if(!cVis(j)) return; c.moveTo(X(CG.P[cSel]), Y(CG.P[cSel])); c.lineTo(X(CG.P[j]), Y(CG.P[j])); }); c.stroke(); }); }
  const order = CG.P.map((p, i) => i).filter(i => cVis(i)).sort((a, b) => CG.r[a] - CG.r[b]);
  order.forEach(i => { const p = CG.P[i]; let fill = C.node, a = 1, big = 1;
    if(cSel >= 0){ if(i === cSel){ fill = C.sel; big = 1.6; } else if(outS.has(i)){ fill = C.out; big = 1.25; } else if(inS.has(i)){ fill = C.inn; big = 1.25; } else a = 0.35; }
    else if(cGrp >= 0){ if(p.g === cGrp){ fill = C.out; } else a = 0.35; }
    c.globalAlpha = a; c.fillStyle = fill; c.beginPath(); c.arc(X(p), Y(p), Math.max(1.3, CG.r[i] * s * big), 0, 6.2832); c.fill();
    if(a === 1 && fill !== C.node){ c.globalAlpha = 1; c.lineWidth = 1; c.strokeStyle = C.panel; c.stroke(); } });
  c.globalAlpha = 1; c.font = '11px ' + C.font; c.lineJoin = 'round';
  const lab = []; if(cSel >= 0){ lab.push(cSel); CG.out[cSel].concat(CG.inn[cSel]).sort((a, b) => CG.r[b] - CG.r[a]).slice(0, 18).forEach(j => lab.push(j)); }
  CD.backbone.slice(0, 12).forEach(pi => { const j = citeNode(pi); if(j >= 0) lab.push(j); });
  const boxes = [[W - 132, 0, W, 46]], done = new Set();
  lab.forEach(j => { if(done.has(j) || !cVis(j)) return; done.add(j); const p = CG.P[j], t = D.papers[p.pi].k, w = c.measureText(t).width;
    let x = X(p) + 6, y = Y(p) + 4; if(x + w > W - 4) x = X(p) - 6 - w; if(x < 0 || y < 10 || y > H) return;
    const b = [x - 2, y - 11, x + w + 2, y + 3]; if(j !== cSel && boxes.some(o => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1])) return;
    boxes.push(b); c.strokeStyle = C.panel; c.lineWidth = 3; c.strokeText(t, x, y); c.fillStyle = C.ink; c.fillText(t, x, y); });
  cv.dataset.nodes = CG.P.length; cv.dataset.links = CD.links.length; cv.dataset.sel = cSel; cv.dataset.shown = order.length;
}
function pLink(pi, extra){ const p = D.papers[pi]; return '<button class="lk" data-cp="' + pi + '">' + esc(p.k) + '</button> ' + esc((p.t || '').slice(0, 90)) + (p.y ? ' (' + p.y + ')' : '') + (extra || ''); }
function cInfo(){ if(cSel < 0){ $('cinfo').innerHTML = ''; return; } const n = CG.P[cSel], p = D.papers[n.pi];
  const li = (arr, f) => arr.length ? '<ol>' + arr.slice(0, 40).map(f).join('') + (arr.length > 40 ? '<li>\u2026 ' + (arr.length - 40) + ' more</li>' : '') + '</ol>' : '<p class="note">none</p>';
  const pin = x => inSec(D.papers[x[0]]), cp = (CD.couple[String(n.pi)] || []).filter(pin), cc = (CD.cocite[String(n.pi)] || []).filter(pin);
  const out = CG.out[cSel].filter(cVis), inn = CG.inn[cSel].filter(cVis);
  $('cinfo').innerHTML = '<strong>' + esc(p.t) + '</strong> <span class="meta" style="display:inline">' + esc(p.k) + ' \u00b7 ' + (p.y || 'n.d.') + ' \u00b7 ' + esc(CD.groups[n.g] || '') + '</span>' +
    acts(p).replace('<button data-act="cites" data-i="' + p._i + '">In the citation map</button>', '') +
    '<div class="cols3"><div><h4><span class="sw2" style="background:var(--c-out)"></span>Cites here (' + out.length + ')</h4>' + li(out, j => '<li>' + pLink(CG.P[j].pi) + '</li>') + '</div>' +
    '<div><h4><span class="sw2" style="background:var(--c-in)"></span>Cited by here (' + inn.length + ')</h4>' + li(inn, j => '<li>' + pLink(CG.P[j].pi) + '</li>') + '</div>' +
    '<div><h4>Shares references with</h4>' + li(cp, x => '<li>' + pLink(x[0], ' \u2014 ' + x[1] + ' shared') + '</li>') +
    '<h4>Cited together with</h4>' + li(cc, x => '<li>' + pLink(x[0], ' \u2014 ' + x[1] + '\u00d7') + '</li>') + '</div></div>'; }
$('cinfo').addEventListener('click', e => { const b = e.target.closest('button[data-cp]'); if(!b) return; const pi = +b.dataset.cp, j = citeNode(pi);
  if(j >= 0) cLocate(j); else openPaper(pi); });
function cLocate(j){ cSelect(j); const p = CG.P[j]; CG.k = Math.max(CG.k, CG.k0 * 2.5); CG.tx = CG.W / 2 - p.x * CG.k; CG.ty = CG.H / 2 - p.y * CG.k; CG.user = true; cDraw();
  try{ $('ccv').scrollIntoView({block: 'center', behavior: 'smooth'}); }catch(e){} }
function openCites(pi){ if(!inSec(D.papers[pi])) setSection(-1, true); go('cites'); const j = citeNode(pi); if(j >= 0 && CG) cLocate(j); }
$('cq').addEventListener('input', () => { const q = $('cq').value.trim().toLowerCase(); if(!CG){ return; }
  const hits = q.length < 2 ? [] : CG.P.map((n, j) => j).filter(j => { const p = D.papers[CG.P[j].pi]; return cVis(j) && (p.k + ' ' + p.t).toLowerCase().indexOf(q) >= 0; });
  $('ccount').textContent = q.length < 2 ? '' : hits.length + ' in the network';
  $('csug').innerHTML = hits.slice(0, 10).map(j => '<button class="tbtn" data-cj="' + j + '">' + esc(D.papers[CG.P[j].pi].k) + '</button>').join(''); });
$('csug').addEventListener('click', e => { const b = e.target.closest('button[data-cj]'); if(b) cLocate(+b.dataset.cj); });
$('cgrp').addEventListener('input', () => { cGrp = +$('cgrp').value; cSel = -1; cInfo(); cDraw(); });
function drawFlows(){ const keep = new Set(scopeTopics()), ix = CD.topics.map((t, i) => i).filter(i => sec < 0 || keep.has(CD.topics[i][0]));
  const T = ix.map(i => CD.topics[i]), M = ix.map(i => ix.map(j => CD.flows[i][j])), n = T.length; if(!n){ $('cflows').innerHTML = ''; return; }
  const rows = M.map(r => r.reduce((a, b) => a + b, 0)), cell = 22, L = 210, Tp = 150, W = L + n * cell + 10, H = Tp + n * cell + 10;
  let max = 0; M.forEach((r, i) => r.forEach(v => { if(rows[i]) max = Math.max(max, v / rows[i]); }));
  let s = '<svg class="flowsvg" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Share of each topic\u2019s in-index citations going to each topic">';
  T.forEach((t, j) => { s += '<text transform="translate(' + (L + j * cell + cell * 0.65) + ',' + (Tp - 6) + ') rotate(-55)" style="fill:var(--muted)">' + esc(t[1]) + '</text>'; });
  T.forEach((t, i) => { s += '<text x="' + (L - 6) + '" y="' + (Tp + i * cell + cell * 0.68) + '" text-anchor="end" style="fill:var(--muted)">' + esc(t[1]) + ' (' + Math.round(rows[i]) + ')</text>';
    M[i].forEach((v, j) => { const sh = rows[i] ? v / rows[i] : 0;
      s += '<rect x="' + (L + j * cell) + '" y="' + (Tp + i * cell) + '" width="' + (cell - 2) + '" height="' + (cell - 2) + '" rx="2" style="fill:var(--c-out);fill-opacity:' + (sh / (max || 1)).toFixed(3) + ';stroke:var(--rule);stroke-width:.5">' +
        '<title>' + esc(t[1]) + ' \u2192 ' + esc(T[j][1]) + ': ' + v.toFixed(1) + ' citations (' + Math.round(sh * 100) + '% of the row)</title></rect>'; }); });
  $('cflows').innerHTML = s + '</svg>'; }
function drawBackbone(){ $('cback').innerHTML = CD.backbone.filter(pi => inSec(D.papers[pi])).map(pi => { const j = citeNode(pi);
  return '<li>' + pLink(pi, j >= 0 ? ' \u2014 cited ' + CG.inn[j].filter(cVis).length + '\u00d7 here' : '') + '</li>'; }).join(''); }
$('cback').addEventListener('click', e => { const b = e.target.closest('button[data-cp]'); if(!b) return; const j = citeNode(+b.dataset.cp); if(j >= 0) cLocate(j); });
// ---------------- figures: index/atlas_figures.js (tools/figure_bank.py); the images ride in the portable bundle
const FG = window.__MWA_FIGS || null;
let FL = null, gShown = 60, gRes = [];
const figsOf = new Map();
if(FG) FG.papers.forEach(r => { if(r[0] in byKey) figsOf.set(byKey[r[0]], r); });
function figCount(pi){ const r = figsOf.get(pi); return r ? r[2].length : 0; }
function figImg(sha, f, i){ return FG.base + sha.slice(0, 2) + '/' + sha + '/' + (f[3] || ('fig' + String(i).padStart(3, '0') + '-' + (f[0] || 'u') + '-p' + f[1] + '.webp')); }
function startFigs(){
  if(!FG){ $('gnote').textContent = 'The figure index (atlas_figures.js) is not beside this page: build it where the figure bank is (build_bwm.py figures), or open the portable bundle.'; return; }
  if(!FL){ FL = []; FG.papers.forEach(r => { const pi = byKey[r[0]]; if(pi === undefined) return; const p = D.papers[pi];
      r[2].forEach((f, i) => FL.push({pi: pi, sha: r[1], f: f, i: i + 1, text: (f[2] + ' ' + p.t + ' ' + p.k).toLowerCase()})); });
    fillSel($('gtop'), 'All topics', scopeTopics());
    $('gnote').innerHTML = esc(FG.figures.toLocaleString()) + ' figures from ' + esc(FG.papers.length.toLocaleString()) + ' papers (' + esc(FG.captioned.toLocaleString()) + ' with their caption), found by a layout detector and captioned from each PDF\u2019s own text (tools/figure_bank.py). ' +
      'Search words must all appear in the caption or the paper\u2019s title. The images ship in the portable bundle (build_bwm.py bundle); without them a card shows the caption only.'; }
  renderFigs(); }
function renderFigs(){ if(!FL) return; const ws = $('gq').value.trim().toLowerCase().split(/\s+/).filter(Boolean), top = $('gtop').value, cap = $('gcap').value;
  gRes = FL.filter(x => inSec(D.papers[x.pi]) && (!cap || x.f[2]) && (!top || D.papers[x.pi].top.indexOf(top) >= 0) && ws.every(w => x.text.indexOf(w) >= 0));
  if(ws.length){ const rr = {core: 0, high: 1, medium: 2, low: 3}, inCap = x => ws.every(w => x.f[2].toLowerCase().indexOf(w) >= 0) ? 0 : 1;
    gRes = gRes.map((x, n) => [inCap(x), rr[D.papers[x.pi].r] ?? 4, -(D.papers[x.pi].y || 0), n, x])    // the caption itself
      .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]).map(a => a[4]); }       // matches first
  $('gcount').textContent = gRes.length.toLocaleString() + ' figures';
  $('glist').innerHTML = gRes.slice(0, gShown).map((x, n) => { const p = D.papers[x.pi];
    return '<div class="fcard" data-g="' + n + '" data-k="' + esc(p.k) + '" tabindex="0" role="button" aria-label="Open figure"><div class="im"><img loading="lazy" alt="" src="' + esc(figImg(x.sha, x.f, x.i)) + '"></div>' +
      '<div class="cap">' + (x.f[2] ? esc(x.f[2]) : '<em>(no caption: a figure-like box)</em>') + '</div><div class="src">' + esc(p.k) + ' \u00b7 ' + (p.y || 'n.d.') + ' \u00b7 p. ' + x.f[1] + (x.f[0] ? ' \u00b7 Fig. ' + esc(x.f[0]) : '') + '</div></div>'; }).join('');
  $('glist').querySelectorAll('img').forEach(im => im.addEventListener('error', () => { im.parentNode.textContent = 'image in the portable bundle'; }, {once: true}));
  $('gmore').hidden = gRes.length <= gShown; }
let gTimer = 0;
$('gq').addEventListener('input', () => { gShown = 60; clearTimeout(gTimer); gTimer = setTimeout(() => {
  try{ history.replaceState(null, '', '#' + hashFor('figs=' + encodeURIComponent($('gq').value.trim()))); }catch(e){} renderFigs(); }, 160); });
['gtop', 'gcap'].forEach(id => $(id).addEventListener('input', () => { gShown = 60; renderFigs(); }));
$('gmore').addEventListener('click', () => { gShown += 120; renderFigs(); });
function figOpen(n){ const x = gRes[n]; if(!x) return; const p = D.papers[x.pi];
  $('glbbox').innerHTML = '<button class="x" id="glbx" aria-label="Close">\u00d7</button><img alt="" src="' + esc(figImg(x.sha, x.f, x.i)) + '">' +
    '<p>' + (x.f[2] ? esc(x.f[2]) : '<em>(no caption)</em>') + '</p><p class="meta">' + esc(p.t) + ' \u00b7 ' + esc(p.k) + ' \u00b7 ' + (p.y || 'n.d.') + ' \u00b7 page ' + x.f[1] + (x.f[0] ? ' \u00b7 Fig. ' + esc(x.f[0]) : '') + '</p>' +
    (p.lp ? '<p><a href="' + esc(p.lp) + '#page=' + x.f[1] + '" target="_blank" rel="noopener">Open the PDF at page ' + x.f[1] + '</a></p>' : '') + acts(p);
  const im = $('glbbox').querySelector('img'); im.addEventListener('error', () => { im.remove(); }, {once: true});
  $('glb').hidden = false; $('glbx').focus(); }
$('glist').addEventListener('click', e => { const c = e.target.closest('.fcard'); if(c) figOpen(+c.dataset.g); });
$('glist').addEventListener('keydown', e => { const c = e.target.closest('.fcard'); if(c && (e.key === 'Enter' || e.key === ' ')){ e.preventDefault(); figOpen(+c.dataset.g); } });
$('glb').addEventListener('click', e => { if(e.target.id === 'glb' || e.target.id === 'glbx') $('glb').hidden = true; if(e.target.closest('button[data-act]')) $('glb').hidden = true; });
document.addEventListener('keydown', e => { if(e.key === 'Escape' && !$('glb').hidden) $('glb').hidden = true; });
function openFigs(pi){ if(!inSec(D.papers[pi])) setSection(-1, true); go('figs'); $('gq').value = D.papers[pi].k; $('gcap').value = ''; gShown = 60; renderFigs(); }
// ---------------- trends
let trendsDone = false;
function svgBars(byYear){
  const ys = Object.keys(byYear).filter(y => +y >= 2010).sort(); if(!ys.length) return '';
  const W = 760, H = 260, L = 44, B = 28, T = 10; const max = Math.max(...ys.map(y => byYear[y].published + byYear[y].preprint));
  const step = (W - L - 8) / ys.length, bw = Math.max(4, step * 0.7), yv = v => H - B - (v / max) * (H - B - T);
  let s = '<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Papers per year, published and preprint">';
  for(let g=0; g<=4; g++){ const v = Math.round(max*g/4); s += '<line x1="'+L+'" x2="'+(W-4)+'" y1="'+yv(v)+'" y2="'+yv(v)+'" style="stroke:var(--rule)"/><text x="'+(L-6)+'" y="'+(yv(v)+4)+'" text-anchor="end" font-size="11" style="fill:var(--muted);font-family:var(--mono)">'+v+'</text>'; }
  ys.forEach((y,i) => { const x = L + i*step + (step-bw)/2, p = byYear[y].published, q = byYear[y].preprint;
    s += '<rect x="'+x+'" y="'+yv(p)+'" width="'+bw+'" height="'+(H-B-yv(p))+'" style="fill:var(--pub)"><title>'+y+': '+p+' published</title></rect>';
    s += '<rect x="'+x+'" y="'+yv(p+q)+'" width="'+bw+'" height="'+(yv(p)-yv(p+q))+'" style="fill:var(--pre)"><title>'+y+': '+q+' preprints</title></rect>';
    if(ys.length < 20 || i % 2 === 0) s += '<text x="'+(x+bw/2)+'" y="'+(H-10)+'" text-anchor="middle" font-size="11" style="fill:var(--muted);font-family:var(--mono)">'+y+'</text>'; });
  return s + '</svg><div class="legend"><span><span class="sw" style="background:var(--pub)"></span>published</span><span><span class="sw" style="background:var(--pre)"></span>preprint</span></div>';
}
function svgLines(series){
  const names = Object.keys(series).filter(k => Object.values(series[k]).some(v => v > 0)).sort(); if(!names.length) return '<p class="note">No complete series recorded.</p>';
  const ys = [...new Set(names.flatMap(n => Object.keys(series[n])))].filter(y => +y >= 2015).sort();
  const W = 760, H = 260, L = 40, B = 28, T = 10, max = Math.max(1, ...names.flatMap(n => ys.map(y => series[n][y] || 0)));
  const xv = i => L + i * (W - L - 12) / Math.max(1, ys.length - 1), yv = v => H - B - (v / max) * (H - B - T);
  const pal = ['var(--depth)','var(--mag)','var(--pub)','var(--pre)','var(--muted)','var(--ink)'];
  let s = '<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="arXiv papers per year by query">';
  for(let g=0; g<=4; g++){ const v = Math.round(max*g/4); s += '<line x1="'+L+'" x2="'+(W-4)+'" y1="'+yv(v)+'" y2="'+yv(v)+'" style="stroke:var(--rule)"/><text x="'+(L-6)+'" y="'+(yv(v)+4)+'" text-anchor="end" font-size="11" style="fill:var(--muted);font-family:var(--mono)">'+v+'</text>'; }
  ys.forEach((y,i) => { s += '<text x="'+xv(i)+'" y="'+(H-10)+'" text-anchor="middle" font-size="11" style="fill:var(--muted);font-family:var(--mono)">'+y+'</text>'; });
  names.forEach((n,k) => { const col = pal[k % pal.length]; s += '<polyline style="fill:none;stroke:'+col+';stroke-width:2" points="'+ys.map((y,i) => xv(i)+','+yv(series[n][y]||0)).join(' ')+'"/>';
    const last = ys.length-1; s += '<circle cx="'+xv(last)+'" cy="'+yv(series[n][ys[last]]||0)+'" r="3" style="fill:'+col+'"/>'; });
  return s + '</svg><div class="legend">' + names.map((n,k) => '<span><span class="sw" style="background:'+pal[k%pal.length]+'"></span>'+esc(n.replace(/_/g,' '))+'</span>').join('') + '</div>';
}
function yearCounts(ps){ const by = {}; ps.forEach(p => { if(!p.y) return; const y = String(p.y);   // build_bwm._index_trends
  by[y] = by[y] || {published: 0, preprint: 0}; by[y][p.pub ? 'published' : 'preprint']++; }); return by; }
let c2noted = false;
function drawTrends(){ if(trendsDone) return; trendsDone = true;
  $('c1').innerHTML = svgBars(sec < 0 ? D.trends.index : yearCounts(D.papers.filter(inSec))); $('c2').innerHTML = svgLines(D.trends.arxiv);
  if(D.trends.meta.length && !c2noted){ c2noted = true; $('c2note').textContent += ' Caveats: ' + D.trends.meta.join(' '); }
  $('docs').innerHTML = D.docs.map(d => '<li>'+esc(d.title)+' <span class="meta" style="display:inline">'+esc(d.file)+'</span></li>').join(''); }
// ---------------- the section banner row (atlas.toml [[sections]]): see `const SEC` above
const TITLE0 = document.querySelector('header h1').textContent, SUB0 = $('sub').textContent;
function renderSecs(){ if(!SEC) return; const box = $('secs'); box.hidden = false;
  const item = (i, label, n, tip) => { const on = i === sec, style = ' style="--sc:' + (i < 0 ? 'var(--muted)' : secColor(i)) + '"';
    const inner = '<span class="dot"></span>' + esc(label) + ' <span class="n">' + n + '</span>';
    if(SEC.mode === 'atlases'){ const href = i < 0 ? SEC.all : SL[i].href;   // its own page; none for an empty section
      return on || !href ? '<span class="sb"' + style + (on ? ' aria-current="page"' : '') + ' title="' + esc(tip) + '">' + inner + '</span>'
        : '<a class="sb"' + style + ' href="' + esc(href) + '" title="' + esc(tip) + '">' + inner + '</a>'; }
    return '<button class="sb"' + style + ' data-sec="' + i + '" aria-pressed="' + on + '"' + (i >= 0 && !n ? ' disabled' : '') +
      ' title="' + esc(i >= 0 && !n ? tip + ' (no papers yet)' : tip) + '">' + inner + '</button>'; };
  box.innerHTML = item(-1, 'All papers', SEC.total || D.papers.length, 'Every paper in the atlas') +
    SL.map((s, i) => item(i, s.t, s.n, s.st || s.t)).join(''); }
function scopedStats(){ const ps = D.papers.filter(inSec), w = ps.filter(p => p.s === 'withdrawn').length;
  let links = 0; if(CD) CD.links.forEach(([a, b]) => { if(cVis(a) && cVis(b)) links++; });
  const groups = LN.filter((n, i) => n.kind !== 'institution' && labIn(i)).length;
  return D.lex.filter(lexIn).length + ' terms \u00b7 ' + ps.length + ' papers (' + ps.filter(p => p.pub).length + ' published, ' + ps.filter(p => p.s === 'preprint').length + ' preprints' +
    (w ? ', ' + w + ' withdrawn' : '') + ') \u00b7 ' + (links ? links + ' citations among them \u00b7 ' : '') + groups + ' groups'; }
function setHeader(){ const s = sec >= 0 ? SL[sec] : null, h = document.querySelector('header');
  if(SEC && SEC.mode === 'banners'){ document.querySelector('header h1').textContent = s ? s.t : TITLE0; document.title = s ? s.t + ' \u2014 ' + TITLE0 : TITLE0; }
  $('sub').textContent = (s && s.st ? s.st + ' \u00b7 ' : '') + (s && SEC.mode === 'banners' ? scopedStats() : SUB0);
  h.classList.toggle('insec', !!s); if(s) h.style.setProperty('--sc', secColor(sec)); }
function setSection(i, quiet){ sec = i; renderSecs(); setHeader();
  const tops = scopeTopics(); ['ftop', 'ltop', 'ptop', 'gtop'].forEach(id => { const el = $(id), v = el.value;
    fillSel(el, 'All topics', tops); if(tops.indexOf(v) >= 0) el.value = v; });
  renderTasks(); fShown = 25; lShown = 120; pShown = 80; gShown = 60;
  renderFind(); renderLex(); renderPapers(); if(FL) renderFigs();
  if(citesReady && CG){ cSel = -1; cInfo(); citesNote(); drawFlows(); drawBackbone(); CG.user = false; CG.W = 0; cDraw(); }
  if(G) renderLabTable(); trendsDone = false; if(!$('p-trends').hidden) drawTrends();
  if(!quiet){ const tab = tabs.find(t => !$('p-' + t).hidden) || 'find'; try{ history.replaceState(null, '', '#' + hashFor(tab)); }catch(e){} } }
$('secs').addEventListener('click', e => { const b = e.target.closest('button[data-sec]'); if(b && !b.disabled) setSection(+b.dataset.sec); });
// a link can open a tab with its search: #find=<task or terms>, #figs=<words>, #cites=<paper key>, #papers=<words>;
// in `banners` mode #in=<section id>&... opens that section first
let start = '', arg = ''; try{ start = decodeURIComponent((location.hash || '').slice(1)); }catch(e){ start = ''; }
if(start.indexOf('in=') === 0){ const amp = start.indexOf('&'), id = start.slice(3, amp < 0 ? undefined : amp);
  start = amp < 0 ? '' : start.slice(amp + 1);
  if(SEC && SEC.mode === 'banners'){ const j = SL.findIndex(s => s.id === id); if(j >= 0 && SL[j].n) sec = j; } }
const eqAt = start.indexOf('='); if(eqAt >= 0){ arg = start.slice(eqAt + 1); start = start.slice(0, eqAt); }
if(!start){ try{ start = localStorage.getItem('mwa-tab') || ''; }catch(e){} }
renderTasks();
if(SEC) setSection(sec, true); else { renderLex(); renderPapers(); renderFind(); }
show(start || 'find');
if(arg){ if(start === 'find'){ $('fq').value = arg; renderFind(); } else if(start === 'figs'){ $('gq').value = arg; renderFigs(); }
  else if(start === 'cites' && arg in byKey){ openCites(byKey[arg]); } else if(start === 'papers'){ $('pq').value = arg; renderPapers(); } }
})();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())
