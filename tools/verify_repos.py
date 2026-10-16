"""verify_repos -- is there REALLY code for this paper, and can we use it? Check claimed repos; search for missing ones.

WHAT IT IS. The companion of `tools/verify_refs.py` for code. Give it the same kind of JSON list (title + arXiv id
and/or DOI, optionally a claimed `repo` / `code_url`), and for each paper it:
  1. CHECKS a claimed GitHub repo through the authenticated API (`gh api`): does it exist, is it archived, what is
     its licence (SPDX), stars, last push -- and does its README actually CITE the paper (the arXiv id, the DOI, or
     most of the title's distinctive words)? A repo that exists but never mentions the paper is reported as
     `EXISTS_UNLINKED`, not as the paper's code.
  2. SEARCHES GitHub for papers with no claimed repo: repositories whose README contains the arXiv id, then the
     DOI. A hit is a CANDIDATE until the same README check links it to the paper. Paper LISTS (awesome-lists,
     indexes citing many arXiv ids) are rejected, `*.github.io` project pages are noted but are not code, and a
     surviving hit is `CANDIDATE_BY_SEARCH` -- a search cannot tell official code from a re-implementation, so it
     never says 'official'. When several hits cite the paper, a live (unarchived) repo beats an archived one and
     more stars beat fewer; the others are named in the reason.

    python tools/verify_repos.py refs.json --out-dir out/repos            # refs.json rows: title, arxiv_id?, doi?, repo?
    python tools/verify_repos.py refs.json --out-dir out/repos --no-search
    python tools/verify_repos.py refs.json --out-dir out/repos --strategies all --cache-dir out/repos/.gh_cache
    python tools/verify_repos.py refs.json --out-dir out/repos --strategies all --search-all   # + strategy_recall.json

SEARCH STRATEGIES (`--strategies`, comma list or `all`; default `id,doi` = the 2026-09-24 behaviour):
  id        `"<arXiv id>" in:readme`
  doi       `"<DOI>" in:readme`
  title     `"<full title>" in:description,readme`            (titles over ~200 chars are cut at a word boundary)
  subtitle  `"<text after the colon>" in:description,readme`  (only when the title has a colon and >= 3 words follow)
  name      the paper's SHORT NAME in repo names: `"<name>" in:name`, plus its squashed form when it has punctuation
            ('V-JEPA 2' -> `vjepa2 in:name`). The name is the row's `name`, else the text before a colon when it is
            <= 3 words, else an acronym in parentheses.
Every hit, whichever strategy found it, passes the SAME README check before it counts. A repo found only by its name
whose README does not cite the paper is listed in `search_hits` (linked: false) for a human to look at, never counted.
`--search-all` searches rows that already have a claimed repo too, and writes `strategy_recall.json`: for every row
whose claimed repo VERIFIED, which strategies retrieved it -- the measurement that says which search to trust.

TWO PRECISION RULES FOR SEARCH HITS (added 2026-10-01, MEASURED). On the first title-search run, a human review of the
20 CANDIDATE_BY_SEARCH hits found 8 that were not the paper's code although their README cited it: daily-arXiv digests
and paper collections, projects that cite the paper as related work, and a fork of a different tool. So a search hit
now also has to (1) not look like a paper list by NAME (`LIST_NAME_HINTS`), and (2) name the paper PROMINENTLY --
in the repo name, its description, or the opening of its README (`prominently_names`) -- or it is kept in
`search_hits` as kind "cites only" and never counted. `--review <json>` measures the result against human verdicts
({"reviews": {repo: {"verdict": official | likely official | unofficial | not code}}}) and `--search-all` also
checks the prominence rule on the claimed repos that VERIFIED (an out-of-sample test of rule 2 on official code).

WHY. A literature sweep hands back `code_url`s that a model may have guessed, and a paper's "official code" is
often an empty placeholder, archived, or GPL -- which matters for flight code (METHODS_IN_USE flags GPL as viral).
The owner asked on 2026-09-23 whether we had found repos for the methods we cite; this answers it from the API.

EXIT CODES (repo contract): 0 = every claimed repo verified and linked; 1 = at least one claimed repo is missing or
does not cite its paper; 2 = something could not be measured (gh missing, rate limit, network).

WHY THE DOI SEARCH (MEASURED 2026-09-24, GitHub API). The arXiv-id-only search reported no code for AONeuS and
Z-Splat and only an archived student copy for SonarSplat, and the world-models review repeated that as "no public
code" (retracted: report/superseded_claims.py, id `wm-no-code-sonar-scene-papers`). AONeuS's README cites the DOI and
title but never the arXiv id, so `"2402.03309" in:readme` returns 0 repos; `"10.1145/3641519.3657446" in:readme`
returns kzhang2/aoneus. SonarSplat's README does carry 2504.00159 (inside badge / link URLs), yet the id search
returned only psu-nittanyaiadvance/26-S-Lockheed-1 (archived, 1 star); the DOI search returns
umfieldrobotics/sonar_splat first. Z-Splat is out of reach of ANY README search: its repo has no README on any
branch (the paper's reference [28] and its project page are the only links) -- see HONEST SCOPE.

HONEST SCOPE. GitHub only (no GitLab / project pages). "Linked" is a README text match, so a repo that cites the
paper only in a docs subfolder -- or has no README at all, like Z-Splat's -- reads as unlinked or is never found: a
false negative, never a false positive. NO_REPO_FOUND therefore means "no README-based search found one", not "the
paper has no code"; check the paper's own code link before saying so. Stars and pushes are a snapshot. A licence
of NOASSERTION means GitHub could not classify it: read the LICENSE file before reuse.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

VERIFIED, UNLINKED, MISSING, FOUND, NONE, UNMEASURED = (
    "VERIFIED_LINKED", "EXISTS_UNLINKED", "REPO_NOT_FOUND", "CANDIDATE_BY_SEARCH", "NO_REPO_FOUND", "COULD_NOT_MEASURE")
#: a README citing more distinct arXiv ids than this is a paper LIST, not one paper's code (measured 2026-09-23: the
#: first search pass returned awesome-lists and a physics-paper index as "the code" for GAIA-1 and others)
LIST_MIN_IDS = 12
VIRAL = ("GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0")
#: a repo NAME containing any of these is a paper list / digest, not one paper's code (2026-10-01 review: DailyArXiv x2,
#: dailypaper-skills, Paper-Collection all passed the README check as "code")
LIST_NAME_HINTS = ("awesome", "daily", "arxiv", "paper", "collection", "digest", "reading-list", "readinglist")
#: share of the title's distinctive words that must appear in the repo's name + description + README opening
PROMINENT_MIN = 0.6
#: how much of the README (markup stripped) counts as its "opening"
OPENING_CHARS = 800
STRATEGIES = ("id", "doi", "title", "subtitle", "name")
DEFAULT_STRATEGIES = ("id", "doi")
#: GitHub rejects a search query over 256 characters; a phrase this long is cut at a word boundary.
MAX_PHRASE = 200
#: set by --cache-dir: `gh api` answers are cached here keyed by the API path (a 404 is cached too). `net_calls`
#: counts real GitHub requests, so the rate-limit sleeps are skipped for answers served from the cache (MEASURED
#: 2026-10-01: a fully cached 96-paper re-run with --sleep 2 slept ~15 min for 3 real calls).
CACHE = {"dir": None, "net_calls": 0}
_STOP = set("the a an of for and with using via from into over under on in to by is are its our we based "
            "towards toward learning model models".split())


class GhError(Exception):
    """gh could not answer (not installed, rate-limited, network). Distinct from a 404."""


# ------------------------------------------------------------------------------------------- pure parts ----

def parse_repo(url):
    """'https://github.com/owner/name(.git)(/tree/...)' or 'owner/name' -> 'owner/name'; None if not GitHub."""
    if not url:
        return None
    s = str(url).strip()
    m = re.match(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:[/#?].*)?$", s, re.I)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.match(r"^([\w.-]+)/([\w.-]+)$", s)
    return f"{m.group(1)}/{m.group(2)}" if m and "." not in m.group(1) else None


def distinctive_words(title):
    return [w for w in re.findall(r"[a-z0-9]+", (title or "").lower()) if len(w) >= 4 and w not in _STOP]


def title_parts(title):
    """'AquaJEPA: Action-Conditioned ...' -> ('AquaJEPA', 'Action-Conditioned ...'); no colon -> (title, ''). Pure."""
    t = re.sub(r"\s+", " ", (title or "")).strip()
    if ":" in t:
        pre, post = t.split(":", 1)
        return pre.strip(), post.strip()
    return t, ""


def short_name(title, name=""):
    """The paper's short name for a repo-NAME search: the row's `name`, else the pre-colon text when it is <= 3 words
    and <= 30 characters, else the first acronym in parentheses ('(DINO-WM)'), else ''. Pure."""
    if name and name.strip():
        return name.strip()
    pre, post = title_parts(title)
    if post and 0 < len(pre.split()) <= 3 and len(pre) <= 30:
        return pre
    m = re.search(r"\(([A-Za-z][A-Za-z0-9-]{1,20})\)", title or "")
    return m.group(1) if m else ""


def _phrase(s, n=MAX_PHRASE):
    s = re.sub(r'["\s]+', " ", s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0]


def search_queries(row, strategies=DEFAULT_STRATEGIES):
    """[(strategy, GitHub search query)] for one row, in strategy order, de-duplicated. Pure."""
    title = row.get("title") or ""
    aid, doi = (row.get("arxiv_id") or "").strip(), (row.get("doi") or "").strip()
    pre, post = title_parts(title)
    out = []
    for s in strategies:
        if s == "id" and aid:
            out.append((s, f'"{aid}" in:readme'))
        elif s == "doi" and doi:
            out.append((s, f'"{doi}" in:readme'))
        elif s == "title" and len(title.split()) >= 2:
            out.append((s, f'"{_phrase(title)}" in:description,readme'))
        elif s == "subtitle" and len(post.split()) >= 3:
            out.append((s, f'"{_phrase(post)}" in:description,readme'))
        elif s == "name":
            nm = short_name(title, row.get("name") or "")
            if nm:
                out.append((s, f'"{nm}" in:name'))
                squashed = re.sub(r"[^A-Za-z0-9]", "", nm).lower()
                if squashed and squashed != nm.lower():
                    out.append((s, f"{squashed} in:name"))
    seen, uniq = set(), []
    for s, q in out:
        if q not in seen:
            seen.add(q)
            uniq.append((s, q))
    return uniq


def parse_strategies(text):
    """'all' / 'id,doi,name' -> a validated tuple in canonical order. Raises ValueError on an unknown name. Pure."""
    raw = (text or "").strip()
    want = STRATEGIES if raw == "all" else tuple(s.strip() for s in raw.split(",") if s.strip())
    bad = [s for s in want if s not in STRATEGIES]
    if bad:
        raise ValueError(f"unknown strategy {bad}; choose from {', '.join(STRATEGIES)} or 'all'")
    return tuple(s for s in STRATEGIES if s in want)


def readme_links_paper(readme, title="", arxiv_id="", doi=""):
    """Does a README cite THIS paper? arXiv id or DOI anywhere wins; otherwise >= 70% of the title's distinctive
    words (at least 3). Returns (bool, reason). Pure."""
    text = (readme or "").lower()
    if arxiv_id and arxiv_id.lower() in text:
        return True, f"README cites arXiv:{arxiv_id}"
    if doi and doi.lower() in text:
        return True, f"README cites doi:{doi}"
    words = distinctive_words(title)
    if len(words) >= 3:
        hit = sum(1 for w in set(words) if w in text) / len(set(words))
        if hit >= 0.7:
            return True, f"README carries {hit:.0%} of the title's distinctive words"
        return False, f"README carries only {hit:.0%} of the title's distinctive words"
    return False, "README does not cite the paper"


def classify_candidate(full, readme):
    """What KIND of repo is a search hit? 'list' (an awesome-list / paper index), 'project page' (*.github.io),
    or 'code (officialness unverified)'. A search cannot tell an official release from a re-implementation, so it
    never claims 'official'. Pure."""
    name = full.split("/")[-1].lower()
    ids = set(re.findall(r"\b\d{4}\.\d{4,5}\b", readme or ""))
    if any(h in name for h in LIST_NAME_HINTS) or "list" in name.split("-") or len(ids) > LIST_MIN_IDS:
        return "list"
    if name.endswith(".github.io"):
        return "project page"
    return "code (officialness unverified)"


def readme_opening(readme, n=OPENING_CHARS):
    """The first `n` characters of a README with HTML tags and image markup removed (link URLs kept: an arXiv id
    often lives in a badge link). Pure."""
    s = re.sub(r"<[^>]+>", " ", readme or "")
    s = re.sub(r"!\[[^\]]*\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:n]


def prominently_names(full, description, readme, title, arxiv_id="", doi="", name=""):
    """Does the repo name the paper where a repo names ITSELF -- its name, its description or its README opening --
    rather than only deeper in the README (a related-work citation)? (bool, reason). Pure."""
    head = " ".join([re.sub(r"[-_.]+", " ", full.split("/")[-1]), description or "", readme_opening(readme)]).lower()
    if arxiv_id and arxiv_id.lower() in head:
        return True, f"names arXiv:{arxiv_id} up front"
    if doi and doi.lower() in head:
        return True, f"names doi:{doi} up front"
    nm = short_name(title, name)
    sq = re.sub(r"[^a-z0-9]", "", nm.lower())
    if len(sq) >= 3 and sq in re.sub(r"[^a-z0-9]", "", head):
        return True, f"names '{nm}' up front"
    words = set(distinctive_words(title))
    if len(words) >= 2:
        hit = sum(1 for w in words if w in head) / len(words)
        if hit >= PROMINENT_MIN:
            return True, f"its name / description / README opening carry {hit:.0%} of the title's distinctive words"
    return False, "cites the paper only deeper in the README (related work?)"


def licence_flag(spdx):
    if not spdx:
        return "no licence file -- all rights reserved by default"
    if spdx == "NOASSERTION":
        return "unclassified -- read LICENSE"
    if spdx in VIRAL:
        return f"{spdx} -- copyleft (viral for flight code)"
    return spdx


def rank_candidates(cands):
    """[(full_name, facts, why, kind, ...)] -> best first: a live (unarchived) repo before an archived one, then more
    stars; ties keep search order. A heuristic for WHICH linked hit to report, never a claim that it is official.
    Pure."""
    return sorted(cands, key=lambda c: (bool((c[1] or {}).get("archived")), -int((c[1] or {}).get("stars") or 0)))


def exit_code(statuses):
    if any(s in (MISSING, UNLINKED) for s in statuses):
        return 1
    if any(s == UNMEASURED for s in statuses):
        return 2
    return 0


# ---------------------------------------------------------------------------------------------- I/O ----

def _cache_file(path):
    import hashlib
    return Path(CACHE["dir"]) / (hashlib.sha1(path.encode("utf-8")).hexdigest() + ".json")


def _cache_put(path, answer):
    if CACHE["dir"]:
        Path(CACHE["dir"]).mkdir(parents=True, exist_ok=True)
        _cache_file(path).write_text(json.dumps({"path": path, "answer": answer}), encoding="utf-8", newline="\n")


def gh_api(path, retries=3):
    """`gh api <path>` -> parsed JSON; None on 404; GhError when it could not be asked. With CACHE['dir'] set, an
    answer (a 404 included) is read from / written to disk, so a re-run asks GitHub nothing."""
    if CACHE["dir"] and _cache_file(path).exists():
        return json.loads(_cache_file(path).read_text(encoding="utf-8")).get("answer")
    if not shutil.which("gh"):
        raise GhError("gh CLI not on PATH")
    last = ""
    for i in range(retries):
        CACHE["net_calls"] = CACHE.get("net_calls", 0) + 1
        r = subprocess.run(["gh", "api", path], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            ans = json.loads(r.stdout or "null")
            _cache_put(path, ans)
            return ans
        last = (r.stderr or r.stdout).strip()
        if "Not Found" in last or "HTTP 404" in last:
            _cache_put(path, None)
            return None
        if "rate limit" in last.lower() or "HTTP 403" in last or "HTTP 5" in last:
            time.sleep(10 * (i + 1))
            continue
        break
    raise GhError(last[:200])


def repo_facts(full):
    d = gh_api(f"repos/{full}")
    if d is None:
        return None
    lic = (d.get("license") or {}).get("spdx_id")
    return {"repo": d.get("full_name"), "url": d.get("html_url"), "archived": bool(d.get("archived")),
            "stars": d.get("stargazers_count"), "pushed_at": (d.get("pushed_at") or "")[:10], "licence": lic,
            "licence_note": licence_flag(lic), "description": (d.get("description") or "")[:140]}


def readme_text(full):
    d = gh_api(f"repos/{full}/readme")
    if not d or "content" not in d:
        return ""
    try:
        return base64.b64decode(d["content"]).decode("utf-8", "replace")
    except Exception:
        return ""


def search_repos(query, limit=5):
    """Repositories matching a raw GitHub search query (`"phrase" in:readme`, `name in:name`, ...)."""
    import urllib.parse
    d = gh_api(f"search/repositories?q={urllib.parse.quote(query)}&per_page={limit}")
    return [it["full_name"] for it in (d or {}).get("items", [])]


def search_readme(term, limit=5):
    """Repositories whose README contains the exact string `term` (an arXiv id or a DOI)."""
    return search_repos(f'"{term}" in:readme', limit)


def collect_hits(row, strategies, limit=5, sleep=0.0):
    """Run every strategy's queries; -> {full_name: [strategies that retrieved it]} in first-retrieval order. Sleeps
    between searches (the search API allows 30 requests a minute)."""
    hits = {}
    calls_before = None
    for i, (strategy, q) in enumerate(search_queries(row, strategies)):
        if i and CACHE.get("net_calls", 0) != calls_before:
            time.sleep(sleep)                                # only after a REAL request, never after a cache hit
        calls_before = CACHE.get("net_calls", 0)
        for full in search_repos(q, limit):
            got = hits.setdefault(full, [])
            if strategy not in got:
                got.append(strategy)
    return hits


def vet_hits(hits, title, aid, doi, name=""):
    """{full_name: [strategies]} -> (linked code hits [(full, facts, why, kind, strategies)], project pages,
    search_hits rows for the record). Every hit gets the same README check, whichever strategy found it; lists are
    dropped, project pages set aside, a hit that cites the paper but does not name it PROMINENTLY becomes kind
    "cites only", and anything not counted stays visible in search_hits."""
    linked_hits, pages, rows = [], [], []
    for full, strategies in hits.items():
        text = readme_text(full)
        linked, why = readme_links_paper(text, title, aid, doi)
        kind = classify_candidate(full, text)
        facts = None
        if linked and kind == "code (officialness unverified)":
            facts = repo_facts(full)
            prom, pwhy = prominently_names(full, (facts or {}).get("description", ""), text, title, aid, doi, name)
            if not prom:
                kind = "cites only"
            why = f"{why}; {pwhy}"
        rows.append({"repo": full, "strategies": list(strategies), "linked": linked, "kind": kind, "why": why})
        if not linked or kind in ("list", "cites only"):
            continue                                       # a paper index / a citing project is never "the code"
        if kind == "project page":
            pages.append(full)
            continue
        linked_hits.append((full, facts or repo_facts(full), why, kind, list(strategies)))
    return linked_hits, pages, rows


def check_row(row, search=True, sleep=0.5, strategies=DEFAULT_STRATEGIES, search_all=False):
    """One paper -> a result record. A claimed repo is CHECKED; without one (or always, with `search_all`) the
    chosen strategies SEARCH. `found_by` names the strategies that retrieved the reported repo; `search_hits` lists
    every repo any strategy returned, linked or not."""
    title, aid, doi = row.get("title", ""), (row.get("arxiv_id") or "").strip(), (row.get("doi") or "").strip()
    claimed = parse_repo(row.get("repo") or row.get("code_url"))
    out = {"title": title, "arxiv_id": aid, "doi": doi, "claimed": claimed, "status": None, "reason": "", "facts": None,
           "found_by": [], "search_hits": []}
    if row.get("name"):
        out["name"] = row["name"]
    try:
        hits = {}
        queries = search_queries(row, strategies) if search else []
        if queries and (search_all or not claimed):
            hits = collect_hits(row, strategies, sleep=sleep)
        if claimed:
            facts = repo_facts(claimed)
            if facts is None:
                out.update(status=MISSING, reason=f"github.com/{claimed} does not exist")
            else:
                linked, why = readme_links_paper(readme_text(claimed), title, aid, doi)
                out.update(status=VERIFIED if linked else UNLINKED, reason=why, facts=facts)
                key = (facts.get("repo") or claimed).lower()
                out["found_by"] = next((s for f, s in hits.items() if f.lower() == key), [])
            if hits:
                out["search_hits"] = vet_hits(hits, title, aid, doi, row.get("name") or "")[2]
            if out["status"] == VERIFIED and search_all:
                text = readme_text(claimed)
                out["prominent"] = prominently_names(facts.get("repo") or claimed, facts.get("description", ""), text,
                                                     title, aid, doi, row.get("name") or "")[0]
            return out
        if not queries:
            out.update(status=NONE, reason="no claimed repo" + ("" if not search else " and nothing to search by"))
            return out
        linked_hits, pages, out["search_hits"] = vet_hits(hits, title, aid, doi, row.get("name") or "")
        if linked_hits:
            (cand, facts, why, kind, found_by), *rest = rank_candidates(linked_hits)
            also = f"; also cite it: {', '.join(c[0] for c in rest)}" if rest else ""
            out.update(status=FOUND, claimed=cand, reason=f"search, {kind}: {why}{also}", facts=facts,
                       found_by=found_by)
            return out
        searched = " or ".join(s for s in (aid and "id" in strategies and f"arXiv:{aid}",
                                           doi and "doi" in strategies and f"doi:{doi}") if s)
        others = [s for s in strategies if s not in ("id", "doi")]
        what = " / ".join(x for x in (searched and f"cites {searched}", others and f"matches by {'/'.join(others)}") if x)
        out.update(status=NONE, reason=f"no code repo's README {what}"
                   + (f" (project page only: {pages[0]})" if pages else ""))
        return out
    except GhError as e:
        out.update(status=UNMEASURED, reason=str(e))
        return out


def strategy_recall(results, strategies):
    """The measurement: over rows whose CLAIMED repo VERIFIED (the yardstick), how many did each strategy retrieve --
    and over rows with no claimed repo, how many linked finds each strategy contributed. Pure."""
    yard = [r for r in results if r["status"] == VERIFIED and r.get("search_hits") is not None]
    out = {"yardstick_rows": len(yard), "per_strategy": {}}
    for s in strategies:
        n = sum(1 for r in yard if s in (r.get("found_by") or []))
        finds = sum(1 for r in results if r["status"] == FOUND and s in (r.get("found_by") or []))
        out["per_strategy"][s] = {"retrieved_known": n, "recall": round(n / len(yard), 3) if yard else None,
                                  "finds_without_a_claimed_repo": finds}
    prom = [r for r in yard if "prominent" in r]
    out["prominence_rule_on_known_repos"] = {"checked": len(prom), "accepted": sum(r["prominent"] for r in prom),
                                             "rejected": [r["title"] for r in prom if not r["prominent"]]}
    union = sum(1 for r in yard if r.get("found_by"))
    out["any_strategy"] = {"retrieved_known": union, "recall": round(union / len(yard), 3) if yard else None}
    out["missed_by_all"] = [r["title"] for r in yard if not r.get("found_by")]
    return out


CODE_VERDICTS = ("official", "likely official", "unofficial")


def review_measure(results, review):
    """Precision of the search against HUMAN verdicts: of the repos reported CANDIDATE_BY_SEARCH, how many a person
    judged to be code; and of the reviewed code repos, how many the search still reports (any status that counts).
    `review` = {repo: {"verdict": ...}}; repos are compared case-insensitively. Pure."""
    rv = {k.lower(): v["verdict"] for k, v in review.items()}
    reported = [((r.get("facts") or {}).get("repo") or r.get("claimed") or "").lower() for r in results
                if r["status"] == FOUND]
    judged = [rv[x] for x in reported if x in rv]
    code_reviewed = [k for k, v in rv.items() if v in CODE_VERDICTS]
    counted = {((r.get("facts") or {}).get("repo") or r.get("claimed") or "").lower() for r in results
               if r["status"] in (FOUND, VERIFIED)}
    return {"reported_candidates": len(reported), "reviewed": len(judged),
            "judged_code": sum(v in CODE_VERDICTS for v in judged),
            "precision": round(sum(v in CODE_VERDICTS for v in judged) / len(judged), 3) if judged else None,
            "reviewed_code_repos": len(code_reviewed),
            "reviewed_code_still_counted": sum(k in counted for k in code_reviewed),
            "reviewed_code_dropped": sorted(k for k in code_reviewed if k not in counted),
            "not_code_still_reported": sorted(x for x in reported if rv.get(x) == "not code"),
            "unreviewed_candidates": sorted(x for x in reported if x not in rv)}


def write_outputs(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "code_repos.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8", newline="\n")
    lines = ["<!-- generated by tools/verify_repos.py from the GitHub API; do not hand-edit -->", "",
             "| paper | status | repo | licence | stars | last push | found by | why |",
             "|---|---|---|---|---|---|---|---|"]
    order = {VERIFIED: 0, FOUND: 1, UNLINKED: 2, MISSING: 3, NONE: 4, UNMEASURED: 5}
    for r in sorted(results, key=lambda r: (order.get(r["status"], 9), r["title"].lower())):
        f = r.get("facts") or {}
        repo = f"[{f['repo']}]({f['url']})" + (" (ARCHIVED)" if f.get("archived") else "") if f else (r.get("claimed") or "-")
        lines.append(f"| {r['title'][:80].replace('|', '/')} | {r['status']} | {repo} | {f.get('licence_note', '-')} | "
                     f"{f.get('stars', '-')} | {f.get('pushed_at', '-')} | {','.join(r.get('found_by') or []) or '-'} | "
                     f"{r['reason'].replace('|', '/')[:90]} |")
    (out_dir / "code_repos.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("refs", help="JSON list (or {'papers': [...]}) of {title, arxiv_id?, doi?, repo|code_url?}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-search", action="store_true", help="only check claimed repos; do not search for missing ones")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between search calls (search API: 30/min)")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES),
                    help=f"comma list from {','.join(STRATEGIES)}, or 'all' (default: id,doi)")
    ap.add_argument("--search-all", action="store_true",
                    help="search rows with a claimed repo too, and write strategy_recall.json (the measurement)")
    ap.add_argument("--cache-dir", default="", help="cache gh api answers here (a re-run then asks GitHub nothing)")
    ap.add_argument("--review", default="", help="human verdicts JSON ({'reviews': {repo: {'verdict': ...}}}): "
                                                 "measure the search's precision against them")
    a = ap.parse_args(argv)
    try:
        strategies = parse_strategies(a.strategies)
    except ValueError as e:
        ap.error(str(e))
    CACHE["dir"] = a.cache_dir or None
    data = json.loads(Path(a.refs).read_text(encoding="utf-8"))
    rows = data.get("papers", data) if isinstance(data, dict) else data
    results = []
    for i, row in enumerate(rows, 1):
        calls_before = CACHE.get("net_calls", 0)
        res = check_row(row, search=not a.no_search, sleep=a.sleep, strategies=strategies, search_all=a.search_all)
        results.append(res)
        print(f"[{i}/{len(rows)}] {res['status']:<18} {(res.get('claimed') or '-')[:40]:40s} "
              f"{','.join(res.get('found_by') or []) or '-':<22} "
              f"{row.get('title', '')[:50].encode('ascii', 'replace').decode()}")
        if CACHE.get("net_calls", 0) != calls_before:     # a fully cached row asked GitHub nothing: no pause
            time.sleep(a.sleep if (a.search_all or not (row.get("repo") or row.get("code_url"))) else 0.3)
    counts = write_outputs(results, a.out_dir)
    print("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if a.search_all or a.review:
        rec = strategy_recall(results, strategies)
        if a.review:
            rev = json.loads(Path(a.review).read_text(encoding="utf-8"))
            rec["review"] = review_measure(results, rev.get("reviews", rev))
            m = rec["review"]
            print(f"vs human review: {m['judged_code']} of {m['reviewed']} reviewed candidates are code "
                  f"(precision {m['precision']}); {m['reviewed_code_still_counted']} of {m['reviewed_code_repos']} "
                  f"reviewed code repos still counted; unreviewed candidates: {len(m['unreviewed_candidates'])}")
        (Path(a.out_dir) / "strategy_recall.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                                                              encoding="utf-8", newline="\n")
        print(f"strategy recall over {rec['yardstick_rows']} verified repos: "
              + ", ".join(f"{s}={v['recall']}" for s, v in rec["per_strategy"].items())
              + f", any={rec['any_strategy']['recall']}")
    return exit_code([r["status"] for r in results])


if __name__ == "__main__":
    sys.exit(main())
