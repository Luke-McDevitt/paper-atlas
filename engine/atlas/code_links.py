"""report/boat_world_model/code_links.py -- the CODE of every paper in the BlueBoat world-model index, found and checked.

WHY (2026-10-12, owner: "improve the github search tool if needed and find github repos for as many of the papers as
we can"). A searched-for repo is a guess; a repo the AUTHORS name is evidence. So candidates are collected from what
the authors state first, every candidate is checked through the GitHub API by tools/verify_repos.py, and only papers
with nothing stated are searched for.

    python report/boat_world_model/build_bwm.py code collect         # stated links -> <cache>/candidates.json
    python report/boat_world_model/build_bwm.py code verify          # check them -> papers/_code.json
    python report/boat_world_model/build_bwm.py code search --relevance high   # GitHub search for the rest
    python report/boat_world_model/build_bwm.py build                # the index, atlas and download list show it

SOURCES, most trusted first (`source` in _code.json):
  paper     a URL in the paper's own full text, in a sentence presenting it as the paper's code / release
            (`papers/_facts.json` code_own, tools/paper_facts.py -- held papers only)
  arxiv     a GitHub link in the arXiv abstract or comments (the authors' own words), via the arXiv API
  hf        Hugging Face Papers' linked repo for the arXiv id (`githubRepo`; mostly auto-detected from the abstract)
  page      a GitHub repo linked from the paper's project page (from the arXiv comments, the full text or the
            agents' links): ONE hop, the repo that page links most often
  agent     the literature agents' `code_url`
STATUS (`status`): VERIFIED_LINKED -- the repo exists and its README cites the paper (arXiv id, DOI or title);
STATED -- it exists and the paper / arXiv record names it, although its README does not cite the paper (an
official repo often never does); CANDIDATE_BY_SEARCH -- found by a GitHub search and passed verify_repos's README +
prominence rules (officialness unverified); EXISTS_UNLINKED (hf / page / agent named it, the README does not cite
the paper: shown, not counted); none when nothing was found.

CACHE: every network answer (arXiv, Hugging Face, project pages, `gh api`) is cached under `<cache>` (default: beside
the project's PDF folder, `<pdfs>_code_cache` -- `boat_wm_code_cache` here), so a re-run, or a run killed by a rate
limit, asks nothing twice.
HONEST SCOPE: GitHub only for checking (a GitLab / Bitbucket link is kept, unverified); "linked" is a README text
match; a search cannot tell an official repo from a re-implementation (verify_repos's rules, measured 2026-10-01).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

UA = {"User-Agent": "sonar-video-fusion-review/bwm-code-links"}
ARXIV_API = "https://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
HF_API = "https://huggingface.co/api/papers/{id}"
_REPO = re.compile(r"(?:https?://)?(?:www\.)?(github\.com|gitlab\.com|bitbucket\.org)/([\w.-]+)/([\w.-]+)", re.I)
_PAGE = re.compile(r"https?://[^\s<>\"')\],]+", re.I)
_NOT_A_REPO_OWNER = {"sponsors", "topics", "orgs", "features", "about", "marketplace", "settings", "login", "join",
                     "pricing", "site", "collections", "explore", "trending", "apps", "readme"}
SOURCE_RANK = {"paper": 0, "arxiv": 1, "hf": 2, "page": 3, "agent": 4}
COUNTED = ("VERIFIED_LINKED", "STATED", "CANDIDATE_BY_SEARCH")


def repo_of(url: str) -> Optional[str]:
    """'github.com/owner/name(.git)(/tree/...)' -> 'github.com/owner/name' (gitlab / bitbucket kept as such)."""
    m = _REPO.search(str(url or ""))
    if not m:
        return None
    host, owner, name = m.group(1).lower(), m.group(2), re.sub(r"\.git$", "", m.group(3)).rstrip(".")
    if owner.lower() in _NOT_A_REPO_OWNER or not name or name.lower() in ("issues", "pulls"):
        return None
    return f"{host}/{owner}/{name}"


# ------------------------------------------------------------------------------------------- the cache + net
class Transient(Exception):
    """A request that failed for a reason that may pass (rate limit, 5xx, network down): never cached."""


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


class Net:
    """GET with a disk cache keyed by URL; `get` is injectable for tests (it returns the body, None for a page that
    does not exist (404 / 410, cached), or raises Transient -- not cached, so the next run asks again)."""

    def __init__(self, cache: Path, get: Optional[Callable[[str], Optional[str]]] = None, pause: float = 0.0):
        self.dir = Path(cache) / "http"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.get_ = get or self._fetch
        self.pause = pause
        self.calls = 0

    def _fetch(self, url: str) -> Optional[str]:
        for i in range(3):
            try:                                         # arxiv.org 15 s, API 3 s; a URL urlsplit rejects
                _load_arxiv_pace(Path(__file__).resolve().parents[2] / "tools").wait(url)
            except ValueError:                           # ('https://[anonymized' from a comment) asks nothing
                return None                              # (fix check 2026-10-14: it ended `code --stage collect`)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                    return r.read(2_000_000).decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                if e.code in (404, 410):
                    return None
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(5 * (i + 1))
                    continue
                return None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                time.sleep(2 * (i + 1))
            except Exception:                            # noqa: BLE001 -- a malformed URL (http.client.InvalidURL)
                return None
        raise Transient(url)

    def get(self, url: str) -> Optional[str]:
        url = str(url or "").strip().split()[0] if str(url or "").strip() else ""   # 'https://x (dataset)' -> the URL
        if not re.match(r"(?i)^https?://", url):
            return None
        f = self.dir / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")
        if f.is_file():
            got = json.loads(f.read_text(encoding="utf-8"))
            if got.get("body") is not None or got.get("final"):   # an empty answer cached before 2026-10-13 may have
                return got.get("body")                            # been a rate limit: asked again
        self.calls += 1
        try:
            body = self.get_(url)
        except Transient:
            return None
        f.write_text(json.dumps({"url": url, "body": body, "final": True}), encoding="utf-8", newline="\n")
        if self.pause:
            time.sleep(self.pause)
        return body


# ------------------------------------------------------------------------------------------------- collect
def arxiv_links(ids: List[str], net: Net, batch: int = 100) -> Dict[str, dict]:
    """arXiv id -> {repos: [...], pages: [...]} from the abstract and comments (the authors' own words)."""
    out: Dict[str, dict] = {}
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        body = net.get(ARXIV_API.format(ids=",".join(chunk), n=len(chunk)))
        if not body:
            continue
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            continue
        for e in root.findall("a:entry", ns):
            aid = re.sub(r"v\d+$", "", (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1])
            text = " ".join(filter(None, [e.findtext("a:summary", "", ns), e.findtext("x:comment", "", ns)]))
            repos = sorted({r for r in (repo_of(m.group(0)) for m in _REPO.finditer(text)) if r})
            pages = sorted({u.rstrip(".") for u in _PAGE.findall(text) if not _REPO.search(u)
                            and "arxiv.org" not in u and "doi.org" not in u})
            if aid:
                out[aid] = {"repos": repos, "pages": pages[:3]}
    return out


def hf_repo(aid: str, net: Net) -> Optional[str]:
    body = net.get(HF_API.format(id=aid))
    if not body:
        return None
    try:
        return repo_of(json.loads(body).get("githubRepo") or "")
    except ValueError:
        return None


def page_repo(url: str, net: Net, title: str = "") -> Optional[str]:
    """The GitHub repo a project page links most often (badge / sponsor links excluded); None if it links none."""
    body = net.get(url)
    if not body:
        return None
    reps = Counter(r for r in (repo_of(m.group(0)) for m in _REPO.finditer(body)) if r and r.startswith("github.com/"))
    if not reps:
        return None
    best = reps.most_common()
    return best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None


def collect(papers: List[dict], facts: Dict[str, dict], net: Net, say=print, hop_limit: int = 400) -> Dict[str, list]:
    """key -> [{repo, source, url}] for every paper with a stated link, best source first."""
    cands: Dict[str, list] = {}

    def add(key, url, source):
        r = repo_of(url)
        if r and not any(c["repo"].lower() == r.lower() for c in cands.get(key, [])):
            cands.setdefault(key, []).append({"repo": r, "source": source, "url": str(url)})

    for p in papers:
        for u in (facts.get(p["key"]) or {}).get("code_own") or []:
            add(p["key"], u, "paper")
    ids = sorted({p["arxiv_id"] for p in papers if p.get("arxiv_id")})
    say(f"  arXiv: {len(ids)} ids")
    ax = arxiv_links(ids, net)
    pages: Dict[str, List[str]] = {}
    for p in papers:
        a = ax.get(p.get("arxiv_id") or "")
        if a:
            for r in a["repos"]:
                add(p["key"], r, "arxiv")
            pages[p["key"]] = list(a["pages"])
    say(f"  Hugging Face: {len(ids)} ids")
    for i, p in enumerate([p for p in papers if p.get("arxiv_id")], 1):
        r = hf_repo(p["arxiv_id"], net)
        if r:
            add(p["key"], r, "hf")
        if i % 250 == 0:
            say(f"    {i}/{len(ids)}")
    for p in papers:
        if p.get("code_url"):
            if repo_of(p["code_url"]):
                add(p["key"], p["code_url"], "agent")
            else:
                pages.setdefault(p["key"], []).append(p["code_url"])
        for u in (facts.get(p["key"]) or {}).get("code_own") or []:
            if not repo_of(u) and "github.io" in u:
                pages.setdefault(p["key"], []).append("https://" + u.split("://")[-1])
    hops = [(k, u) for k, us in sorted(pages.items()) if not cands.get(k) for u in us[:2]][:hop_limit]
    say(f"  project pages: {len(hops)} one-hop fetches")
    for k, u in hops:
        r = page_repo(u, net)
        if r:
            add(k, r, "page")
    for k in cands:
        cands[k].sort(key=lambda c: SOURCE_RANK[c["source"]])
    return cands


# -------------------------------------------------------------------------------------------------- verify
def verify(papers: List[dict], cands: Dict[str, list], VR, say=print) -> Dict[str, dict]:
    """key -> the best checked candidate (VERIFIED_LINKED > STATED > EXISTS_UNLINKED), with every candidate tried."""
    out: Dict[str, dict] = {}
    by_key = {p["key"]: p for p in papers}
    order = {"VERIFIED_LINKED": 0, "STATED": 1, "EXISTS_UNLINKED": 2, "REPO_NOT_FOUND": 3, "UNCHECKED": 4,
             "COULD_NOT_MEASURE": 5}
    for i, (key, cs) in enumerate(sorted(cands.items()), 1):
        p = by_key.get(key) or {}
        tried = []
        for c in cs:
            row = {"repo": c["repo"], "source": c["source"], "url": c["url"]}
            if not c["repo"].startswith("github.com/"):
                row.update(status="UNCHECKED", reason="not on GitHub: kept as stated, not checked")
                tried.append(row)
                continue
            full = c["repo"].split("/", 1)[1]
            try:
                facts = VR.repo_facts(full)
                if facts is None:
                    row.update(status="REPO_NOT_FOUND", reason=f"github.com/{full} does not exist")
                else:
                    linked, why = VR.readme_links_paper(VR.readme_text(full), p.get("title", ""),
                                                        p.get("arxiv_id") or "", p.get("doi") or "")
                    status = "VERIFIED_LINKED" if linked else ("STATED" if c["source"] in ("paper", "arxiv")
                                                               else "EXISTS_UNLINKED")
                    row.update(status=status, reason=why, facts=facts)
            except VR.GhError as e:
                row.update(status="COULD_NOT_MEASURE", reason=str(e)[:160])
            tried.append(row)
            if row["status"] == "VERIFIED_LINKED":
                break
        best = sorted(tried, key=lambda r: (order.get(r["status"], 9), SOURCE_RANK[r["source"]]))[0]
        out[key] = {**best, "tried": [{k: r[k] for k in ("repo", "source", "status")} for r in tried]}
        if i % 100 == 0:
            say(f"  checked {i}/{len(cands)} papers")
    return out


#: the TITLE phrase is searched too (every strategy's query runs for each paper): MEASURED by verify_repos on 36 known
#: repos (2026-10-01), the title retrieves 89 %, the arXiv id 42 %, the DOI 6 % -- id + doi alone missed most
SEARCH_STRATEGIES = ("title", "id", "doi")


def search(papers: List[dict], done: Dict[str, dict], VR, relevance=("core", "high"), strategies=SEARCH_STRATEGIES,
           sleep: float = 2.1, limit: Optional[int] = None, say=print, save=None, every: int = 50) -> Dict[str, dict]:
    """GitHub search (verify_repos.check_row) for the papers with no counted repo yet, most relevant first. `save`
    (if given) is called with the finds so far every `every` papers and at the end, so a stopped run keeps them (and
    every GitHub answer is cached: a re-run asks only what it has not asked)."""
    rank = {"core": 0, "high": 1, "medium": 2, "low": 3}
    todo = [p for p in papers if p["relevance"] in relevance and (done.get(p["key"]) or {}).get("status") not in COUNTED
            and (p.get("arxiv_id") or p.get("doi") or "title" in strategies)]
    todo.sort(key=lambda p: (rank.get(p["relevance"], 9), p["key"]))
    todo = todo[:limit] if limit else todo
    out = {}
    for i, p in enumerate(todo, 1):
        res = VR.check_row({"title": p["title"], "arxiv_id": p.get("arxiv_id") or "", "doi": p.get("doi") or ""},
                           search=True, sleep=sleep, strategies=tuple(strategies))
        if res["status"] == VR.FOUND:
            f = res.get("facts") or {}
            out[p["key"]] = {"repo": "github.com/" + (f.get("repo") or res["claimed"]), "source": "search",
                             "url": f.get("url", ""), "status": "CANDIDATE_BY_SEARCH", "reason": res["reason"],
                             "facts": f, "found_by": res.get("found_by") or []}
        elif res["status"] == VR.UNMEASURED:
            say(f"  stopped at {i}/{len(todo)}: {res['reason'][:120]} (re-run: answers so far are cached)")
            break
        if i % every == 0:
            say(f"  searched {i}/{len(todo)}: {len(out)} found")
            if save:
                save(out)
    if save:
        save(out)
    return out


def record(key_rows: Dict[str, dict]) -> List[dict]:
    """_code.json rows (one per paper, sorted), compact."""
    rows = []
    for key in sorted(key_rows):
        r = key_rows[key]
        f = r.get("facts") or {}
        rows.append({"key": key, "repo": r["repo"], "url": f.get("url") or ("https://" + r["repo"]),
                     "status": r["status"], "source": r["source"], "stars": f.get("stars"),
                     "licence": f.get("licence"), "licence_note": f.get("licence_note", ""),
                     "archived": f.get("archived"), "pushed_at": f.get("pushed_at", ""),
                     "reason": str(r.get("reason", ""))[:200], "tried": r.get("tried", []),
                     "found_by": r.get("found_by", [])})
    return rows
