r"""arXiv's own HTML (LaTeXML) for every held arXiv paper: fetched ONCE, version-matched to the held PDF, cached gzipped,
and parsed into what the PDF text layer cannot give -- the section tree, every equation as the authors' LaTeX, figures
with captions and image paths, tables, the bibliography, and WHICH SECTION cites WHAT.

    python tools/arxiv_html.py fetch --bwm [--limit N] [--retry]     # cache the pages (resumable; one request per 15 s)
    python tools/arxiv_html.py parse --bwm                             # -> <cache>/parsed/<id>.json.gz (no network)
    python tools/arxiv_html.py stats --bwm                             # coverage, versions, conversion errors
    python tools/arxiv_html.py show <key>                              # one paper's sections / equations / figures
    python tools/arxiv_html.py equations --bwm                         # lexicon equations vs the cited paper's LaTeX

Why (LIT_PAPER_MINING.md sec. 2.2, 2026-10-13): 40 of 40 sampled papers had HTML, 39 with their mathematics; the
lexicon's equations match the cited paper's LaTeX EXACTLY only 3 of 78 times, and token similarity cannot tell a right
equation from a one-symbol-wrong one -- so provenance needs the source, not a fuzzy match. arXiv converts every paper
since Dec 2023 (older ones on request); about a quarter of conversions carry some error (ltx_ERROR nodes, counted).

VERSIONS. arXiv serves the LATEST version by default; a held PDF may be older. The version comes from the arXiv stamp
on the held PDF's first page (`arXiv:2401.01234v2 [cs.RO] ...`; found on 1,383 of 1,414 held arXiv PDFs, 2026-10-14);
`<id>v<k>` is asked first, the latest only when that version has no HTML -- and the manifest records which was served.

CACHE (outside git): `<pdfs>_html/` beside the papers folder: `manifest.json` (one row per arXiv id: status ok /
no_html / error, the version asked and served, bytes, date), `pages/<id>v<k>.html.gz`, `parsed/<id>.json.gz`.
Courtesy: ONE REQUEST EVERY 15 s -- arxiv.org/robots.txt sets `Crawl-delay: 15` for every crawler (read
2026-10-14; the 3 s rule is only the API's, export.arxiv.org, whose robots.txt disallows crawling pages) -- a
descriptive User-Agent, Retry-After honoured on 429 / 503, nothing asked twice. ~1,400 papers = ~6 h: run it in
chunks (`--limit 400` is ~1.7 h).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import html as H
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_text as PT  # noqa: E402  (the shared text cache and the BlueBoat index)
if str(Path(__file__).resolve().parents[1]) not in sys.path:   # run by path: the repo root, for tools/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.atomic_replace import replace_retry  # noqa: E402

UA = "svfr-research/1.0 (academic literature mining; one request per 15 s)"
#: arxiv.org/robots.txt: "Crawl-delay: 15" for User-agent * (and ar5iv: the file covers *.arxiv.org mirrors)
CRAWL_DELAY = 15.0
STAMP = re.compile(r"arXiv:\s*([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(v\d+)")
#: LaTeXML puts each figure / object under `<id>v<k>/...`: the version actually served
SERVED = re.compile(r"""(?:src|data)="(?:https?://arxiv\.org/html/)?([^"/]*?)(v\d+)/""")


def default_cache(pdfs: Path) -> Path:
    p = Path(pdfs)
    return p.parent / (p.name + "_html")


def flat(aid: str) -> str:
    return aid.replace("/", "_")


# --------------------------------------------------------------------------------------------- what to fetch
def index_arxiv() -> Dict[str, str]:
    """{arXiv id (no version): key} of every index paper with an arXiv id, held here or not."""
    P = json.loads((PT.BWM / "index" / "papers.json").read_text(encoding="utf-8"))
    return {re.sub(r"v\d+$", "", p["arxiv_id"]): p["key"] for p in P["papers"] if p.get("arxiv_id")}


def index_aliases() -> Dict[str, str]:
    """{alias key: index key} -- a row filed under a key since folded into another is that paper's, not wrong."""
    P = json.loads((PT.BWM / "index" / "papers.json").read_text(encoding="utf-8"))
    return {a: p["key"] for p in P["papers"] for a in p.get("aliases") or []}


def held_arxiv(text_cache: Path) -> List[dict]:
    """Every held arXiv paper of the BlueBoat index: {key, arxiv_id (no version), version ('' if unknown), sha}."""
    names = PT.bwm_names()
    P = json.loads((PT.BWM / "index" / "papers.json").read_text(encoding="utf-8"))
    ax = {p["key"]: re.sub(r"v\d+$", "", p["arxiv_id"]) for p in P["papers"] if p.get("arxiv_id")}
    out = {}
    for sha, n in names.items():
        aid = ax.get(n["key"])
        if not aid or aid in out:
            continue
        f = PT.cache_file(text_cache, sha)
        ver = ""
        if f.is_file():
            page1 = f.read_text(encoding="utf-8").split(PT.FF)[0]
            ver = next((v for i, v in STAMP.findall(page1) if i == aid), "")
        out[aid] = {"key": n["key"], "arxiv_id": aid, "version": ver, "sha": sha}
    return [out[a] for a in sorted(out)]


# ----------------------------------------------------------------------------------------------------- fetch
def http_get(url: str, timeout: float = 90.0) -> Tuple[int, bytes, Dict[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b""), dict(e.headers or {})


def _load_manifest(cache: Path) -> Dict[str, dict]:
    f = Path(cache) / "manifest.json"
    return json.loads(f.read_text(encoding="utf-8")).get("papers", {}) if f.is_file() else {}


def _save_manifest(cache: Path, rows: Dict[str, dict], touched: Optional[set] = None) -> None:
    """Write the manifest. With `touched`, MERGE: rows on disk are kept and only the touched ids are replaced -- two
    writers can never erase each other's rows (2026-10-14: an orphaned fetcher and a new one rewrote their own
    snapshots and the manifest lost 300 fetched papers; the page files survived and `repair` re-listed them)."""
    f = Path(cache) / "manifest.json"
    if touched is not None:
        rows = {**_load_manifest(cache), **{k: rows[k] for k in touched if k in rows}}
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"schema": "svfr.arxiv_html/1", "papers": {k: rows[k] for k in sorted(rows)}},
                              indent=0, ensure_ascii=False), encoding="utf-8", newline="\n")
    # Windows refuses the replace while ANOTHER process is reading the manifest (2026-10-14: a 15-s-paced fetch died
    # on WinError 32 after 46 papers because an eval script had it open): retried, then raised naming it.
    replace_retry(tmp, f)


def pid_alive(pid: int) -> bool:
    """Is this exact pid a live process? Repeated from tools/octree_cache.pid_alive (the repo keeps such checks local):
    Windows ignores a pid's low two bits, so the handle is asked which process it names; an OpenProcess failure other
    than ERROR_INVALID_PARAMETER means 'cannot tell' and reads as alive (fail closed)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetProcessId.restype = wintypes.DWORD
        k32.GetProcessId.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x1000, False, pid)
        if not h:
            return ctypes.get_last_error() != 87
        try:
            named = k32.GetProcessId(h)
            if named and named != pid:
                return False
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FetchLock:
    """ONE fetcher per cache (2026-10-14: a stopped task's python process kept fetching at the old 3 s pace beside the
    new 15 s one). `<cache>/fetch.lock` holds the owner's pid; a live owner makes a second fetcher refuse to start."""

    def __init__(self, cache: Path):
        self.path = Path(cache) / "fetch.lock"

    STALE_S = 3600                                     # every request refreshes the lock (fetch.ask), so an hour with
    #                                                    no request means its owner is gone or its pid was reused

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:                                       # O_EXCL: two fetchers started together cannot both win
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    text = self.path.read_text(encoding="utf-8")
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue                           # released between our two looks: try again
                except OSError:
                    text, age = "", self.STALE_S + 1
                try:
                    pid = int(text.split()[0])
                except (ValueError, IndexError):
                    pid = 0
                if pid and pid != os.getpid() and pid_alive(pid) and age < self.STALE_S:
                    raise SystemExit(f"fetch: another fetcher (pid {pid}) holds {self.path} -- one at a time, so the "
                                     f"crawl delay holds; stop it first (or delete {self.path} if pid {pid} is not one)")
                moved = self.path.with_name(f"{self.path.name}.{os.getpid()}.stale")
                try:                                   # break it ATOMICALLY: move it aside, then check that what moved
                    replace_retry(self.path, moved)       # is the lock judged stale -- not one another fetcher has just
                except FileNotFoundError:              # taken (fix check: an unlink deleted a fresh lock)
                    continue
                if moved.read_text(encoding="utf-8") != text:
                    replace_retry(moved, self.path)       # someone else's fresh lock: put it back, and look again
                    continue
                moved.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()} {datetime.datetime.now().isoformat(timespec='seconds')}\n")
            return self
        raise SystemExit(f"fetch: could not take {self.path}")

    def __exit__(self, *exc):
        try:
            if self.path.read_text(encoding="utf-8").split()[0] == str(os.getpid()):
                self.path.unlink()
        except (OSError, IndexError):
            pass
        return False


def repair(cache: Path, items: List[dict], today: str = "", index: Optional[Dict[str, str]] = None,
           aliases: Optional[Dict[str, str]] = None) -> dict:
    """No network. (1) Re-list cached pages the manifest lost: every pages/<id>v<k>.html.gz whose id has no 'ok' row
    gets one (version from the file name, version_held from the PDF stamp, its size). (2) Re-validate every 'ok' row's
    page with today's is_latexml: a placeholder (ar5iv's 'No content available') becomes no_html. Saves MERGE, under
    the fetch lock."""
    cache = Path(cache)
    with FetchLock(cache):
        rows = _load_manifest(cache)
        by_flat = {flat(it["arxiv_id"]): it for it in items}
        touched, added, demoted = set(), 0, 0
        for f in sorted((cache / "pages").glob("*.html.gz")):
            m = re.match(r"(.+?)(v\d+)?\.html\.gz$", f.name)
            it = by_flat.get(m.group(1)) if m else None
            if it is None or rows.get(it["arxiv_id"], {}).get("status") == "ok":
                continue
            body = gzip.decompress(f.read_bytes())
            if not is_latexml(body):
                continue
            served, ver = m.group(2) or "", it.get("version") or ""
            rows[it["arxiv_id"]] = {"key": it["key"], "version_held": ver, "status": "ok", "date": today,
                                    "url": "", "version_served": served, "file": f.name, "bytes": len(body),
                                    "version_match": bool(ver) and served == ver, "source": "repaired from the cache"}
            touched.add(it["arxiv_id"])
            added += 1
        retired = revived = 0                          # (3) with `index` ({arXiv id: key}, index_arxiv()): retire a
        for aid, r in rows.items():                    # row whose id is no longer that key's -- a WRONG id removed
            if index is None:                          # from the index (review 2026-10-14); by the INDEX, never by
                break                                  # what this machine holds
            if index.get(aid) and r.get("status") == "not_indexed":   # the index gives a retired id to a key AGAIN:
                r.update(key=index[aid], status="error",               # live, so `fetch --retry` asks (third fix
                         why=f"the index gives arXiv {aid} to {index[aid]} again (repair): fetch --retry asks again")
                touched.add(aid)                                       # check 2026-10-15: it stayed dead for good)
                revived += 1
                continue
            if index.get(aid) and index.get(aid) != r.get("key"):
                r["key"] = index[aid]                  # the page IS that id's paper: filed under the key the index
                touched.add(aid)                       # gives it now (an alias, a renamed key), never retired
                continue
            if r.get("status") in ("ok", "no_html", "error") and index.get(aid) != r.get("key"):
                r.update(status="not_indexed", why=f"arXiv {aid} is not {r.get('key')}'s id in the index (repair)")
                touched.add(aid)
                retired += 1
        for aid, r in rows.items():
            if r.get("status") != "ok":
                continue
            f = cache / "pages" / r.get("file", "-")
            if not f.is_file():                        # a lost file is not 'arXiv has no HTML': --retry asks again
                r.update(status="error", why="page file missing (repair): fetch --retry asks again")
                touched.add(aid)
                demoted += 1
                continue
            body = gzip.decompress(f.read_bytes())
            if not is_latexml(body):
                r.update(status="no_html", why="placeholder page (re-validated by repair)")
                touched.add(aid)
                demoted += 1
            elif not r.get("bytes"):
                r["bytes"] = len(body)
                touched.add(aid)
        _save_manifest(cache, rows, touched=touched)
    return {"re_listed": added, "now_no_html": demoted, "retired_not_indexed": retired, "revived": revived,
            "manifest": len(rows)}


def is_latexml(body: bytes) -> bool:
    """A real LaTeXML rendering: the document class AND some content -- ar5iv answers 200 with an empty
    <article class="ltx_document"> titled 'No content available' (13 rows had been cached as 'ok'; review
    2026-10-14)."""
    if not (b"ltx_document" in body or b"ltx_page_main" in body) or b"No content available" in body[:20000]:
        return False
    return any(m in body for m in (b"ltx_para", b"ltx_section", b"ltx_title_document"))


def retry_after_s(wait: str, default: float, cap: float = 3600.0) -> float:
    """A Retry-After header -> seconds: delta-seconds or an HTTP date (fix check 2026-10-14: a date was ignored and the
    retry went out ~45 s after a 503 that asked for 120 s); anything else -> default."""
    w = (wait or "").strip()
    try:
        return min(cap, max(0.0, float(w)))
    except ValueError:
        pass
    try:
        import email.utils
        dt = email.utils.parsedate_to_datetime(w)
        return min(cap, max(0.0, (dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()))
    except (TypeError, ValueError, IndexError):
        return default


def fetch(items: List[dict], cache: Path, pause: float = CRAWL_DELAY, limit: int = 0, retry: bool = False,
          get: Callable = http_get, sleep: Callable = time.sleep, say=print, today: str = "") -> dict:
    """Fetch what the manifest lacks (or, with retry, what failed). Resumable: the manifest is written after each paper.
    One request per `pause` seconds, always -- also after a 404."""
    cache = Path(cache)
    (cache / "pages").mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(cache)
    todo = [it for it in items if it["arxiv_id"] not in rows
            or (retry and rows[it["arxiv_id"]].get("status") == "error")]
    if limit:
        todo = todo[:limit]
    n_req = 0
    stamp, lock = cache / "last_request", cache / "fetch.lock"

    def ask(url):
        nonlocal n_req
        for attempt in range(4):
            if n_req:
                sleep(pause)
            elif stamp.is_file():                        # the previous RUN's last request counts too (review: a new
                try:                                     # chunk's first request went out 0.6 s after the last one)
                    since = time.time() - float(stamp.read_text(encoding="utf-8").strip() or 0)
                except (OSError, ValueError):
                    since = pause
                if since < pause:
                    sleep(pause - since)
            n_req += 1
            stamp.write_text(f"{time.time():.3f}", encoding="utf-8")
            if lock.is_file():                           # a live fetcher keeps its lock fresh (FetchLock.STALE_S)
                os.utime(lock)
            st, body, hdr = get(url)
            if st in (429, 503) and attempt < 3:              # told to slow down: wait as asked, then once more
                wait = hdr.get("Retry-After") or hdr.get("retry-after") or ""
                sleep(retry_after_s(wait, 30.0 * (attempt + 1)))
                continue
            return st, body
        return st, body

    for i, it in enumerate(todo, 1):
        aid, ver = it["arxiv_id"], it.get("version") or ""
        urls = ([f"https://arxiv.org/html/{aid}{ver}"] if ver else []) + [f"https://arxiv.org/html/{aid}",
                                                                          f"https://ar5iv.labs.arxiv.org/html/{aid}"]
        row = {"key": it["key"], "version_held": ver, "status": "no_html", "date": today}
        for url in urls:
            try:
                st, body = ask(url)
            except Exception as e:                                # noqa: BLE001 -- recorded, retried with --retry
                row.update(status="error", why=f"{type(e).__name__}: {e}"[:200], url=url)
                break
            if st == 200 and is_latexml(body):
                m = SERVED.search(body[:400000].decode("utf-8", "replace"))
                served = m.group(2) if m and m.group(1) == aid.split("/")[-1] else (ver if url.endswith(ver) and ver else "")
                name = f"{flat(aid)}{served}.html.gz"
                (cache / "pages" / name).write_bytes(gzip.compress(body, 9))
                row.update(status="ok", url=url, version_served=served, file=name, bytes=len(body),
                           version_match=bool(ver) and served == ver,
                           source="ar5iv" if "ar5iv" in url else "arxiv")
                row.pop("why", None)
                row.pop("http", None)                    # the 404 of an earlier URL is not this row's (fix check)
                break
            if st not in (200, 404):                     # not a definite 'no HTML': stop, --retry asks the
                row.update(status="error", why=f"HTTP {st}", url=url)   # held version again (a latest version
                break                                        # served after a hiccup was saved as 'ok' -- review)
            row.update(status="no_html", url=url, http=st)
        rows[aid] = row
        _save_manifest(cache, rows, touched={aid})
        say(f"  [{i}/{len(todo)}] {aid}{ver or ''} {it['key']}: {row['status']}"
            + (f" (served {row.get('version_served') or '?'})" if row["status"] == "ok" else f" {row.get('why', '')}"))
    c = collections.Counter(r["status"] for r in rows.values())
    return {"asked": len(todo), "requests": n_req, "manifest": dict(c)}


# ----------------------------------------------------------------------------------------------------- parse
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class Node:
    __slots__ = ("tag", "attrs", "kids", "parent")

    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.kids, self.parent = tag, attrs, [], parent

    def cls(self) -> List[str]:
        return (self.attrs.get("class") or "").split()

    def has(self, c: str) -> bool:
        return c in self.cls()

    def iter(self):
        st = [self]
        while st:
            n = st.pop()
            yield n
            st.extend(k for k in reversed(n.kids) if isinstance(k, Node))


class _Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {}, None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, {k: (v or "") for k, v in attrs}, self.cur)
        self.cur.kids.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.kids.append(Node(tag, {k: (v or "") for k, v in attrs}, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:          # close up to the matching open tag; a stray end is ignored
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.kids.append(data)


def tree(html: str) -> Node:
    t = _Tree()
    t.feed(html)
    t.close()
    return t.root


def text_of(n: Node, math: bool = True) -> str:
    """Visible text; a <math> becomes $<its LaTeX>$ (the alttext), never its MathML spelling-out."""
    out = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
            return
        if x.tag == "math":
            if math and x.attrs.get("alttext"):
                out.append(" $" + x.attrs["alttext"] + "$ ")
            return
        if x.tag in ("script", "style", "annotation", "annotation-xml") or x.has("ltx_tag_equation"):
            return
        for k in x.kids:
            walk(k)
    walk(n)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def raw_text(n: Node) -> str:
    """Every text node under n (text_of skips equation tags so "(1)" never lands in a paragraph; this does not)."""
    out = [x for x in _texts(n)]
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _texts(n):
    for k in n.kids:
        if isinstance(k, str):
            yield k
        else:
            yield from _texts(k)


#: a section title -> its role; first match wins, a child section without a match inherits its parent's role
ROLES = [("references", r"^(?:references?|bibliography|works cited|literature cited)$"),
         ("acknowledgments", r"^acknowledge?ments?\b"),
         ("appendix", r"\bappendi|supplementa"),
         ("related_work", r"related work|prior work|previous work|literature|background|state of the art|preliminar"),
         ("introduction", r"^(?:introduction|motivation)\b"),
         ("conclusion", r"conclu|future work|limitation|discussion|summary"),
         ("experiments", r"experiment|evaluat|result|simulat|field|trial|test|setup|dataset|benchmark|ablation|"
                         r"implementation|deploy|case stud|validation|comparison"),
         ("method", r"method|approach|proposed|model|system|framework|architecture|design|algorithm|formulation|"
                    r"problem|theor|control|planning|learning|estimation|navigation|perception|mapping")]


def role_of(title: str) -> str:
    """A section title -> its role. 'references' / 'acknowledgments' / 'introduction' only for a whole title (review
    2026-10-14: 'IV-A Reference Trajectory Tracking' read as the bibliography, 'System Overview' as the introduction)."""
    # numbering only: '2.3', 'IV-A', 'III', 'A' -- not an all-caps first word ('RELATED WORK' kept; fix check)
    t = re.sub(r"^(?:(?:\d+|[IVXLC]+|[A-Z])(?:[.\-]\d+|-[A-Z])*\.?)\s+", "", (title or "").strip()).lower().strip(" .:")
    return next((r for r, pat in ROLES if re.search(pat, t)), "")


def _in_eq_table(n: Node) -> bool:
    p = n.parent
    while p is not None:
        if p.tag == "table" and (p.has("ltx_equation") or p.has("ltx_equationgroup")):
            return True
        p = p.parent
    return False


#: a relation sign: a row without one that starts with an operator continues the row above
RELATION = re.compile(r"=|<|>|\\(?:leq?|geq?|leqslant|geqslant|approx|sim|simeq|propto|equiv|neq?|in|subset|subseteq|"
                      r"cong|triangleq|doteq|coloneqq|mapsto|to|gets|ll|gg|prec|succ)(?![a-zA-Z])")
CONTINUES = re.compile(r"^(?:\s|&|\\q?quad(?![a-zA-Z])|\\[,;:!]|\{\})*(?:[+\-]|\\(?:cdot|times|pm|mp)(?![a-zA-Z]))")


def _equation_rows(table: Node, section: str) -> List[dict]:
    r"""An equation table -> one entry per EQUATION: LaTeXML writes each alignment cell of an align / eqnarray as its own
    <math>, so a row's LaTeX is its cells joined by a space (review 2026-10-14: 2,821 of 18,844 display equations were
    stored as fragments no lexicon equation could equal; '' fused '\leq' into the next cell). A row with no relation
    that starts with an operator ('\quad+\lambda\|\theta\|^2') CONTINUES the row above and is joined to it, never an
    equation of its own (fix check: alone, the first line matched an equation missing its last term). A row of 4+
    cells also offers each cell pair ('m\dot{v}' '=mg+...') -- align often sets two equations side by side. A multi-row
    group adds one entry for the whole block, flagged 'group' (left out of the equation counts)."""
    rows: List[dict] = []
    for tr in (k for k in table.iter() if k.tag == "tr"):
        cells = [re.sub(r"[\\](?:displaystyle)\s*", "", H.unescape(m.attrs.get("alttext", ""))).strip()
                 for m in tr.iter() if m.tag == "math" and m.attrs.get("alttext")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        holder = tr.parent if tr.parent is not None and tr.parent.tag == "tbody" else tr
        tag = next((raw_text(m) for m in tr.iter() if m.tag == "span" and m.has("ltx_tag_equation")), "").strip("()")
        text = " ".join(cells)
        if rows and CONTINUES.match(text) and not RELATION.search(text):
            last = rows[-1]                            # a continuation line: part of the equation above
            last["latex"] = [last["latex"][0] + " " + text]
            last["tag"] = last["tag"] or tag
            continue
        latex = [text]
        if len(cells) >= 4:
            latex += [" ".join(cells[i:i + 2]) for i in range(0, len(cells) - 1, 2)]
        rows.append({"id": holder.attrs.get("id") or table.attrs.get("id", ""), "tag": tag, "latex": latex,
                     "section": section})
    if len(rows) > 1:
        rows.append({"id": table.attrs.get("id", ""), "tag": "", "section": section, "group": True,
                     "latex": [" \\\\ ".join(r["latex"][0] for r in rows)]})
    return rows


def _nearest_figure(n: Node) -> Optional[Node]:
    p = n.parent
    while p is not None and p.tag != "figure":
        p = p.parent
    return p


def _section_of(n: Node, sec_ids: Dict[int, str]) -> str:
    p = n.parent
    while p is not None:
        if id(p) in sec_ids:
            return sec_ids[id(p)]
        p = p.parent
    return ""


def parse(html: str) -> dict:
    """LaTeXML HTML -> {title, abstract, sections, equations, inline_math, figures, tables, bib, cites, errors}."""
    root = tree(html)
    secs, sec_ids, by_id = [], {}, {}
    for n in root.iter():
        if n.tag == "section" or (n.tag == "div" and n.has("ltx_appendix")):
            sid = n.attrs.get("id") or f"sec{len(secs)}"
            head = next((k for k in n.kids if isinstance(k, Node) and re.fullmatch(r"h[1-6]", k.tag)), None)
            title = text_of(head, math=False) if head is not None else ""
            parent = _section_of(n, sec_ids)
            level = by_id[parent]["level"] + 1 if parent in by_id else 1
            own = role_of(title)
            if n.has("ltx_bibliography"):
                own = "references"
            elif n.has("ltx_appendix"):
                own = "appendix"
            inherited = by_id[parent]["role"] if parent in by_id else ""
            # a subsection takes its parent's role ("3.2 Control in a planar system" is under Experiments) unless
            # its own title says it is related work / acknowledgments / references
            if inherited == "references":
                inherited = ""                           # never pass the bibliography's role down
            role = own if (not inherited or own in ("related_work", "acknowledgments", "references")) else inherited
            row = {"id": sid, "title": title, "level": level, "parent": parent, "role": role or own, "text": ""}
            secs.append(row)
            sec_ids[id(n)] = sid
            by_id[sid] = row
    paras = collections.defaultdict(list)
    title = abstract = ""
    for n in root.iter():
        if n.tag == "h1" and n.has("ltx_title_document"):
            title = text_of(n, math=False)
        elif n.tag == "div" and n.has("ltx_abstract"):
            abstract = text_of(n)
        elif n.tag == "p" and n.has("ltx_p"):
            sid = _section_of(n, sec_ids)
            inside_fig = False
            p = n.parent
            while p is not None and p is not root:
                if p.tag == "figure" or (p.tag == "div" and p.has("ltx_abstract")):
                    inside_fig = True
                    break
                p = p.parent
            if not inside_fig:
                paras[sid].append(text_of(n))
    for s in secs:
        s["text"] = "\n".join(paras.get(s["id"], []))
    eqs, inline, figs, tabs, bib, cites = [], [], [], [], [], []
    for n in root.iter():
        if n.tag == "table" and (n.has("ltx_equation") or n.has("ltx_equationgroup")):
            if _in_eq_table(n):
                continue
            eqs += _equation_rows(n, _section_of(n, sec_ids))
        elif n.tag == "math" and n.attrs.get("display") != "block" and not _in_eq_table(n):
            inline.append(H.unescape(n.attrs.get("alttext", "")))
        elif n.tag == "figure" and (n.has("ltx_figure") or n.has("ltx_table")):
            # the figure's OWN caption: the figcaption whose nearest <figure> is n (a sub-figure's "(a)" caption comes
            # first in document order -- 2026-10-14, thomas2021asvlite listed only its panels' captions)
            cap = next((k for k in n.iter() if k.tag == "figcaption" and _nearest_figure(k) is n), None)
            tag = next((text_of(k) for k in (cap.iter() if cap is not None else [])
                        if k.tag == "span" and (k.has("ltx_tag_figure") or k.has("ltx_tag_table"))), "")
            imgs = [k.attrs.get("src") or k.attrs.get("data") for k in n.iter()
                    if k.tag in ("img", "object") and (k.attrs.get("src") or k.attrs.get("data"))]
            par = n.parent
            while par is not None and not (par.tag == "figure" and par.has("ltx_figure")):
                par = par.parent
            tag = tag.rstrip(": ").strip()
            caption = text_of(cap) if cap is not None else ""
            if tag and caption.startswith(tag):
                caption = caption[len(tag):].lstrip(": ").strip()
            row = {"id": n.attrs.get("id", ""), "tag": tag, "caption": caption, "section": _section_of(n, sec_ids)}
            if n.has("ltx_table"):
                row["rows"] = [[text_of(c) for c in r.iter() if c.tag in ("td", "th")]
                               for r in n.iter() if r.tag == "tr"][:200]
                tabs.append(row)
            else:
                row["images"] = imgs
                row["inline_svg"] = any(k.tag == "svg" for k in n.iter())     # TikZ / pgfplots: drawn, no file
                row["parent"] = par.attrs.get("id", "") if par is not None else ""
                figs.append(row)
        elif n.tag == "li" and n.has("ltx_bibitem"):
            t = text_of(n)
            bib.append({"id": n.attrs.get("id", ""), "text": t[:600],
                        "arxiv": sorted(set(re.findall(r"(?<![\d.])(\d{4}\.\d{4,5})(?![\d])", t))
                                        | set(re.findall(r"(?i)10\.48550/arxiv\.(\d{4}\.\d{4,5})", t))),
                        "doi": sorted(set(d.rstrip(".,;)") for d in re.findall(r"10\.\d{4,9}/[^\s\"<>]+", t)))})
        elif n.tag == "cite":
            sid = _section_of(n, sec_ids)
            for a in n.iter():
                href = a.attrs.get("href", "") if a.tag == "a" else ""
                if "#bib." in href:
                    cites.append([sid, href.split("#", 1)[1]])
    errors = sum(1 for n in root.iter() if n.has("ltx_ERROR"))
    return {"schema": "svfr.arxiv_html.parsed/1", "title": title, "abstract": abstract, "sections": secs,
            "equations": eqs, "inline_math": inline, "figures": figs, "tables": tabs, "bib": bib,
            "cites": cites, "errors": errors}


#: bump when parse() changes: a parsed file written by another version is parsed again
PARSER_VERSION = "3"


def parse_all(cache: Path, say=print, force: bool = False) -> dict:
    cache = Path(cache)
    rows = _load_manifest(cache)
    (cache / "parsed").mkdir(parents=True, exist_ok=True)
    n = new = 0
    empty: Dict[str, dict] = {}
    for aid, r in sorted(rows.items()):
        if r.get("status") != "ok":
            continue
        n += 1
        out = cache / "parsed" / f"{flat(aid)}.json.gz"
        src = cache / "pages" / r["file"]
        if out.is_file() and not force and out.stat().st_mtime_ns >= src.stat().st_mtime_ns:
            try:
                if json.loads(gzip.decompress(out.read_bytes())).get("parser") == PARSER_VERSION:
                    continue
            except (OSError, ValueError, EOFError):
                pass
        d = parse(gzip.decompress(src.read_bytes()).decode("utf-8", "replace"))
        if not has_body(d):                            # 'See pages 1-last of paper.pdf': a PDF-only submission's empty
            r.update(status="no_html", why="empty LaTeXML body (a PDF-only submission)")    # shell, not HTML (fix
            empty[aid] = r                                                                   # check 2026-10-14)
            out.unlink(missing_ok=True)
            continue
        d.update(arxiv_id=aid, key=r["key"], version=r.get("version_served", ""), parser=PARSER_VERSION)
        out.write_bytes(gzip.compress(json.dumps(d, ensure_ascii=False).encode("utf-8"), 6))
        new += 1
        if new % 100 == 0:
            say(f"  parsed {new}")
    if empty:
        _save_manifest(cache, rows, touched=set(empty))
    return {"ok_pages": n, "newly_parsed": new, "empty_now_no_html": len(empty)}


def has_body(d: Optional[dict]) -> bool:
    """A parsed page with text: a section with text, or an abstract."""
    return bool(d) and (bool((d.get("abstract") or "").strip()) or
                        any((s.get("text") or "").strip() for s in d.get("sections") or []))


def load_parsed(cache: Path, aid: str) -> Optional[dict]:
    f = Path(cache) / "parsed" / f"{flat(aid)}.json.gz"
    return json.loads(gzip.decompress(f.read_bytes()).decode("utf-8")) if f.is_file() else None


# ---------------------------------------------------------------------------------------- equation provenance
FONTS = r"mathrm|text|operatorname|textrm|mathit|mathbf|boldsymbol|bm|mathsf|mathbb|mathcal|mathnormal|textbf|rm"
ENVS = r"aligned|align\*?|gathered|split|cases|array\}\{[^}]*|eqnarray\*?"


def norm_tex(s: str) -> str:
    """Spelling-neutral LaTeX: wrappers, sizing, spacing, braces and definition signs (:=, \\doteq) removed, and the
    layout a paper puts around an equation: a \\makebox / \\text LABEL ending in ':' ("Transition model:") and the frame
    of \\boxed / \\makebox (review 2026-10-14: once rows had to be EQUAL, PlaNet's RSSM, each row labelled with a
    \\makebox, read as 'not in source')."""
    s = H.unescape(s or "")
    s = re.sub(r"\\(?:makebox|mathmakebox|mbox|text|textrm)\s*(?:\[[^\]]*\])*\s*\{[^{}]*:\s*\}", "", s)  # a label
    s = re.sub(r"\\(?:boxed|fbox|makebox|mathmakebox|mbox)(?![a-zA-Z])\s*(?:\[[^\]]*\])*", "", s)        # a frame
    for _ in range(3):
        s = re.sub(r"\\(?:" + FONTS + r")\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)l?r?(?![a-zA-Z])", "", s)
    s = re.sub(r"\\(?:doteq|coloneqq|triangleq|eqqcolon|stackrel\{[^{}]*\}\{=\})|:=|=:", "=", s)
    s = re.sub(r"\\(?:displaystyle|textstyle|nonumber|notag|label\{[^{}]*\}|quad|qquad)(?![a-zA-Z])|\\[,;:!]", "", s)
    s = re.sub(r"\s|\{|\}|\\ ", "", s)
    return s.rstrip(".,;")


def rows_of(lx: str) -> List[str]:
    r"""An aligned / multi-line block -> its rows; a ', \qquad' pair -> both halves; else the equation itself."""
    body = re.sub(r"\\(?:begin|end)\{(?:" + ENVS + r")\}", "", lx)
    parts = [r.replace("&", "") for r in re.split(r"\\\\(?:\[[^\]]*\])?", body) if r.strip()]
    rows: List[str] = []
    for r in parts:                                    # a continuation line belongs to the row above (fix check
        if rows and CONTINUES.match(r) and not RELATION.search(r):      # 2026-10-14: a block cut at a row boundary
            rows[-1] += " " + r                                          # read as the whole equation)
        else:
            rows.append(r)
    # a ', \qquad' pair is two equations; a '\quad + ...' is one equation going on
    parts = [q for r in rows for q in re.split(r",?\s*\\q?quad(?!\s*(?:[+\-]|\\(?:cdot|times|pm|mp)(?![a-zA-Z])))\s*", r)
             if norm_tex(q)]
    return parts or [lx]


def find_equation(latex: str, d: dict) -> Optional[dict]:
    """The paper equation an agent-written equation IS (every row EQUAL to one of the paper equation's rows after
    norm_tex), else None. Display equations are searched first, then inline math."""
    cands = [(e, x) for e in d["equations"] for x in e["latex"]] + [(None, x) for x in d["inline_math"]]
    rows = [norm_tex(r) for r in rows_of(latex)]
    for e, x in cands:
        nx = {norm_tex(x)} | {norm_tex(r) for r in rows_of(x)}
        if all(r in nx for r in rows):                 # EQUAL rows only: a substring rule accepted an equation missing
            #                                          its last term or a hat (review 2026-10-14)
            return {"id": e["id"] if e else "", "tag": e["tag"] if e else "", "section": e["section"] if e else "",
                    "display": e is not None}
    return None


GREEK_SWAP = {"alpha": "beta", "beta": "alpha", "gamma": "delta", "delta": "gamma", "theta": "phi", "phi": "theta",
              "sigma": "tau", "tau": "sigma", "mu": "nu", "nu": "mu", "lambda": "kappa", "pi": "rho", "omega": "psi"}


def mutate(latex: str) -> str:
    """One symbol changed -- a control that a provenance check must REJECT: a standalone letter, else a digit, else a
    Greek letter (theta -> phi). Returns the input unchanged when it has none of them (no control possible)."""
    m = re.sub(r"(?<![\\A-Za-z])([a-zA-Z])(?![A-Za-z])", lambda m: "Q" if m.group(1) != "Q" else "W", latex, count=1)
    if m != latex:
        return m
    m = re.sub(r"(?<![\\A-Za-z])(\d)", lambda m: str((int(m.group(1)) + 1) % 10), latex, count=1)
    if m != latex:
        return m
    return re.sub(r"\\(" + "|".join(GREEK_SWAP) + r")(?![A-Za-z])", lambda m: "\\" + GREEK_SWAP[m.group(1)], latex,
                  count=1)


def mutants(latex: str) -> Dict[str, str]:
    """Controls a provenance check must REJECT: one symbol swapped (`mutate`), the last term dropped, an accent
    (hat / dot / tilde / bar) removed. A kind with nothing to change is left out."""
    out = {"symbol": mutate(latex)}
    t = re.sub(r"\s*[+\-]\s*[^+\-=]+$", "", latex.strip().rstrip(".,;"), count=1)
    out["last_term"] = t
    out["accent"] = re.sub(r"\\(?:hat|dot|ddot|tilde|bar|widehat|widetilde|overline)\s*(\{[^{}]*\}|\\?[A-Za-z]+)", r"\1",
                           latex, count=1)
    return {k: v for k, v in out.items() if v != latex and norm_tex(v) != norm_tex(latex)}


def equation_provenance(cache: Path, today: str = "") -> dict:
    """Every lexicon equation with a source paper: is it the source's own equation (status exact), absent from the
    source's LaTeX (not_in_source), or uncheckable (no_html / not_arxiv / no_source)? Mutants must not match."""
    P = json.loads((PT.BWM / "index" / "papers.json").read_text(encoding="utf-8"))
    alias, papers = P["aliases"], {p["key"]: p for p in P["papers"]}
    M = json.loads((PT.BWM / "lexicon" / "_math.json").read_text(encoding="utf-8"))["terms"]
    rows, no_control, parsed, tried, matched = [], 0, {}, {}, {}
    for t in M:
        for i, e in enumerate(t["math"]):
            k = alias.get(e.get("source_key") or "", e.get("source_key") or "")
            aid = re.sub(r"v\d+$", "", (papers.get(k) or {}).get("arxiv_id") or "")
            row = {"term": t.get("id") or t.get("term", ""), "i": i, "source_key": k, "arxiv": aid,
                   "status": "no_source"}
            if k and not aid:
                row["status"] = "not_arxiv"
            elif aid:
                if aid not in parsed:
                    parsed[aid] = load_parsed(cache, aid)
                d = parsed[aid]
                if d is None or not (d["equations"] or d["inline_math"]):
                    row["status"] = "no_html"
                else:
                    hit = find_equation(e["latex"], d)
                    row["status"] = "exact" if hit else "not_in_source"
                    if hit:
                        row.update(eq=hit["id"], tag=hit["tag"], section=hit["section"], display=hit["display"])
                        ms = mutants(e["latex"])
                        if not ms:
                            no_control += 1
                        for kind, mut in ms.items():
                            tried[kind] = tried.get(kind, 0) + 1
                            matched[kind] = matched.get(kind, 0) + (find_equation(mut, d) is not None)
            rows.append(row)
    c = collections.Counter(r["status"] for r in rows)
    return {"schema": "svfr.bwm.math_provenance/1", "checked_on": today, "summary": dict(sorted(c.items())),
            "control": {"exact_matches": c.get("exact", 0), "no_mutant_possible": no_control,
                        "mutants": {k: {"tried": tried[k], "also_matched": matched.get(k, 0)} for k in sorted(tried)}},
            "equations": rows}


# ----------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fetch", "parse", "stats", "show", "equations", "repair"])
    ap.add_argument("key", nargs="?", default="", help="show: a paper key or arXiv id")
    ap.add_argument("--bwm", action="store_true", help="the BlueBoat world-model index's held arXiv papers")
    ap.add_argument("--pdfs", type=Path, default=PT.DEFAULT_PDFS, help="the folder of paper PDFs")
    ap.add_argument("--cache", type=Path, default=None, help="HTML cache (default: <pdfs>_html beside the folder)")
    ap.add_argument("--limit", type=int, default=0, help="fetch: at most this many papers this run")
    ap.add_argument("--retry", action="store_true", help="fetch: ask again for papers that failed (not no_html)")
    ap.add_argument("--pause", type=float, default=CRAWL_DELAY,
                    help="fetch: seconds between requests (arxiv.org/robots.txt: Crawl-delay 15)")
    ap.add_argument("--force", action="store_true", help="parse: re-parse every page")
    ap.add_argument("--write", action="store_true", help="equations: write lexicon/_math_provenance.json")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    cache = a.cache or default_cache(a.pdfs)
    today = datetime.date.today().isoformat()
    if a.cmd == "fetch":
        if not a.bwm:
            ap.error("fetch needs --bwm (the papers to fetch come from the BlueBoat index)")
        items = held_arxiv(PT.default_cache(a.pdfs))
        print(f"{len(items)} held arXiv papers; {sum(bool(i['version']) for i in items)} with the held version "
              f"stamped on page 1 -> {cache}")
        with FetchLock(cache):
            res = fetch(items, cache, pause=a.pause, limit=a.limit, retry=a.retry, today=today)
        print(json.dumps(res))
        return 0
    if a.cmd == "repair":
        print(json.dumps(repair(cache, held_arxiv(PT.default_cache(a.pdfs)), today, index=index_arxiv(),
                                aliases=index_aliases())))
        return 0
    if a.cmd == "parse":
        print(json.dumps(parse_all(cache, force=a.force)))
        return 0
    rows = _load_manifest(cache)
    if a.cmd == "stats":
        c = collections.Counter(r["status"] for r in rows.values())
        ok = [r for r in rows.values() if r["status"] == "ok"]
        err = figs = eqs = tabs = secs = with_math = 0
        for aid, r in rows.items():
            d = load_parsed(cache, aid) if r["status"] == "ok" else None
            if d:
                err += d["errors"] > 0
                figs += len(d["figures"])
                eqs += sum(1 for e in d["equations"] if not e.get("group"))
                tabs += len(d["tables"])
                secs += len(d["sections"])
                with_math += bool(d["equations"] or d["inline_math"])
        print(f"{len(rows)} arXiv papers in the manifest: {dict(c)}; held version served for "
              f"{sum(bool(r.get('version_match')) for r in ok)} of {sum(bool(r.get('version_held')) for r in ok)} "
              f"with a known held version; {sum(r.get('bytes') or 0 for r in ok) / 1e6:.0f} MB of HTML "
              f"({sum((cache / 'pages' / r['file']).stat().st_size for r in ok if (cache / 'pages' / r['file']).is_file()) / 1e6:.0f}"
              f" MB gzipped)")
        print(f"parsed: {with_math} with math, {eqs} display equations, {figs} figures, {tabs} tables, {secs} sections;"
              f" {err} pages carry LaTeXML conversion errors")
        return 0
    if a.cmd == "show":
        aid = a.key
        if aid not in rows:                            # a key: its 'ok' row first, else any row -- its status is
            mine = [x for x, r in rows.items() if r.get("key") == a.key]   # what to report (third fix check)
            aid = min(mine, key=lambda x: {"ok": 0, "not_indexed": 2}.get(rows[x].get("status"), 1), default="")
            #   ok, then a live no_html / error row, then a retired one -- never by the ids' order (fix check)
        if aid and rows.get(aid, {}).get("status") == "not_indexed":
            print(f"show: arXiv {aid} is not an index paper's id any more ({rows[aid].get('why', '')})")
            return 1
        d = load_parsed(cache, aid) if aid and rows.get(aid, {}).get("status") == "ok" else None
        if not d:
            print(f"show: no parsed HTML for {a.key!r} (fetch + parse first; status: {rows.get(aid, {}).get('status')})")
            return 1
        print(f"{d['key']} -- arXiv {aid}{d['version']}: {d['title']}")
        for s in d["sections"]:
            print(f"  {'  ' * (s['level'] - 1)}{s['id']:<10} [{s['role'] or '-'}] {s['title'][:80]}  ({len(s['text'])} chars)")
        print(f"  {sum(1 for e in d['equations'] if not e.get('group'))} display equations, "
              f"{len(d['inline_math'])} inline math, {len(d['figures'])} "
              f"figures, {len(d['tables'])} tables, {len(d['bib'])} references, {len(d['cites'])} citations, "
              f"{d['errors']} conversion errors")
        for f in d["figures"][:12]:
            print(f"    {f['tag'] or f['id']}: {f['caption'][:110]}  {f['images'][:2]}")
        return 0
    res = equation_provenance(cache, today)
    print(json.dumps({k: res[k] for k in ("summary", "control")}))
    if a.write:
        out = PT.BWM / "lexicon" / "_math_provenance.json"
        body = json.dumps({k: v for k, v in res.items() if k != "equations"}, ensure_ascii=False, indent=1)[:-2]
        body += ',\n "equations": [\n' + ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in res["equations"])
        out.write_text(body + "\n ]\n}\n", encoding="utf-8", newline="\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
