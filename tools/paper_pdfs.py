"""paper_pdfs -- which papers' full-text PDFs do we HAVE, and which are still to download? Match a folder of PDFs to a
paper list, whatever the PDFs are named, and write a checklist with a download link for every missing one.

    python tools/paper_pdfs.py --refs report/world_models/options/wm_options.json --dir report/world_models/papers
    python tools/paper_pdfs.py --refs report/world_models/verified_refs.json --dir <pdfs> --out <dir>/CHECKLIST.md
    python tools/paper_pdfs.py --refs <refs> --dir <pdfs> --fetch-arxiv          # + download every MISSING open arXiv PDF

FETCH (`--fetch-arxiv`, added 2026-10-01): every missing paper with an arXiv id is downloaded from arxiv.org/pdf/<id>
to `<id>_<key>.pdf` (so it matches by id), one request every `--sleep` seconds (default 15: arxiv.org/robots.txt Crawl-delay), 429/503 backed
off and retried, the body checked to BE a PDF (`%PDF-` magic, > 20 KB -- an HTML error page is refused), written to a
temp name and renamed, so a failed or interrupted download never leaves a half file. Journal-only papers are never
fetched (they need the institution). Exit 2 when any download failed; the checklist is written either way.

WHY. Full texts make a review and its slides better (hardware, code links, numbers straight from the paper), but
publishers name files by their own ids (`1-s2.0-S1566253526006408-main.pdf` is Elsevier's PII for PIG-Dyn), so "do we
have it?" was a manual hunt. Each PDF is identified from its OWN metadata -- the XMP `dc:title` and the DOI tags
(`prism:doi`, `crossmark:DOI`, ...) that publishers embed as plain text -- then matched to the list by DOI, arXiv id
(from the file name or metadata), else a near-exact title (`--min-ratio`). No PDF library needed; nothing is
downloaded. Refs: `wm_options.json` (`options` rows) or any `tools/verify_refs.py` output.

OUTPUT: CHECKLIST.md in --dir (or --out): HAVE / MISSING per paper with its link (arXiv abstract for open preprints, the
DOI otherwise -- "needs institution" marked), plus any PDF that matched nothing. The --dir is for local full texts:
keep it gitignored (publisher PDFs are copyrighted and usually > 5 MB). Exit 0; with --require-all, 1 while any is
missing.

HONEST SCOPE: metadata can be absent or wrong (a scanned PDF, a preprint server's generic title); such a PDF lands in
"unmatched" -- rename it `<option key>.pdf` or include the arXiv id / DOI in its name and it matches. A first DOI found
in raw bytes can be a CITED paper's, so only DOI-specific tags count, never the first DOI in the text.
"""
from __future__ import annotations

import argparse
import difflib
import http.client
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:   # run by path: the repo root, for tools/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.atomic_replace import replace_retry  # noqa: E402

#: a bare "application/pdf" Accept got HTTP 406 from arxiv.org on 2 of 41 papers (2026-10-01); */* is the fallback
def _user_agent(tool: str) -> str:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_paper_pdfs_atlas_project", Path(__file__).resolve().parent / "atlas_project.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.user_agent(tool)


UA = {"User-Agent": _user_agent("paper_pdfs"),     # no owner address in an exported kit (atlas_project)
      "Accept": "application/pdf,*/*;q=0.8"}
MIN_PDF_BYTES = 20_000

_TITLE = re.compile(rb"<dc:title>\s*<rdf:Alt>\s*<rdf:li[^>]*>([^<]*)</rdf:li>")
_DOI_TAG = re.compile(rb"<(?:prism:doi|crossmark:DOI|pdfx:doi|dc:identifier)>(?:\s*doi:\s*|https?://(?:dx\.)?doi\.org/)?"
                      rb"(10\.\d{4,9}/[^<\s]+)<", re.I)
_ARXIV = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
#: an old-style id as `fetch_missing` writes it into a file name (cs/0102001 -> cs_0102001) or as printed
_ARXIV_OLD = re.compile(r"(?<![A-Za-z])((?:astro-ph|cond-mat|gr-qc|hep-[a-z]{2}|math-ph|nlin|nucl-[a-z]{2}|physics|"
                        r"quant-ph|cs|math|q-bio|q-fin|stat|eess|econ)(?:\.[A-Z]{2})?)[_/](\d{7})(?:v\d+)?(?!\d)")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def pdf_identity(path):
    """{file, title, doi, arxiv_id} read from a PDF's own metadata + its file name. Pure apart from the read."""
    p = Path(path)
    head = p.read_bytes()
    t = _TITLE.search(head)
    d = _DOI_TAG.search(head)
    title = t.group(1).decode("utf-8", "replace").strip() if t else ""
    a = _ARXIV.search(p.stem)                                  # an arXiv id in the file name wins
    old = None if a else _ARXIV_OLD.search(p.stem)             # cs_0102001_<key>.pdf -> cs/0102001
    m = None if (a or old) else re.search(rb"arXiv:\s*(\d{4}\.\d{4,5})", head[:200000])
    return {"file": p.name, "title": title or p.stem.replace("_", " "), "title_from": "metadata" if title else "file name",
            "doi": d.group(1).decode("ascii", "replace").rstrip(").,;").lower() if d else "",
            "arxiv_id": a.group(1) if a else (f"{old.group(1)}/{old.group(2)}" if old
                                              else (m.group(1).decode() if m else ""))}


def paper_rows(data):
    """wm_options.json ({'options': [...]}) or verify_refs output -> [{key, title, arxiv_id, doi, venue, year}]."""
    if isinstance(data, dict) and "options" in data:
        out = []
        for r in data["options"]:
            ref = r.get("ref", "")
            if not ref.lower().startswith(("arxiv:", "doi:")):
                continue                                     # our own demo: not a paper
            kind, ident = ref.split(":", 1)
            out.append({"key": r["key"], "title": r.get("title") or r["name"], "name": r["name"],
                        "arxiv_id": ident if kind.lower() == "arxiv" else "",
                        "doi": ident.lower() if kind.lower() == "doi" else "", "venue": r.get("venue", ""),
                        "year": str(r.get("year", ""))})
        return out
    rows = data.get("papers", data) if isinstance(data, dict) else data
    out = []
    for r in rows:
        rec = r.get("record") or r
        out.append({"key": r.get("key") or rec.get("key", ""), "title": rec.get("title", ""), "name": rec.get("title", ""),
                    "arxiv_id": r.get("arxiv_id") or rec.get("arxiv_id", ""),
                    "doi": (rec.get("doi") or "").lower(), "venue": rec.get("venue", ""), "year": str(rec.get("year", ""))})
    return out


#: a library-style file name: `Surname_Year_Title_Words` (tools/paper_rename.py) -- the prefix is not title text
_LIB_PREFIX = re.compile(r"^(?:[A-Za-z]+_)?(?:19|20)\d\d_(?=\w)")


#: prefixes that turn a word into its opposite or a different thing (review 2026-10-13: 'Unsupervised X' ~ 'Supervised X')
_NEG = frozenset("un non anti semi self de dis in im ir il multi".split())


def same_words(a, b):
    """No whole word of one title is missing from the other. A near-identical RATIO is not enough: MEASURED
    2026-10-13, 'Mildly Conservative Q-Learning for Offline RL' vs 'Conservative Q-Learning for Offline RL' scores
    0.891 and 'Anti-Exploration by Random Network Distillation' vs 'Exploration by ...' 0.876 -- two different
    papers taken as held, so never downloaded. Short words (<= 2 letters) are ignored, a word joined or split by a
    hyphen still matches (it is looked for in the other title with its spaces removed), and a typo (>= 0.8 to a
    word of the other) still matches. Pure."""
    wa, wb = norm(a).split(), norm(b).split()

    def joins(words):                                # 'Q-Learning' vs 'QLearning': up to 3 adjacent words joined
        return {"".join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}

    def negated(w, o):                               # 'unsupervised' vs 'supervised': a typo-close OPPOSITE
        for x, y in ((w, o), (o, w)):
            if x != y and x.endswith(y) and x[:len(x) - len(y)] in _NEG:
                return True
        return False

    def covered(words, other):
        ok = joins(other)
        for w in words:
            if len(w) <= 2 and not any(c.isdigit() for c in w):
                continue                              # 'of', 'a'; but '2d' vs '3d' is compared
            if w in ok:
                continue
            if any(len(o) >= 4 and len(w) >= 4 and not negated(w, o)
                   and difflib.SequenceMatcher(None, w, o).ratio() >= 0.8 for o in other):
                continue
            return False
        return True
    return covered(wa, wb) and covered(wb, wa)


def match(papers, pdfs, min_ratio=0.85):
    """{paper key: pdf file} by DOI, then arXiv id, then the key in the file name, then a near-exact title. Pure."""
    found, used = {}, set()
    for p in papers:
        for f in pdfs:
            if f["file"] in used:
                continue
            hit = ((p["doi"] and f["doi"] == p["doi"]) or (p["arxiv_id"] and f["arxiv_id"] == p["arxiv_id"])
                   or (p["key"] and norm(Path(f["file"]).stem) == norm(p["key"])))
            if not hit and norm(p["title"]):
                # the embedded title first, then the FILE NAME read as a title: publishers ship PDFs whose metadata
                # title is a placeholder ('frobt-2021-706646 1..11'), which used to hide a renamed file (2026-10-07)
                for cand in (f["title"], _LIB_PREFIX.sub("", Path(f["file"]).stem).replace("_", " ")):
                    if (norm(cand) and difflib.SequenceMatcher(None, norm(cand), norm(p["title"])).ratio() >= min_ratio
                            and same_words(cand, p["title"])):
                        hit = True
                        break
            if hit:
                found[p["key"] or p["title"]] = f["file"]
                used.add(f["file"])
                break
    return found


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


ARXIV_PACE = _load_arxiv_pace(Path(__file__).resolve().parent)


def http_get(url, retries=4, pause=10.0):
    """GET -> bytes; backs off on 429 / 5xx (Retry-After honoured), retries a dropped transfer (IncompleteRead: 3 of
    41 arXiv PDFs on 2026-10-01) and answers a 406 with Accept */*. Raises the last error when retries run out."""
    last, headers = None, dict(UA)
    for i in range(retries):
        ARXIV_PACE.wait(url)                             # arxiv.org: 15 s since ANY request to it, retries included
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 406 and headers.get("Accept") != "*/*":
                headers["Accept"] = "*/*"
                continue
            if e.code not in (429, 500, 502, 503, 504):
                raise
            wait = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(wait) if wait and str(wait).isdigit() else pause * (i + 1))
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as e:
            last = e
            time.sleep(pause * (i + 1))
    raise last


def curl_get(url, workdir):
    """GET via the curl executable -> bytes (the fallback: MEASURED 2026-10-01, urllib failed 5 of 41 arXiv PDFs on the
    lab network -- 3 truncated at exactly 3 / 4 / 8 MiB, 2 HTTP 406 even with Accept */* -- and curl fetched them
    whole). Writes next to the destination, never to the system temp (C: is nearly full on the lab computer)."""
    exe = shutil.which("curl")
    if not exe:
        raise OSError("curl not on PATH")
    out = Path(workdir) / f".curl-{abs(hash(url))}.part"
    ARXIV_PACE.wait(url)                                 # the fallback is a request too (it went out 0 s after)
    try:
        # --retry-delay 15: curl's own retries would otherwise ask arxiv.org again at +1, +3, +7 s, unseen by the
        # pacer (fix check 2026-10-14); a fixed delay also turns off curl's backoff doubling
        r = subprocess.run([exe, "-sSL", "--fail", "--retry", "3", "--retry-delay", "15", "-A", UA["User-Agent"],
                            "-o", str(out), url],
                           capture_output=True, text=True, timeout=900)
        if r.returncode:
            raise OSError(f"curl exit {r.returncode}: {(r.stderr or '').strip()[:100]}")
        return out.read_bytes()
    finally:
        out.unlink(missing_ok=True)


def _ends(data: bytes) -> bool:
    """A whole PDF ends with %%EOF (allowing trailing whitespace / a few bytes of junk)."""
    return b"%%EOF" in data[-2048:]


def fetch_arxiv_pdf(aid, dest, get=http_get, fallback=curl_get):
    """Download arXiv:<aid> to `dest` (atomic): urllib first, curl on failure. (True, size) or (False, reason).
    Never leaves a partial file."""
    dest = Path(dest)
    url = f"https://arxiv.org/pdf/{aid}"
    try:
        data = get(url)
    except Exception as e:                                   # network / HTTP: try curl, then report, never raise
        try:
            data = fallback(url, dest.parent) if fallback else None
        except Exception as e2:
            data = None
            e = f"{e}; curl: {e2}"
        if data is None:
            return False, f"download failed: {str(e)[:160]}"
    if not data.startswith(b"%PDF-") or len(data) < MIN_PDF_BYTES:
        return False, f"not a PDF ({len(data)} bytes, starts {data[:15]!r})"
    if not _ends(data) and fallback:                     # cut off in transfer (urllib, 2026-10-01): curl once more
        try:
            data = fallback(url, dest.parent) or data
        except Exception:                                # noqa: BLE001 -- judged below
            pass
    if not _ends(data):
        return False, f"cut off: {len(data)} bytes and no %%EOF at the end -- not saved"
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    replace_retry(tmp, dest)
    return True, len(data)


def withdrawn_versions(aid, get=http_get):
    """(the withdrawal notice, [versions]) when arXiv's abstract page says the paper was WITHDRAWN, else None.
    MEASURED 2026-10-13: 4 of 790 arXiv papers 404'd on /pdf/<id> -- each abstract page reads "This paper has been
    withdrawn by <author>", the withdrawal version has no PDF, and every earlier version's PDF is still served."""
    try:
        html = get(f"https://arxiv.org/abs/{aid}").decode("utf-8", "replace")
    except Exception:                                    # noqa: BLE001 -- unknown is not withdrawn
        return None
    m = re.search(r"This paper has been withdrawn[^<]{0,160}", html)
    if not m:
        return None
    return " ".join(m.group(0).split()), sorted({int(v) for v in re.findall(r"\[v(\d+)\]", html)})


def fetch_missing(papers, found, folder, get=http_get, sleep=15.0, limit=None, say=print, min_free_gb=0.0,
                  free=lambda d: shutil.disk_usage(d).free, fallback=curl_get):
    """Download every missing paper that has an arXiv id. -> [(key, ok, detail)]. Stops (every later paper reported
    'not tried') when the disk holding `folder` has less than `min_free_gb` free -- a nearly full system disk is the
    case this guards (the laptop's C: had 7.3 GB free on 2026-10-13)."""
    todo = [p for p in papers if (p["key"] or p["title"]) not in found and p["arxiv_id"]][:limit]
    out = []
    for i, p in enumerate(todo):
        left = free(folder) if min_free_gb else None
        if left is not None and left < min_free_gb * 1e9:
            say(f"  STOP: {left / 1e9:.2f} GB free on the disk of {folder}, below --min-free-gb "
                f"{min_free_gb}; {len(todo) - i} paper(s) not tried")
            out += [(q["key"], False, "not tried: disk nearly full") for q in todo[i:]]
            break
        if i:
            time.sleep(sleep)
        flat = p["arxiv_id"].replace("/", "_")             # an old-style id (cs/0102001) is a path otherwise
        ok, detail = fetch_arxiv_pdf(p["arxiv_id"], Path(folder) / f"{flat}_{p['key'] or 'paper'}.pdf", get,
                                     fallback)
        if not ok and "404" in str(detail):               # withdrawn? keep the last version that HAS a PDF
            wd = withdrawn_versions(p["arxiv_id"], get)
            if wd and len(wd[1]) >= 2:
                keep = wd[1][-2]
                time.sleep(sleep)
                ok, got = fetch_arxiv_pdf(f"{p['arxiv_id']}v{keep}",
                                          Path(folder) / f"{flat}v{keep}_{p['key'] or 'paper'}.pdf", get, fallback)
                detail = (f"WITHDRAWN ({wd[0]}; v{wd[1][-1]}); kept v{keep}, {got // 1024} KB" if ok
                          else f"WITHDRAWN ({wd[0]}); v{keep} failed too: {got}")
        out.append((p["key"], ok, detail))
        say(f"  [{i + 1}/{len(todo)}] {'OK  ' if ok else 'FAIL'} {p['arxiv_id']} {p['key']} "
            f"{detail if not ok or isinstance(detail, str) else f'{detail // 1024} KB'}")
    return out


def open_pdf_urls(dois, get=http_get, batch=50, sleep=1.0):
    """DOI -> [candidate open PDF URLs, best first] from OpenAlex (best_oa_location, then every location with a
    pdf_url; OpenAlex merges Unpaywall, repositories and publisher OA). The review behind this (2026-10-13,
    report/boat_world_model/LIT_PAPER_MINING.md): no single source finds every open copy, so the candidates are tried
    in order and only a real, whole PDF is kept. A DOI OpenAlex does not know maps to []."""
    import urllib.parse
    out = {}
    dois = sorted({d.lower() for d in dois if d})
    for i in range(0, len(dois), batch):
        chunk = dois[i:i + batch]
        q = urllib.parse.urlencode({"filter": "doi:" + "|".join(chunk), "per-page": batch,
                                    "select": "doi,best_oa_location,locations"})
        try:
            res = json.loads(get("https://api.openalex.org/works?" + q).decode("utf-8"))["results"]
        except Exception:                                    # noqa: BLE001 -- unknown, never 'no copy':
            continue                                         # these DOIs stay out of `out` = NOT CHECKED
        for d in chunk:
            out.setdefault(d, [])                            # asked: [] now means 'OpenAlex knows no open PDF'
        for w in res:
            d = (w.get("doi") or "").lower().replace("https://doi.org/", "")
            urls = out.setdefault(d, [])                     # two works with one DOI: MERGE their locations
            for loc in [w.get("best_oa_location") or {}] + list(w.get("locations") or []):
                u = (loc or {}).get("pdf_url")
                if u and u not in urls:
                    urls.append(u)
        if sleep and i + batch < len(dois):
            time.sleep(sleep)
    return out


def fetch_open(papers, found, folder, get=http_get, sleep=3.0, limit=None, say=print, min_free_gb=0.0,
               free=lambda d: shutil.disk_usage(d).free, urls_for=None):
    """Download every missing paper WITHOUT an arXiv id whose DOI has an open PDF (OpenAlex), saved as `<key>.pdf`.
    Each candidate URL must give a real PDF that ends (%PDF- magic, %%EOF) or the next is tried. -> [(key, ok, detail)]."""
    todo = [p for p in papers if (p["key"] or p["title"]) not in found and not p["arxiv_id"] and p["doi"]][:limit]
    urls = (urls_for or open_pdf_urls)([p["doi"] for p in todo]) if todo else {}
    out = []
    for i, p in enumerate(todo):
        left = free(folder) if min_free_gb else None
        if left is not None and left < min_free_gb * 1e9:
            out += [(q["key"], False, "not tried: disk nearly full") for q in todo[i:]]
            break
        if p["doi"].lower() not in urls:
            out.append((p["key"], False, "NOT CHECKED: OpenAlex could not be asked"))
            say(f"  [{i + 1}/{len(todo)}] SKIP {p['key']} OpenAlex could not be asked")
            continue
        cands = urls[p["doi"].lower()]
        ok, detail = False, "no open PDF known to OpenAlex"
        for u in cands:
            try:
                data = get(u)
            except Exception as e:                           # noqa: BLE001
                detail = f"{u}: {str(e)[:80]}"
                time.sleep(sleep)
                continue
            time.sleep(sleep)
            if data.startswith(b"%PDF-") and len(data) >= MIN_PDF_BYTES and _ends(data):
                dest = Path(folder) / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', p['key'] or 'paper')}.pdf"
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                replace_retry(tmp, dest)
                ok, detail = True, f"{len(data) // 1024} KB from {u}"
                break
            detail = f"{u}: not a whole PDF ({len(data)} bytes)"
        out.append((p["key"], ok, detail))
        say(f"  [{i + 1}/{len(todo)}] {'OK  ' if ok else 'FAIL'} {p['key']} {detail[:140]}")
    return out


def link(p):
    if p["arxiv_id"]:
        return f"https://arxiv.org/abs/{p['arxiv_id']}", "open (arXiv)"
    if p["doi"].startswith("10.48550/"):
        return f"https://doi.org/{p['doi']}", "open (arXiv)"
    return f"https://doi.org/{p['doi']}", "needs institution (check for an open copy)"


def checklist(papers, found, pdfs):
    have = [p for p in papers if (p["key"] or p["title"]) in found]
    miss = [p for p in papers if (p["key"] or p["title"]) not in found]
    lines = [f"# Full-text checklist -- {len(have)} of {len(papers)} PDFs here", "",
             "*Generated by `tools/paper_pdfs.py`; re-run it after downloading. Files may keep any name: they are matched "
             "by the DOI / title in their own metadata.*", ""]
    for head, rows in (("Missing -- needs institution access", [p for p in miss if not link(p)[1].startswith("open")]),
                       ("Missing -- open on arXiv", [p for p in miss if link(p)[1].startswith("open")]),
                       ("Have", have)):
        if not rows:
            continue
        lines += [f"## {head} ({len(rows)})", ""]
        for p in rows:
            url, _ = link(p)
            extra = f" -- `{found[p['key'] or p['title']]}`" if p in have else ""
            lines.append(f"- [{p['title']}]({url}) ({p.get('venue') or 'arXiv'} {p.get('year', '')}){extra}")
        lines.append("")
    loose = sorted(set(f["file"] for f in pdfs) - set(found.values()))
    if loose:
        lines += [f"## PDFs that matched nothing ({len(loose)}) -- rename `<key>.pdf` or add the DOI / arXiv id", ""]
        lines += [f"- `{f}`" for f in loose] + [""]
    return "\n".join(lines), have, miss


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refs", required=True, help="wm_options.json or a verify_refs.py JSON")
    ap.add_argument("--dir", required=True, help="folder of full-text PDFs (keep it gitignored)")
    ap.add_argument("--out", default="", help="checklist path (default: <dir>/CHECKLIST.md)")
    ap.add_argument("--min-ratio", type=float, default=0.85, help="title similarity for a metadata-title match")
    ap.add_argument("--require-all", action="store_true", help="exit 1 while any paper is missing")
    ap.add_argument("--fetch-arxiv", action="store_true", help="download every missing paper that has an arXiv id")
    ap.add_argument("--sleep", type=float, default=15.0,
                    help="seconds between arxiv.org requests (default 15: its robots.txt Crawl-delay)")
    ap.add_argument("--limit", type=int, default=None, help="download at most this many")
    ap.add_argument("--fetch-open", action="store_true",
                    help="download missing NON-arXiv papers that have an open PDF (OpenAlex locations), as <key>.pdf")
    ap.add_argument("--min-free-gb", type=float, default=2.0,
                    help="stop downloading when the disk has less than this free (default 2)")
    a = ap.parse_args(argv)
    papers = paper_rows(json.loads(Path(a.refs).read_text(encoding="utf-8")))
    folder = Path(a.dir)
    folder.mkdir(parents=True, exist_ok=True)
    pdfs = [pdf_identity(p) for p in sorted(folder.glob("*.pdf"))]
    found = match(papers, pdfs, a.min_ratio)
    failed = []
    if a.fetch_arxiv:
        res = fetch_missing(papers, found, folder, sleep=a.sleep, limit=a.limit, min_free_gb=a.min_free_gb)
        failed = [k for k, ok, _ in res if not ok]
    if a.fetch_open:
        res = fetch_open(papers, found, folder, sleep=a.sleep, limit=a.limit, min_free_gb=a.min_free_gb)
        print(f"open copies: {sum(ok for _, ok, _ in res)} of {len(res)} non-arXiv papers fetched")
    if a.fetch_arxiv or a.fetch_open:                    # re-scan after ANY download (review 2026-10-13: a plain
        pdfs = [pdf_identity(p) for p in sorted(folder.glob("*.pdf"))]   # --fetch-arxiv listed its downloads as missing)
        found = match(papers, pdfs, a.min_ratio)
    text, have, miss = checklist(papers, found, pdfs)
    out = Path(a.out) if a.out else folder / "CHECKLIST.md"
    out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"{len(have)} of {len(papers)} papers have a PDF in {folder}; {len(miss)} missing -> {out}")
    if failed:
        print(f"WARNING: {len(failed)} arXiv download(s) failed: {', '.join(failed)}")
        return 2
    return 1 if (a.require_all and miss) else 0


if __name__ == "__main__":
    sys.exit(main())
