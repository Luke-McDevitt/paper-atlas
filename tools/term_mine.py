r"""term_mine -- grow the BlueBoat world-model LEXICON from the papers themselves: mine every "long form (SHORT FORM)"
definition in the held papers' full text, keep the pairs the lexicon does not have yet, rank them by how many papers
define them, pull verbatim quotes for the chosen ones, and VERIFY every quote of a lexicon file against the cited
paper's own text.

    python tools/term_mine.py candidates [--min-papers 4] [--top 300] [--out cands.tsv]   # needs the text cache
    python tools/term_mine.py quotes "Real-time kinematic" RTK [--n 8]   # verbatim <= 15-word quotes, key + page
    python tools/term_mine.py verify report/boat_world_model/lexicon/fulltext_terms.json  # every quote on its page

WHY (owner, 2026-10-15): "feel free to add new terms to the atlas lexicon, but check each one against the papers
first". The 2026-10-13 paper-mining review (report/boat_world_model/LIT_PAPER_MINING.md, IE table) rated deterministic
abbreviation mining -- Schwartz & Hearst (PSB 2003): 96% precision / 82% recall with no training data -- ADOPT for
lexicon growth. `candidates` is that algorithm over all held papers; a person (or a session) chooses which candidates
become terms and writes the definitions; `quotes` and `verify` make sure every citation is a verbatim sentence of the
cited paper, on the page it names -- the same rule the lexicon's agent-written quotes were held to.

ALGORITHM (Schwartz & Hearst 2003). A short form is a parenthesised span of 2-10 characters with at most two words,
starting with a letter or digit, holding at least two upper-case letters (acronyms only: '(2019)', '(a)', '(see ...)'
never qualify). Its long form is found in the min(|SF| + 5, 2 |SF|) words before the parenthesis by matching the short
form's characters right to left, the first one at the start of a word; a long form that contains the short form, or is
not longer than it, is refused.

HONEST SCOPE: an acronym miner finds ABBREVIATED terms only; a concept written out in full every time is invisible to
it. Counts are papers that DEFINE the pair, not papers that use the term. The ranking is a shortlist: whether a
candidate is worth a lexicon entry is a judgment, and every entry's definition is written by a person or a session --
`verify` checks its quotes and that the cited paper uses the term, not that the definition is right.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_text as PT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BWM = PT.BWM                                  # the atlas project (tools/atlas_project.py)
REL = {"core": 0, "high": 1, "medium": 2, "low": 3}
MAX_QUOTE_WORDS = 15
_SF = re.compile(r"\(([^()\[\]]{2,12})\)")
_SENT_END = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z(\[])")


# ------------------------------------------------------------------------------------------ the algorithm (pure)
def is_short_form(sf: str) -> bool:
    """Schwartz & Hearst's short-form test, narrowed to acronyms: 2-10 characters, at most two words, starting with a
    letter or digit, at least one letter, at least two upper-case letters, not a bare year or roman numeral list."""
    s = sf.strip()
    if not (2 <= len(s) <= 10) or len(s.split()) > 2 or not s[0].isalnum():
        return False
    if sum(c.isupper() for c in s) < 2 or not any(c.isalpha() for c in s):
        return False
    return not re.fullmatch(r"[IVXL]+|\d{4}[a-z]?|[A-Z]\d*|e\.g\.|i\.e\.", s)


def best_long_form(sf: str, lf: str) -> Optional[str]:
    """Schwartz & Hearst's backward match: each short-form character (right to left) found in the long-form candidate
    (right to left), the FIRST one at the start of a word. -> the long form (from that word on) or None."""
    s, i = len(sf) - 1, len(lf) - 1
    while s >= 0:
        c = sf[s].lower()
        if not c.isalnum():
            s -= 1
            continue
        while i >= 0 and (lf[i].lower() != c or (s == 0 and i > 0 and lf[i - 1].isalnum())):
            i -= 1
        if i < 0:
            return None
        i -= 1
        s -= 1
    start = lf.rfind(" ", 0, i + 1) + 1
    out = lf[start:].strip()
    return out or None


def pairs_in(text: str) -> List[Tuple[str, str, int]]:
    """Every (short form, long form, offset) definition in one flattened page of text."""
    out = []
    for m in _SF.finditer(text):
        sf = m.group(1).strip()
        if not is_short_form(sf):
            continue
        before = text[:m.start()].rstrip()
        cut = max(before.rfind(". "), before.rfind("; "), before.rfind(": "), before.rfind("? "))
        sent = before[cut + 2:] if cut >= 0 else before
        words = sent.split()
        n = min(len(sf) + 5, 2 * len(sf))
        cand = " ".join(words[-n:])
        lf = best_long_form(sf, cand)
        if not lf:
            continue
        lw = lf.split()
        if len(lf) <= len(sf) or sf.lower() in lf.lower() or len(lw) < 2 or len(lw) > n:
            continue
        if not re.match(r"^[A-Za-z]", lf) or re.search(r"[()\[\]{}=<>|]", lf):
            continue
        out.append((sf, lf, m.start()))
    return out


def norm_lf(lf: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\u2010-\u2015-]+\s*", "-", lf.strip().lower()))


# ------------------------------------------------------------------------------------------ the index + corpus
def _index(base: Path = BWM):
    papers = json.loads((base / "index" / "papers.json").read_text(encoding="utf-8"))
    lex = json.loads((base / "index" / "lexicon.json").read_text(encoding="utf-8"))["terms"]
    return {p["key"]: p for p in papers["papers"]}, papers["aliases"], lex


def known_names(lex: Sequence[dict]) -> Tuple[set, set]:
    """(exact names, folded names) of every lexicon term and alias -- also each name with a trailing '(XYZ)' removed
    and the XYZ itself, so 'Terrain-aided navigation (TAN)' knows both 'terrain-aided navigation' and 'TAN'."""
    exact, folded = set(), set()
    for t in lex:
        for n in [t["term"]] + list(t.get("aliases") or []):
            n = str(n).strip()
            parts = [n] + re.findall(r"\(([^()]+)\)", n) + [re.sub(r"\s*\([^()]*\)\s*", " ", n).strip()]
            for p in parts:
                for q in re.split(r"\s*/\s*", p):
                    if q.strip():
                        exact.add(q.strip())
                        folded.add(norm_lf(q))
    return exact, folded


def candidates(cache: Path, base: Path = BWM, min_papers: int = 4, say=print,
               known: Optional[Sequence[dict]] = None) -> List[dict]:
    """Every (SF, long form) pair defined in >= min_papers held papers that the lexicon lacks (neither the short form
    nor the long form is a term or alias), ranked by papers of core / high relevance, then all papers. `known`: the
    terms to treat as the lexicon (default: the built index/lexicon.json)."""
    names = PT.bwm_names()
    by_key, alias, lex = _index(base)
    exact, folded = known_names(lex if known is None else known)
    hits: Dict[Tuple[str, str], dict] = {}
    lf_forms: Dict[Tuple[str, str], collections.Counter] = {}
    for n, (info, pages) in enumerate(PT.corpus(cache, names), 1):
        key = alias.get(info["key"], info["key"])
        seen = set()
        for pno, pg in enumerate(pages, 1):
            text = PT._flat_ws(pg)
            for sf, lf, off in pairs_in(text):
                k = (sf, norm_lf(lf))
                lf_forms.setdefault(k, collections.Counter())[lf] += 1
                h = hits.setdefault(k, {"sf": sf, "lf": k[1], "papers": set(), "core_high": set(), "example": None})
                if key in seen and k in seen:
                    continue
                h["papers"].add(key)
                if REL.get((by_key.get(key) or {}).get("relevance"), 9) <= 1:
                    h["core_high"].add(key)
                if h["example"] is None or (REL.get((by_key.get(key) or {}).get("relevance"), 9) <
                                            REL.get((by_key.get(h["example"]["key"]) or {}).get("relevance"), 9)):
                    a, b = max(0, off - 160), min(len(text), off + 60)
                    h["example"] = {"key": key, "page": pno, "text": text[a:b]}
                seen.add(k)
        if n % 500 == 0:
            say(f"  read {n} papers")
    out = []
    for k, h in hits.items():
        if len(h["papers"]) < min_papers:
            continue
        if h["sf"] in exact or h["lf"] in folded or norm_lf(h["sf"]) in folded:
            continue
        out.append({"sf": h["sf"], "lf": lf_forms[k].most_common(1)[0][0], "papers": len(h["papers"]),
                    "core_high": len(h["core_high"]), "keys": sorted(h["papers"])[:12], "example": h["example"]})
    out.sort(key=lambda r: (-r["core_high"], -r["papers"], r["sf"].lower(), r["lf"].lower()))
    return out


def quotes(cache: Path, long_form: str, short_form: str = "", n: int = 8, base: Path = BWM) -> List[dict]:
    """Verbatim windows of <= 15 words around the term, one per paper, defining occurrences ('long form (SF)') first,
    then core / high relevance papers: [{key, page, quote, defines, relevance}]."""
    return quotes_many(cache, [(long_form, short_form)], n, base)[0]


def quotes_many(cache: Path, pairs: Sequence[Tuple[str, str]], n: int = 8, base: Path = BWM) -> List[List[dict]]:
    """quotes() for many (long form, short form) pairs in ONE pass over the corpus (a pass reads ~120 M characters):
    per pair, the best occurrence per paper -- a defining one ('long form (SF)') first, then the earliest page --
    sorted defining first, then by relevance (core first), then key."""
    names = PT.bwm_names()
    by_key, alias, _ = _index(base)
    rxs = [re.compile(r"(?<![A-Za-z0-9])" + r"[\s-]*".join(map(re.escape, re.findall(r"[A-Za-z0-9]+", lf)))
                      + r"(?![A-Za-z0-9])", re.I) for lf, _ in pairs]
    found: List[Dict[str, tuple]] = [{} for _ in pairs]
    for info, pages in PT.corpus(cache, names):
        key = alias.get(info["key"], info["key"])
        low = None
        for pno, pg in enumerate(pages, 1):
            text = PT._flat_ws(pg)
            low = text.lower()
            for i, ((lf, sf), rx) in enumerate(zip(pairs, rxs)):
                first = re.findall(r"[a-z0-9]+", lf.lower())[:1]
                if first and first[0] not in low:     # substring prefilter (C speed) before the regex
                    continue
                best = found[i].get(key)
                if best is not None and best[0] is False:
                    continue
                for m in rx.finditer(text):
                    tail = text[m.end():m.end() + len(sf) + 4].lstrip()
                    defines = bool(sf) and tail.startswith("(" + sf)
                    cand = (not defines, pno, window(text, m.start(), m.end()))
                    if best is None or cand < best:
                        best = cand
                    if defines:
                        break
                if best is not None:
                    found[i][key] = best
    out = []
    for f in found:
        rows = [{"key": k, "page": b[1], "quote": b[2], "defines": not b[0],
                 "relevance": (by_key.get(k) or {}).get("relevance", "")} for k, b in f.items()]
        rows.sort(key=lambda r: (not r["defines"], REL.get(r["relevance"], 9), r["key"]))
        out.append(rows[:n])
    return out


def window(text: str, a: int, b: int, words: int = MAX_QUOTE_WORDS) -> str:
    """The <= `words`-word verbatim span of `text` around [a, b): the match's sentence, cut to the words nearest it."""
    s0 = max(text.rfind(". ", 0, a), text.rfind("? ", 0, a), text.rfind("! ", 0, a))
    s0 = s0 + 2 if s0 >= 0 else 0
    e = [i for i in (text.find(". ", b), text.find("? ", b), text.find("! ", b)) if i >= 0]
    s1 = min(e) + 1 if e else len(text)
    sent = text[s0:s1].split()
    pos = len(text[s0:a].split())                     # the match's first word
    span = len(text[a:b].split())
    left = max(0, min(pos, pos - (words - span) // 2))
    return " ".join(sent[left:left + words])


# ----------------------------------------------------------------------------------------- the auto glossary
GLOSSARY_FILE = "auto_glossary.json"
#: a long form that names a VENUE or an ORGANISATION, not a term: reference lists define "International Conference on
#: Robotics and Automation (ICRA)" in every robotics paper (2026-10-15: 3 of the first real kit glossary's 18 entries)
_NOT_A_TERM = re.compile(r"\b(?:conference|symposium|workshop|proceedings|transactions|journal|letters|society|"
                         r"institute|institution|school|college|association|university|laboratory|agency|"
                         r"administration|foundation|council|"
                         r"ministry|department|organi[sz]ation|committee|union|corporation|company)\b", re.I)
#: venue acronyms whose long form a reference list truncates past the words above ('Intelligent Robots and Systems
#: (IROS)'): never glossary terms
_VENUE_SF = {"IROS", "ICRA", "ICML", "NEURIPS", "NIPS", "CVPR", "ICCV", "ECCV", "ICLR", "AAAI", "IJCAI", "CORL", "WACV",
             "BMVC", "EMNLP", "NAACL", "SIGGRAPH", "IFAC", "SPIE", "IEEE", "ACM", "OCEANS", "ISER", "ISRR", "ITSC",
             "IJRR", "RAL"}


def glossary(cache: Path, base: Path = BWM, min_papers: int = 3, top: int = 200, per_term: int = 2,
             say=print) -> Tuple[List[dict], int]:
    """The AUTOMATIC glossary of a project (tools/atlas.py update, stage `terms`): the abbreviations its papers define
    (Schwartz-Hearst pairs in >= `min_papers` held papers that no lexicon file names), each with up to `per_term`
    verbatim quotes from different papers (defining ones first), checked by `verify` -- an entry with any problem is
    left out. Its definition says plainly that it is automatic. An entry a person has reviewed (`"auto": false` in
    lexicon/auto_glossary.json) is KEPT as written and its names are not mined again. -> (entries, dropped)."""
    f = Path(base) / "lexicon" / GLOSSARY_FILE
    old = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else []
    kept = [e for e in old if isinstance(e, dict) and e.get("auto") is False]
    known = other_terms(base, GLOSSARY_FILE) + kept
    cands = [c for c in candidates(cache, base, min_papers, say=say, known=known)
             if not _NOT_A_TERM.search(c["lf"]) and c["sf"].upper() not in _VENUE_SF][:top]
    rows = quotes_many(cache, [(c["lf"], c["sf"]) for c in cands], n=per_term * 3, base=base) if cands else []
    new = []
    for c, qs in zip(cands, rows):
        cites, seen = [], set()
        for q in qs:
            if q["key"] in seen or not q["quote"]:
                continue
            seen.add(q["key"])
            cites.append({"key": q["key"], "quote": q["quote"], "where": f"p. {q['page']}", "source": "fulltext"})
            if len(cites) >= per_term:
                break
        if not cites:
            continue
        lf = c["lf"][0].upper() + c["lf"][1:]
        new.append({"term": f"{lf} ({c['sf']})", "aliases": [c["sf"]], "category": "abbreviation", "topic": "glossary",
                    "definition": (f"{c['lf']} ({c['sf']}), as {c['papers']} papers of this atlas define it. An "
                                   f"automatic entry: the quotes are the papers' own words; write a definition here "
                                   f"and set \"auto\": false to keep it."),
                    "citations": cites, "auto": True})
    ok = []
    for e in new:                                       # one entry at a time: a problem drops that entry only
        if not verify([e], cache, base, own_file=GLOSSARY_FILE):
            ok.append(e)
    say(f"glossary: {len(kept)} reviewed entries kept, {len(ok)} automatic ({len(new) - len(ok)} dropped by verify)")
    return kept + ok, len(new) - len(ok)


def write_glossary(entries: Sequence[dict], base: Path = BWM) -> Path:
    f = Path(base) / "lexicon" / GLOSSARY_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(list(entries), indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return f


# ------------------------------------------------------------------------------------------------- verify
def other_terms(base: Path, skip_file: str) -> List[dict]:
    """Every lexicon SOURCE entry (lexicon/<topic>.json) except those of `skip_file` -- what a new file must not
    duplicate. Read from the sources, not index/lexicon.json, so a rebuilt index holding the new terms is no 'duplicate'."""
    out = []
    for f in sorted((Path(base) / "lexicon").glob("*.json")):
        if f.name.startswith("_") or f.name == skip_file:
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        out += [e for e in (data if isinstance(data, list) else []) if isinstance(e, dict) and e.get("term")]
    return out


def verify(entries: Sequence[dict], cache: Path, base: Path = BWM, own_file: str = "") -> List[str]:
    """Problems with a lexicon file's entries ([] = every check passed): each has a definition and citations; each
    citation's key is an index paper whose text is cached; its quote has <= 15 words and occurs VERBATIM in that
    paper's text (whitespace and line-break hyphens aside) on the page its `where` names; the paper uses the term or an
    alias; neither the term (its name with or without the '(ABBR)') nor an alias is already a term or alias in another
    lexicon file -- the build would merge it into that entry instead of adding a term."""
    names = PT.bwm_names()
    sha_of = {n["key"]: sha for sha, n in names.items()}
    by_key, alias, _ = _index(base)
    exact, folded = known_names(other_terms(base, own_file))
    problems = []
    for e in entries:
        term = str(e.get("term") or "").strip()
        if not term:
            problems.append("an entry without a term")
            continue
        names_e = [term] + list(e.get("aliases") or [])
        dup = [n for n in names_e if n.strip() in exact or norm_lf(re.sub(r"\s*\([^()]*\)\s*", " ", n)) in folded]
        if dup:
            problems.append(f"{term}: already in the lexicon as {', '.join(repr(d) for d in dup)}")
        if not str(e.get("definition") or "").strip():
            problems.append(f"{term}: no definition")
        if not e.get("citations"):
            problems.append(f"{term}: no citation")
        for c in e.get("citations") or []:
            key = alias.get(c.get("key"), c.get("key"))
            q = str(c.get("quote") or "").strip()
            if key not in by_key:
                problems.append(f"{term}: {c.get('key')} is not an index paper")
                continue
            if len(q.split()) > MAX_QUOTE_WORDS or not q:
                problems.append(f"{term}: quote from {key} has {len(q.split())} words (1..{MAX_QUOTE_WORDS})")
            sha = sha_of.get(key)
            f = PT.cache_file(PT.default_cache(PT.DEFAULT_PDFS) if cache is None else cache, sha) if sha else None
            if not (f and f.is_file()):
                problems.append(f"{term}: no cached text for {key} (cannot verify)")
                continue
            pages = [PT._flat_ws(p) for p in f.read_text(encoding="utf-8").split(PT.FF)]
            found = [i for i, p in enumerate(pages, 1) if PT._flat_ws(q) in p]
            if not found:
                problems.append(f"{term}: quote NOT found verbatim in {key}: \"{q}\"")
                continue
            where = str(c.get("where") or "")
            m = re.fullmatch(r"p\.\s*(\d+)", where)
            if not m or int(m.group(1)) not in found:
                problems.append(f"{term}: {key} quote is on p. {found[0]}, `where` says {where!r}")
            body = " ".join(pages).lower()
            if not any(_uses(body, n) for n in names_e):
                problems.append(f"{term}: {key} never uses the term or an alias")
    return problems


def _uses(body: str, name: str) -> bool:
    """The lower-cased text uses this name: its words in order, joined by spaces, hyphens or nothing ('side-scan' ==
    'side scan' == 'sidescan'), on word boundaries; a '(ABBR)' in the name counts on its own too."""
    n = re.sub(r"\s*\([^()]*\)\s*", " ", str(name)).strip()
    for part in [n] + re.findall(r"\(([^()]+)\)", str(name)):
        toks = re.findall(r"[a-z0-9]+", part.lower())
        if toks and len("".join(toks)) >= 2 and \
                re.search(r"(?<![a-z0-9])" + r"[\s-]*".join(map(re.escape, toks)) + r"(?![a-z0-9])", body):
            return True
    return False


# --------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["candidates", "quotes", "verify", "glossary"])
    ap.add_argument("args", nargs="*", help="quotes: LONG_FORM [SHORT_FORM]; verify: lexicon JSON file(s)")
    ap.add_argument("--pdfs", type=Path, default=PT.DEFAULT_PDFS)
    ap.add_argument("--min-papers", type=int, default=4)
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="candidates: write a TSV here")
    a = ap.parse_args(argv)
    cache = PT.default_cache(a.pdfs)
    if a.cmd == "glossary":                          # the project's automatic glossary (tools/atlas.py update)
        entries, _ = glossary(cache, BWM, min_papers=a.min_papers, top=a.top)
        print(f"wrote {write_glossary(entries, BWM)} ({len(entries)} entries)")
        return 0
    if a.cmd == "candidates":
        rows = candidates(cache, min_papers=a.min_papers)[:a.top]
        lines = ["sf\tlong_form\tpapers\tcore_high\tkeys\texample_key\tpage\texample"]
        for r in rows:
            ex = r["example"] or {}
            lines.append(f"{r['sf']}\t{r['lf']}\t{r['papers']}\t{r['core_high']}\t{','.join(r['keys'])}\t"
                         f"{ex.get('key', '')}\t{ex.get('page', '')}\t{ex.get('text', '')}")
        text = "\n".join(lines) + "\n"
        if a.out:
            a.out.write_text(text, encoding="utf-8", newline="\n")
            print(f"{len(rows)} candidates -> {a.out}")
        else:
            print(text)
        return 0
    if a.cmd == "quotes":
        if not a.args:
            ap.error("quotes needs a long form (and optionally its short form), or --batch FILE")
        pairs = [(a.args[0], a.args[1] if len(a.args) > 1 else "")]
        if a.args[0] == "--batch" or (len(a.args) == 1 and Path(a.args[0]).is_file()):
            src = Path(a.args[-1])
            pairs = [tuple((ln.split("\t") + [""])[:2]) for ln in src.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.startswith("#")]
        for (lf, sf), rows in zip(pairs, quotes_many(cache, pairs, a.n)):
            print(f"## {lf} ({sf})" if sf else f"## {lf}")
            for r in rows:
                print(f"{r['key']}\tp. {r['page']}\t{'DEFINES' if r['defines'] else 'uses'}\t{r['relevance']}\t"
                      f"{r['quote']}")
        return 0
    bad = 0
    for f in a.args:
        entries = json.loads(Path(f).read_text(encoding="utf-8"))
        problems = verify(entries, cache, own_file=Path(f).name)
        n_c = sum(len(e.get("citations") or []) for e in entries)
        print(f"{f}: {len(entries)} terms, {n_c} citations, {len(problems)} problem(s)")
        for p in problems:
            print("  PROBLEM: " + p)
        bad += len(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
