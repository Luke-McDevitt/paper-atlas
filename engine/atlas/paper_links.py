"""report/boat_world_model/paper_links.py -- WHICH papers we hold as PDFs, and WHERE to get the ones we do not.

WHY (2026-10-12, owner): "upgrade the list of papers to download (ones that we are missing) to also include links to
download those papers so it is easier to hunt for them", and a portable .zip of the atlas with every paper we hold. Two
committed data files make both possible without a network or the PDF folder at build time:

  papers/_held.json   `scan`: every PDF in the papers folder matched to its index key (arXiv id or key in the file
                      name, else the DOI / title in its own metadata via tools/paper_pdfs.py), with bytes + sha256 --
                      what the build marks "held", what `bundle` packs, and (later) what the lab computer can check
                      its own copies against.
  papers/_links.json  `links`: for every paper NOT held and not on arXiv, what the open-access indexes know --
                      Semantic Scholar's batch API (open-access PDF, an arXiv version of a journal paper, the S2 page)
                      and OpenAlex (best open location, every location with a PDF). Re-run it to fill gaps (OpenAlex's
                      free daily budget runs out; the file keeps what each source returned and when).

    python report/boat_world_model/build_bwm.py add "<download folder>" [--as FILE=KEY]   # matcher + add_pdfs
    python report/boat_world_model/build_bwm.py scan  --pdfs C:/svfr/papers/boat_wm [--prune]
    python report/boat_world_model/build_bwm.py links [--source s2|openalex|all]
    python report/boat_world_model/build_bwm.py build

hunt_links(paper) is the ONE place the download list's links are made (the atlas page builds the same ones in JS):
publisher (DOI), every open copy found, an arXiv version, a Google Scholar title search (its "All versions" finds
author copies), a web search for the title as a PDF, and the Semantic Scholar page. No library-proxy link: UF's
proxy prefix could not be confirmed (2026-10-12, login.lp.hscl.ufl.edu passed straight through) -- on campus or on
the UF VPN the publisher link opens the full text.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

UA = {"User-Agent": "sonar-video-fusion-review/build_bwm-links", "Content-Type": "application/json"}
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=externalIds,openAccessPdf,url"
OPENALEX = "https://api.openalex.org/works?filter=doi:{dois}&select=doi,open_access,best_oa_location,locations&per-page=50"
_ARXIV_NAME = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")


# ------------------------------------------------------------------------------------------------------- held
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdfs_in(folder: Path) -> List[Path]:
    """The PDFs directly in `folder`, by name (the .pdf suffix in any case: Linux globs are case-sensitive)."""
    return sorted((f for f in Path(folder).iterdir() if f.is_file() and f.suffix.lower() == ".pdf"), key=lambda x: x.name)


def _flat(s: str) -> str:
    """Lower-case letters and digits only, for comparing titles. Both sides of every comparison go through it:
    compatibility-decomposed first (pdfminer returns the fi / fl / ffl ligature glyphs as single characters, which
    the plain filter used to DROP -- 'Offline' became 'Oine'), accents dropped, HTML entities and tags removed
    (Crossref titles carry '&amp;' and '<scp>')."""
    s = re.sub(r"</?[A-Za-z][^<>]{0,40}>", "", html.unescape(html.unescape(str(s or ""))))  # some are escaped twice
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


_DOI_TEXT = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")
#: `Bingham_2019_Toward_Maritime_...`, `2024_Title_...` (no author), with an optional `_2` collision number
_LIBRARY_NAME = re.compile(r"^(?:[A-Za-z]+_)?(?:19|20)\d\d_(.+)$")
# a notice ABOUT a paper prints that paper's title and DOI on its page 1 -- never let it stand in for the paper. It
# names itself at the START of a heading line ("Correction to ...", "Corrigendum:", "RETRACTED ARTICLE:", a lone
# "Erratum" label); prose ("in response to the agent", "geometric correction of side-scan") never starts one.
_NOTICE_LINE = re.compile(r"^(?:(?:(?:author|publisher|editorial|editor'?s?)\s+)?"
                          r"(?:corrigend(?:um|a)|errat(?:um|a)|addendum|expression of concern"
                          r"|corrections?(?: notice)?"
                          r"|notice of retraction|retraction(?: note| notice)?|retracted(?: article)?"
                          r"|withdrawal notice|withdrawn)"
                          r"(?:\s*$|\s*[:.]|\s+(?:to|on)\b)"
                          r"|(?:corrigend(?:um|a)|errat(?:um|a)|retraction|notice of retraction)\s+(?:of|for)\b"
                          r"|(?:(?:authors?'?s?|general)\s+)?"
                          r"(?:comments?|commentary|reply|response|rejoinder|rebuttal)\s*$"
                          r"|(?:comments?|commentary|reply|response|rejoinder|rebuttal)\s+(?:to|on)\b"
                          r"|(?:comments?|commentary|reply|response|rejoinder|rebuttal|clarification|discussion"
                          r"|closure)\s*:"
                          r"|(?:discussion|closure)\s+(?:of|to)\s*(?:$|[\"'\u201c\u2018]))")   # ASCE: a QUOTED title
#   (or the line end) -- never prose ('Discussion of important unanswered questions ...' in a book's preface)
#   (fix check 2026-10-14: Elsevier 'WITHDRAWN: <title>', ASCE 'Discussion of "<title>"' / 'Closure to', 'Comment:'
#   passed as the paper with 'title+doi'; third fix check 2026-10-15: Frontiers' 'Commentary: <title>', ICE's
#   'Discussion: <title>', 'Rejoinder:', a lone "Authors' Reply" / 'Erratum for' / 'Retraction of' label line too.
#   'Correction of / for' is NOT one: real titles start that way)
#: a COMPANION file of a paper (its supplement, appendix, slides) prints the paper's title and authors too
_COMPANION_LINE = re.compile(r"^(?:(?:electronic|online)\s+)?(?:supplementary|supplemental|supporting)\s+"
                             r"(?:materials?|information|document|file|data|appendix|appendices|text|note)s?"
                             r"\s*(?:$|:|\s(?:for|to|of)\b)"
                             r"|(?:appendix|appendices)\s*(?:$|:|\s(?:for|to)\b)"
                             r"|online resource\b|(?:presentation\s+)?slides\s*$")
#: the label ALONE on its line ('Supporting Information:') -- on a repository cover or in a journal's page-1 SIDEBAR
#: it is a FIELD of the paper, not the label of a companion file
_BARE_COMPANION = re.compile(r"^(?:(?:electronic|online)\s+)?(?:supplementary|supplemental|supporting)\s+"
                             r"(?:materials?|information|data)\s*:?$")
_SIDEBAR_FIELD = re.compile(r"^(?:key points|plain language summary|correspondence to|citation|received|accepted"
                            r"|published online)\s*:?")


def page1_text(f: Path) -> str:
    """The text of a PDF's first page (pdfminer.six, which the repo's declared pdfplumber pulls in), '' when it cannot
    be read -- a scanned page, or the library missing."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(f), maxpages=1) or ""
    except Exception:                                    # any parser failure is just "no text"
        return ""


def _clean_title(t: str) -> str:
    """An index title as it would be PRINTED: Crossref's HTML entities (some escaped twice) and tags, and LaTeX
    wrappers, dropped (review 2026-10-14: '<scp>3D</scp>', '&amp;#x201C;Seatrac&amp;#x201D;', '${\\cal K}$' sent 8 right
    PDFs to be read)."""
    t = html.unescape(html.unescape(str(t or "")))
    t = re.sub(r"</?[A-Za-z][^<>]{0,40}>", " ", t)
    t = re.sub(r"\\[A-Za-z]+", " ", t)                # \cal, \mathbf, \infty, ... (the argument stays)
    return re.sub(r"[${}\\]", " ", t)


def _id_streams(text: str) -> List[List[str]]:
    """The words of a text, twice: as printed, and with a line-end hyphen joined ('inter-\nnational'); accents and
    ligatures folded (NFKD), lower case."""
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()
    joined = re.sub(r"-[ \t]*\r?\n\s*(?=[a-z])", "", t)
    return [re.findall(r"[a-z0-9]+", t), re.findall(r"[a-z0-9]+", joined)]


#: where a paper's TITLE BLOCK ends: an abstract / article-info / keywords / introduction heading at a line start
_BLOCK_END = re.compile(r"^[ \t]*(?:a[ \t]*b[ \t]*s[ \t]*t[ \t]*r[ \t]*a[ \t]*c[ \t]*t\b"
                        r"|a[ \t]*r[ \t]*t[ \t]*i[ \t]*c[ \t]*l[ \t]*e[ \t]+i[ \t]*n[ \t]*f[ \t]*o\b"
                        r"|summary\b|key[ \t-]*words\b|index[ \t]+terms\b|(?:1|i)\.?[ \t]+introduction\b"
                        r"|introduction[ \t]*$)", re.I | re.M)
#: a references / bibliography heading: nothing after it is the paper's own identity (a cited title or DOI)
_REFS_HEAD = re.compile(r"^[ \t]*(?:\d+\.?[ \t]*)?(?:references|bibliography|literature cited|works cited)"
                        r"[ \t]*:?[ \t]*$", re.I | re.M)
#: a page that only WRAPS the paper (a repository or thesis cover, a download stamp): the title may be on the next
_COVER = re.compile(r"collaborating with jstor|this content downloaded from|researchgate|see discussions, stats"
                    r"|escholarship|permalink|proquest", re.I)
#: a THESIS cover says so near its top -- in its first 15 lines, never in an acknowledgement or abstract sentence
#: (third fix check 2026-10-15: '... based on the author's thesis' in page 1 prose made page 2 a title block)
_THESIS = re.compile(r"\bdissertation\b|\bthesis\b", re.I)   # whole words: never
#: 'synthesis' / 'hypothesis'; no per-page publisher stamp ('Authorized licensed use ...' is on EVERY IEEE page -- fix
#: check 2026-10-14: 191 held PDFs let pages 2-3 prose count as a title block)
#: a one-word acronym / name before a colon on a PRINTED title line ('COM-PACT: COMponent-Aware ...') that the index
#: title lacks. ONE token only -- 'Part I:' (a space inside) is never dropped
_PREFIX = re.compile(r"^\s*[A-Za-z0-9][A-Za-z0-9.-]{1,24}\s*:\s*")   # ('Corrigendum: <title>' is a NOTICE first:
#                                                                            identity_evidence returns 'notice')
#: what may differ between an index title and the printed line: a footnote mark / affiliation letter, a stop word
_NOISE = re.compile(r"\d{1,3}|[a-z]")
_STOPW = frozenset("a an and as at by for from in into of on or the to via with".split())


def _title_blocks(pages: List[str], n_pages: int = 3) -> List[str]:
    """The text that can hold the TITLE: page 1 up to its abstract / introduction / references heading (its first
    2,000 characters when there is none), and the next page's when the page before is a cover (a repository or thesis
    wrapper, a download stamp, or under 600 letters) -- never body prose, a reference list, or page 3 of a normal
    paper (review 2026-10-14: a generic phrase, a citing paper's references, an abstract all 'confirmed' wrong PDFs)."""
    out = []
    for pg in pages[:n_pages]:
        pg = pg or ""
        cut = [next((m.start() for m in rx.finditer(pg) if m.start() > 0), len(pg)) for rx in (_BLOCK_END, _REFS_HEAD)]
        out.append(pg[:min(cut + [2000])])
        top = "\n".join([ln for ln in pg.splitlines() if ln.strip()][:15])
        if not (_COVER.search(pg[:3000]) or _THESIS.search(top) or sum(c.isalnum() for c in pg) < 600):
            break
    return out


def _variant(x: str, y: str) -> bool:
    """A spelling variant ('manoeuvring' / 'maneuvering'), or a footnote mark glued on ('vessels1')."""
    import difflib
    if re.fullmatch(re.escape(x) + r"\d{1,3}", y):
        return True
    return (x.isalpha() and y.isalpha() and min(len(x), len(y)) >= 4 and x[:3] == y[:3]
            and abs(len(x) - len(y)) <= 2 and difflib.SequenceMatcher(None, x, y).ratio() >= 0.75)
    # (fix check 2026-10-14: with ratio alone, 754 index titles were 'confirmed' by a page saying the OPPOSITE --
    # offline/online, centralized/decentralized, implicit/explicit, sonar/solar)


def _aligned(t: List[str], w: List[str]) -> Tuple[bool, float]:
    """Does the word list `w` (some whole printed lines) say the title `t`? Every difference must be a footnote mark /
    affiliation letter or a stop word put in, a stop word left out, or a spelling variant (at most one per 8 title
    words). ANY other word -- before ('Mildly ...'), inside, after, or instead ('Part 2' for 'Part 1') -- fails.
    -> (matched, word-sequence similarity)."""
    import difflib
    sm = difflib.SequenceMatcher(None, t, w, autojunk=False)
    sim = sm.ratio()
    if sim < 0.8:
        return False, sim
    variants = 0
    ops = sm.get_opcodes()
    for n, (op, a0, a1, b0, b1) in enumerate(ops):
        ta, wb = t[a0:a1], w[b0:b1]
        if op == "equal":
            continue
        edge = n == 0 or n == len(ops) - 1             # the edges: 'of Tethered ...' ends 'Framework for ... Planning'
        if op == "insert" and edge and not (n == len(ops) - 1 and n > 0 and all(_NOISE.fullmatch(x) for x in wb)):
            return False, sim                          # (fix check 2026-10-14) -- only a footnote mark after the end
        if op == "delete" and edge:
            return False, sim
        if op == "insert" and all(_NOISE.fullmatch(x) or x in _STOPW for x in wb):
            continue
        if op == "delete" and all(x in _STOPW for x in ta):
            continue
        if op == "replace" and len(ta) == len(wb) and all(_variant(x, y) for x, y in zip(ta, wb)):
            variants += len(ta)
            continue
        if op == "replace" and len(ta) + len(wb) <= 4 and _variant("".join(ta), "".join(wb)):
            variants += 1                              # one word split or merged ('NavM rph' for 'NavMorph': a glyph)
            continue
        return False, sim
    return variants <= max(1, len(t) // 8), sim


def _title_match(t: List[str], block: str, similar: bool = True) -> Tuple[str, float, str]:
    """('exact' | 'aligned' | '', best similarity, the 4 lines before and 8 after the match -- where its author line
    is): some run
    of 1-8 WHOLE lines of `block` is the title -- squashed
    (no spaces, hyphens or line ends) EQUAL to it, footnote digits after it allowed, or word-aligned (`_aligned`, titles
    of 4+ words). A title inside a longer line or phrase never matches (review 2026-10-14: 'world model synthesizes'
    held 'World Models' once squashed; 'Mildly Conservative Q-Learning ...' held 'Conservative Q-Learning ...')."""
    st = "".join(t)
    lines = [ln for ln in re.split(r"[\r\n]+", block) if ln.strip()]
    best = 0.0
    for i in range(len(lines)):
        for j in range(i + 1, min(i + 9, len(lines) + 1)):
            span = "\n".join(lines[i:j])
            streams = _id_streams(span)
            cut = _PREFIX.sub("", span, count=1)
            nums = [re.match(r"^\s*(\d{1,4})\s+", ln) for ln in lines[i:j]]   # a line-numbered manuscript: '1 A ...'
            plain = ("\n".join(ln[m.end():] for ln, m in zip(lines[i:j], nums))  # '2 Guidance ...' (2+ lines,
                     if j - i >= 2 and all(nums) and [int(m.group(1)) for m in nums]  # consecutive numbers only)
                     == list(range(int(nums[0].group(1)), int(nums[0].group(1)) + j - i)) else "")
            extra = (_id_streams(cut)[:1] if cut != span else []) + (_id_streams(plain)[:1] if plain else [])
            for words in streams + extra:
                sq = "".join(words)
                around = "\n".join(lines[max(0, i - 4):i] + lines[j:j + 8])   # a book cover prints its
                if sq and (sq == st or re.fullmatch(re.escape(st) + r"\d{1,3}", sq)):   # authors ABOVE the title
                    return "exact", 1.0, around
                if similar and words:
                    ok, sim = _aligned(t, words)
                    best = max(best, sim)
                    if ok:
                        return "aligned", sim, around
            if len(streams[0]) > 2 * len(t) + 8:
                break                                  # a longer span only adds words
    return "", best, ""


def _id_printed(ident: str, text: str, kind: str) -> bool:
    """Is this DOI / arXiv id printed in `text` (NFKD, lower case, whitespace KEPT) -- as a whole id, never the start of
    a longer one ('10.1/a' inside '10.1/ab'; review 2026-10-14), and allowed to break across a line?"""
    i = (ident or "").strip().lower()
    if kind == "arxiv":
        i = re.sub(r"v\d+$", "", i)
    if not i or (kind == "doi" and i.startswith("10.48550/")):     # arXiv's DOI is not a publisher DOI
        return False
    body = r"\s*".join(re.escape(c) for c in i)
    pat = (r"(?<![0-9])" + body + r"(?![0-9a-z_])") if kind == "doi" else (r"(?<![\d.])" + body + r"(?:v\d+)?(?!\d)")
    return re.search(pat, text) is not None


_PARTICLES = frozenset("van von der den de del della da di dos du la le ter ten bin al el".split())


def _author_after(authors: Sequence[str], after: str, surnames: frozenset = frozenset()) -> bool:
    """One of the first 3 authors on the lines right around the title (its author line) AS A NAME: the surname with a
    word of the given name's initial right before it ('Amir Bar', 'A. Bar', 'Jun-Yan Zhu', 'Arturo Gomez Chavez') or
    right after it ('Chen, W.', 'MA Yi'); a glued footnote mark allowed ('Kirillov1,2,4' -> 'kirillov1'), a glued
    affiliation LETTER only after a surname of 4+ letters that does not make ANOTHER surname of `surnames` (the index's:
    'Chen'+'g' is 'Cheng'; 'Ha'+'o' is 'Hao'); never a citation further down. A surname alone is not enough: 'Bar' is in
    'Bar-Ilan University', 'Ma' in 'Cambridge, MA', 'He' in 'He received' (third fix check 2026-10-15)."""
    people = []
    for a in list(authors or [])[:3]:
        w = _id_streams(str(a))[0]
        if a != "et al." and w and len(w[-1]) >= 2:
            people.append((w[-1], w[0][0] if len(w) >= 2 else "", frozenset(w[1:-1])))
    toks = _id_streams(after)[0]
    skip = lambda x, own: x in _PARTICLES or x in own or (len(x) > 1 and x in _STOPW)   # ('A. Wald': 'a' an initial)
    for nm, g0, own in people:                         # (own: the rest of a compound name -- 'L. Abril Torres-Mendez')
        for p, tk in enumerate(toks):
            if not (re.fullmatch(re.escape(nm) + r"(?:\d{1,3})?", tk)
                    or (len(nm) >= 4 and len(tk) == len(nm) + 1 and tk.startswith(nm) and tk[-1].isalpha()
                        and tk not in surnames)):
                continue
            if not g0:                                 # the index gives no given name: the surname must be long
                if len(nm) >= 4:
                    return True
                continue
            before = [x for x in toks[max(0, p - 5):p] if not skip(x, own) and x != nm]
            nxt = toks[p + 1] if p + 1 < len(toks) else ""
            if any(x.startswith(g0) for x in before[-2:]) or (nxt.startswith(g0) and not skip(nxt, frozenset())):
                return True
    return False


def identity_evidence(title: str, doi: str, arxiv_id: str, pages: List[str], n_pages: int = 3,
                      authors: Sequence[str] = (), longer: Sequence[Tuple[str, str]] = (),
                      surnames: frozenset = frozenset()) -> dict:
    """Is this PDF the paper the index says? Pure (the caller reads the text cache). The index title (markup dropped)
    must BE some whole lines of the TITLE BLOCK (`_title_blocks`: page 1 before its abstract, and a cover page's next
    page) -- `_title_match`: 'exact' (squashed equal) or 'aligned' (word-aligned, only footnote marks, stop words and a
    spelling variant may differ; titles of 4+ words). `doi` / `arxiv`: the index's DOI / arXiv id printed as a whole id
    on pages 1-3 before any references heading. verdict: 'title' / 'title+doi', else 'doi only' or 'weak' (READ it: a
    notice or a citing paper prints another paper's DOI; a retitled version or an odd layout is a false alarm), or 'no
    text' (a scan: OCR it first). Review 2026-10-14 MEASURED the first design (the title anywhere on pages 1-3, or 70 %
    of its words in order): 700 (held PDF, other index title) pairs passed -- a generic phrase, a citing paper's
    references, a longer title containing it. The title alone is not enough ('World Models' is the last line of
    'Transformers are Sample-Efficient / World Models'; 'Conservative Q-Learning ...' the second line of 'Mildly /
    Conservative ...'): it also needs one of the first 3 `authors`' surnames on the lines after it, or the DOI / arXiv id
    printed. A correction / retraction notice is verdict 'notice' (READ it), never the paper. `longer` = [(key, title)]
    of OTHER index papers whose title holds this one as a run of words: one of them matching too is verdict 'sibling'
    (READ it) -- a same-author sibling ('SAM 2: / Segment Anything / in Images and Videos' for 'Segment Anything')
    meets the author rule (fix check 2026-10-14). A companion file -- the paper's supplement, appendix or slides --
    is verdict 'supplement' (READ it; third fix check 2026-10-15). `surnames`: every index author's surname
    (`_author_after`)."""
    window = "\n".join(pages[:n_pages])
    m = _REFS_HEAD.search(window)
    if m:
        window = window[:m.start()]
    if len(_id_streams(window)[0]) < 20:
        return {"verdict": "no text", "how": "", "sim": 0.0, "n": 0, "doi": False, "arxiv": False}
    t = _id_streams(_clean_title(title))[0]
    text = unicodedata.normalize("NFKD", window).lower()
    doi_hit, ax_hit = _id_printed(doi, text, "doi"), _id_printed(arxiv_id, text, "arxiv")
    how, sim, after = "", 0.0, ""
    if len("".join(t)) >= 4:                           # (never the index title minus its prefix: 'MOReL : Model-Based
        for block in _title_blocks(pages, n_pages):    # Offline RL' minus 'MOReL' is a wrapped LINE of another title)
            h, s, after = _title_match(t, block, similar=len(t) >= 4)
            sim = max(sim, s)
            if h:
                how = h
                break
    if how and not (doi_hit or ax_hit) and not _author_after(authors, after, surnames):
        how = ""                                       # the title alone: read it (a longer, different title can hold it
    #                                                    on lines of their own -- fix check 2026-10-14)
    blocks = _title_blocks(pages, n_pages)
    out = {"how": how, "sim": round(sim, 3), "n": len(t), "doi": doi_hit, "arxiv": ax_hit}
    if any(looks_like_a_notice(b) for b in blocks):   # a correction / retraction notice prints the original's title
        return {"verdict": "notice", **out}            # (in ANY title block: page 2 behind a repository cover too)
    if any(looks_like_a_companion(b) for b in blocks):
        return {"verdict": "supplement", **out}
    if how:
        for k, other in longer:
            ot = _id_streams(_clean_title(other))[0]
            if any(_title_match(ot, b, similar=len(ot) >= 4)[0] for b in blocks):
                return {"verdict": "sibling", "sibling": k, **out}
    verdict = ("title+doi" if how and doi_hit else "title" if how else "doi only" if doi_hit or ax_hit else "weak")
    return {"verdict": verdict, "how": how, "sim": round(sim, 3), "n": len(t), "doi": doi_hit, "arxiv": ax_hit}


def _head_lines(text: str) -> List[str]:
    """Page 1's first 25 lines, lower case, typeset apostrophes folded ('Authors\u2019 Reply' -- fix check 2026-10-15:
    the label as PDFs extract it never matched)."""
    fold = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})
    return [ln.strip().lower().translate(fold).lstrip("\"'([") for ln in text.splitlines() if ln.strip()][:25]


def looks_like_a_notice(text: str) -> bool:
    """A correction / corrigendum / erratum / retraction / comment / reply: one of page 1's first 25 lines starts by
    naming that article type."""
    return any(_NOTICE_LINE.match(ln) for ln in _head_lines(text))


def looks_like_a_companion(text: str) -> bool:
    """A paper's supplement / appendix / slides: one of page 1's first 25 lines is that label ('Supplementary
    Material', 'Supplementary Material:', 'Supplementary Material for', 'Appendix', 'Slides' -- anywhere in a title
    block the supplement copied from its paper) -- never a sentence ('Supplementary material is available online'), and
    never a bare FIELD label: a repository cover's ('Supplemental Material' + its URL on an eScholarship cover) or a
    journal sidebar's ('Supporting Information:' beside 'Key Points:' / 'Correspondence to:' on an AGU paper). Fix
    checks 2026-10-15: the first rule refused three held AGU papers in `add`'s page-1 matcher; the next, a line
    window and a whole-cover exemption, let a supplement with a long title block, a 'Supplementary Material:' line or
    a ResearchGate cover through."""
    lines = _head_lines(text)
    cover = bool(_COVER.search(text[:3000]))
    sidebar = any(_SIDEBAR_FIELD.match(ln) for ln in lines)
    for ln in lines:
        if not _COMPANION_LINE.match(ln):
            continue
        if _BARE_COMPANION.match(ln) and (cover or (sidebar and ln.startswith("supporting information"))):
            continue                                   # a field of the paper / of its cover
        return True
    return False


def matcher(papers: List[dict], aliases: Dict[str, str], identity: Optional[Callable[[Path], dict]] = None,
            page_text: Optional[Callable[[Path], str]] = None) -> Callable[[Path], Tuple[Optional[str], str]]:
    """-> match(pdf) -> (index key or None, how). The FILE NAME first (an arXiv id; a key / alias as the stem or after
    `<arXiv id>_`), then the PDF's own metadata via `identity` (tools/paper_pdfs.pdf_identity: arXiv id, a DOI from a
    DOI-specific tag, the embedded title -- exact after `_flat`, >= 24 characters), then the file name read as a title,
    the same way (IEEE Xplore saves `Title_Words_Like_This.pdf`; metadata titles can be junk like "untitled"), then --
    via `page_text` -- what PAGE 1 prints (2026-10-12: MDPI / Copernicus / MTS name files `jmse-12-02173.pdf`, `s7.pdf`
    and embed no DOI tag). Page 1 never names a correction / comment / reply notice (it prints the ORIGINAL's title
    and DOI). Else: the ONE indexed paper whose DOI and title are both on page 1 (a cited DOI alone never counts; two
    such papers = ambiguous, None); else the ONE indexed title in the title block -- page 1 up to "abstract" (at most
    1,500 characters), or its first 600 when no abstract heading is found -- after dropping a hit that is part of a
    longer hit's title. Never a guess. Measured 2026-10-12 on 676 real PDFs of known identity: page 1 alone names
    650, never a wrong one; with each paper's OWN entry removed from the index (a download the index does not know)
    it names another paper once -- a title containing a sibling paper's whole title (buchholz2025framework /
    buchholz2025tethersim), a limit only a paper outside the index can hit."""
    by_arx = {p["arxiv_id"]: p["key"] for p in papers if p.get("arxiv_id")}
    by_doi = {str(p["doi"]).lower(): p["key"] for p in papers if p.get("doi")}
    keys = {p["key"] for p in papers}
    by_title = {_flat(p.get("title")): p["key"] for p in papers if len(_flat(p.get("title"))) >= 24}
    title_of = {p["key"]: _flat(p.get("title")) for p in papers}
    canon = lambda k: aliases.get(k, k) if aliases.get(k, k) in keys else None
    # Springer / Nature / Copernicus name a download by its DOI suffix (978-0-387-36945-7.pdf, s41597-025-06347-0.pdf)
    suffixes: Dict[str, List[str]] = {}
    for p in papers:
        if p.get("doi") and "/" in str(p["doi"]):
            sfx = _flat(str(p["doi"]).split("/", 1)[1])
            if len(sfx) >= 8:
                suffixes.setdefault(sfx, []).append(p["key"])
    by_suffix = {s: ks[0] for s, ks in suffixes.items() if len(ks) == 1}

    def from_page1(f: Path) -> Tuple[Optional[str], str]:
        text = page_text(f) if page_text else ""
        flat = _flat(text)
        if len(flat) < 40 or looks_like_a_notice(text) or looks_like_a_companion(text):
            return None, ""
        with_doi = sorted({k for d in _DOI_TEXT.findall(text) for k in [by_doi.get(d.rstrip(").,;").lower())]
                           if k and len(title_of[k]) >= 24 and title_of[k] in flat})
        if len(with_doi) == 1:
            return with_doi[0], "DOI and title printed on page 1"
        if with_doi:
            return None, ""
        cut = flat.find("abstract")
        block = flat[:cut if 0 < cut < 1500 else 600]      # no "abstract" heading: the top 600 only
        hits = {k: t for t, k in by_title.items() if t in block}
        hits = {k: t for k, t in hits.items() if not any(t != u and t in u for u in hits.values())}
        if len(hits) == 1:
            return next(iter(hits)), "title printed on page 1"
        return None, ""

    def match(f: Path) -> Tuple[Optional[str], str]:
        stem = f.stem
        m = _ARXIV_NAME.search(stem)
        if m and m.group(1) in by_arx:
            return by_arx[m.group(1)], "arXiv id in the file name"
        for cand in (stem, stem.split("_", 1)[1] if "_" in stem and m else ""):
            if cand and canon(cand):
                return canon(cand), "key in the file name"
        if identity is not None:
            ident = identity(f)
            if ident.get("arxiv_id") in by_arx:
                return by_arx[ident["arxiv_id"]], "arXiv id in the PDF"
            if str(ident.get("doi") or "").lower() in by_doi:
                return by_doi[str(ident.get("doi")).lower()], "DOI in the PDF metadata"
            if _flat(ident.get("title")) in by_title:
                return by_title[_flat(ident.get("title"))], "title in the PDF metadata"
        if _flat(stem) in by_title:
            return by_title[_flat(stem)], "title as the file name"
        if _flat(stem) in by_suffix:
            return by_suffix[_flat(stem)], "DOI as the file name"
        lib = _LIBRARY_NAME.match(stem)                 # Surname_2019_Title_Words (tools/paper_rename.py)
        for part in ((lib.group(1), re.sub(r"_\d$", "", lib.group(1))) if lib else ()):
            t = _flat(part)                           # (a trailing _2 may be a collision number)
            if t in by_title:
                return by_title[t], "title in a library-style file name"
            if len(t) >= 40:                          # a long title is cut in the name: a unique prefix
                hits = [k for title, k in by_title.items() if title.startswith(t)]
                if len(hits) == 1:
                    return hits[0], "title in a library-style file name (cut)"
        return from_page1(f)
    return match


def _is_rejected(rejected: Optional[dict], sha: Optional[str], key: str) -> bool:
    """Is this file (sha256) recorded as NOT this key? `rejected` maps a sha to one key or a set of keys."""
    got = (rejected or {}).get(str(sha or "").lower())
    return bool(got) and (key == got if isinstance(got, str) else key in got)


def scrub_link(link: Optional[dict], bad: Iterable[str]) -> Optional[dict]:
    """A _links.json row without any trace of arXiv ids known to be ANOTHER paper (papers/_rejected.json): Semantic
    Scholar merges a citing preprint into the cited paper's record (review 2026-10-14: aghili2013driftless got arXiv
    2208.11229 back as its 'free copy' the day it was removed)."""
    bad = {re.sub(r"v\d+$", "", str(b).lower()) for b in bad or () if b}
    if not link or not bad:
        return link
    hit = lambda u: bool(u) and any(re.search(r"(?:arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv\.)" + re.escape(b)
                                              + r"(?:v\d+)?(?!\d)", str(u).lower()) for b in bad)   # + arXiv's DOI
    out = dict(link)
    if isinstance(out.get("s2"), dict):
        s2 = dict(out["s2"])
        if re.sub(r"v\d+$", "", str(s2.get("arxiv") or "").lower()) in bad:
            s2.pop("arxiv", None)
        if hit(s2.get("pdf")):
            s2.pop("pdf", None)
        out["s2"] = s2
    if isinstance(out.get("openalex"), dict):
        oa = dict(out["openalex"])
        for f in ("pdf", "landing"):
            if hit(oa.get(f)):
                oa.pop(f, None)
        if oa.get("more"):
            oa["more"] = [u for u in oa["more"] if not hit(u)]
        out["openalex"] = oa
    return out


def scan(pdf_dir: Path, papers: List[dict], aliases: Dict[str, str],
         identity: Optional[Callable[[Path], dict]] = None, page_text: Optional[Callable[[Path], str]] = None,
         previous: Optional[dict] = None, prune: bool = False, rejected: Optional[Dict[str, str]] = None) -> dict:
    """Match every PDF in `pdf_dir` to an index key (`matcher`). Two files for one key: the larger is kept, the other
    listed as a duplicate. A row of the `previous` _held.json whose file is NOT in this folder is KEPT (listed under
    not_in_folder) unless `prune`: a scan on a machine with only part of the papers -- the lab box, an unplugged
    drive -- must never erase what another machine holds. For the same reason the `previous` `pruned` list (what the
    last `--prune` dropped) is kept by a scan that does not prune (2026-10-15: every lab `add` deleted main's).
    `rejected` {sha256: key}: a file KNOWN not to be that paper (papers/_rejected.json) is never held as it, wherever
    it turns up and whatever it is called (review 2026-10-14)."""
    rejected = rejected or {}                        # {sha: key or {keys}} -- `_is_rejected`
    match = matcher(papers, aliases, identity, page_text)
    keys = {p["key"] for p in papers}
    known = {}                                       # the same CONTENT keeps its identity, whatever the file is now called
    for r in (previous or {}).get("files") or []:
        if isinstance(r, dict) and r.get("sha256") and aliases.get(r.get("key"), r.get("key")) in keys:
            known[r["sha256"]] = (aliases.get(r["key"], r["key"]), r.get("matched_by") or "")
    files: Dict[str, dict] = {}
    unmatched, dupes, refused = [], [], []
    present = set()
    for f in pdfs_in(pdf_dir):
        present.add(f.name)
        sha = _sha256(f)
        key, how = known.get(sha) or match(f)
        if key and _is_rejected(rejected, sha, key):
            refused.append(f.name)
            continue
        if not key:
            unmatched.append(f.name)
            continue
        row = {"key": key, "file": f.name, "bytes": f.stat().st_size, "sha256": sha, "matched_by": how}
        if key in files:
            keep, drop = sorted((files[key], row), key=lambda r: (-r["bytes"], r["file"]))
            files[key] = keep
            dupes.append(drop["file"])
        else:
            files[key] = row
    away, pruned = [], []
    for r in (previous or {}).get("files") or []:
        if not isinstance(r, dict) or not r.get("key") or r.get("file") in present:
            continue
        key = aliases.get(r["key"], r["key"])
        if key in files:
            continue
        if _is_rejected(rejected, r.get("sha256"), key):   # a row from before the file was found to be another paper
            refused.append(r["file"])
            continue
        if prune:
            pruned.append(r["file"])
        else:
            files[key] = {**r, "key": key}
            away.append(r["file"])
    doc = {"schema": "svfr.bwm.held/1", "files": [files[k] for k in sorted(files)],
           "duplicates": sorted(dupes), "unmatched": unmatched, "not_in_folder": sorted(away)}
    last = (previous or {}).get("pruned")
    if pruned:
        doc["pruned"] = sorted(pruned)
    elif not prune and isinstance(last, list) and last:   # this scan pruned nothing: the last prune's list stands
        doc["pruned"] = sorted(str(x) for x in last)
    if refused:
        doc["rejected"] = sorted(set(refused))
    return doc


def pdf_problem(f: Path) -> str:
    """'' for a whole PDF, else why not: an HTML error / login page saved as .pdf, a stub, or a cut-off download (no
    %%EOF in its last 4 KB -- an interrupted transfer, e.g. when a publisher pauses access mid-download)."""
    size = f.stat().st_size
    with open(f, "rb") as fh:
        head = fh.read(1024)
        fh.seek(max(0, size - 4096))
        tail = fh.read()
    if not head.startswith(b"%PDF-"):
        return f"not a PDF (starts {head[:15]!r}) -- an error or login page saved as .pdf?"
    if size < 20_000:
        return f"only {size} bytes -- a stub, not a paper"
    if b"%%EOF" not in tail:
        return "no %%EOF at the end -- the download looks cut off; download it again"
    return ""


def suggest(f: Path, papers: List[dict], n: int = 2) -> List[Tuple[float, str]]:
    """The index papers whose titles best match the file name, for an unmatched file's hint (difflib ratio, key)."""
    import difflib
    t = re.sub(r"[^a-z0-9]+", " ", f.stem.lower()).strip()
    if len(t) < 12:
        return []
    scored = sorted(((difflib.SequenceMatcher(None, t, re.sub(r"[^a-z0-9]+", " ", (p.get("title") or "").lower())).ratio(),
                      p["key"]) for p in papers), key=lambda x: (-x[0], x[1]))
    return [(round(r, 2), k) for r, k in scored[:n] if r >= 0.5]


_DASHES = {cp: "-" for cp in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212)}


def file_id(name: str) -> str:
    """How an `--as FILE` is compared with a file in the inbox: the bare name (a tab-completed path works), any
    Unicode dash as '-', case-folded (Windows file names are case-insensitive)."""
    base = name.replace(chr(92), "/").rstrip("/").rsplit("/", 1)[-1]
    return unicodedata.normalize("NFC", base).translate(_DASHES).casefold()


def audit_folder(inbox: Path, pdf_dir: Path, papers: List[dict], aliases: Dict[str, str], held_rows: List[dict],
                 identity: Optional[Callable[[Path], dict]] = None, page_text: Optional[Callable[[Path], str]] = None,
                 data_dir: Optional[Path] = None, obtained: Optional[List[dict]] = None,
                 rejected: Optional[Dict[str, str]] = None) -> List[dict]:
    """READ-ONLY: is every file in a download folder saved somewhere else? (owner, 2026-10-13: "verify that we have
    the papers saved and renamed somewhere else other than that more papers folder"). One row per file:
      saved         -- the SAME BYTES are in the papers folder (or its _duplicates/), under `where`
      saved (data)  -- the same bytes are an obtained dataset file (`obtain`), under `where`
      paper saved   -- a different download of a paper that IS held (`where`); this exact file is not kept
      NOT saved     -- `why`: not identified / refused (broken, cut off) / identified but not copied / not a PDF
    Identity by sha256 first (renames cannot fool it), then the same matcher `add` uses. Nothing is written."""
    present: Dict[str, str] = {}
    for f in sorted(Path(pdf_dir).rglob("*.pdf")):
        present.setdefault(_sha256(f), f.relative_to(pdf_dir).as_posix())
    held_file = {aliases.get(r["key"], r["key"]): r["file"] for r in held_rows
                 if r.get("key") and r.get("file") and (Path(pdf_dir) / r["file"]).is_file()}
    data: Dict[str, str] = {}
    for item in obtained or []:
        for fr in item.get("files") or []:
            if data_dir and (Path(data_dir) / item["key"] / fr["name"]).is_file():
                data[fr["sha256"]] = f"{item['key']}/{fr['name']}"
    match = matcher(papers, aliases, identity, page_text)
    rows: List[dict] = []
    for f in sorted((p for p in Path(inbox).iterdir() if p.is_file()), key=lambda p: p.name.lower()):
        sha = _sha256(f)
        row = {"file": f.name, "bytes": f.stat().st_size, "sha256": sha}
        if sha in present:
            row.update(status="saved", where=present[sha])
        elif sha in data:
            row.update(status="saved (data)", where=data[sha])
        elif f.suffix.lower() != ".pdf":
            row.update(status="NOT saved", why="not a PDF -- only in this folder (`obtain` keeps a dataset)")
        else:
            bad = pdf_problem(f)
            key, how = (None, "") if bad else match(f)
            if bad:
                row.update(status="NOT saved", why=f"refused by add: {bad}")
            elif key and _is_rejected(rejected, sha, key):
                row.update(status="NOT saved", key=key, why=f"rejected: this file is NOT {key} (papers/_rejected.json)"
                                                            f" -- not a paper of the index")
            elif key and key in held_file:
                row.update(status="paper saved", where=held_file[key], key=key,
                           why=f"a different download of {key} ({how}); the held copy is kept, this file is not")
            elif key:
                row.update(status="NOT saved", key=key, why=f"identified as {key} ({how}) but not copied -- run add")
            else:
                row.update(status="NOT saved", why="not identified",
                           hints=[f"{k} ({r})" for r, k in suggest(f, papers)])
        rows.append(row)
    return rows


def _plain_pdf_name(name: object) -> str:
    """`name` when it is a bare `.pdf` file name safe on every machine (no folder, drive or reserved character), else
    '' -- a name read from the committed record never becomes a path."""
    s = name if isinstance(name, str) else ""
    ok = s.lower().endswith(".pdf") and len(s) <= 240 and not re.search(r'[<>:"/\\|?*\x00-\x1f]', s)
    return s if ok and s.strip(" .") == s and Path(s).name == s else ""


def add_pdfs(inbox: Path, pdf_dir: Path, papers: List[dict], aliases: Dict[str, str], held: Iterable[str],
             identity: Optional[Callable[[Path], dict]] = None, overrides: Optional[Dict[str, str]] = None,
             page_text: Optional[Callable[[Path], str]] = None,
             name_for: Optional[Callable[[str], str]] = None, held_shas: Optional[Dict[str, str]] = None,
             rejected: Optional[Dict[str, str]] = None, recorded: Optional[Iterable[dict]] = None) -> dict:
    """COPY every new paper PDF in `inbox` (e.g. the Downloads folder they landed in) into the papers folder, named
    `name_for(key)` (build_bwm: `Surname_Year_Title.pdf`, tools/paper_rename.py) or `<key>.pdf`; `scan` keeps the
    identity by content, so the name is for people. out["names"] = {key: the file name used}. The inbox is never modified.
    `held` = the papers whose PDF is already IN `pdf_dir`. `overrides` {file: key} (`--as FILE=KEY`) names a file the
    matcher cannot, or corrects it; an override naming no inbox file stops the run before anything is copied
    (bad_overrides). Refused, never copied: a broken or cut-off PDF (`pdf_problem`), an override naming no index key,
    a second file for a paper already copied in this run, a different file already at the destination. Skipped: a
    paper already held, and a file whose very bytes are already in `pdf_dir` under ANY name (one copy per content).
    `recorded` = the shared record's rows (papers/_held.json): a file whose bytes the record already holds AS THE PAPER
    this run names keeps that row's `matched_by` and its file name -- the identity was made where the evidence was (the
    original download's name, a person's --as), and re-deriving it from a renamed copy writes over it (2026-10-15, the
    lab's copy of the laptop's 2,690 PDFs: 1,230 rows re-derived, a human --as among them, and one file renamed where
    the index lists no first author). -> {added, already_held, unmatched, refused, bad_overrides} lists."""
    match = matcher(papers, aliases, identity, page_text)
    keys = {p["key"] for p in papers}
    held = set(held)
    record: Dict[Tuple[str, str], dict] = {}
    for r in recorded or ():
        if isinstance(r, dict) and r.get("sha256") and r.get("key"):
            record[(str(r["sha256"]).lower(), aliases.get(r["key"], r["key"]))] = r
    by_size: Optional[Dict[int, List[Path]]] = None  # the papers folder's PDFs by size, read once: candidates for
    shas: Dict[Path, str] = {}                       # "these bytes are already here", hashed only on a size match

    def already_here(f: Path, sha: str) -> Optional[Path]:
        nonlocal by_size
        if by_size is None:
            by_size = {}
            for p in pdfs_in(pdf_dir):
                by_size.setdefault(p.stat().st_size, []).append(p)
        for p in by_size.get(f.stat().st_size, ()):
            if shas.setdefault(p, _sha256(p)) == sha:
                return p
        return None

    out = {"added": [], "already_held": [], "unmatched": [], "refused": [], "bad_overrides": []}
    inbox_files = pdfs_in(inbox)
    ids = {file_id(f.name) for f in inbox_files}
    over = {}
    for name, key in (overrides or {}).items():
        if file_id(name) in ids:
            over[file_id(name)] = key
        else:
            out["bad_overrides"].append(name)
    if out["bad_overrides"]:
        return out
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)
    claimed: Dict[str, str] = {}
    for f in inbox_files:
        bad = pdf_problem(f)
        if bad:
            out["refused"].append((f.name, bad))
            continue
        sha = _sha256(f)
        if file_id(f.name) in over:
            asked = over[file_id(f.name)]
            key, how = aliases.get(asked, asked), "named with --as"
            if key not in keys:
                out["refused"].append((f.name, f"--as names {asked!r}, which is no paper in the index"))
                continue
        else:
            key, how = match(f)
        if not key and held_shas:                    # unreadable (a scan) but byte-identical to a PDF of the index:
            hit = held_shas.get(sha, "")             # the sha IDENTIFIES it (2026-10-14: 'noaa florida report
            if hit in keys:                          # thing.pdf' WAS whitall2019); whether it is already held HERE is
                key, how = hit, "sha256 of a held PDF"   # still decided below by this folder (review: on the lab
        if not key:                                  # a laptop-only row must not skip a copy the lab lacks)
            out["unmatched"].append((f.name, suggest(f, papers)))
            continue
        if _is_rejected(rejected, sha, key):         # even with --as: papers/_rejected.json says it is ANOTHER paper
            out["refused"].append((f.name, f"this file is NOT {key} (papers/_rejected.json: found by `confirm`)"))
            continue
        if key in claimed:
            out["refused"].append((f.name, f"the same paper ({key}) as {claimed[key]}, copied first -- compare the "
                                           f"two; if this one is a different paper, name it with --as"))
            continue
        if key in held:
            out["already_held"].append((f.name, key))
            continue
        rec = record.get((sha, key))                 # the record holds THESE bytes as THIS paper: its provenance
        if rec is not None:                          # stands, whatever this machine can still read from the copy
            how = rec.get("matched_by") or how
        here = already_here(f, sha)
        if here is not None:                         # the very bytes are in the folder already (not yet scanned,
            out["already_held"].append((f.name, key))   # or under another name): never a second copy of them
            out.setdefault("rows", []).append({"key": key, "file": here.name, "bytes": here.stat().st_size,
                                               "sha256": sha, "matched_by": how})
            continue
        want = _plain_pdf_name(rec.get("file")) if rec is not None else ""
        if want and not (Path(pdf_dir) / want).exists():
            dest = Path(pdf_dir) / want              # the record's name: every machine finds the paper by it
        else:
            dest = Path(pdf_dir) / (name_for(key) if name_for else f"{key}.pdf")
        if dest.exists():
            if not name_for:
                out["refused"].append((f.name, f"{dest.name} already exists with different bytes -- not overwritten"))
                continue
            stem, n = dest.stem, 2                   # a title name another paper already uses: number it
            while dest.exists():
                dest, n = dest.with_name(f"{stem}_{n}.pdf"), n + 1
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(f, tmp)
        replace_retry(tmp, dest)
        by_size.setdefault(dest.stat().st_size, []).append(dest)   # (read by already_here above: never None here)
        shas[dest] = sha
        claimed[key] = f.name
        out.setdefault("names", {})[key] = dest.name
        out.setdefault("rows", []).append({"key": key, "file": dest.name, "bytes": dest.stat().st_size,
                                           "sha256": _sha256(dest), "matched_by": how})
        out["added"].append((f.name, key, how))
    return out


# ------------------------------------------------------------------------------------------------------ links
def _post(url: str, body: dict, retries: int = 5, pause: float = 4.0) -> object:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=UA, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                raise
            wait = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(wait) if wait and str(wait).isdigit() else pause * (i + 1))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(pause * (i + 1))
    raise last


# the OpenAlex key's rules live in ONE place (tools/openalex_key.py), shared with verify_refs / lit_sweep
_TOOLS = str(Path(__file__).resolve().parents[2] / "tools")
if _TOOLS not in sys.path:
    sys.path.append(_TOOLS)
import openalex_key as _OK  # noqa: E402
if str(Path(__file__).resolve().parents[2]) not in sys.path:   # run by path: the repo root, for tools/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.atomic_replace import replace_retry  # noqa: E402

user_env, openalex_auth, openalex_request, openalex_urlopen = _OK.user_env, _OK.auth, _OK.request, _OK.urlopen


def _get(url: str, retries: int = 4, pause: float = 4.0) -> object:
    """GET -> JSON, retrying 5xx and dropped connections like _post; a 429 (OpenAlex's spent daily budget) is raised
    at once -- retrying it only burns the same budget."""
    last = None
    for i in range(retries):
        try:
            with openalex_urlopen(url, {"User-Agent": UA["User-Agent"]}, timeout=120) as r:   # key: openalex_key
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (500, 502, 503, 504):
                raise
            time.sleep(pause * (i + 1))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(pause * (i + 1))
    raise last


# ------------------------------------------------------------------------------- withdrawal / retraction status
#: the authors' own words in an arXiv comment -- MEASURED 2026-10-13: catches 2 of the 4 withdrawn papers we found by
#: hand ('Withdrawn to reorganize ...', 'Withdrawal for further improvement'); the other two never say it
_WITHDRAW_WORDS = re.compile(r"\b(?:withdraw\w*|retract\w*)\b", re.I)
#: arXiv's own notice on the abstract page -- caught 4 of 4 (the authoritative signal)
_WITHDRAWN_NOTICE = re.compile(r"This paper has been withdrawn[^<]{0,160}")


def withdrawn_comment(entry_xml: str) -> str:
    """The arXiv API entry's comment when it says the paper was withdrawn / retracted, else ''."""
    c = " ".join(re.findall(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>", entry_xml or "", re.S))
    return re.sub(r"\s+", " ", c).strip() if _WITHDRAW_WORDS.search(c) else ""


def withdrawn_notice(abs_html: str) -> str:
    """arXiv's 'This paper has been withdrawn by ...' notice on an abstract page, else ''."""
    m = _WITHDRAWN_NOTICE.search(abs_html or "")
    return " ".join(m.group(0).split()) if m else ""


def openalex_retracted(dois: Iterable[str], get: Callable = None, batch: int = 50, pause: float = 1.0):
    """-> ({doi: title} flagged is_retracted by OpenAlex (fed by Retraction Watch), [dois looked up], why-not)."""
    import urllib.parse
    get = get or (lambda u: _get(u))
    dois = sorted({d.lower() for d in dois if d})
    flagged, looked, why = {}, [], ""
    for i in range(0, len(dois), batch):
        chunk = dois[i:i + batch]
        q = urllib.parse.urlencode({"filter": "doi:" + "|".join(chunk), "per-page": batch,
                                    "select": "doi,is_retracted,title"})
        try:
            res = get("https://api.openalex.org/works?" + q)
        except Exception:                                # noqa: BLE001 -- unchecked, never 'clean'
            res = None
        if not isinstance(res, dict):
            why = "OpenAlex could not be asked (rate limit or network) -- the rest is UNCHECKED, not clean"
            break
        looked += chunk
        for w in res.get("results") or []:
            if w.get("is_retracted"):
                flagged[(w.get("doi") or "").lower().replace("https://doi.org/", "")] = w.get("title") or ""
        if pause and i + batch < len(dois):
            time.sleep(pause)
    return flagged, looked, why


def http_url(u: object) -> str:
    """The URL if it is http(s), else '' -- an API answer never becomes a javascript: / file: link."""
    u = str(u or "").strip()
    return u if re.match(r"(?i)^https?://", u) else ""


def to_look_up(papers: List[dict], held: Iterable[str]) -> List[dict]:
    """Papers with no PDF held and no arXiv id (an arXiv paper's PDF is always at arxiv.org) that have a DOI."""
    h = set(held)
    return [p for p in papers if p["key"] not in h and not p.get("arxiv_id") and p.get("doi")]


def s2_links(papers: List[dict], post: Callable = _post, batch: int = 400,
             say=print) -> Tuple[Dict[str, dict], List[str], str]:
    """Semantic Scholar batch lookup by DOI -> ({key: {pdf, status, arxiv, url}} for the papers S2 found, the keys
    actually looked up, stop reason or ''). A failed batch stops the run; what was asked before it is kept."""
    out: Dict[str, dict] = {}
    looked: List[str] = []
    for i in range(0, len(papers), batch):
        chunk = papers[i:i + batch]
        try:
            res = post(S2_BATCH, {"ids": [f"DOI:{p['doi']}" for p in chunk]})
        except (urllib.error.URLError, OSError, ValueError) as e:
            return out, looked, f"Semantic Scholar stopped at {i}/{len(papers)}: {str(e)[:120]}"
        looked += [p["key"] for p in chunk]
        for p, r in zip(chunk, res if isinstance(res, list) else []):
            if not isinstance(r, dict):
                continue
            oa = r.get("openAccessPdf") or {}
            ext = r.get("externalIds") or {}
            out[p["key"]] = {"pdf": http_url(oa.get("url")), "status": oa.get("status") or "",
                             "arxiv": str(ext.get("ArXiv") or ""), "url": http_url(r.get("url"))}
        say(f"  semantic scholar: {min(i + batch, len(papers))}/{len(papers)} looked up, {len(out)} found")
        time.sleep(1.2)
    return out, looked, ""


def openalex_links(papers: List[dict], get: Callable = _get, batch: int = 40,
                   say=print) -> Tuple[Dict[str, dict], List[str], str]:
    """OpenAlex by DOI (OR filter, `batch` per request) -> ({key: {pdf, landing, status, more}}, the keys actually
    looked up, stop reason or ''). A refused request (the free daily budget) stops the run; what was asked is kept."""
    by_doi = {p["doi"]: p["key"] for p in papers}
    out: Dict[str, dict] = {}
    looked: List[str] = []
    dois = sorted(by_doi)
    for i in range(0, len(dois), batch):
        chunk = dois[i:i + batch]
        try:
            res = get(OPENALEX.format(dois="|".join(urllib.parse.quote(d, safe="/") for d in chunk)))
        except urllib.error.HTTPError as e:
            why = (" (the free daily budget, reset at midnight UTC; run `links --source openalex` again then)"
                   if e.code == 429 else "")
            return out, looked, f"OpenAlex stopped at {i}/{len(dois)}: HTTP {e.code}{why}"
        except (urllib.error.URLError, OSError, ValueError) as e:      # a ValueError's text can quote a header
            why = type(e).__name__ if isinstance(e, ValueError) else str(e)[:120]   # value -- the key: never shown
            return out, looked, f"OpenAlex stopped at {i}/{len(dois)}: {why}"
        looked += [by_doi[d] for d in chunk]
        for w in (res or {}).get("results") or []:
            d = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", str(w.get("doi") or "")).lower()
            if d not in by_doi:
                continue
            best = w.get("best_oa_location") or {}
            more = sorted({http_url(loc.get("pdf_url")) for loc in (w.get("locations") or [])
                           if http_url(loc.get("pdf_url")) and loc.get("pdf_url") != best.get("pdf_url")})[:4]
            out[by_doi[d]] = {"pdf": http_url(best.get("pdf_url")), "landing": http_url(best.get("landing_page_url")),
                              "status": (w.get("open_access") or {}).get("oa_status") or "", "more": more}
        say(f"  openalex: {min(i + batch, len(dois))}/{len(dois)} looked up, {len(out)} found")
        time.sleep(0.2)
    return out, looked, ""


def merge_links(old: dict, source: str, found: Dict[str, dict], looked_up: Iterable[str], date: str) -> dict:
    """Fold one source's answers into _links.json: a paper looked up again gets that source's new answer (or none)."""
    looked = sorted(set(looked_up))
    rows = {r["key"]: r for r in (old.get("papers") or [])}
    for k in looked:
        row = rows.setdefault(k, {"key": k})
        if k in found:
            row[source] = found[k]
        else:
            row.pop(source, None)
    srcs = dict(old.get("sources") or {})
    srcs[source] = {"checked": date, "looked_up": len(looked), "found": sum(1 for k in looked if k in found)}
    keep = [rows[k] for k in sorted(rows) if len(rows[k]) > 1]
    return {"schema": "svfr.bwm.links/1", "sources": srcs, "papers": keep}


# ------------------------------------------------------------------------------------------------------- hunt
def open_copy(p: dict, link: Optional[dict]) -> Tuple[str, str]:
    """(url, label) of the best free copy we know of, or ('', '')."""
    link = link or {}
    s2, oa = link.get("s2") or {}, link.get("openalex") or {}
    at_publisher = lambda u: re.match(r"(?i)^https?://(dx\.)?doi\.org/", u) is not None
    for url in (http_url(s2.get("pdf")), http_url(oa.get("pdf"))):
        if url:
            return url, ("free at publisher" if at_publisher(url) else "open PDF")
    u = http_url(p.get("oa_pdf_url"))
    if u:
        return u, ("free at publisher" if at_publisher(u) else "open copy")
    u = http_url(oa.get("landing"))
    if u and oa.get("status") not in ("", "closed"):
        # OpenAlex "green" = a repository copy; gold / hybrid / bronze / diamond = free on the publisher's site
        return u, ("open copy" if oa.get("status") == "green" else "free at publisher")
    return "", ""


def arxiv_version(p: dict, link: Optional[dict]) -> str:
    a = ((link or {}).get("s2") or {}).get("arxiv") or ""
    return a if a and not p.get("arxiv_id") else ""


def availability(p: dict, held: bool, link: Optional[dict], obtained: bool = False) -> str:
    """'held' | 'obtained' (not a PDF: a dataset kept by `obtain`) | 'open' (an arXiv paper, a free copy, or an
    arXiv version) | 'library'."""
    if held:
        return "held"
    if obtained:
        return "obtained"
    if p.get("arxiv_id") or open_copy(p, link)[0] or arxiv_version(p, link):
        return "open"
    return "library"


def hunt_links(p: dict, link: Optional[dict] = None) -> List[Tuple[str, str]]:
    """[(label, url)] for finding a paper's full text, best first. Mirrored by `huntLinks` in the atlas page."""
    out: List[Tuple[str, str]] = []
    if p.get("arxiv_id"):
        out.append(("arXiv PDF", f"https://arxiv.org/pdf/{p['arxiv_id']}"))
    url, label = open_copy(p, link)
    if url:
        out.append((label, url))
    av = arxiv_version(p, link)
    if av:
        out.append(("arXiv version", f"https://arxiv.org/abs/{av}"))
    for extra in ((link or {}).get("openalex") or {}).get("more") or []:
        if http_url(extra) and extra != url:
            out.append(("other copy", extra))
    if p.get("doi"):
        out.append(("publisher", f"https://doi.org/{p['doi']}"))
    elif not p.get("arxiv_id") and http_url(p.get("url")):
        out.append(("landing page", http_url(p.get("url"))))       # no DOI, no arXiv: keep the one link we have
    t = (p.get("title") or "").strip()
    if t:
        out.append(("Scholar", "https://scholar.google.com/scholar?q=" + urllib.parse.quote(f'"{t}"', safe="")))
        out.append(("PDF search", "https://www.google.com/search?q=" + urllib.parse.quote(f'"{t}" filetype:pdf', safe="")))
    s2u = http_url(((link or {}).get("s2") or {}).get("url"))
    if s2u:
        out.append(("Semantic Scholar", s2u))
    seen, uniq = set(), []
    for lab, u in out:
        if u not in seen:
            seen.add(u)
            uniq.append((lab, u))
    return uniq
