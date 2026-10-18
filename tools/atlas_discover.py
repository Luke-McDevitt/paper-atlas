r"""atlas_discover -- from a TASK to its papers: the ones this atlas already holds, the ones out there (OpenAlex,
Semantic Scholar, arXiv), which of those are open (fetched by `--add`) and which are not (a list with the links to get
them) -- then, on request, the new ones brought into the atlas as a SECTION of their own.

    python tools/atlas.py discover "a few sentences on the task" [--title "Short name"] [--slug s]
    python tools/atlas.py discover <slug> --add [--top 25] [--section]     # bring the top ones (or add_list.txt) in
    python tools/atlas_discover.py bench --project <atlas> [--split dev] [--replay | --tune | --rescore]
                                                                  # the measured comparison (LIT_PAPER_DISCOVERY.md)

WHY (owner, 2026-10-16): "build out an upgrade to the atlas that is a tool that takes a task and searches databases,
google scholar, openalex, etc for applicable papers based on generating sets of keywords from the task / problem
(investigate good ways to do this) and has the ability to auto-scrape the open ones and generate a list of closed ones
with links to download them ... (if they are not already found in the user's cache / index ...). after that it has
the ability to update the atlas with these new papers / terms based on this new problem (and the ability to add this
new problem as a new section)."

HOW -- three channels, because no single one finds enough (the research and the measurement:
report/boat_world_model/LIT_PAPER_DISCOVERY.md):
  1. SEMANTIC -- the task's own words to OpenAlex `search.semantic` (GTE-Large over title + abstract, <= 50 hits) and
     to Semantic Scholar's relevance search (<= 100). LitSearch (EMNLP 2024): dense retrieval beat BM25 by ~25 recall
     points on literature-search questions, and Google-style engines trailed both.
  2. KEYWORDS -- boolean queries built from the task: its key phrases (scored by where they sit in the task, by the
     lexicon, and by how many SEED papers -- the best local and semantic hits -- use them), the lexicon's aliases, and
     the seed papers' own vocabulary for each phrase: words over-represented in the seeds against this atlas's papers
     (the "objective approach" of Hausner et al. 2015: search terms from a seed set's term frequencies, non-inferior
     to expert-built strategies). Pairs of facets are ANDed, each facet an OR of its words; OpenAlex and arXiv rank by
     relevance, Semantic Scholar's bulk search most-cited first; every hit is re-checked on its title + abstract
     (tools/lit_sweep.py's query builders and check).
  3. CITATIONS -- backward and forward from the seeds on OpenAlex (Wohlin et al. 2022: database search plus
     snowballing found ~30% more primary studies than the search alone); a paper linked to several seeds ranks higher.
  Every hit is merged across sources (arXiv id / DOI / title), checked against what you HAVE (this atlas; the other
  projects or lists in atlas.toml [discover] also), and ranked by weighted reciprocal-rank fusion (paper_find's) of:
  dense similarity to the task (the project's embedding model, when a GPU / torch Python is set up), BM25 of the task
  over title + abstract, the semantic engines' order, keyword-query hits, citation links to the seeds, and citation
  count. Google Scholar has NO API and its terms forbid automated queries: it is never asked -- each closed paper gets
  a Scholar link for a person to click, next to its DOI, the publisher's page and the library proxy atlas.toml names.

OUTPUTS -- <project>/discover/<slug>/ (small text, fine to commit): DISCOVER.md (what you have, the new open and the
new closed papers, each with WHY it is listed, and the best starting points), GET_THESE.md (the closed ones with their
links), add_list.txt (the top new papers in `atlas add` list form: edit it), plan.json (key phrases, seed words,
facets, queries: edit it and pass --plan) and run.json (every call asked, hits, failures, OpenAlex cost). The full
candidate list and the raw API answers stay on this machine (<pdfs>_atlas/discover/), so a re-run costs nothing.

HONEST SCOPE: the measured recall (bench) is against our reviews' reference lists -- an upper bound on recall against
what exists, and the reviews' own searches used similar engines. The facets are built by rules, not by a reader:
plan.json is there to be corrected. OpenAlex without a key allows ~$0.10 of calls a day (~100 searches; a free key
gives 10x: `setx OPENALEX_API_KEY <key>`, tools/openalex_key.py) and Semantic Scholar's keyless pool returns 429 often:
a source that could not be asked is reported as COULD_NOT_MEASURE in run.json and DISCOVER.md, never as 'nothing found'.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_project as AP  # noqa: E402
import lit_sweep as LS  # noqa: E402
import paper_find as PF  # noqa: E402

VR = LS.VR
HELD, INDEXED, ELSEWHERE, NEW = "HELD", "INDEXED", "ELSEWHERE", "NEW"
CHANNELS = ("semantic", "keywords", "citations")
SOURCES = ("openalex", "s2", "arxiv")
OA_WORKS = "https://api.openalex.org/works"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper"
ARXIV_API = "https://export.arxiv.org/api/query"
OA_SELECT = LS.OPENALEX_SELECT + ",best_oa_location"
OA_SELECT_REFS = OA_SELECT + ",referenced_works"
S2_FIELDS = LS.S2_FIELDS + ",openAccessPdf"
#: reciprocal-rank-fusion weights per signal (tools/paper_find.py's fusion). MEASURED 2026-10-18 (LIT_PAPER_DISCOVERY.md
#: sec. 4): `bench --tune` chose these on DEV (11 reviews); TEST (11 other reviews, scored once with them) gave R@100
#: 0.021 vs 0.020 for the provisional judgments they replace (dense 1, bm25 1, semantic 1, keywords 0.5, citations
#: 0.5, cited 0.25): +0.0008, 95% CI +0.0000..+0.0023, better on 1 review and worse on none -- a small gain, since at
#: this recall what the channels find limits the list more than its order. `local` cannot be measured there (the
#: benchmark's answers are kept out of the atlas's own search): its weight stays a judgment.
WEIGHTS: Dict[str, float] = {"dense": 2.0, "bm25": 1.0, "semantic": 0.5, "keywords": 0.5, "citations": 1.0,
                             "cited": 0.0, "local": 1.0}
SIGNALS = tuple(WEIGHTS)
#: the atlas page's own section palette (build_bwm.py BROWSER_TEMPLATE `SPAL`: a section with no colour takes the one
#: of its position); a new section takes the first one no section shows -- test_atlas_discover keeps the two equal
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#9b2a64", "#9a6a12", "#5b5bd6", "#0f766e", "#c2410c", "#6b7280",
           "#b5179e", "#3a86ff", "#588157")

#: words that name no topic on their own: they end a key phrase like a stop word does
GENERIC = frozenset("""
able about across actually additionally aim aims allow allows already also although always among another any anything
appropriate around automatic automatically available avoid basic become becomes best better build building built can
cannot certain challenge challenges choose common compare compared comparing complete current currently deal deals
different difficult do does doing done easily easy effective effectively efficient efficiently enable enables
especially etc even every existing explore exploring extend fast find finding first focus following found four from
future general generally get gets give given go goal goals good great hard help helps high higher how however idea
ideas identify important improve improved improves improving include includes including increase instead interested
interesting issue issues just kind kinds know large larger least less like likely look looking lot lots low lower
main make makes making manage many may maybe might more most much must need needed needs new next novel now often one
only order other others our out over own part particular particularly perform possible potential potentially present
presents previous previously propose proposed provide provides quickly rather real really recent recently related
relevant require required requires research result results right same second see seem several should show shows
similar simple since small so some something specific starting start still strong such suggest suitable support take
takes than their them then there these they thing things think third those three through thus together toward
towards try trying two typical typically under understand up upon use used useful uses using usually various very via
want wants way ways well what when where whether which while who whose why will within without work working would yet
paper papers study studies approach approaches problem problems task tasks point points no not nor non each per vs
versus eg ie
""".split())
#: research verbs and the nouns their problems go by: 'estimate the pose' -> 'pose estimation', 'extract figures' ->
#: 'figure extraction' (a seed paper's wording, which the support score then confirms or drops)
NOMINAL = {
    "estimate": "estimation", "extract": "extraction", "detect": "detection", "track": "tracking",
    "classify": "classification", "segment": "segmentation", "localize": "localization", "localise": "localisation",
    "map": "mapping", "predict": "prediction", "schedule": "scheduling", "generate": "generation",
    "reconstruct": "reconstruction", "register": "registration", "navigate": "navigation", "plan": "planning",
    "recognize": "recognition", "simulate": "simulation", "calibrate": "calibration", "fuse": "fusion",
    "compress": "compression", "label": "labeling", "annotate": "annotation", "retrieve": "retrieval",
    "rank": "ranking", "cluster": "clustering", "filter": "filtering", "denoise": "denoising", "enhance": "enhancement",
    "train": "training", "evaluate": "evaluation", "measure": "measurement", "optimize": "optimization",
    "optimise": "optimisation", "deploy": "deployment", "monitor": "monitoring", "inspect": "inspection",
    "explore": "exploration", "avoid": "avoidance", "dock": "docking", "grasp": "grasping", "manipulate": "manipulation",
    "mine": "mining", "parse": "parsing", "index": "indexing", "link": "linking", "match": "matching",
    "align": "alignment", "stitch": "stitching", "mosaic": "mosaicking", "summarize": "summarization",
    "translate": "translation", "transcribe": "transcription", "identify": "identification",
    "quantify": "quantification", "characterize": "characterization", "model": "modeling", "control": "control",
    "count": "counting", "infer": "inference", "forecast": "forecasting", "partition": "partitioning",
    "balance": "balancing", "verify": "verification", "validate": "validation", "test": "testing",
    "benchmark": "benchmarking", "visualize": "visualization", "render": "rendering", "synthesize": "synthesis",
    "augment": "augmentation", "adapt": "adaptation", "distribute": "distribution", "parallelize": "parallelization",
    "accelerate": "acceleration", "cache": "caching", "survey": "survey", "search": "search", "follow": "following",
}
#: head nouns of compounds ('world model', 'search method', 'data association'): kept inside a phrase, never alone
WEAK = frozenset("""algorithm algorithms analysis application applications architecture architectures data dataset
datasets design designs evaluation framework frameworks method methods model models process processes scheme schemes
setting settings solution solutions strategy strategies system systems technique techniques technology technologies
tool tools learning""".split())
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_BREAK = re.compile(r"[.,;:!?()\[\]{}\"|/]")
_SURVEY = re.compile(r"\b(survey|review|overview|tutorial|primer|state[- ]of[- ]the[- ]art|systematic (literature )?"
                     r"review|meta[- ]analysis)\b", re.I)
_CODE = re.compile(r"github\.com|gitlab\.com|code (is|are|will be) (made )?(publicly |freely )?available|open[- ]sourced?\b",
                   re.I)
_DATASET = re.compile(r"\b(we (release|introduce|present|collect)|publicly available|new) [^.]{0,40}\b(dataset|"
                      r"benchmark)s?\b", re.I)


def _ascii(s) -> str:
    """Console text (the repo's rule: ASCII only)."""
    return str(s if s is not None else "").encode("ascii", "replace").decode("ascii")


def _stems(word: str) -> List[str]:
    """A word's S-stems, a hyphenated word split ('side-scan' -> side, scan), a possessive dropped. Pure."""
    return [PF.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", str(word).lower()))]


def _is_break_word(w: str) -> bool:
    lw = w.lower()
    return lw in PF.STOP or lw in GENERIC or lw.isdigit() or len(re.sub(r"[^a-z0-9]", "", lw)) < 2


def slug_of(s: str, default: str = "task") -> str:
    """A topic / section / task id from a title: lower-case ASCII letters, digits and '_', at most 40 characters,
    cut at a word boundary. Pure."""
    words = [w for w in re.findall(r"[a-z0-9]+", str(s or "").lower()) if w not in PF.STOP and w not in GENERIC]
    out = ""
    for w in words:
        if len(out) + len(w) + (1 if out else 0) > 40:
            break
        out = f"{out}_{w}" if out else w
    return out or default


# --------------------------------------------------------------------------------------------- key phrases
def runs(text: str) -> List[List[Tuple[str, int]]]:
    """The maximal runs of content words in `text` -- (surface form, token position) -- broken by stop words, GENERIC
    words and punctuation. Pure."""
    out: List[List[Tuple[str, int]]] = []
    cur: List[Tuple[str, int]] = []
    prev = 0
    for i, m in enumerate(_TOKEN.finditer(text or "")):
        if cur and _BREAK.search(text[prev:m.start()]):
            out.append(cur)
            cur = []
        w = m.group(0)
        if _is_break_word(w):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append((w, i))
        prev = m.end()
    if cur:
        out.append(cur)
    return out


def _gram_key(words: Sequence[str]) -> Tuple[str, ...]:
    return tuple(s for w in words for s in _stems(w))


def verb_base(word: str) -> str:
    """A research verb's base form when `word` is one ('estimating' / 'estimates' / 'estimated' -> 'estimate',
    'mapping' -> 'map'), else ''. Pure."""
    w = word.lower()
    if w in NOMINAL:
        return w
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            b = w[: -len(suf)]
            for cand in (b, b + "e", b[:-1] if len(b) > 2 and b[-1] == b[-2] else ""):
                if cand in NOMINAL:
                    return cand
    return ""


def singular(word: str) -> str:
    """'PDFs' -> 'PDF', 'figures' -> 'figure', 'AUVs' -> 'AUV'; other words unchanged in case. Pure."""
    if len(word) > 2 and word.endswith("s") and word[:-1].isupper():
        return word[:-1]
    return PF.stem(word.lower()) if word.islower() else word


def candidate_phrases(text: str, max_len: int = 4) -> List[dict]:
    """Every contiguous 1..max_len-word piece of every content run -- {phrase, key (stems), pos (first token), n,
    span (token positions), tf} -- plus, for a research verb, its problem noun after the verb's object ('estimate the
    pose' -> 'pose estimation'). A piece made only of WEAK words ('model', 'data') or a bare research verb is not a
    candidate. Pure."""
    seen: Dict[Tuple[str, ...], dict] = {}

    def add(words: List[str], span: Sequence[int], form: str) -> None:
        key = _gram_key(words)
        if not key:
            return
        if key in seen:
            seen[key]["tf"] += 1
            return
        seen[key] = {"phrase": " ".join(words), "key": key, "pos": min(span), "n": len(words), "span": sorted(span),
                     "tf": 1, "form": form}

    rs = runs(text)
    for run in rs:
        for n in range(1, min(max_len, len(run)) + 1):
            for i in range(len(run) - n + 1):
                words = [w for w, _ in run[i:i + n]]
                if all(w.lower() in WEAK for w in words) or (n == 1 and verb_base(words[0])):
                    continue
                # an English noun phrase ends in its head: 'lab desktops' names a thing, 'idle lab' does not
                form = "verbal" if verb_base(words[0]) else ("head" if i + n == len(run) else "part")
                add(words, [p for _, p in run[i:i + n]], form)
    toks = [m.group(0).lower() for m in _TOKEN.finditer(text or "")]
    for i, w in enumerate(toks):
        base = verb_base(w)
        if not base:
            continue
        own = next((r for r in rs if any(p == i for _, p in r)), None)
        objs = [[(x, p) for x, p in (own or []) if p > i]]           # 'extract figures': the rest of its run
        end = objs[0][-1][1] if objs[0] else i
        nxt = next((r for r in rs if end < r[0][1] <= end + 4), None)
        if nxt is not None and (not objs[0] or set(toks[end + 1:nxt[0][1]]) <= {"and", "or"}):
            objs.append(nxt)                          # 'estimate the pose', or '... figures and tables'
        for obj in objs:
            obj = [(x, p) for x, p in obj if not verb_base(x)]
            if not obj:
                continue
            for head in [obj[-1:]] + ([obj[-2:]] if len(obj) >= 2 else []):
                add([singular(x) for x, _ in head] + [NOMINAL[base]], [i] + [p for _, p in head], "nominal")
    return list(seen.values())


#: a phrase's shape, as a prior when no seed says which wording papers use: the noun form of a verb phrase ('figure
#: extraction') and a piece ending in its head noun beat the verb phrase ('extract figures') and a fragment
FORM_PRIOR = {"nominal": 1.15, "head": 1.1, "part": 0.9, "verbal": 0.85, "lexicon": 1.0}
#: words that make what follows a constraint, not a topic: 'without a DVL' asks for papers that do not need one
NEGATION = frozenset({"without", "no", "not", "except", "excluding", "lacking", "absent"})
NEGATED = 0.6                                         # a negated phrase's weight: it can still be a facet, last


def negated_positions(text: str) -> set:
    """The token positions of the content runs a negation word governs ('without a DVL' -> DVL's). Pure."""
    toks = [m.group(0).lower() for m in _TOKEN.finditer(text or "")]
    out = set()
    for run in runs(text):
        j = run[0][1] - 1
        while j >= 0 and toks[j] in {"a", "an", "the", "any"}:
            j -= 1
        if j >= 0 and toks[j] in NEGATION:
            out |= {p for _, p in run}
    return out


def _contains(doc_keys: str, key: Sequence[str]) -> bool:
    return f" {' '.join(key)} " in doc_keys


def doc_key_text(text: str) -> str:
    """A document as ' stem stem stem ' for phrase containment (hyphens split, S-stemmed). Pure."""
    return " " + " ".join(s for w in _TOKEN.findall(text or "") for s in _stems(w)) + " "


def key_phrases(text: str, seeds: Sequence[str] = (), terms: Optional["PF.Terms"] = None,
                lexicon: Optional[Dict[str, dict]] = None, k: int = 8, background: Sequence[str] = ()) -> List[dict]:
    """The task's key phrases, best first: [{phrase, key, score, support, bg_share, lexicon, aliases}]. A candidate
    scores by where it sits (earlier is better), its length (2-3 words best), an acronym in it, a lexicon term it IS
    (whose aliases come along), how DISTINCTIVE it is among the atlas's own papers (`background`: 'underwater' in a
    marine atlas restricts little), and -- the evidence -- the share of SEED documents that use it (a phrase no paper
    uses, like 'looking sonar odometry', falls away). Chosen greedily: no two share a task word, and none is a part of
    another chosen one. Pure."""
    cands = candidate_phrases(text)
    seed_keys = [doc_key_text(s) for s in seeds if s]
    bg_keys = [doc_key_text(s) for s in background if s] if len(background) >= 200 else []
    ref = math.log((len(bg_keys) + 1) / (0.01 * len(bg_keys) + 1)) if bg_keys else 1.0
    starts = [m.start() for m in _TOKEN.finditer(text or "")]
    lex_spans: Dict[Tuple[str, ...], dict] = {}
    for m in (terms.find(text) if terms is not None else []):
        t = (lexicon or {}).get(m["id"]) or {}
        words = _TOKEN.findall(m["matched"])
        key = _gram_key(words)
        if key:
            lex_spans.setdefault(key, {"id": m["id"], "term": m["term"], "matched": m["matched"],
                                       "pos": sum(s < m["start"] for s in starts), "n": len(words),
                                       "aliases": [a for a in (t.get("aliases") or []) if isinstance(a, str)]})
    for key, lx in lex_spans.items():
        if not any(c["key"] == key for c in cands):
            cands.append({"phrase": lx["matched"], "key": key, "pos": lx["pos"], "n": lx["n"], "tf": 1,
                          "span": list(range(lx["pos"], lx["pos"] + lx["n"])), "form": "lexicon"})
    negpos = negated_positions(text)
    out = []
    for c in cands:
        support = (sum(_contains(d, c["key"]) for d in seed_keys) / len(seed_keys)) if seed_keys else 0.0
        share = (sum(_contains(d, c["key"]) for d in bg_keys) / len(bg_keys)) if bg_keys else 0.0
        lx = lex_spans.get(c["key"])
        length = {1: 1.0, 2: 1.4, 3: 1.3}.get(c["n"], 1.0)
        acro = 1.3 if re.search(r"\b[A-Z][A-Z0-9]{1,6}s?\b", c["phrase"]) else 1.0
        pos = 1.0 / (1.0 + c["pos"] / 25.0)
        score = (1.0 + math.log(c["tf"])) * length * acro * (0.5 + 0.5 * pos) * (1.6 if lx else 1.0) \
            * FORM_PRIOR.get("lexicon" if lx else c.get("form", "part"), 1.0)
        if bg_keys:
            idf = math.log((len(bg_keys) + 1) / (share * len(bg_keys) + 1))
            score *= 0.5 + 0.5 * min(1.0, idf / ref)
        if seed_keys:
            score *= 0.35 + support if not lx else max(0.35 + support, 1.0)
        neg = bool(set(c["span"]) & negpos)
        if neg:
            score *= NEGATED
        out.append(dict(c, score=round(score, 4), support=round(support, 3), bg_share=round(share, 4),
                        lexicon=(lx or {}).get("id", ""), aliases=(lx or {}).get("aliases", []), negated=neg))
    # the task's own two-word phrase beats its pieces once ANY seed uses it: the seeds of a side-scan atlas all say
    # 'sonar', and 'forward-looking sonar' -- what the task asks for -- had lost to it (the kit demo, 2026-10-16).
    # Two words only: a promoted 'forward-looking sonar odometry' would be one over-specific quoted phrase where two
    # facets ANDed find more.
    for c in out:
        if c["n"] == 2 and c["support"] >= 0.1 and not c["negated"]:
            subs = [o["score"] for o in out if set(o["key"]) < set(c["key"]) and set(o["span"]) <= set(c["span"])]
            if subs and max(subs) >= c["score"]:
                c["score"] = round(max(subs) * 1.01, 4)
    out.sort(key=lambda c: (-c["score"], c["pos"], c["phrase"]))
    chosen: List[dict] = []
    used: set = set()
    words: set = set()                                # the topic words already taken (shared head nouns stay free)
    weak = {PF.stem(w) for w in WEAK}
    for c in out:
        span = set(c["span"])
        if span & used and not c["lexicon"]:
            continue
        if any(set(c["key"]) <= set(o["key"]) or set(o["key"]) <= set(c["key"]) for o in chosen):
            continue                                  # 'pose' and 'pose estimation': the better-scored one only
        if (set(c["key"]) - weak) & words:            # 'pose data' beside 'pose estimation': one concept, once
            continue
        chosen.append(c)
        used |= span
        words |= set(c["key"]) - weak
        if len(chosen) >= k:
            break
    return [{k2: v for k2, v in c.items() if k2 not in ("tf", "span")} | {"key": list(c["key"])} for c in chosen]


# ----------------------------------------------------------------------- seed vocabulary (the objective approach)
def doc_grams(text: str, max_n: int = 3) -> Dict[Tuple[str, ...], str]:
    """{stems: surface} for every 1..max_n-word piece of every content run (no piece of WEAK words only). Pure."""
    out: Dict[Tuple[str, ...], str] = {}
    for run in runs(text):
        for n in range(1, min(max_n, len(run)) + 1):
            for i in range(len(run) - n + 1):
                words = [w for w, _ in run[i:i + n]]
                if all(w.lower() in WEAK for w in words):
                    continue
                key = _gram_key(words)
                if key:
                    out.setdefault(key, " ".join(words))
    return out


def seed_terms(seeds: Sequence[str], background: Sequence[str], k: int = 15, min_df: int = 2) -> List[dict]:
    """Words the SEED documents share and this atlas's papers mostly do not: score = (share of seeds using it) x
    log((N + 1) / (background df + 1)) x a small length bonus -- the objective approach's 'frequent in the test set,
    rare in the population set'. A piece contained in a longer chosen piece with the same seed count is dropped.
    [{term, key, seed_df, bg_df, score}], best first. Pure."""
    seed_sets = [doc_grams(s) for s in seeds if s]
    if not seed_sets:
        return []
    df: Counter = Counter()
    surface: Dict[Tuple[str, ...], Counter] = {}
    for g in seed_sets:
        for key, surf in g.items():
            df[key] += 1
            surface.setdefault(key, Counter())[surf] += 1
    cand = {key for key, n in df.items() if n >= min_df}
    bg: Counter = Counter()
    for text in background:
        for key in doc_grams(text):
            if key in cand:
                bg[key] += 1
    n_bg, n_seed = len(background), len(seed_sets)
    # a piece every seed using it uses only inside a longer piece ('side scan' in 'side scan sonar') adds nothing
    dominated = {key[i:i + n] for key in cand for n in range(1, len(key)) for i in range(len(key) - n + 1)
                 if key[i:i + n] in cand and df[key[i:i + n]] <= df[key]}
    scored = []
    for key in cand - dominated:
        idf = math.log((n_bg + 1) / (bg[key] + 1)) if n_bg else 1.0
        score = (df[key] / n_seed) * max(idf, 0.0) * {1: 1.0, 2: 1.3, 3: 1.2}.get(len(key), 1.0)
        if score > 0:
            scored.append({"term": surface[key].most_common(1)[0][0], "key": list(key), "seed_df": df[key],
                           "bg_df": bg[key], "score": round(score, 4)})
    scored.sort(key=lambda t: (-t["score"], -len(t["key"]), t["term"]))
    out: List[dict] = []
    for t in scored:
        if any(set(t["key"]) < set(o["key"]) and t["seed_df"] <= o["seed_df"] for o in out):
            continue
        out.append(t)
        if len(out) >= k:
            break
    return out


# ------------------------------------------------------------------------------------------- facets and queries
def _usable(term: str) -> bool:
    return len(re.sub(r"[^A-Za-z0-9]", "", term or "")) >= 3 and not all(w.lower() in WEAK | GENERIC
                                                                          for w in _TOKEN.findall(term))


def build_facets(phrases: Sequence[dict], prf: Sequence[dict], max_facets: int = 3, max_terms: int = 6) -> List[dict]:
    """Facets -- the task's concepts, each an OR of its words: a key phrase, its lexicon aliases, and the seed words
    that share a (non-weak) word with it ('figure extraction' for 'extract figures'). Phrases used by no seed at all
    are kept only when no better facet exists. [{name, terms, words}]. Pure."""
    facets: List[dict] = []
    weak = {PF.stem(w) for w in WEAK}
    evidence = any(p.get("support", 0) > 0 for p in phrases)     # no seeds at all: every phrase is a guess alike
    positive = sum(not p.get("negated") for p in phrases)
    for p in phrases:
        if len(facets) >= max_facets:
            break
        if p.get("negated") and positive >= 2:        # 'without a DVL': a constraint for the ranking, not a facet
            continue
        if facets and evidence and p.get("support", 0) == 0 and not p.get("lexicon") and len(phrases) > max_facets:
            continue
        terms = [p["phrase"]] + [a for a in p.get("aliases", []) if _usable(a)]
        facets.append({"name": p["phrase"], "terms": _dedupe_terms(terms)[:max_terms], "key": list(p["key"]),
                       "words": sorted(set(p["key"]) - weak)})
    for t in prf:
        tk = set(t["key"])
        best, best_n = None, 0
        for f in facets:
            if tk <= set(f["key"]):                   # the facet's own words, or fewer: broader, adds nothing
                continue
            n = len((tk - weak) & set(f["words"]))
            if n >= max(1, math.ceil(len(f["words"]) / 2)) and n > best_n:    # 'sonar data' is not a variant of
                best, best_n = f, n                                             # 'forward-looking sonar'
        if best is not None and best_n and len(best["terms"]) < max_terms and _usable(t["term"]):
            best["terms"] = _dedupe_terms(best["terms"] + [t["term"]])
    return facets


def _dedupe_terms(terms: Sequence[str]) -> List[str]:
    """Terms with the same stems ('Forward-Looking Sonar', 'forward-looking sonar'): the first spelling only. Pure."""
    seen, out = set(), []
    for t in terms:
        k = _gram_key(_TOKEN.findall(t))
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def build_queries(facets: Sequence[dict], prf: Sequence[dict], context_terms: int = 6) -> List[dict]:
    """Boolean queries for lit_sweep's builders: every PAIR of facets ANDed (recall), and the first two with the seed
    words no facet took as a third group (precision). One facet: that facet alone. Pure."""
    fs = [f for f in facets if f["terms"]]
    qs: List[dict] = []
    if len(fs) == 1:
        qs.append({"id": "k1", "groups": [fs[0]["terms"]]})
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            qs.append({"id": f"k{len(qs) + 1}", "groups": [fs[i]["terms"], fs[j]["terms"]]})
    taken = {_gram_key(_TOKEN.findall(t)) for f in fs for t in f["terms"]}
    extra = [t["term"] for t in prf if tuple(t["key"]) not in taken and _usable(t["term"])][:context_terms]
    if len(fs) >= 2 and extra:
        qs.append({"id": f"k{len(qs) + 1}", "groups": [fs[0]["terms"], fs[1]["terms"], extra]})
    return qs


# ---------------------------------------------------------------------------------------------- the network
def _wid(x) -> str:
    """'https://openalex.org/W123' -> 'W123'. Pure."""
    return str(x or "").rstrip("/").rsplit("/", 1)[-1]


def _s2_headers() -> dict:
    key = os.environ.get("S2_API_KEY", "")
    return dict(VR.HEADERS, **({"x-api-key": key} if key else {}))


class Net:
    """Every call discover makes: through tools/lit_sweep.py's on-disk cache (a re-run asks nothing again), paced per
    host, and logged. A call that fails is COULD_NOT_MEASURE in the log -- its channel's rows are missing, which the
    report says -- never 'no hits'. `cache_only` answers from the cache and asks nothing: a saved run re-derived
    exactly, its calls that were refused then (never cached) refused again. `get(url, headers=None)` is the test
    seam."""

    def __init__(self, cache: Path, refresh: bool = False, offline: bool = False, get: Optional[Callable] = None,
                 cache_only: bool = False):
        self.cache, self.refresh, self.offline, self.cache_only = Path(cache), refresh, offline, cache_only
        self._get = get or self._cached_get
        self.log: List[dict] = []
        self.cost = 0.0
        self._last: Dict[str, float] = {}
        self.blocked: Dict[str, str] = {}             # host -> the 429 that closed it for this run

    def _path(self, url: str) -> Path:
        return self.cache / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".txt")   # lit_sweep.cached_get's layout

    def _cached(self, url: str) -> bool:
        return not self.refresh and self._path(url).exists()

    def _cached_get(self, url: str, headers: Optional[dict] = None) -> Optional[str]:
        """lit_sweep.cached_get with ONE retry (3 s): a 429 here is mostly a spent daily budget (OpenAlex keyless:
        $0.10), which four long retries per call only made a 5-minute wait (measured 2026-10-16)."""
        p = self._path(url)
        if p.exists() and not self.refresh:
            return p.read_text(encoding="utf-8")
        txt = VR._get(url, retries=2, pause=3.0, headers=headers or VR.HEADERS)
        if txt is not None:
            self.cache.mkdir(parents=True, exist_ok=True)
            p.write_text(txt, encoding="utf-8", newline="\n")
        return txt

    def ask(self, what: str, url: str, headers: Optional[dict] = None, pace: float = 0.0) -> Optional[str]:
        entry = {"what": what, "url": url, "fresh": False, "error": ""}
        self.log.append(entry)
        if self.offline:
            entry["error"] = "offline"
            return None
        fresh = not self._cached(url)
        host = urllib.parse.urlparse(url).netloc
        if fresh and self.cache_only:
            entry["error"] = f"{LS.UNMEASURED}: not in the cache (a cache-only replay asks nothing)"
            return None
        if fresh and host in self.blocked:
            entry["error"] = f"{LS.UNMEASURED}: not asked -- {host} refused an earlier call ({self.blocked[host]})"
            return None
        if fresh and pace:
            wait = self._last.get(host, 0.0) + pace - time.time()
            if wait > 0:
                time.sleep(wait)
        try:
            txt = self._get(url, headers=headers)
        except (VR.FetchError, OSError, ValueError) as e:
            entry["error"] = f"{LS.UNMEASURED}: {str(e)[:160]}"
            if "429" in str(e) and host == "api.openalex.org":    # OpenAlex 429 = the day's budget is spent;
                self.blocked[host] = "HTTP 429: the day's budget"   # S2's shared pool 429s one call, not the next
            txt = None
        if fresh:
            self._last[host] = time.time()
        entry["fresh"] = fresh
        if txt and fresh and host == "api.openalex.org":
            try:
                self.cost += float((json.loads(txt).get("meta") or {}).get("cost_usd") or 0.0)
            except (ValueError, AttributeError):
                pass
        return txt

    def failed(self, prefix: str = "", since: int = 0) -> List[str]:
        """The calls that were meant to be asked and were not (offline runs ask nothing on purpose) -- from log entry
        `since` on: one Net serves every task of a benchmark, and a task owns only the calls it made."""
        return [e["what"] for e in self.log[since:] if e["error"] and e["error"] != "offline"
                and e["what"].startswith(prefix)]


def oa_records(txt: Optional[str]) -> List[dict]:
    """An OpenAlex /works answer -> records (tools/verify_refs.py's normalisation + date, arXiv id, best open PDF,
    landing page, type and -- when selected -- reference ids). Pure."""
    try:
        d = json.loads(txt or "{}")
    except ValueError:
        return []
    out = []
    for w in d.get("results") or []:
        r = VR.openalex_norm(w)
        best = w.get("best_oa_location") or {}
        prim = w.get("primary_location") or {}
        r.update({"date": w.get("publication_date") or "", "arxiv_id": LS.doi_to_arxiv(r.get("doi") or ""),
                  "pdf_url": best.get("pdf_url") or "",
                  "landing": prim.get("landing_page_url") or best.get("landing_page_url") or "",
                  "type": w.get("type") or "", "openalex_id": _wid(w.get("id")),
                  "refs": [_wid(x) for x in w.get("referenced_works") or []]})
        out.append(r)
    return out


def s2_records(txt: Optional[str]) -> List[dict]:
    """A Semantic Scholar search answer -> records (lit_sweep's parser + the open PDF and the S2 page). Pure."""
    try:
        raw = json.loads(txt or "{}").get("data") or []
    except ValueError:
        return []
    recs, _, _ = LS.parse_s2(json.dumps({"data": raw}))
    for r, p in zip(recs, raw):
        r["pdf_url"] = (p.get("openAccessPdf") or {}).get("url") or ""
        r["s2_url"] = f"https://www.semanticscholar.org/paper/{p['paperId']}" if p.get("paperId") else ""
    return recs


def arxiv_records(txt: Optional[str]) -> List[dict]:
    try:
        recs, _ = LS.parse_arxiv_feed(txt or "<feed xmlns='http://www.w3.org/2005/Atom'/>")
    except Exception:                                 # noqa: BLE001 -- a malformed feed is no hits, logged by Net
        return []
    for r in recs:
        r["pdf_url"] = f"https://arxiv.org/pdf/{r['arxiv_id']}" if r.get("arxiv_id") else ""
    return recs


def _hit(rec: dict, src: str, hit: list) -> dict:
    return dict(rec, sources=[src], hits=[hit])


def semantic_channel(net: Net, text: str, s2_query: str, n_oa: int = 50, n_s2: int = 100) -> List[dict]:
    """The task's own words to OpenAlex semantic search and Semantic Scholar's relevance search, in their order."""
    out = []
    url = (f"{OA_WORKS}?search.semantic={urllib.parse.quote(text[:2000])}&per-page={min(n_oa, 50)}"
           f"&select={OA_SELECT_REFS}")
    for i, r in enumerate(oa_records(net.ask("semantic/openalex", url, pace=1.1)), 1):
        out.append(_hit(r, "openalex", ["semantic", "openalex", i]))
    if s2_query:
        url = (f"{S2_PAPER}/search?query={urllib.parse.quote(s2_query)}&limit={min(n_s2, 100)}"
               f"&fields={S2_FIELDS}")
        for i, r in enumerate(s2_records(net.ask("semantic/s2", url, headers=_s2_headers(), pace=1.5)), 1):
            out.append(_hit(r, "s2", ["semantic", "s2", i]))
    return out


def keyword_channel(net: Net, queries: Sequence[dict], sources: Sequence[str] = SOURCES, cap: int = 100) -> List[dict]:
    """Each boolean query on each source -- OpenAlex and arXiv by relevance, Semantic Scholar's bulk search most-cited
    first -- and every hit re-checked on its own title + abstract (lit_sweep.match_query)."""
    out = []
    for q in queries:
        g = q["groups"]
        for src in sources:
            if src == "openalex":
                url = (f"{OA_WORKS}?search={urllib.parse.quote(LS.openalex_query(g))}&per-page={min(cap, 200)}"
                       f"&select={OA_SELECT}")
                recs = oa_records(net.ask(f"keywords/{q['id']}/openalex", url, pace=0.2))
            elif src == "arxiv":
                url = (f"{ARXIV_API}?search_query={urllib.parse.quote(LS.arxiv_query(g))}&sortBy=relevance"
                       f"&sortOrder=descending&start=0&max_results={min(cap, 200)}")
                recs = arxiv_records(net.ask(f"keywords/{q['id']}/arxiv", url))
            elif src == "s2":
                url = (f"{S2_PAPER}/search/bulk?query={urllib.parse.quote(LS.s2_query(g))}&fields={S2_FIELDS}"
                       f"&sort=citationCount:desc")
                recs = s2_records(net.ask(f"keywords/{q['id']}/s2", url, headers=_s2_headers(), pace=1.5))[:cap]
            else:
                continue
            for i, r in enumerate(recs, 1):
                hit = LS.match_query(f"{r.get('title', '')} {r.get('abstract', '')}", g)
                if hit is not None:
                    out.append(dict(_hit(r, src, ["keywords", q["id"], src, i]),
                                    matched=sorted({t for grp in hit for t in grp})))
    return out


def _chunks(xs: Sequence, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield list(xs[i:i + n])


def citation_channel(net: Net, seeds: Sequence[dict], n_back: int = 60, n_fwd: int = 50) -> Tuple[List[dict], dict]:
    """Backward (the works most of the seeds cite) and forward (the works citing the seeds: most cited, and newest)
    on OpenAlex. A seed without an OpenAlex id is found by its DOI (an arXiv paper by arXiv's DOI). -> (records,
    {seed ids, resolved})."""
    ids: Dict[str, dict] = {s["openalex_id"]: s for s in seeds if s.get("openalex_id")}
    dois = []
    for s in seeds:
        if s.get("openalex_id"):
            continue
        d = VR.norm_doi(s.get("doi") or "") or (f"10.48550/arxiv.{s['arxiv_id']}" if s.get("arxiv_id") else "")
        if d and "," not in d and "|" not in d:
            dois.append(d.lower())
    for chunk in _chunks(sorted(set(dois)), 40):
        url = (f"{OA_WORKS}?filter=doi:{urllib.parse.quote('|'.join(chunk), safe='/|.:')}&per-page=50"
               f"&select=id,doi,title,referenced_works,cited_by_count")
        for r in oa_records(net.ask("citations/seeds", url, pace=0.2)):
            if r.get("openalex_id"):
                ids.setdefault(r["openalex_id"], r)
    refs: Counter = Counter()
    first: Dict[str, int] = {}                        # the first seed (the best one) that cites it
    for pos, s in enumerate(ids.values()):
        for w in sorted(set(s.get("refs") or [])):
            refs[w] += 1
            first.setdefault(w, pos)
    # ties broken by the citing seed's place, then the id -- never by set order: Python salts str hashes per process,
    # so most_common() over a set's order chose a different 60 (another URL, a cache miss) on every run (2026-10-18)
    back = sorted((w for w in refs if w not in ids), key=lambda w: (-refs[w], first[w], w))[:n_back]
    out: List[dict] = []
    for chunk in _chunks(back, 50):
        url = f"{OA_WORKS}?filter=openalex_id:{'|'.join(chunk)}&per-page=50&select={OA_SELECT}"
        for r in oa_records(net.ask("citations/back", url, pace=0.2)):
            out.append(_hit(r, "openalex", ["citations", "back", refs.get(r["openalex_id"], 0)]))
    seed_ids = set(ids)
    if seed_ids:
        cites = "|".join(sorted(seed_ids)[:50])
        for how, sort in (("fwd", "cited_by_count:desc"), ("fwd", "publication_date:desc")):
            url = f"{OA_WORKS}?filter=cites:{cites}&sort={sort}&per-page={min(n_fwd, 200)}&select={OA_SELECT_REFS}"
            for r in oa_records(net.ask(f"citations/forward/{sort.split(':')[0]}", url, pace=0.2)):
                n = len(set(r.get("refs") or []) & seed_ids)
                if n and r["openalex_id"] not in seed_ids:
                    out.append(_hit(r, "openalex", ["citations", how, n]))
    return out, {"seeds": sorted(seed_ids), "asked": len(seeds)}


# ------------------------------------------------------------------------------------------- merge and status
def merge(raw: Sequence[dict]) -> List[dict]:
    """One record per paper: records sharing ANY identity key (arXiv id, DOI, normalised title -- lit_sweep's
    record_keys) are joined transitively; fields fill from the first record having them; sources, hits and matched
    terms are unioned. Order: first appearance. Pure."""
    parent = list(range(len(raw)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: Dict[str, int] = {}
    for i, r in enumerate(raw):
        for k in LS.record_keys(r):
            if k in first:
                a, b = find(first[k]), find(i)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[k] = i
    groups: Dict[int, List[int]] = {}
    for i in range(len(raw)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for root in sorted(groups):
        m: dict = {"sources": [], "hits": [], "matched": []}
        for i in groups[root]:
            for k, v in raw[i].items():
                if k in ("sources", "matched"):
                    m[k] = sorted(set(m[k]) | set(v or []))
                elif k == "hits":
                    m["hits"].extend(v or [])
                elif k == "refs":
                    continue
                elif v not in (None, "", [], {}) and m.get(k) in (None, "", [], {}):
                    m[k] = v
        m["arxiv_id"] = VR.clean_arxiv_id(m.get("arxiv_id") or "") or LS.doi_to_arxiv(m.get("doi") or "")
        keys = LS.record_keys(m)
        m["id"] = keys[0] if keys else f"row:{root}"
        out.append(m)
    return out


class Have:
    """What you already have, by identity key: this atlas's index (HELD = its PDF is here, INDEXED = listed without
    one) and every project or list in `also` (ELSEWHERE): an atlas project folder (its index/papers.json), a JSON list
    (rows with title / doi / arxiv_id) or a TSV / CSV with such columns (the lab library's papers.tsv works)."""

    def __init__(self, project: Optional[Path] = None, also: Sequence[Path] = (), exclude: Iterable[str] = ()):
        self.map: Dict[str, Tuple[str, str, str]] = {}
        self.exclude = set(exclude)
        self.where: List[str] = []
        if project is not None:
            self._project(Path(project), mine=True)
        for a in also:
            a = Path(a)
            if (a / "index" / "papers.json").is_file():
                self._project(a, mine=False)
            elif a.is_file():
                self._list(a)

    def _add(self, rec: dict, value: Tuple[str, str, str]) -> None:
        for k in LS.record_keys(rec):
            self.map.setdefault(k, value)

    def _project(self, d: Path, mine: bool) -> None:
        try:
            doc = json.loads((d / "index" / "papers.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.where.append(d.name)
        for p in doc.get("papers") or []:
            if mine and p.get("key") in self.exclude:
                continue
            st = (HELD if p.get("pdf_held") else INDEXED) if mine else ELSEWHERE
            self._add({"title": p.get("title"), "doi": p.get("doi"), "arxiv_id": p.get("arxiv_id")},
                      (st, p.get("key") or "", d.name))

    def _list(self, f: Path) -> None:
        rows: List[dict] = []
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if f.suffix.lower() == ".json":
                data = json.loads(text)
                rows = data.get("papers", data) if isinstance(data, dict) else data
            else:
                import csv
                rows = list(csv.DictReader(text.splitlines(), delimiter="\t" if f.suffix.lower() == ".tsv" else ","))
        except (OSError, ValueError):
            return
        self.where.append(f.name)
        for r in rows or []:
            if isinstance(r, dict):
                rec = r.get("record") if isinstance(r.get("record"), dict) else r
                self._add({"title": rec.get("title"), "doi": rec.get("doi"), "arxiv_id": rec.get("arxiv_id")},
                          (ELSEWHERE, str(r.get("key") or r.get("rel") or ""), f.name))

    def status(self, rec: dict) -> Tuple[str, str, str]:
        for k in LS.record_keys(rec):
            if k in self.map:
                return self.map[k]
        return (NEW, "", "")


# --------------------------------------------------------------------------------------------------- ranking
def features(c: dict) -> dict:
    """The signals a candidate's hits give: best semantic rank, keyword queries + sources hit, citation links. Pure."""
    sem = [h[2] for h in c.get("hits") or [] if h[0] == "semantic"]
    kw = [h for h in c.get("hits") or [] if h[0] == "keywords"]
    back = max([h[2] for h in c.get("hits") or [] if h[0] == "citations" and h[1] == "back"], default=0)
    fwd = max([h[2] for h in c.get("hits") or [] if h[0] == "citations" and h[1] == "fwd"], default=0)
    loc = [h[2] for h in c.get("hits") or [] if h[0] == "local"]
    return {"sem": min(sem) if sem else 0, "sem_n": len(sem), "kw": len({h[1] for h in kw}),
            "kw_src": len({h[2] for h in kw}), "back": back, "fwd": fwd, "links": back + fwd,
            "local": min(loc) if loc else 0}


def rank(cands: List[dict], task: str, weights: Optional[Dict[str, float]] = None,
         dense: Optional[Dict[str, float]] = None, expand: Sequence[str] = ()) -> List[dict]:
    """Weighted reciprocal-rank fusion (paper_find.fuse_scores) of the signals; sets c['score'], c['rank'] and
    c['signals'] (each signal's rank of the candidate), best first. Pure."""
    lists = signal_lists(cands, task, dense=dense, expand=expand)
    fused = fuse_lists(lists, weights)
    pos = {n: {i: k + 1 for k, i in enumerate(r)} for n, r in lists.items()}
    for c in cands:
        c["score"] = round(fused.get(c["id"], 0.0), 6)
        c["signals"] = {n: pos[n][c["id"]] for n in lists if c["id"] in pos[n]}
    cands.sort(key=lambda c: (-c["score"], c["id"]))
    for i, c in enumerate(cands, 1):
        c["rank"] = i
    return cands


def fuse_lists(lists: Dict[str, List[str]], weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """{candidate id: fused score} from the signals' rankings (weights over WEIGHTS). Pure."""
    w = dict(WEIGHTS, **(weights or {}))
    return dict(PF.fuse_scores([(w.get(n, 0.0), r) for n, r in lists.items()], k=PF.RRF_K, depth=10 ** 6))


def signal_lists(cands: List[dict], task: str, dense: Optional[Dict[str, float]] = None,
                 expand: Sequence[str] = ()) -> Dict[str, List[str]]:
    """Each signal's ranking of the candidates (ids, best first): dense, bm25, semantic, keywords, citations, cited,
    local. Sets c['f'] (features). Pure."""
    for c in cands:
        c["f"] = features(c)
    lists: Dict[str, List[str]] = {}
    if dense:
        lists["dense"] = [i for i, _ in sorted(dense.items(), key=lambda kv: (-kv[1], kv[0]))]
    bm = PF.MetaBM25({c["id"]: " ".join([c.get("title") or "", c.get("title") or "", c.get("abstract") or ""])
                      for c in cands})
    lists["bm25"] = bm.rank(" ".join([task] + list(expand)), limit=len(cands))
    lists["semantic"] = [c["id"] for c in sorted((c for c in cands if c["f"]["sem"]),
                                                 key=lambda c: (c["f"]["sem"], -c["f"]["sem_n"], c["id"]))]
    lists["keywords"] = [c["id"] for c in sorted((c for c in cands if c["f"]["kw"]),
                                                 key=lambda c: (-c["f"]["kw"], -c["f"]["kw_src"], c["id"]))]
    lists["citations"] = [c["id"] for c in sorted((c for c in cands if c["f"]["links"]),
                                                  key=lambda c: (-c["f"]["links"], c["id"]))]
    lists["cited"] = [c["id"] for c in sorted((c for c in cands if c.get("cited_by_count")),
                                              key=lambda c: (-int(c.get("cited_by_count") or 0), c["id"]))]
    lists["local"] = [c["id"] for c in sorted((c for c in cands if c["f"]["local"]),
                                              key=lambda c: (c["f"]["local"], c["id"]))]
    return lists


def flags(c: dict) -> List[str]:
    """Starting-point hints: a survey / review, code mentioned in the abstract, a new dataset or benchmark. Pure."""
    t, a = c.get("title") or "", c.get("abstract") or ""
    out = []
    if c.get("type") == "review" or _SURVEY.search(t):
        out.append("survey")
    if _CODE.search(a):
        out.append("code")
    if _DATASET.search(f"{t}. {a}"):
        out.append("dataset")
    return out


def access(c: dict, proxy: str = "") -> dict:
    """Where to read it: an open copy (arXiv, else the open PDF OpenAlex / Semantic Scholar know, else the open landing
    page) or, closed, the links a person uses -- DOI, publisher page, Google Scholar (a link: Scholar is never queried),
    Semantic Scholar, and the library proxy atlas.toml names. Pure."""
    doi = c.get("doi") if c.get("doi") and not LS.doi_to_arxiv(c.get("doi") or "") else ""
    open_url, how = "", ""
    if c.get("arxiv_id"):
        open_url, how = f"https://arxiv.org/pdf/{c['arxiv_id']}", "arXiv"
    elif c.get("pdf_url"):
        open_url, how = c["pdf_url"], "open PDF"
    elif c.get("oa_url"):
        open_url, how = c["oa_url"], "open (publisher or repository page)"
    links = {}
    if doi:
        links["doi"] = f"https://doi.org/{doi}"
    if c.get("landing") and c["landing"] != links.get("doi"):
        links["publisher"] = c["landing"]
    if c.get("title"):
        links["scholar"] = "https://scholar.google.com/scholar?q=" + urllib.parse.quote(f'"{c["title"]}"')
    if c.get("s2_url"):
        links["semantic_scholar"] = c["s2_url"]
    if proxy and doi:
        links["library"] = proxy + urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    return {"open": bool(open_url), "open_url": open_url, "how": how, "links": links}


def why(c: dict, queries: Sequence[dict] = ()) -> List[str]:
    """Why a candidate is listed, in words. Pure."""
    out = []
    by_q = {q["id"]: q for q in queries}
    for h in (h for h in c.get("hits") or [] if h[0] == "local"):
        out.append(f"this atlas's own search (paper_find) #{h[2]}")
    for h in sorted((h for h in c.get("hits") or [] if h[0] == "semantic"), key=lambda h: h[2]):
        out.append(f"semantic search #{h[2]} on {'OpenAlex' if h[1] == 'openalex' else 'Semantic Scholar'}")
    kw = sorted({h[1] for h in c.get("hits") or [] if h[0] == "keywords"})
    if kw:
        src = sorted({h[2] for h in c.get("hits") or [] if h[0] == "keywords"})
        desc = "; ".join(" AND ".join(g[0] for g in by_q[q]["groups"]) for q in kw if q in by_q)
        out.append(f"keywords ({desc}) on {', '.join(src)}" + (f": {', '.join(c.get('matched') or [])[:120]}"
                                                                  if c.get("matched") else ""))
    f = c.get("f") or features(c)
    if f["back"]:
        out.append(f"cited by {f['back']} of the seed papers")
    if f["fwd"]:
        out.append(f"cites {f['fwd']} of the seed papers")
    return out


# --------------------------------------------------------------------------------------------- the whole run
def local_find(project: Path, task: str, pdfs: Optional[Path] = None, k: int = 12, exclude: Iterable[str] = ()) -> List[dict]:
    """paper_find on this atlas (its full text when the text cache is here, else the atlas's own Find) -> the index
    rows of the best papers, minus `exclude` (the benchmark's own answers)."""
    ex = set(exclude)
    try:
        hits = PF.find(task, k=k + len(ex), base=project, pdfs=pdfs, snippets=False)
    except Exception:                                 # noqa: BLE001 -- no usable index: no local seeds
        return []
    ix = PF.load_index(str(project))
    out = []
    for h in hits:
        if h["key"] in ex:
            continue
        p = ix.by_key.get(h["key"]) or {}
        out.append({"key": h["key"], "title": p.get("title") or h.get("title") or "", "doi": p.get("doi") or "",
                    "arxiv_id": p.get("arxiv_id") or "", "abstract": p.get("summary") or "",
                    "year": p.get("year"), "venue": p.get("venue") or "", "url": p.get("url") or "",
                    "cited_by_count": p.get("cited_by"), "held": bool(p.get("pdf_held")), "rank": len(out) + 1,
                    "why": h.get("why") or []})
        if len(out) >= k:
            break
    return out


def gate(text: str, task_words: set, min_words: int = 0) -> bool:
    """Is a would-be seed about the task at all: does its title + abstract use enough of the task's content words --
    `min_words`, by default 30% of them and at least 2? (A task the atlas knows nothing about must not seed the search
    with the atlas's nearest papers: 'lab' and 'test' are not a topic.) Pure."""
    need = min_words or max(2, math.ceil(0.3 * len(task_words)))
    return len(task_words & set(doc_key_text(text).split())) >= need


def background_texts(project: Path) -> List[str]:
    """This atlas's papers as the PRF background: title + summary per paper."""
    try:
        doc = json.loads((Path(project) / "index" / "papers.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [f"{p.get('title') or ''}. {p.get('summary') or ''}" for p in doc.get("papers") or []]


def _lexicon(project: Path) -> Tuple[Optional["PF.Terms"], Dict[str, dict]]:
    try:
        lex = json.loads((Path(project) / "index" / "lexicon.json").read_text(encoding="utf-8")).get("terms") or []
    except (OSError, ValueError):
        return None, {}
    return PF.Terms(lex), {t["id"]: t for t in lex}


def make_plan(task: str, title: str, seeds: Sequence[str], background: Sequence[str],
              terms: Optional["PF.Terms"] = None, lexicon: Optional[Dict[str, dict]] = None) -> dict:
    """The keyword half of the plan from the task and the seed texts (pure: the network part fills `seeds`)."""
    phrases = key_phrases(task, seeds, terms, lexicon, background=background)
    prf = seed_terms(seeds, background)
    facets = build_facets(phrases, prf)
    return {"task": task, "title": title, "phrases": phrases, "seed_terms": prf, "facets": facets,
            "queries": build_queries(facets, prf), "new_terms": new_terms(phrases, prf, terms)}


def new_terms(phrases: Sequence[dict], prf: Sequence[dict], terms: Optional["PF.Terms"] = None, k: int = 12) -> List[str]:
    """The task's vocabulary the atlas's lexicon does not have yet: its key phrases and the seed papers' words that
    match no lexicon term or alias -- candidates for the lexicon, each to be checked against the papers. Pure."""
    out, seen = [], set()
    for t in [p["phrase"] for p in phrases if not p.get("lexicon")] + [t["term"] for t in prf]:
        key = _gram_key(_TOKEN.findall(t))
        if not key or key in seen or not _usable(t) or (terms is not None and terms.find(t)):
            continue
        seen.add(key)
        out.append(t)
    return out[:k]


def s2_query_of(task: str, title: str = "", n: int = 10) -> str:
    """Semantic Scholar's relevance search wants a few words, not a paragraph (and no hyphens): the title, else the
    task's first `n` content words. Pure."""
    src = title or task
    words = [w for run in runs(src) for w, _ in run][:n]
    return " ".join(w.replace("-", " ") for w in words)


def discover(task: str, project: Path, title: str = "", pdfs: Optional[Path] = None, net: Optional[Net] = None,
             channels: Sequence[str] = CHANNELS, sources: Sequence[str] = SOURCES, plan: Optional[dict] = None,
             also: Sequence[Path] = (), exclude: Iterable[str] = (), cap: int = 100, n_seeds: int = 10,
             dense: Optional[Callable[[str, List[dict]], Dict[str, float]]] = None,
             weights: Optional[Dict[str, float]] = None, proxy: str = "", say: Callable = print) -> dict:
    """The whole search for one task -> {plan, candidates (ranked, with status / access / why / flags), local, net log}.
    Writes nothing: `write_outputs` does."""
    project = Path(project)
    ex = set(exclude)
    net = net or Net(Path(tempfile.gettempdir()) / "atlas_discover_cache")
    t0 = len(net.log)                                 # this task's calls start here (a benchmark shares one Net)
    task_words = {s for s in doc_key_text(task).split() if s not in {PF.stem(w) for w in GENERIC | PF.STOP}}
    local = local_find(project, task, pdfs=pdfs, k=12, exclude=ex)
    local_seeds = [s for s in local if gate(f"{s['title']} {s['abstract']}", task_words)][:n_seeds]
    if net.offline:
        say("  offline: this atlas only")
    raw: List[dict] = []
    soft = max(2, math.ceil(0.2 * len(task_words)))   # listing what you have: a softer bar than seeding the search
    for s in local:
        if gate(f"{s['title']} {s['abstract']}", task_words, min_words=soft):
            raw.append({"title": s["title"], "doi": s["doi"], "arxiv_id": s["arxiv_id"], "abstract": s["abstract"],
                        "year": s.get("year"), "venue": s.get("venue"), "url": s.get("url"),
                        "cited_by_count": s.get("cited_by_count"), "sources": ["atlas"],
                        "hits": [["local", "atlas", s["rank"]]]})
    sem: List[dict] = []
    if "semantic" in channels:
        sem = semantic_channel(net, task, (plan or {}).get("s2_query") or s2_query_of(task, title))
        raw += sem
        say(f"  semantic: {len(sem)} hits" + (f" ({', '.join(net.failed('semantic', t0))} COULD NOT be asked)"
                                              if net.failed("semantic", t0) else ""))
    online_seeds = []
    for r in sem:
        if gate(f"{r.get('title', '')} {r.get('abstract', '')}", task_words) and len(online_seeds) < n_seeds:
            if not any(set(LS.record_keys(r)) & set(LS.record_keys(o)) for o in online_seeds):
                online_seeds.append(r)
    seed_texts = [f"{s['title']}. {s['abstract']}" for s in local_seeds] + \
                 [f"{r.get('title', '')}. {r.get('abstract', '')}" for r in online_seeds]
    terms, lex = _lexicon(project)
    if plan is None:
        plan = make_plan(task, title, seed_texts, background_texts(project), terms, lex)
    plan = dict(plan, s2_query=plan.get("s2_query") or s2_query_of(task, title),
                seeds={"local": [s["key"] for s in local_seeds],
                       "online": [r.get("doi") or r.get("arxiv_id") or r.get("title") for r in online_seeds]})
    if "keywords" in channels and plan.get("queries"):
        kw = keyword_channel(net, plan["queries"], sources, cap=cap)
        raw += kw
        say(f"  keywords: {len(plan['queries'])} queries x {len(sources)} sources -> {len(kw)} hits kept"
            + (f" ({len(net.failed('keywords', t0))} calls COULD NOT be asked)" if net.failed("keywords", t0) else ""))
    cite_info = {"seeds": [], "asked": 0}
    if "citations" in channels and (local_seeds or online_seeds):
        cit, cite_info = citation_channel(net, list(local_seeds) + list(online_seeds))
        raw += cit
        say(f"  citations: {len(cite_info['seeds'])} seeds on OpenAlex -> {len(cit)} linked works"
            + (f" ({len(net.failed('citations', t0))} calls COULD NOT be asked)" if net.failed("citations", t0)
               else ""))
    cands = merge(raw)
    have = Have(project, also=also, exclude=ex)
    for c in cands:
        c["status"], c["local_key"], c["where"] = have.status(c)
    expand = [t for f in plan.get("facets") or [] for t in f["terms"]]
    dscores = None
    if dense is not None and cands:
        try:
            dscores = dense(task, cands)
        except Exception as e:                        # noqa: BLE001 -- dense is optional: rank without it
            say(f"  dense ranking skipped: {type(e).__name__}: {str(e)[:120]}")
    rank(cands, task, weights, dense=dscores, expand=expand)
    for c in cands:
        c["access"] = access(c, proxy)
        c["flags"] = flags(c)
        c["why"] = why(c, plan.get("queries") or [])
    return {"plan": plan, "candidates": cands, "local": local, "cite": cite_info, "dense": bool(dscores),
            "log": net.log[t0:], "cost_usd": round(net.cost, 4), "channels": list(channels), "sources": list(sources)}


# --------------------------------------------------------------------------------------------------- dense
def dense_scorer(py_gpu: str, model: str, cache: Path) -> Callable[[str, List[dict]], Dict[str, float]]:
    """Cosine similarity of the task to each candidate's title + abstract with the project's embedding model,
    computed in the Python that has torch (`_dense` below runs there); scores cached by (model, task, text)."""
    cache = Path(cache)

    def score(task: str, cands: List[dict]) -> Dict[str, float]:
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"dense_{model}.json"
        try:
            memo = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            memo = {}
        tq = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]
        texts = {c["id"]: f"{c.get('title') or ''}. {c.get('abstract') or ''}"[:2000] for c in cands}
        keyof = {i: f"{tq}:{hashlib.sha256(t.encode('utf-8')).hexdigest()[:16]}" for i, t in texts.items()}
        todo = {i: t for i, t in texts.items() if keyof[i] not in memo}
        if todo:
            with tempfile.TemporaryDirectory(dir=str(cache)) as tmp:
                src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
                src.write_text(json.dumps({"task": task, "texts": todo}), encoding="utf-8")
                r = subprocess.run([py_gpu, "-c", DENSE_JOB, str(Path(__file__).resolve().parent), model, str(src),
                                    str(dst)], capture_output=True, text=True, timeout=3600)
                if r.returncode or not dst.is_file():
                    raise RuntimeError((r.stderr or r.stdout or "no output").strip().splitlines()[-1][:200])
                got = json.loads(dst.read_text(encoding="utf-8"))
            memo.update({keyof[i]: float(v) for i, v in got.items()})
            f.write_text(json.dumps(memo), encoding="utf-8", newline="\n")
        return {i: memo[keyof[i]] for i in texts}

    return score


#: run by the torch Python (`python -c DENSE_JOB <tools dir> <model> <in.json> <out.json>`): only tools/paper_embed.py
#: is imported there -- {task, texts: {id: text}} -> {id: cosine of the task and the text}
DENSE_JOB = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import paper_embed as PE
model, src, dst = sys.argv[2], sys.argv[3], sys.argv[4]
job = json.loads(Path(src).read_text(encoding="utf-8"))
m = PE.load_model(model)
spec = PE.MODELS[model]
ids = sorted(job["texts"])
q = PE.encode_queries(m, spec, [job["task"]], form="definition")[0]
d = PE.encode_docs(m, spec, [job["texts"][i] for i in ids])
Path(dst).write_text(json.dumps({i: float(v) for i, v in zip(ids, d @ q)}), encoding="utf-8")
"""


# --------------------------------------------------------------------------------------------------- outputs
def _md(s, n: int = 0) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).replace("|", "/").strip()
    return s if not n or len(s) <= n else s[: n - 3].rstrip() + "..."


def _ident(c: dict) -> str:
    """The id `atlas add` resolves best: a (non-arXiv) DOI, else the arXiv id, else the title. Pure."""
    if c.get("doi") and not LS.doi_to_arxiv(c["doi"]):
        return c["doi"]
    return c.get("arxiv_id") or c.get("title") or ""


def _line(c: dict, n: int, links: bool = True) -> str:
    a = c["access"]
    meta = ", ".join(x for x in (str(c.get("year") or ""), _md(c.get("venue"), 50)) if x)
    head = f"{n}. **{_md(c.get('title'), 160)}**" + (f" ({meta})" if meta else "")
    tags = [t for t in c.get("flags") or []]
    if c.get("cited_by_count"):
        tags.append(f"cited {c['cited_by_count']}")
    out = [head + (f" -- {', '.join(tags)}" if tags else "")]
    if c["status"] in (HELD, INDEXED, ELSEWHERE):
        out.append(f"   - you have it: `{c.get('local_key')}` in {c.get('where')}"
                   + (" (PDF held)" if c["status"] == HELD else " (listed, no PDF)" if c["status"] == INDEXED else ""))
    if links and c["status"] != HELD:                 # a held PDF needs no link to get it
        if c["status"] in (INDEXED, ELSEWHERE) and not a["open"]:
            out.append("   - to get its PDF: " + " / ".join(f"[{k}](<{v}>)" for k, v in a["links"].items()))
        elif c["status"] in (INDEXED, ELSEWHERE):
            out.append(f"   - its PDF: <{a['open_url']}> ({a['how']})")
    if links and c["status"] == NEW:
        if a["open"]:
            out.append(f"   - open: <{a['open_url']}> ({a['how']})")
        else:
            out.append("   - closed: " + " / ".join(f"[{k}](<{v}>)" for k, v in a["links"].items()))
    if c.get("why"):
        out.append(f"   - why: {'; '.join(c['why'])[:400]}")
    return "\n".join(out)


def discover_md(slug: str, res: dict, today: str, top: int = 30, me: str = "atlas") -> str:
    plan, cands = res["plan"], res["candidates"]
    have = [c for c in cands if c["status"] in (HELD, INDEXED)]
    else_ = [c for c in cands if c["status"] == ELSEWHERE]
    new_open = [c for c in cands if c["status"] == NEW and c["access"]["open"]]
    new_closed = [c for c in cands if c["status"] == NEW and not c["access"]["open"]]
    failed = sorted({e["what"] for e in res["log"] if e["error"] and e["error"] != "offline"})
    offline = any(e["error"] == "offline" for e in res["log"])
    L = [f"# Papers for: {_md(plan.get('title') or plan['task'], 120)}", "",
         f"*`atlas discover` on {today}: {len(cands)} papers from {', '.join(res['channels'])} "
         f"({', '.join(res['sources'])}); ranked by {'dense similarity + ' if res.get('dense') else ''}BM25 + the "
         f"engines' order + keyword hits + citation links + citations (weighted reciprocal-rank fusion). "
         f"OpenAlex cost reported: ${res.get('cost_usd', 0):.3f}.*", "", f"> {_md(plan['task'], 1200)}", ""]
    if offline:
        L += ["**OFFLINE:** only this atlas was searched (run without --offline to ask OpenAlex, Semantic Scholar and "
              "arXiv).", ""]
    if failed:
        L += [f"**COULD NOT ASK:** {', '.join(failed)} -- those rows are missing (a re-run asks only what failed; "
              "OpenAlex keyless = ~$0.10 / day, Semantic Scholar's keyless pool returns 429 often).", ""]
    L += [f"**{len(have)} already in this atlas** ({sum(c['status'] == HELD for c in have)} with a PDF), "
          f"**{len(else_)} elsewhere** in your collections, **{len(new_open)} new and open**, "
          f"**{len(new_closed)} new and closed**.", ""]
    starts = [c for c in cands[:80] if c["status"] == NEW and ("survey" in c["flags"] or "code" in c["flags"]
                                                               or "dataset" in c["flags"])][:8]
    sections = [("Already in this atlas", have[:top]), ("Elsewhere in your collections", else_[:top]),
                ("Good starting points (new: surveys, code, datasets -- among the top 80)", starts),
                ("New and open (`--add` fetches these)", new_open[:top]),
                ("New and closed (the links: GET_THESE.md)", new_closed[:top])]
    for head, rows in sections:
        if not rows:
            continue
        L += [f"## {head}", ""]
        L += [_line(c, i) for i, c in enumerate(rows, 1)]
        L.append("")
    L += ["## How it searched (plan.json: edit and re-run with `--plan`)", "",
          "- key phrases: " + "; ".join(f"{p['phrase']} (seeds {p['support']:.0%})" for p in plan.get("phrases") or []),
          "- seed words (frequent in the seeds, rare in this atlas): "
          + ", ".join(t["term"] for t in plan.get("seed_terms") or []),
          "- facets: " + " | ".join(f"[{' OR '.join(f['terms'])}]" for f in plan.get("facets") or []),
          "- keyword queries: " + "; ".join(f"{q['id']}: " + " AND ".join("(" + " OR ".join(g) + ")" for g in q["groups"])
                                           for q in plan.get("queries") or []),
          f"- seeds: {len(plan['seeds']['local'])} from this atlas ({', '.join(plan['seeds']['local'][:8])}), "
          f"{len(plan['seeds']['online'])} from the semantic search; citations chased from {len(res['cite']['seeds'])}",
          "- terms this atlas's lexicon lacks: " + (", ".join(plan.get("new_terms") or []) or "none") + " -- check one "
          "against the papers before it enters a curated lexicon (`python tools/term_mine.py quotes \"<term>\"`); a "
          "project with `[pipeline] terms = true` grows its glossary from the new papers' own definitions on `update`",
          "", "## Next", "",
          f"- bring the new ones in -- `add_list.txt` lists the top new papers: delete what you do not want, add DOIs / "
          f"arXiv ids -- then `{me} discover {slug} --add` (each one checked against its registry; open copies "
          f"downloaded and checked; closed ones listed in the atlas with where to get them)",
          f"- ... as a SECTION of the atlas (its own banner; the papers you already had included): add `--section`",
          f"- then `{me} update`", ""]
    return "\n".join(L)


def get_these_md(slug: str, res: dict, today: str, top: int = 60) -> str:
    rows = [c for c in res["candidates"] if c["status"] == NEW and not c["access"]["open"]][:top]
    L = [f"# Closed papers for: {_md(res['plan'].get('title') or res['plan']['task'], 120)}", "",
         f"*{today}: the {len(rows)} best-ranked papers with no open copy found. Download the ones you want into the "
         "project's `inbox` folder, then `atlas add inbox` files them (each PDF is checked to be its paper). Google "
         "Scholar's link often finds an author's copy ('All versions').*", ""]
    for i, c in enumerate(rows, 1):
        L.append(f"- [ ] {i}. **{_md(c.get('title'), 160)}** ({c.get('year') or 'n.d.'}"
                 + (f", {_md(c.get('venue'), 60)}" if c.get("venue") else "") + ")")
        L.append("      " + " / ".join(f"[{k}](<{v}>)" for k, v in c["access"]["links"].items()))
    return "\n".join(L) + "\n"


def add_list(slug: str, res: dict, top: int = 25) -> str:
    """`atlas add` list form: one id per line with the topic as its second column; '#' lines say what each is."""
    L = [f"# atlas discover {slug}: the top new papers -- delete a line to skip it, add DOIs / arXiv ids / titles",
         "# then: atlas discover " + slug + " --add"]
    n = 0
    for c in res["candidates"]:
        if c["status"] != NEW or not _ident(c):
            continue
        n += 1
        L.append(f"# {n}. {_md(c.get('title'), 150)} ({c.get('year') or 'n.d.'}; "
                 f"{'open' if c['access']['open'] else 'closed'}; score {c['score']})")
        L.append(f"{_ident(c)} | {slug}")
        if n >= top:
            break
    return "\n".join(L) + "\n"


def write_outputs(project: Path, state: Path, slug: str, res: dict, today: str, me: str = "atlas",
                  top: int = 25) -> Path:
    out = Path(project) / "discover" / slug
    out.mkdir(parents=True, exist_ok=True)
    plan = {k: v for k, v in res["plan"].items()}
    (out / "plan.json").write_text(json.dumps(dict(plan, slug=slug, date=today), indent=1, ensure_ascii=False) + "\n",
                                   encoding="utf-8", newline="\n")
    (out / "run.json").write_text(json.dumps({"date": today, "channels": res["channels"], "sources": res["sources"],
                                              "cost_usd": res.get("cost_usd", 0), "dense": res.get("dense", False),
                                              "calls": [{k: e[k] for k in ("what", "url", "fresh", "error")}
                                                        for e in res["log"]],
                                              "counts": dict(Counter(c["status"] for c in res["candidates"]))},
                                             indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "DISCOVER.md").write_text(discover_md(slug, res, today, me=me), encoding="utf-8", newline="\n")
    (out / "GET_THESE.md").write_text(get_these_md(slug, res, today), encoding="utf-8", newline="\n")
    (out / "add_list.txt").write_text(add_list(slug, res, top=top), encoding="utf-8", newline="\n")
    st = Path(state) / "discover" / slug
    st.mkdir(parents=True, exist_ok=True)
    keep = ("id", "title", "authors", "year", "venue", "doi", "arxiv_id", "cited_by_count", "type", "status",
            "local_key", "where", "score", "rank", "signals", "access", "flags", "why", "abstract")
    rows = [{k: (c.get(k) if k != "abstract" else (c.get(k) or "")[:600]) for k in keep} for c in res["candidates"]]
    (st / "candidates.json").write_text(json.dumps({"date": today, "plan": plan, "candidates": rows}, indent=1,
                                                   ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return out


# ---------------------------------------------------------------------------------------------- the section
def section_block(slug: str, title: str, task: str, papers: Sequence[str], color: str, today: str) -> str:
    """The atlas.toml text for a task's section: its topic label, its Find task and the section (its new topic plus
    the papers you already had). Pure."""
    q = lambda v: json.dumps(v, ensure_ascii=False)
    lines = ["", f"# added by `atlas discover` on {today} (discover/{slug}/DISCOVER.md)", f"[topics.{slug}]",
             f"label = {q(title)}", "", "[[tasks]]", f"id = {q(slug)}", f"title = {q(title)}",
             f"query = {q(task)}", "", "[[sections]]", f"id = {q(slug)}", f"title = {q(title)}", f"color = {q(color)}",
             f"topics = [{q(slug)}]"]
    if papers:
        lines.append(f"papers = {q(list(papers))}")
    lines += [f"tasks = [{q(slug)}]", ""]
    return "\n".join(lines)


def add_section(project: Path, slug: str, title: str, task: str, papers: Sequence[str], today: str) -> str:
    """Append the section to atlas.toml -- refused (ValueError, nothing written) when the topic, task or section id
    exists, the banner row is full, or the result does not validate (tools/atlas_project.py)."""
    project = Path(project)
    cfg = AP.config(project)
    if slug in cfg["topics"] or slug in {t["id"] for t in cfg["tasks"]} or slug in {s["id"] for s in cfg["sections"]}:
        raise ValueError(f"atlas.toml already has a topic, task or section {slug!r}: pick another --slug")
    if len(cfg["sections"]) >= AP.MAX_SECTIONS:
        raise ValueError(f"atlas.toml already has {AP.MAX_SECTIONS} sections (one banner row): merge or remove one")
    used = {(s.get("color") or PALETTE[i % len(PALETTE)]).lower() for i, s in enumerate(cfg["sections"])}
    color = next((c for c in PALETTE if c not in used), PALETTE[len(cfg["sections"]) % len(PALETTE)])
    f = project / AP.CONFIG
    text = f.read_text(encoding="utf-8") if f.is_file() else ""
    new = text.rstrip("\n") + "\n" + section_block(slug, title, task, papers, color, today)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / AP.CONFIG).write_text(new, encoding="utf-8")
        AP.config(Path(tmp))                         # raises ConfigError, naming the problem
    f.write_text(new, encoding="utf-8", newline="\n")
    return color


# --------------------------------------------------------------------------------------------- the benchmark
def review_tasks(project: Path, min_rel: int = 10) -> List[dict]:
    """The benchmark's tasks: each LIT_*.md review of the project -- its H1 title as the task (what a person would
    type) and the index papers it cites by [key] as the relevant set; DEV / TEST by review (paper_find.split_of)."""
    project = Path(project)
    ix = PF.load_index(str(project))
    out = []
    for f in sorted(project.glob("LIT_*.md")):
        text = f.read_text(encoding="utf-8")
        h1 = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), f.stem)
        rel = sorted({ix.alias.get(k, k) for k in PF.KEY_REF.findall(text)} & set(ix.by_key))
        if len(rel) < min_rel:
            continue
        out.append({"id": f.stem, "doc": f.name, "task": h1, "rel": rel, "split": PF.split_of(f.name, base=project)})
    return out


def recall_at(order: Sequence[str], rel: set, ks: Sequence[int] = (25, 50, 100, 200, 10 ** 9)) -> Dict[str, float]:
    """Share of the relevant set in the first k of `order`, per k ('all' = the whole list). Pure."""
    out = {}
    for k in ks:
        got = len(set(order[:k]) & rel)
        out["all" if k >= 10 ** 9 else str(k)] = round(got / len(rel), 4) if rel else 0.0
    return out


def bench_baseline(net: Net, task: str, n: int = 200) -> List[dict]:
    """The do-nothing baseline: the task typed into OpenAlex's keyword search, its first `n` in its own order."""
    url = f"{OA_WORKS}?search={urllib.parse.quote(task)}&per-page={min(n, 200)}&select={OA_SELECT}"
    return oa_records(net.ask("baseline/openalex", url, pace=0.2))


def _rel_order(cands: Sequence[dict], key_of: Callable[[dict], str]) -> List[str]:
    out = []
    for c in cands:
        k = key_of(c)
        out.append(k or f"~{c.get('id')}")
    return out


VARIANTS = {"fused": CHANNELS, "semantic": ("semantic",), "keywords": ("keywords",), "citations": ("citations",),
            "no-semantic": ("keywords", "citations"), "no-keywords": ("semantic", "citations"),
            "no-citations": ("semantic", "keywords")}


def variant_pool(candidates: Sequence[dict], chans: Sequence[str]) -> List[dict]:
    """Copies of the candidates the given channels found, with only those channels' hits. Pure."""
    pool = [dict(c) for c in candidates if any(h[0] in chans for h in c.get("hits") or [])]
    for c in pool:
        c["hits"] = [h for h in c["hits"] if h[0] in chans]
    return pool


def _key_of(have: Have) -> Callable[[dict], str]:
    def key_of(c):
        st, k, _ = have.status(c)
        return k if st in (HELD, INDEXED) else ""
    return key_of


def pool_dense(pool: Sequence[dict], dense: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """The dense scores of a variant's own candidates only: a variant is ranked as if only its channels had run, so
    the papers the other channels found take no dense ranks above its own (they did until 2026-10-18). Pure."""
    ids = {c["id"] for c in pool}
    return {i: v for i, v in (dense or {}).items() if i in ids} or None


def bench_scores(res: dict, rel: set, have: Have, weights: Optional[Dict[str, float]] = None,
                 dense: Optional[Dict[str, float]] = None,
                 variants: Sequence[str] = tuple(VARIANTS)) -> Dict[str, Dict[str, float]]:
    """Recall@k of each of `variants` (the candidates found by those channels, re-ranked with `weights`). Pure."""
    out = {}
    for name in variants:
        pool = variant_pool(res["candidates"], VARIANTS[name])
        rank(pool, res["plan"]["task"], weights, dense=pool_dense(pool, dense),
             expand=[t for f in res["plan"].get("facets") or [] for t in f["terms"]])
        out[name] = recall_at(_rel_order(pool, _key_of(have)), rel)
    return out


#: the grid `bench --tune` searches on DEV (dense only when the runs have dense scores)
TUNE_GRID = {"dense": (0.0, 0.5, 1.0, 2.0), "bm25": (0.0, 0.5, 1.0), "semantic": (0.5, 1.0, 2.0),
             "keywords": (0.25, 0.5, 1.0), "citations": (0.25, 0.5, 1.0), "cited": (0.0, 0.25, 0.5)}


def tune(runs: Sequence[dict], have: Have, grid: Optional[Dict[str, Sequence[float]]] = None) -> List[tuple]:
    """Every weight combination of the grid on the saved runs (the full fusion's pool), best first by mean R@100,
    then R@50, then the nearest to WEIGHTS: [(mean R@100, mean R@50, weights)]. No network. Pure."""
    import itertools
    grid = dict(grid or TUNE_GRID)
    if not any(r.get("dense") for r in runs):
        grid["dense"] = (0.0,)
    key_of = _key_of(have)
    pre = []
    for run in runs:
        pool = variant_pool(run["candidates"], CHANNELS)
        lists = signal_lists(pool, run["task"], dense=pool_dense(pool, run.get("dense")),
                             expand=run.get("expand") or [])
        keys = {c["id"]: (key_of(c) or f"~{c['id']}") for c in pool}
        pre.append((lists, keys, set(run["rel"])))
    names = list(grid)
    out = []
    for combo in itertools.product(*(grid[n] for n in names)):
        w = dict(zip(names, combo), local=0.0)
        r100 = r50 = 0.0
        for lists, keys, rel in pre:
            fused = fuse_lists(lists, w)
            order = [keys[i] for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
            order += [keys[i] for i in sorted(set(keys) - set(fused))]
            rc = recall_at(order, rel, ks=(50, 100))
            r100, r50 = r100 + rc["100"], r50 + rc["50"]
        n = max(len(pre), 1)
        out.append((round(r100 / n, 4), round(r50 / n, 4), w))
    out.sort(key=lambda x: (-x[0], -x[1], sum(abs(x[2][k] - WEIGHTS.get(k, 0.0)) for k in names)))
    return out


def bootstrap_diff(a: Sequence[float], b: Sequence[float], n: int = 4000, seed: int = 7) -> Tuple[float, float, float]:
    """Mean of a - b over paired tasks, with a 95% bootstrap interval. Pure."""
    import random
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return round(sum(d) / len(d), 4), round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="recall of each channel and the fusion on the project's LIT_*.md reviews")
    b.add_argument("--project", default="")
    b.add_argument("--split", default="all", choices=["dev", "test", "all"])
    b.add_argument("--only", default="", help="comma list of review ids (LIT_... stems)")
    b.add_argument("--dense", default="auto", choices=["auto", "yes", "no"])
    b.add_argument("--refresh", action="store_true")
    b.add_argument("--replay", action="store_true", help="re-derive the runs from the cache and ask nothing (a call "
                                                         "not cached -- refused when the run was made -- is "
                                                         "COULD_NOT_MEASURE again)")
    b.add_argument("--tune", action="store_true", help="choose the fusion weights on the SAVED runs of --split "
                                                       "(dev when 'all'); no network")
    b.add_argument("--rescore", action="store_true", help="score the SAVED runs of --split with --weights (else the "
                                                          "tuned ones, else WEIGHTS); no network")
    b.add_argument("--weights", default="", help="JSON {signal: weight}, or a file holding it")
    b.add_argument("--out", default="", help="results JSON (default <pdfs>_atlas/discover/_bench/bench_<split>.json)")
    a = ap.parse_args(argv)
    return bench_main(a)


def _weights_arg(text: str, tuned: Path) -> Tuple[Dict[str, float], str]:
    """--weights as JSON text or a file; else the tuned weights; else WEIGHTS -> (weights, where they came from)."""
    if text:
        p = Path(text)
        raw = p.read_text(encoding="utf-8") if p.is_file() else text
        w = json.loads(raw)
        w = w.get("weights", w) if isinstance(w, dict) else w
        bad = sorted(set(w) - set(SIGNALS))
        if bad:
            raise SystemExit(f"bench: unknown signal(s) {bad} in --weights (known: {list(SIGNALS)})")
        return {k: float(v) for k, v in w.items()}, "--weights"
    if tuned.is_file():
        return json.loads(tuned.read_text(encoding="utf-8"))["weights"], str(tuned)
    return dict(WEIGHTS), "WEIGHTS (provisional)"


def score_run(run: dict, have: Have, weights: Dict[str, float],
              variants: Sequence[str] = tuple(VARIANTS)) -> Dict[str, Dict[str, float]]:
    """A saved run's recall per VARIANT and for the baseline (no network)."""
    res = {"plan": {"task": run["task"], "facets": [{"terms": run.get("expand") or []}]},
           "candidates": run["candidates"]}
    sc = bench_scores(res, set(run["rel"]), have, weights=weights, dense=run.get("dense") or None, variants=variants)
    sc["baseline"] = recall_at(run.get("baseline") or [], set(run["rel"]))
    return sc


def score_row(run: dict, have: Have, weights: Dict[str, float]) -> dict:
    """A saved run's benchmark row: recall per VARIANT and of the baseline under `weights`, and the full fusion's
    under WEIGHTS as well ('weights_fused') -- the decision rule's other side: weights tuned on DEV replace WEIGHTS only
    when they beat them on TEST too. No network."""
    return {"id": run["id"], "split": run["split"], "n_rel": len(run["rel"]), "n_cand": len(run["candidates"]),
            "scores": score_run(run, have, weights),
            "weights_fused": score_run(run, have, dict(WEIGHTS), variants=("fused",))["fused"]}


def summarize(rows: Sequence[dict]) -> dict:
    summary = {}
    for split in ("dev", "test", "all"):
        rs = [r for r in rows if split == "all" or r["split"] == split]
        if not rs:
            continue
        summary[split] = {v: {k: round(sum(r["scores"][v][k] for r in rs) / len(rs), 4)
                              for k in rs[0]["scores"][v]} for v in rs[0]["scores"]}
        summary[split]["fused_vs_baseline_R@100"] = bootstrap_diff([r["scores"]["fused"]["100"] for r in rs],
                                                                   [r["scores"]["baseline"]["100"] for r in rs])
        summary[split]["fused_vs_semantic_R@100"] = bootstrap_diff([r["scores"]["fused"]["100"] for r in rs],
                                                                   [r["scores"]["semantic"]["100"] for r in rs])
        if all("weights_fused" in r for r in rs):     # the full fusion under the code's WEIGHTS, paired
            summary[split]["WEIGHTS_fused"] = {k: round(sum(r["weights_fused"][k] for r in rs) / len(rs), 4)
                                               for k in rs[0]["weights_fused"]}
            summary[split]["fused_vs_WEIGHTS_R@100"] = bootstrap_diff([r["scores"]["fused"]["100"] for r in rs],
                                                                      [r["weights_fused"]["100"] for r in rs])
        summary[split]["n"] = len(rs)
    return summary


def bench_main(a) -> int:
    import atlas as A                                  # the project, its state folder and its GPU Python
    P = A.project_of(a.project or None)
    state = P.state / "discover" / "_bench"
    runs_dir, tuned = state / "runs", state / "tuned.json"
    split = a.split if not (a.tune and a.split == "all") else "dev"
    have = Have(P.dir)
    if a.tune or a.rescore:
        runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(runs_dir.glob("*.json"))]
        runs = [r for r in runs if split == "all" or r["split"] == split]
        if a.only:
            runs = [r for r in runs if r["id"] in {x.strip() for x in a.only.split(",")}]
        if not runs:
            print(f"bench: no saved runs for split {split} in {runs_dir} -- run `bench --split {split}` first")
            return 2
        if a.tune:
            best = tune(runs, have)
            for r100, r50, w in best[:8]:
                print(f"  R@100 {r100:.4f}  R@50 {r50:.4f}  {json.dumps(w)}")
            r100, r50, w = best[0]
            state.mkdir(parents=True, exist_ok=True)
            tuned.write_text(json.dumps({"date": datetime.date.today().isoformat(), "split": split, "n": len(runs),
                                         "dev_R@100": r100, "dev_R@50": r50, "weights": w,
                                         "combinations": len(best)}, indent=1) + "\n", encoding="utf-8", newline="\n")
            print(f"tuned on {len(runs)} {split} runs ({len(best)} combinations): {json.dumps(w)} -> {tuned}")
            return 0
        weights, where = _weights_arg(a.weights, tuned)
        rows = [dict(score_row(r, have, weights), failed=r.get("failed") or []) for r in runs]
        return _report(rows, weights, where, Path(a.out) if a.out else state / f"bench_{split}_rescored.json",
                       cost=0.0)
    if a.replay and a.refresh:
        print("bench: --replay answers from the cache and --refresh ignores it -- pick one")
        return 2
    tasks = [t for t in review_tasks(P.dir) if split in ("all", t["split"])]
    if a.only:
        want = {x.strip() for x in a.only.split(",")}
        tasks = [t for t in tasks if t["id"] in want]
    if not tasks:
        print("bench: no LIT_*.md review with >= 10 cited index papers in this project")
        return 2
    weights, where = _weights_arg(a.weights, tuned)
    net = Net(P.state / "discover_cache", refresh=a.refresh, cache_only=a.replay)
    dense = None
    if a.dense != "no":
        c = A.context(P, gpu="" if a.dense == "auto" else "yes")
        if c.model:
            dense = dense_scorer(c.py_gpu, c.model, P.state / "discover_cache")
    rows = []
    runs_dir.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        print(f"bench {t['id']} ({t['split']}, {len(t['rel'])} relevant): {_ascii(t['task'][:80])}")
        start = len(net.log)                          # this task's calls, the baseline's included, start here
        res = discover(t["task"], P.dir, title=t["task"], pdfs=P.papers, net=net, exclude=t["rel"], dense=dense,
                       say=lambda s: print(_ascii(s)))
        dsc = dense(t["task"], res["candidates"]) if dense is not None and res["dense"] else {}
        base_order = []
        for r in bench_baseline(net, t["task"]):
            st, k, _ = have.status(r)
            base_order.append(k if st in (HELD, INDEXED) else f"~{(LS.record_keys(r) or ['?'])[0]}")
        run = {"id": t["id"], "split": t["split"], "task": t["task"], "rel": t["rel"], "baseline": base_order,
               "expand": [x for f in res["plan"].get("facets") or [] for x in f["terms"]], "dense": dsc,
               "queries": res["plan"].get("queries"), "failed": sorted(set(net.failed(since=start))),
               "candidates": [{k: c.get(k) for k in ("id", "title", "doi", "arxiv_id", "abstract", "cited_by_count",
                                                     "hits")} for c in res["candidates"]]}
        (runs_dir / f"{t['id']}.json").write_text(json.dumps(run) + "\n", encoding="utf-8", newline="\n")
        row = dict(score_row(run, have, weights), failed=run["failed"], dense=bool(dsc))
        rows.append(row)
        sc = row["scores"]
        print("   " + "  ".join(f"{v}: R@100 {sc[v]['100']:.2f} all {sc[v]['all']:.2f}" for v in ("baseline", "fused",
                                                                                                "semantic"))
              + (f"  (COULD NOT ASK: {', '.join(run['failed'])})" if run["failed"] else ""))
    return _report(rows, weights, where, Path(a.out) if a.out else state / f"bench_{split}.json", cost=net.cost)


def _report(rows: Sequence[dict], weights: Dict[str, float], where: str, out: Path, cost: float) -> int:
    summary = summarize(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"date": datetime.date.today().isoformat(), "cost_usd": round(cost, 4),
                               "weights": weights, "weights_from": where, "rows": list(rows), "summary": summary},
                              indent=1) + "\n", encoding="utf-8", newline="\n")
    for split, s in summary.items():
        print(f"{split} (n={s['n']}): " + "  ".join(f"{v} R@100 {s[v]['100']:.3f}" for v in s if isinstance(s[v], dict))
              + f"  fused-baseline {s['fused_vs_baseline_R@100']}  fused-semantic {s['fused_vs_semantic_R@100']}"
              + (f"  fused-WEIGHTS {s['fused_vs_WEIGHTS_R@100']}" if "fused_vs_WEIGHTS_R@100" in s else ""))
    print(f"weights ({where}): {json.dumps(weights)}")
    print(f"-> {out}  (OpenAlex cost reported: ${cost:.3f})")
    if any(r.get("failed") for r in rows):
        print("NOTE: some calls COULD NOT be asked (see 'failed' per row): those rows' recall is a lower bound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
