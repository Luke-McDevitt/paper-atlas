"""Pure logic shared by the literature-review pipeline scripts.

The pipeline scripts are top-to-bottom one-offs with hardcoded paths and
import-time side effects, so nothing in them is importable or testable. The
predicates that actually decide what ends up in the review -- DOI
normalisation, the "is this the same paper" test, the PRISMA dedup key --
are lifted here verbatim (behaviour-preserving except where noted) so they
have one definition and a test suite.

Lifted from:
  pdf_acquisition/oa_final.py   -- norm(), STOP, same_paper()
  pdf_acquisition/*.py          -- the four divergent DOI parsers
  build_prisma_review.py        -- nt(), key()

Fixes applied during the lift (see docs/TESTING_GAPS.md A1/A2):
  * norm_doi strips *any* number of doi.org/ prefixes (match_pdfs.py used
    split("doi.org/", 1)[1], which left the second prefix in place).
  * same_paper normalises both DOIs before comparing, so a URL-prefix
    variant can no longer defeat the distinct-DOI guard.
  * prisma_key normalises the DOI (a bare "10.1109/x" now matches
    "https://doi.org/10.1109/x") and falls back to the *full* normalised
    title instead of an 80-character truncation used as an exact key.

No I/O, no side effects: safe to import from anywhere.
"""

import difflib
import re
import unicodedata

# ---------------------------------------------------------------- DOIs ----

# One prefix hop: an optional scheme, an optional dx./www. host prefix and
# the doi.org host, or a bare "doi:" label. Applied repeatedly so that
# doubled prefixes ("https://doi.org/https://doi.org/10.1109/x", which the
# free-DB rows really do contain) collapse completely.
_DOI_PREFIX_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:dx\.)?doi\.org/|^doi:\s*", re.I)
_DOI_TRAILING = " \t\r\n.,;"


def norm_doi(s):
    """Canonical DOI: prefixes stripped, lowercased, trailing punctuation gone.

    Returns "" for None/empty. Idempotent: norm_doi(norm_doi(x)) == norm_doi(x).
    """
    if s is None:
        return ""
    d = str(s).strip()
    prev = None
    while d != prev:                     # strip *every* prefix, not just one
        prev = d
        d = _DOI_PREFIX_RE.sub("", d).strip()
    return d.lower().strip(_DOI_TRAILING)


# -------------------------------------------------------------- titles ----

def norm(s):
    """Title squashed to bare alphanumerics (oa_final.py:39-41)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def nt(s):
    """Title normalised to lowercase alphanumeric words (build_prisma_review.py:36).

    NOT truncated: the original truncated to 80 chars and used the result as
    an exact dedup key, which collapses "... Part I" and "... Part II".
    """
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# Generic words that are not distinctive enough to be a paper's acronym.
STOP = set("slam sonar fusion auv uuv rov imu dvl data deep pose mapping visual acoustic optical "
           "underwater vehicle system network survey camera image based using".split())


def title_tokens(title):
    """Content tokens of a title (oa_final.py:54)."""
    return set(w for w in re.findall(r"[a-z0-9]+", str(title).lower()) if len(w) >= 4)


def title_words(title):
    """All lowercase words of a title, in order (oa_final.py:55)."""
    return re.findall(r"[a-z0-9]+", str(title).lower())


def derive_match_fields(r):
    """Attach the fields same_paper() reads (oa_final.py:53-55). Mutates and returns r."""
    r["tn"] = norm(r.get("title"))
    r["tok"] = title_tokens(r.get("title"))
    r["words"] = title_words(r.get("title"))
    return r


def make_paper(title, doi=None, **extra):
    """Build a match-ready record. Convenience for callers/tests."""
    r = dict(title=title, doi=doi, **extra)
    return derive_match_fields(r)


def _derived(r):
    """same_paper's view of a record: use cached fields if the caller prepared them."""
    if "tn" in r and "tok" in r and "words" in r:
        return r
    return derive_match_fields(dict(r))


def _acronym(title):
    """Leading acronym: the phrase before the first colon (handles hyphens)."""
    t = str(title)
    pre = t.split(":", 1)[0] if ":" in t else (t.split()[0] if t.split() else "")
    return norm(pre)


def same_paper(a, b):
    """Are these two records the same study? (lifted from oa_final.py:58-75)

    Records need "title" and "doi"; "tn"/"tok"/"words" are used if present.
    """
    d_a, d_b = norm_doi(a.get("doi")), norm_doi(b.get("doi"))
    if d_a and d_b:
        return d_a == d_b   # authoritative: distinct DOIs => distinct papers
    a, b = _derived(a), _derived(b)
    if difflib.SequenceMatcher(None, a["tn"], b["tn"]).ratio() >= 0.90:
        return True
    if len(a["tn"]) >= 12 and len(b["tn"]) >= 12 and (a["tn"] in b["tn"] or b["tn"] in a["tn"]):
        return True
    for x, y in ((a, b), (b, a)):  # bare acronym <-> full title
        if 1 <= len(x["words"]) <= 2 and len(x["words"][0]) >= 4 \
                and x["words"][0] not in STOP and x["words"][0] in y["tok"]:
            return True
    # shared leading acronym ("VIMS: ..." / "FAR-AVIO: ..." twins) + token overlap.
    aa, ba = _acronym(a["title"]), _acronym(b["title"])
    if aa and aa == ba and 4 <= len(aa) <= 16:
        ov = len(a["tok"] & b["tok"]) / max(1, min(len(a["tok"]), len(b["tok"])))
        if ov >= 0.4:
            return True
    return False


# --------------------------------------------------------------- PRISMA ---

def prisma_key(r):
    """Cross-database dedup key (build_prisma_review.py:45-47).

    DOI when there is one (normalised, so bare and URL forms collide as they
    should), otherwise the full normalised title.
    """
    d = norm_doi(r.get("doi") if r.get("doi") is not None else r.get("d"))
    return d if d else nt(r.get("title") or r.get("t"))
