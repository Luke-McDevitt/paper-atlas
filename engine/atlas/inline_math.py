r"""report/boat_world_model/inline_math.py -- the plain-text symbol explanations of the lexicon's equations, turned into
INLINE LaTeX: "c_i: colour of Gaussian i; alpha_i: opacity" -> "$\mathbf{c}_i$: colour of Gaussian $i$; $\alpha_i$: ...".

WHY (2026-10-13, owner: "The word document still shows things like c_i as text instead of the proper word equation
format outside of the main equations"; and "Instead of fanning out agents to do the fix can we just make a tool that
fixes them when it detects them?"). The equations themselves are display LaTeX (lexicon/_math.json), but each one's
"where ..." explanation is typed ASCII. This converts it DETERMINISTICALLY at build time -- the stored text stays
plain and editable, the Word document, the atlas and LEXICON.md all get real math.

    convert(symbols, display_latex) -> the explanation with $...$ spans   (pure; idempotent on $...$ already there)
    residue(text) -> the math-looking tokens still outside $...$         (what the tests hold near zero)

WHAT IS MATH (each a span, merged when only operators / numbers / spaces separate two spans):
  * a symbol with a subscript, superscript or prime: c_i, X_udot, M_RB, p_k^ref, n^2, G'_i, z_{1-alpha/2}, theta_-3dB
  * a Greek name used as a symbol: alpha, Sigma, psi, sigma_i, eta[pi], alpha-hat, hat eta
  * a function of those: R(psi), K_T(J), p_j(x), c(z), x-hat_0(x_t)
  * a single-letter variable: "colour at sample i", "N: number of lines" -- but never the article a / A or the
    pronoun I, a unit after a number ("10 m", "0.1 dB/m"), a letter in "e.g." / "i.e." / "2-D", or a letter joined
    to a word by a hyphen or apostrophe
  * the left side of "lhs: description" when it is all math ("M = M_RB + M_A: inertia matrix")
NOTATION follows the display equation: a symbol the equation sets bold / bold-italic / calligraphic / blackboard
(\mathbf{c}, \boldsymbol{\eta}, \mathcal{N}) -- and never also plain -- is set the same way inline; a long
subscript (path, ref, env) becomes \text{...}; ^T becomes ^\top when the equation uses \top; <= >= != -> ~ +- become
\leq \geq \neq \to \sim \pm; cos sin exp log max min ... become \cos ...; diag / softmax / argmin -> \operatorname.
HONEST SCOPE: a heuristic. It never rewrites words, only wraps and translates math-looking tokens; a formula typed
as prose ("twenty log R") stays prose. The tests run every converted span of the real lexicon through pandoc.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi rho sigma "
         "tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega").split()
_GREEK_SET = set(GREEK)
FUNCS = {"cos": r"\cos", "sin": r"\sin", "tan": r"\tan", "exp": r"\exp", "log": r"\log", "ln": r"\ln",
         "max": r"\max", "min": r"\min", "det": r"\det", "tanh": r"\tanh", "sinh": r"\sinh", "cosh": r"\cosh",
         "arccos": r"\arccos", "arcsin": r"\arcsin", "arctan": r"\arctan", "sup": r"\sup", "inf": r"\inf"}
OPNAMES = {"diag", "softmax", "argmin", "argmax", "sign", "sgn", "tr", "var", "cov", "rank", "erf", "clip",
           "relu", "sigmoid", "atan2", "mod", "round", "floor", "ceil", "KL", "trans", "vec"}
_STYLE = re.compile(r"\\(mathbf|boldsymbol|bm|mathcal|mathbb|mathsf)\{(\\?[A-Za-z]+|\d)\}")
_ACCENT = {"hat": r"\hat", "tilde": r"\tilde", "bar": r"\bar", "dot": r"\dot", "ddot": r"\ddot", "vec": r"\vec"}
_IDENT = r"[A-Za-z][A-Za-z0-9]*"
_SUP_SUB = r"(?:_(?:\{[^{}]*\}|-?[A-Za-z0-9]+)|\^(?:\{[^{}]*\}|\([^()]*\)|-?[A-Za-z0-9]+)|'(?![A-Za-z]))"
# a symbol token: optional accent word, identifier, at least one of _x / ^x / '   (or a Greek name with none)
_SCRIPTED = re.compile(r"(?<![\w\\$'])(?:(?:hat|tilde|bar)\s)?" + _IDENT + r"(?:\*(?=[_^]))?(?:-hat|-tilde|-bar)?" + _SUP_SUB + r"+"
                       r"(?:-hat|-tilde|-bar)?")
_GREEK_WORD = re.compile(r"(?<![\w\\$])(?:(?:hat|tilde|bar)\s)?(?:" + "|".join(sorted(GREEK, key=len, reverse=True)) +
                         r")(?:-hat|-tilde|-bar)?(?![\w'])")
_ACCENTED = re.compile(r"(?<![\w\\$])(?:(?:hat|tilde|bar)\s+[A-Za-z]|[A-Za-z]-(?:hat|tilde|bar))(?![\w'])")
_INDICATOR = re.compile(r"(?<![\w.])1\[\.\]")
_FUNC_BEFORE = re.compile(r"(?:(?<![\w.])\d+(?:\.\d+)?\s*)?\b(?:log|ln|exp|sqrt|cos|sin|tan)\s+$")
_SINGLE = re.compile(r"(?<![\w\\$'.\-/])([b-zB-HJ-Z])(?![\w'\-/]|\.\w)")
_UNIT_AFTER_NUMBER = re.compile(r"(?:(?<![\w.^_])\d+(?:\.\d+)?|\x00)\s*$")   # a NUMBER (not the 0 of W_0), or existing math ("$c = 1$ m")
_ARGS = re.compile(r"\((?:[^()]|\([^()]*\)){0,40}\)")
_BRACKET_ARGS = re.compile(r"\[(?:[^\[\]]){0,24}\]")
_ENGLISH = re.compile(r"\b[A-Za-z]{4,}\b")
#: short English words that are never a symbol in these explanations ('a' is: "s, a: state and action")
_STOP = frozenset("an at and of the for to is on or by as if it its per not are was all any but can has may one two "
                  "use via see we our".split())
#: a stand-in for an existing $...$ span while the rest is converted (never in real text, never \w)
_MASK = "\x00"


def _has_words(s: str) -> bool:
    """Does `s` hold an English word (4+ letters that are not a Greek, function or operator name)?"""
    return any(w not in _GREEK_SET and w not in FUNCS and w not in OPNAMES and w not in ("hat", "tilde", "bar")
               for w in _ENGLISH.findall(s or ""))


def style_map(latex: str) -> Dict[str, str]:
    """base symbol -> its styled form, for symbols the display equation styles and never writes plain."""
    styled: Dict[str, set] = {}
    for cmd, base in _STYLE.findall(latex or ""):
        styled.setdefault(base.lstrip("\\"), set()).add(f"\\{cmd}{{{base}}}")
    bare_src = _STYLE.sub(" ", latex or "")
    bare_src = re.sub(r"\\(?:text|operatorname|mathrm|label|begin|end)\{[^{}]*\}", " ", bare_src)
    bare = set(re.findall(r"\\([A-Za-z]+)", bare_src)) | set(re.findall(r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])",
                                                                         bare_src))
    return {b: next(iter(s)) for b, s in styled.items() if len(s) == 1 and b not in bare}


def _base_tex(name: str, styles: Dict[str, str]) -> str:
    if name in styles:
        return styles[name]
    if name in _GREEK_SET:
        return "\\" + name
    if name in FUNCS:
        return FUNCS[name]
    if name in OPNAMES:
        return r"\operatorname{" + name + "}"
    if len(name) > 1 and not name.isupper() and not re.fullmatch(r"[A-Za-z]\d+", name):
        return r"\text{" + name + "}"                     # a multi-letter name ('TL', 'OA' stay upright caps)
    return name if len(name) == 1 or name.isupper() else r"\mathrm{" + name + "}"


def _script(s: str, styles: Dict[str, str], top: bool) -> str:
    """The text of a subscript / superscript -> LaTeX."""
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return "{" + expr(s[1:-1], styles, top) + "}"
    if s.startswith("(") and s.endswith(")"):
        return "{(" + expr(s[1:-1], styles, top) + ")}"
    if top and s == "T":
        return r"\top"
    m = re.fullmatch(r"([a-z])dot", s)                    # X_udot -> X_{\dot{u}}
    if m:
        return r"{\dot{" + m.group(1) + "}}"
    if s in ("max", "min"):
        return "{\\" + s + "}"
    if s in _GREEK_SET:
        return "{\\" + s + "}"
    if re.fullmatch(r"-?\d+", s) or len(s) == 1:
        return s if len(s) == 1 else "{" + s + "}"
    if re.fullmatch(r"-?\d*[A-Za-z]{2,}", s) and not re.fullmatch(r"[a-z]{2}", s) and not s.isupper():
        num = re.match(r"-?\d*", s).group(0)
        return "{" + num + r"\text{" + s[len(num):] + "}}"   # path, ref, env, -3dB
    return "{" + s + "}"                                   # ij, RB, ta, k1


def symbol(tok: str, styles: Dict[str, str], top: bool = True) -> str:
    """One symbol token (c_i, X_udot, alpha-hat, hat eta, p_k^ref, G'_i) -> LaTeX."""
    tok = tok.strip()
    accent = None
    m = re.match(r"(hat|tilde|bar)\s+(.*)$", tok)
    if m:
        accent, tok = m.group(1), m.group(2)
    m = re.match(r"(" + _IDENT + r")(\*(?=[_^]))?(-hat|-tilde|-bar)?(.*)$", tok)
    name, star, acc2, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    if rest.endswith(("-hat", "-tilde", "-bar")):
        acc2, rest = rest[rest.rindex("-"):], rest[:rest.rindex("-")]
    accent = accent or (acc2[1:] if acc2 else None)
    pieces = [sm.group(0) for sm in re.finditer(_SUP_SUB, rest)]
    for acc in ("hat", "tilde", "bar"):                  # x_hat_k / x_T_hat: the accent is on the base, not a subscript
        if "_" + acc in pieces:
            pieces.remove("_" + acc)
            accent = accent or acc
    if name.endswith("dot") and len(name) > 3 and name[:-3] in _GREEK_SET | set("uvwrpqxyz"):
        base = r"\dot{" + _base_tex(name[:-3], styles) + "}"
    else:
        base = _base_tex(name, styles)
    if accent:
        base = _ACCENT[accent] + "{" + base + "}"
    if star:                                              # a*_t: the optimal action -> {a^{*}}_t
        base = "{" + base + "^{*}}"
    subs = [_script(p[1:], styles, False) for p in pieces if p[0] == "_"]
    sub = subs[0] if len(subs) == 1 else ""
    for s in (reversed(subs) if len(subs) > 1 else ()):   # pi_theta_old -> \pi_{\theta_{old}}: a chain NESTS
        s = s[1:-1] if s.startswith("{") and s.endswith("}") else s
        sub = "{" + s + ("_" + sub if sub else "") + "}"
    out = base + ("'" * sum(p == "'" for p in pieces)) + ("_" + sub if sub else "")
    for p in pieces:
        if p[0] == "^":
            out += "^" + _script(p[1:], styles, True)
    return out


_TOKEN = re.compile(r"(?:(?:hat|tilde|bar)\s)?" + _IDENT + r"(?:\*(?=[_^]))?(?:-hat|-tilde|-bar)?" + _SUP_SUB +
                    r"*(?:-hat|-tilde|-bar)?"
                    r"|[_^](?:\{[^{}]*\}|-?[A-Za-z0-9]+)"
                    r"|<=|>=|!=|->|\+-|[-+*/=<>~^|,.;:()\[\]{}!]|\d+(?:\.\d+)?|\s+|.", re.S)


def expr(s: str, styles: Dict[str, str], top: bool = True) -> str:
    """A math expression typed as text ('M = M_RB + M_A', 'R(psi)', 'd <= lambda/2', '[x, y, psi]^T') -> LaTeX."""
    out: List[str] = []
    for tok in _TOKEN.findall(s):
        if tok.isspace():
            out.append(" ")
        elif tok in ("<=", ">=", "!=", "->", "+-", "~"):
            out.append({"<=": r" \leq ", ">=": r" \geq ", "!=": r" \neq ", "->": r" \to ", "+-": r" \pm ",
                        "~": r" \sim "}[tok])
        elif tok == "*":
            out.append(r" \cdot ")
        elif tok in ("{", "}", "%", "#", "&", "$"):
            out.append("\\" + tok)
        elif tok == "^" or tok == "_":
            out.append(tok)
        elif tok[0] in "^_":                              # ]^{d_x}, )^2
            out.append(tok[0] + _script(tok[1:], styles, top and tok[0] == "^"))
        elif re.match(r"[A-Za-z]", tok):
            if tok == "inf" or tok == "infinity":
                out.append(r"\infty")
            elif re.search(_SUP_SUB, tok) or re.search(r"-(?:hat|tilde|bar)$", tok) or tok.startswith(("hat ", "tilde ", "bar ")):
                out.append(symbol(tok, styles, top))
            elif tok == "in":
                out.append(r" \in ")                             # h in R^d
            elif tok.lower() in _STOP:
                out.append(r"\text{ " + tok + " }")              # spaced: math mode drops the spaces around words
            else:
                out.append(_base_tex(tok, styles) if (tok in styles or tok in _GREEK_SET or tok in FUNCS or
                                                      tok in OPNAMES or len(tok) == 1) else r"\text{" + tok + "}")
        else:
            out.append(tok)
    tex = re.sub(r"\^T\b", r"^\\top", "".join(out)) if top else "".join(out)
    tex = re.sub(r"\[\.\]", r"[\\cdot]", tex)
    tex = re.sub(r"(?<=[(\[])\s*\.(?=\s|[|)\],])", r"\\cdot", tex)
    tex = re.sub(r"\s\|\s", r" \\mid ", tex)
    if "1" in styles:
        tex = re.sub(r"(?<![\d.\w])1\[", lambda m: styles["1"] + "[", tex)
    tex = re.sub(r"(\\(?:cos|sin|tan|exp|log|ln|tanh)) +(?=[A-Za-z\\])", r"\1 ", tex)
    return re.sub(r" {2,}", " ", tex).strip()


def _protected(text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in re.finditer(r"(?<!\\)\$[^$]+(?<!\\)\$", text)]


def _spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) of every math span in a stretch of prose (no $...$ in it)."""
    cands: List[Tuple[int, int]] = []

    def extend(a, b):
        ext = _ARGS.match(text, b) or _BRACKET_ARGS.match(text, b)
        if ext and not _has_words(ext.group(0)[1:-1]):
            b = ext.end()
            tail = re.match(_SUP_SUB + "+", text[b:])          # x-hat_0(x_t)  /  c(z)^2
            if tail:
                b += tail.end()
        return a, b
    for rx in (_SCRIPTED, _GREEK_WORD, _ACCENTED):
        cands += [extend(*m.span()) for m in rx.finditer(text)
                  if not re.fullmatch(r"[A-Za-z]{3,}'+", m.group(0))]   # members' / agents': a possessive, not a prime
    cands += [m.span() for m in _INDICATOR.finditer(text)]
    for m in re.finditer(r"(?<![\w$])(?:\[[^\[\]]{1,24}\]|\([^()]{1,24}\))(?:[_^]\{[^{}]{1,24}\}|\^-?[A-Za-z0-9]+)", text):
        if not _has_words(m.group(0)):                    # [-1, 1]^{d_x}, (1 - p)^2
            cands.append(m.span())
    for m in re.finditer(r"(?<![\w\\$'.\-/])([A-Za-z])(\((?:[^()]|\([^()]*\)){1,40}\))", text):  # c(z), P(. | s, a)
        if not _has_words(m.group(2)):
            cands.append(m.span())
    for m in _SINGLE.finditer(text):
        a, b = m.span()
        if _UNIT_AFTER_NUMBER.search(text[:a]):              # '10 m', '0.1 dB/m': a unit, not a variable
            continue
        if text[a - 1:a] == "(" and text[b:b + 1] == ")":     # '(m)': a unit in parentheses
            continue
        cands.append((a, b))
    braced = [m.span() for m in re.finditer(r"[_^]\{[^{}]*\}", text)]
    cands = [(a, b) for a, b in cands if not any(x < a and b <= y for x, y in braced)]
    found: List[Tuple[int, int]] = []
    for a, b in sorted(set(cands), key=lambda ab: (-(ab[1] - ab[0]), ab[0])):
        if not any(a < y and b > x for x, y in found):
            found.append((a, b))
    found.sort()
    for _ in range(2):                                       # merge, then fold enclosing parentheses, then merge
        merged: List[List[int]] = []
        for a, b in found:
            if merged:
                gap = text[merged[-1][1]:a]
                if re.fullmatch(r"\s*(?:(?:<=|>=|!=|->|[-+*/=<>~^|])\s*|\d+(?!\d)(?:\.\d+(?!\d))?\s*)*", gap):
                    merged[-1][1] = b                        # only operators / numbers / spaces between: one formula
                    continue
            merged.append([a, b])
        found = []
        for a, b in merged:
            m = re.match(r"\s*(?:=|<=|>=|<|>|~)\s*-?\d+(?:\.\d+)?(?![\w.])", text[b:])   # 'lambda = 0.2'
            if m:
                b += m.end()
            m = re.search(r"(?<![\w.])-?\d+(?:\.\d+)?\s*[-+*/]\s*$", text[:a])        # '1 - lambda'
            if m:
                a = m.start()
            m = _FUNC_BEFORE.search(text[:a])                                            # '20 log R'
            if m:
                a = m.start()
            while a > 0 and text[a - 1] == "(" and text[b:b + 1] == ")" and not _has_words(text[a:b]):
                a, b = a - 1, b + 1                                                      # '(X_udot - Y_vdot)'
            m = re.search(r"(?<![\w\\])(?:" + "|".join(FUNCS) + r")$", text[:a]) if text[a:a + 1] == "(" else None
            if m:
                a = m.start()                                                            # 'sin(omega_0 W_0 x + b_0)'
            found.append((a, b))
    return found


def _math_lhs(lhs: str) -> bool:
    """Is 'lhs' of 'lhs: description' all math (symbols, operators, numbers -- no English words)?"""
    words = [w for w in _ENGLISH.findall(lhs) if w not in _GREEK_SET and w not in FUNCS and w not in OPNAMES
             and w not in ("hat", "tilde", "bar") and not re.fullmatch(r"[a-z]dot|[A-Z]+", w)]
    words += [w for w in re.findall(r"(?<![\w\\])[A-Za-z]{2,3}(?![\w(])", lhs) if w.lower() in _STOP]
    return (bool(lhs.strip()) and not words and len(lhs) <= 90 and bool(re.search(r"[A-Za-z]", lhs))
            and _MASK not in lhs)


def convert(text: str, latex: str = "", definitions: bool = False) -> str:
    """The explanation with its math as $...$ (existing $...$ kept). `definitions=True` is the conservative mode
    for definition prose: only scripted symbols (X_u, sigma_i) are converted."""
    if not text:
        return text
    styles = style_map(latex)
    masked = text
    for a, b in _protected(text):                        # existing math is opaque: found spans never enter or touch it
        masked = masked[:a] + _MASK * (b - a) + masked[b:]
    spans: List[Tuple[int, int]] = []
    if definitions:                                      # prose: only a SYMBOL -- a 1-2 letter or Greek base
        spans = [m.span() for m in _SCRIPTED.finditer(masked)   # with a script (X_u, sigma_i), never a code name
                 if re.match(r"(?:[A-Za-z]|[A-Z][A-Za-z]|" + "|".join(GREEK) + r")[_^]", m.group(0))]   # (is_last) or a possessive
    else:                                                # clause by clause: an all-math "lhs: description" lhs is one span
        off = 0
        for i, clause in enumerate(re.split(r"(;\s+|\.\s+(?=[A-Z]))", masked)):
            m = None if i % 2 else re.match(r"(\s*)([^:]{1,90}?)(:\s)(.*)$", clause, re.S)
            if m and _math_lhs(m.group(2)) and not re.fullmatch(r"\s*\d+\s*", m.group(2)):
                spans.append((off + m.start(2), off + m.end(2)))
                spans += [(off + m.start(4) + a, off + m.start(4) + b) for a, b in _spans(m.group(4))]
            elif not i % 2:
                spans += [(off + a, off + b) for a, b in _spans(clause)]
            off += len(clause)
    keep = []
    for a, b in spans:
        raw = masked[a:b]
        a2, b2 = a + len(raw) - len(raw.lstrip()), b - (len(raw) - len(raw.rstrip()))
        if b2 <= a2 or _MASK in raw or masked[a2 - 1:a2] == _MASK or masked[b2:b2 + 1] == _MASK:
            continue                                     # inside or touching existing math: left as it is
        keep.append((a, b))
    return _wrap(text, keep, styles)


def _wrap(s: str, spans: List[Tuple[int, int]], styles: Dict[str, str]) -> str:
    out, last = [], 0
    for a, b in spans:
        raw = s[a:b]
        lead = raw[:len(raw) - len(raw.lstrip())]
        trail = raw[len(raw.rstrip()):]
        body = raw.strip()
        out.append(s[last:a] + lead + ("$" + expr(body, styles) + "$" if body else "") + trail)
        last = b
    return "".join(out) + s[last:]


def residue(text: str) -> List[str]:
    """Math-looking tokens (scripted symbols, Greek names) still OUTSIDE $...$ -- what the conversion missed."""
    prose = re.sub(r"(?<!\\)\$[^$]+(?<!\\)\$", " ", text or "")
    return [m.group(0) for rx in (_SCRIPTED, _GREEK_WORD) for m in rx.finditer(prose)
            if not re.fullmatch(r"[A-Za-z]{3,}'+", m.group(0))]
