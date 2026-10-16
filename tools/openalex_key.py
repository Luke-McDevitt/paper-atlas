"""The owner's OpenAlex API key, handled in ONE place for every tool that asks OpenAlex.

OpenAlex's keyless budget is $0.10 a day (1,000 list / filter calls -- MEASURED 2026-10-14: HTTP 429 after, reset at
midnight UTC); a free key (openalex.org/settings/api) is 10x that. The owner sets it once, in their own terminal:
`setx OPENALEX_API_KEY <key>`. The rules (review + fix check 2026-10-14):
  * only https://api.openalex.org (port 443, no userinfo) ever gets it -- as an Authorization header, never in a URL
    (no URL cache, log or cache pack may hold it), and UNREDIRECTED: urllib copies ordinary headers to wherever a
    redirect points, any host, even plain http;
  * a key that is not one plain token is ignored, and its value is never printed;
  * on Windows the value STORED with `setx` wins over a stale one this process was started with (the lab's TEMP
    lesson: a process keeps the environment it started with), `%NAME%` references expanded against the stored values
    first;
  * OpenAlex answering 401 / 403 to a keyed request (a typo, a revoked or rotated key): one note naming the variable
    (never its value), the key is dropped for the rest of the process, and that request is retried keyless.
"""
from __future__ import annotations

import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional

NAME = "OPENALEX_API_KEY"
_KEY_SHAPE = re.compile(r"[A-Za-z0-9._~+/=-]{8,256}")
_NOTED: set = set()
_REFUSED: list = []
#: under pytest (PYTEST_VERSION, set for every session, inherited by xdist workers and test subprocesses) the owner's
#: REAL key is never read -- not from this process's environment, not from the registry -- whichever conftest did or
#: did not load (fix check 2026-10-14: `--rootdir` below the repo root skipped the root conftest's guard, and a failed
#: assertion printed the stored key). A test that fakes the registry sets _TEST_REGISTRY_OK.
_TEST_REGISTRY_OK: list = []
if os.environ.get("PYTEST_VERSION"):
    os.environ.pop(NAME, None)


def _note(tag: str, msg: str) -> None:
    if tag not in _NOTED:
        _NOTED.add(tag)
        print(f"NOTE: {msg}")


def _registry(name: str) -> Optional[tuple]:
    """(value, kind) of HKCU\\Environment\\name, None when absent (or not Windows, or under the test suite: the
    repo-root conftest sets SVFR_OPENALEX_NO_REGISTRY so no test can read -- and print -- the owner's real key)."""
    if os.name != "nt" or os.environ.get("SVFR_OPENALEX_NO_REGISTRY") or (
            os.environ.get("PYTEST_VERSION") and not _TEST_REGISTRY_OK):
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)
    except OSError:
        return None


def _expand(v: str) -> str:
    """'%NAME%' -> the STORED user value first (what a new process would see), else this process's, else left as is."""
    def one(m):
        got = _registry(m.group(1))
        if got is not None and isinstance(got[0], str):
            return got[0]
        return os.environ.get(m.group(1), m.group(0))
    return re.sub(r"%([^%\s]+)%", one, v)


def user_env(name: str, env=None) -> str:
    """A user environment variable. `env` given (tests): that mapping only. On Windows the value stored with `setx`
    (HKCU\\Environment, REG_SZ / REG_EXPAND_SZ) wins over this process's -- with a note, never the values, when they
    differ; elsewhere the process's."""
    if env is not None:
        return (env.get(name) or "").strip()
    here = (os.environ.get(name) or "").strip()
    got = _registry(name)
    if got is None:
        return here
    v, kind = got
    wr = sys.modules.get("winreg")                       # (the real module, or a test's stand-in)
    if kind == getattr(wr, "REG_EXPAND_SZ", 2):
        v = _expand(str(v or ""))
    elif kind != getattr(wr, "REG_SZ", 1):
        return here                                      # a list / binary value is no variable
    stored = str(v or "").strip()
    if stored and here and stored != here:
        _note("stale", f"{name} in this process differs from the one stored with setx -- using the stored one "
                       f"(values not printed)")
    return stored or here


def auth(url: str, env=None) -> Dict[str, str]:
    """{'Authorization': 'Bearer <key>'} for https://api.openalex.org only, {} otherwise / without a usable key."""
    try:
        u = urllib.parse.urlsplit(str(url or ""))
        port = u.port
    except ValueError:
        return {}
    if (u.scheme != "https" or (u.hostname or "") != "api.openalex.org" or port not in (None, 443)
            or "@" in u.netloc):                       # ANY userinfo -- 'https://:pw@api...' too (third fix check)
        return {}
    if _REFUSED and env is None:
        return {}
    key = user_env(NAME, env)
    if key and not _KEY_SHAPE.fullmatch(key):
        _note("shape", f"{NAME} is not one plain token (a space or line break inside?) -- ignored; its value is not "
                       f"printed")
        return {}
    return {"Authorization": f"Bearer {key}"} if key else {}


def request(url: str, headers: Optional[Dict[str, str]] = None) -> urllib.request.Request:
    """A GET for `url` carrying the key as an UNREDIRECTED header (never copied to a redirect target)."""
    req = urllib.request.Request(url, headers=dict(headers or {}))
    for k, v in auth(url).items():
        req.add_unredirected_header(k, v)
    return req


def urlopen(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 60):
    """urllib.request.urlopen of `request(url, headers)`; a 401 / 403 to a KEYED request: one note, the key dropped for
    the rest of this process, the same request again keyless (fix check 2026-10-14: a refused key failed every call,
    reported as 'rate limit or network')."""
    req = request(url, headers)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403) and "Authorization" in req.unredirected_hdrs:
            _REFUSED.append(e.code)
            _note("refused", f"OpenAlex refused {NAME} (HTTP {e.code}) -- fix or remove it; continuing WITHOUT it "
                             f"(its value is not printed)")
            return urllib.request.urlopen(request(url, headers), timeout=timeout)
        raise
