"""ONE pace for arXiv, shared by every tool in a process: before EVERY request to an arXiv host -- first attempts,
retries and fallbacks alike -- call `wait(url)`.

    arxiv.org, www.arxiv.org, ar5iv.labs.arxiv.org, ar5iv.org   15 s apart   robots.txt: `Crawl-delay: 15` (the file
                                                                            covers *.arxiv.org mirrors; read 2026-10-14)
    export.arxiv.org (the API: query, OAI-PMH)                   3 s apart    the API's terms of use

WHY (2026-10-14 review): tools paced arxiv.org at 3 s (the API's number, not the pages'); a PDF fetch's curl fallback
and abstract-page check went out 0 s after the request before them; a retry loop asked again after 5 s; a failed
download skipped its pause. A pacer per HOST, consulted per REQUEST, makes all of those impossible to get wrong one
call site at a time. Consumers load it BY PATH into sys.modules["arxiv_pace"] (report/ modules have no tools/ on their
path), so every tool in one process shares the same clocks. Pure stdlib.
"""
from __future__ import annotations

import threading
import time
import urllib.parse

DELAYS = {"arxiv.org": 15.0, "www.arxiv.org": 15.0, "ar5iv.labs.arxiv.org": 15.0, "ar5iv.org": 15.0,
          "export.arxiv.org": 3.0}
_last: dict = {}
_lock = threading.Lock()


def delay_for(url: str) -> float:
    return DELAYS.get((urllib.parse.urlsplit(str(url)).hostname or "").lower(), 0.0)


def wait(url: str, sleep=time.sleep, clock=time.monotonic) -> float:
    """Block until this host's delay has passed since the last request to it in this process; -> seconds waited."""
    d = delay_for(url)
    if not d:
        return 0.0
    host = (urllib.parse.urlsplit(str(url)).hostname or "").lower()
    with _lock:
        last = _last.get(host)
        w = max(0.0, last + d - clock()) if last is not None else 0.0
        if w:
            sleep(w)
        _last[host] = clock()
    return w

