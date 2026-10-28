"""tools/atomic_replace.py -- THE replace for publishing a file other processes may hold open: `replace_retry`.

WHY ONE HELPER (owner, 2026-10-19: "can this be fixed for good?"). On Windows `os.replace(tmp, dst)` raises
PermissionError -- WinError 5 -- when ANY other handle holds `dst` open, whatever its share mode, and WinError 32
when one holds `tmp` without share-delete. MEASURED on the lab computer 2026-10-16 (memory
windows-os-replace-foreign-handle): with no handle of ours open, 7 of 15,000 read -> write -> replace cycles failed
that way, more once another session's suite ran beside them, and the Restart Manager named no holder -- inferred: a
virus scan. So any reader of a published file (a viewer, a poller, a scan) can crash its publisher, rarely, more
under load, never in a quiet run: it reads as a test flake. It refused landings through sim/test_shots.py
(7faea8e9), tools/claude_sync/core.py::write_json and, on 2026-10-19, smoke_tracks.save_ledger -- each refusal cost
a whole pre-push sweep (up to ~1 h) and a new wait in the landing queue both machines share. Four separate retry
loops had grown (tools/farm/index.py::replace_retry, sim/worldgen/shots.py::_replace_held,
sim/assets/lockfile.py::_replace, sim/stitch_chunks.py::_write_json_atomic); they now all call this one.

WHAT IT DOES. `replace_retry(src, dst)` is `os.replace(src, dst)` that, on Windows, rides out a PermissionError for
up to REPLACE_WAIT_S (backing off from 10 ms to 250 ms), says so ONCE through `log` when it starts waiting, and then:
  * a RECORD (the default) raises PermissionError naming `dst`, how long it waited, and that the new content is
    still in `src` -- nothing is lost and nothing is silently stale;
  * a STATUS (`may_drop=True`: a heartbeat, a progress file, a cache) gives up after STATUS_WAIT_S, removes `src` and
    returns False -- a status file must never kill the run it describes.
Any other OSError (a cross-volume WinError 17, a missing directory) raises at once: those are not a hold. On POSIX
a replace over an open file succeeds, so a PermissionError there is a real one and is not retried.

THE GUARD. tools/tests/test_no_bare_replace.py AST-scans every tracked .py file for `os.replace(...)` and one-argument
`.replace(...)` calls (a `Path.replace(target)`; `str.replace` takes two) and fails when a file holds more than
bare_replace_baseline.json records -- a ratchet that only goes down. A line that is genuinely safe says why with
`# replace-ok: <why>` (a log rotation inside a logger that never raises, a test pinning os.replace's own error).

HONEST SCOPE: retrying cannot help a holder that never lets go (a viewer left open on a record): that raises after
the bound, naming the file, as it should. `os.rename` and `shutil.move` are not covered (on Windows `os.rename` onto
an existing file fails anyway). Standard library only: sim/ and runtime/ are published to the lab repo with this
file beside them (tools/manifests/lab_repo.toml), never with the rest of tools/.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

#: How long a RECORD write rides out a held destination before it raises naming the file. CHOSEN, not measured: the
#: bound shots.POST_REPLACE_WAIT_S and the farm's REPLACE_WAIT_S already used -- seconds lost on a held record, never
#: the record, and a holder that never lets go still fails visibly.
REPLACE_WAIT_S = 10.0
#: How long a STATUS write (`may_drop=True`) tries before it drops: stitch_chunks' old 8 tries came to ~1.8 s, and a
#: heartbeat a supervisor polls must not stall the tick loop that writes it.
STATUS_WAIT_S = 2.0
FIRST_PAUSE_S, MAX_PAUSE_S = 0.01, 0.25


def retries_by_default(os_name: str = os.name) -> bool:
    """Retry only where the hold exists: POSIX replaces over an open file, so a PermissionError there is a real one."""
    return os_name == "nt"


RETRIES_ON = retries_by_default()


def replace_retry(src, dst, *, may_drop: bool = False, wait_s: Optional[float] = None,
                  log: Optional[Callable[[str], object]] = None, sleep: Optional[Callable[[float], object]] = None,
                  retries_on: Optional[bool] = None) -> bool:
    """`os.replace(src, dst)`, riding out another process's handle on either file. -> True once replaced.

    A RECORD (default): retried for up to `wait_s` (REPLACE_WAIT_S), then PermissionError naming `dst` -- `src` still
    holds the new content. A STATUS (`may_drop=True`): retried for up to `wait_s` (STATUS_WAIT_S), then `src` is
    removed and False returned. `log` hears one line when the first refusal starts the wait (never one per retry).
    Only PermissionError is retried; anything else raises at once. `sleep` and `retries_on` are for tests (and the
    farm's REPLACE_RETRIES_ON switch); `None` means time.sleep and RETRIES_ON, looked up at call time."""
    budget = float((STATUS_WAIT_S if may_drop else REPLACE_WAIT_S) if wait_s is None else wait_s)
    retry = RETRIES_ON if retries_on is None else bool(retries_on)
    nap = time.sleep if sleep is None else sleep
    t0, pause, said = time.monotonic(), FIRST_PAUSE_S, False
    while True:
        try:
            os.replace(src, dst)  # replace-ok: the one retried replace every other call site goes through
            return True
        except PermissionError as e:
            held = time.monotonic() - t0
            if not retry or held >= budget:
                if may_drop:
                    try:
                        os.unlink(src)
                    except OSError:
                        pass
                    if log is not None:
                        log(f"{dst}: held open by another process for {held:.1f} s; this status write was dropped")
                    return False
                if not retry:
                    raise
                raise PermissionError(
                    e.errno, f"{e.strerror} -- replacing {dst} was refused for {held:.1f} s: held open by another "
                    f"process past the {budget:g} s bound (tools.atomic_replace); the new content is still in {src}",
                    str(dst)) from e
            if not said and log is not None:
                log(f"{dst}: held open by another process; waiting up to {budget:g} s to replace it")
            said = True
            nap(pause)
            pause = min(pause * 2, MAX_PAUSE_S)
