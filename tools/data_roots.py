"""tools.data_roots -- find this project's big EXTERNAL data folders (the ones that live outside git) on this machine.

WHY (owner, 2026-09-27): "I moved the lake waburg files on to E to save space on C, make any updates to tests and the
like needed to cope with this". Three places assumed the Lake Wauburg survey folder sat BESIDE the repo -- animlab's
`${LAKE_SCENE}` root, the ICRA reel's `LAKE_SCENE_DEFAULT`, and `report/ICRA_ARTIFACTS.json`, which recorded the four
lake inputs by ABSOLUTE `C:/Users/...` path. After the move `report/test_icra_freeze.py` failed in every checkout, and
the pre-push hook refuses any push whose `report` suite is red. A drive letter cannot be the fix
(`tools/tests/test_no_hardcoded_drive_defaults.py`): the data has already moved once (C: -> E:), a 5 TB capture drive is
on order, and a worktree under `C:\\wt\\` was never beside the data in the first place.

    from tools import data_roots
    hit = data_roots.find("LAKE_DATA")      # -> Found(path=..., how="drive root E:\\", looked=[...])
    lake = hit.path                         # None when the folder is not on this machine
    data_roots.expand("${LAKE_SCENE}/mbes_returns.ply")   # -> Path, or None when the root is not found

    python -m tools.data_roots              # print every root: where it was found, or everywhere it was looked for

RESOLUTION ORDER for a root (first hit wins; `Found.how` says which):
  1. its ENVIRONMENT VARIABLE (`SVFR_LAKE_DATA`, `SVFR_LAKE_SCENE`) -- an explicit setting always wins and is returned
     even when the folder is missing, because a typo in a setting must surface as "missing", never be silently
     replaced by a folder found somewhere else;
  2. a DERIVED root inside its parent root (the lake scene is `<LAKE_DATA>/Lake Scene (3D layers)`), so pointing
     `SVFR_LAKE_DATA` at a folder is enough for both;
  3. beside THIS checkout: `<repo>/../<folder>` (where the data lived until 2026-09-27);
  4. beside the SHARED checkout, found by reading the worktree's `.git` file, so a worktree under `C:\\wt\\` sees what
     sits beside the main clone. No `git` subprocess: a hook's inherited GIT_DIR would aim one at the wrong repo
     (report/CI_STATE_AND_HIDDEN_SUITES.md section 7);
  5. the ROOT OF EVERY LOCAL DRIVE (Windows: fixed + removable only). Network and optical drives are skipped, because a
     stat on a disconnected share can block for tens of seconds.
  6. LAST, inside the LAB LIBRARY (owner, 2026-10-12: one master copy of every dataset on the 5 TB disk, found by its
     `_LIBRARY.toml` marker or `$SVFR_LIBRARY`; tools/library), at the root's `lib` path -- so a travel drive's
     original can be retired without any code naming the library's drive, while a machine's WORKING copy (D: on the
     lab computer, which stays its development disk) still wins over the archive.
A folder counts as FOUND only when its MARKER exists inside it (`Data From The Boat` for the survey folder,
`mbes_returns.ply` for the scene layers): a same-named empty folder is not the data.

HONEST:
  * Step 5 can wake a sleeping USB drive (seconds, once). Results are cached per process; `find(..., refresh=True)`
    re-searches.
  * Only folders registered in `ROOTS` are searched. A new external dataset gets one row there, not a new idiom.
  * This finds data; it does not check it. Byte identity is `report/icra_freeze.py --verify --deep`'s job.
"""
from __future__ import annotations

import ntpath
import os
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Root:
    """One external data folder. `folder` is searched for by name; `parent` + `sub` derive it from another root."""
    env: str
    marker: str
    folder: Optional[str] = None
    parent: Optional[str] = None
    sub: Optional[str] = None
    about: str = ""
    lib: Optional[str] = None          # where it is filed INSIDE the lab library (tools/library), library-relative


#: Every external data root the repo knows how to find. Add a row here rather than a new `root.parent / "..."`.
ROOTS: Dict[str, Root] = {
    "LAKE_DATA": Root(env="SVFR_LAKE_DATA", folder="lake waburg  09.12.2026", marker="Data From The Boat",
                      lib="10_field/2026-09-12_lake-wauburg_blueboat",
                      about="Lake Wauburg 2026-09-12 survey folder: boat logs, rig bags, derived scenes "
                            "(two spaces before the date, as the owner named it)"),
    "LAKE_SCENE": Root(env="SVFR_LAKE_SCENE", parent="LAKE_DATA", sub="Lake Scene (3D layers)",
                       marker="mbes_returns.ply",
                       about="the Lake Wauburg 3-D layers lake_scene writes (MBES + side-scan PLYs, boat path)"),
    # Added 2026-09-29 for the lab computer, where the clone sits on D: (C: is full) instead of beside the repo;
    # the laptop's `<repo>/../holoocean` is still found by step 3, so nothing changes there. The marker is the
    # client package itself: `D:\holoocean` on that box is $HOLODECKPATH (the downloaded worlds), not the source.
    "HOLOOCEAN_SRC": Root(env="SVFR_HOLOOCEAN_SRC", folder="holoocean", marker="client/src/holoocean/__init__.py",
                          about="the HoloOcean source clone (v2.3.0; the Python client under client/src/holoocean) "
                                "that tools/holoocean_devbox.py checks and sim/holoocean_realism.py was verified "
                                "against"),
    # Added 2026-10-06 for the Hollywood reef v2 worlds (report/HOLLYWOOD_REEF_V2.md). On the lab computer the owner's
    # deliverables folder sits at a drive root (found by step 5) and the working data one level down; the variables
    # move either. A `folder` may hold a sub-path: every candidate is `<base> / folder`.
    "HOLLYWOOD_3D": Root(env="SVFR_HOLLYWOOD_3D", folder="Hollywood_Reef_3D_2026-09-19/Hollywood_Reef_3D",
                         marker="modules/site3/gallery.json",
                         about="the Hollywood Beach 2026-09-19 reef deliverables: fused per-module clouds "
                               "(modules/site{3,4}/files), sections, boat looks, the v1 reconstruction"),
    "HOLLYWOOD_REEF": Root(env="SVFR_REEF_ROOT", folder="svfr/reef/hollywood", marker="fuse/s3/site_model.json",
                           about="the Hollywood reef working data (boat svlz, NemoSens, fuse, DEM, identity, meshes, "
                                 "renders); the v2 rebuild writes under v2/"),
    # Added 2026-10-13: these field datasets were filed into the lab library (tools/library) and their travel-drive
    # originals retired, which left hard-coded `E:`/`F:` paths in three callers pointing at nothing.
    "POOL_MAPPING": Root(env="SVFR_POOL_MAPPING", folder="Mapping in pool 09.04.2026", marker="rec_1788538282/rosbag2",
                         lib="10_field/2026-09-04_pool-mapping",
                         about="the 2026-09-04 pool mapping session: rig bag rec_1788538282 with its derived pathbench / "
                               "locmatrix / player outputs inside it, and the GoPro footage"),
    "HOLLYWOOD_FIELD": Root(env="SVFR_HOLLYWOOD_FIELD", folder="Hollywood Beach sea trial 09.19.2026",
                            marker="Data from nemosens", lib="10_field/2026-09-19_hollywood-beach/raw",
                            about="the Hollywood Beach 2026-09-19 sea trial's RAW recordings: rig bags, GoPro, the "
                                  "NemoSens AUV logs (.xtf), the boat's site 3/4 side-scan"),
    # Added 2026-10-08 for the NemoSens reverse-engineering (report/NEMOSENS_SOFTWARE_INTERNALS.md). On the lab computer
    # the working copy of the owner's "files on nemosens laptop" folder (a drive root on F:/E:) is
    # D:\svfr\nemosens\laptop_files, found by step 5; point the variable at the original on any other machine.
    "NEMOSENS_LAPTOP": Root(env="SVFR_NEMOSENS_LAPTOP", folder="svfr/nemosens/laptop_files",
                            lib="10_field/2026-09-28_nemosens-laptop-files",
                            marker="rtsys/cousto/bin/libRTProtoCpp.so",
                            about="a copy of the NemoSens laptop's files: the Cousto 2.4.7 install (bin/libRTProtoCpp.so "
                                  "holds the vehicle's message schemas), its downloaded vehicle logs (rtsys/cousto/Logs), "
                                  "the team's mission files and the RTSYS manuals (NemoSens_docs)"),
    # Added 2026-10-14 for runtime.apps.reef_sonar_map. The laptop's reef WORKING root (temp/reef on its system drive) holds the
    # only copy of every per-window section_recon output (recon/site{3,4}/<window>/section.npz: each window's MBES
    # returns in its own frame) and of module_merge's per-module records (modules/site{3,4}/<module>/merge.json +
    # fuse.json: where each window sits in its module). The 2026-10-12 laptop triage put it in the library's attic,
    # so `lib` points there until it is filed elsewhere.
    "HOLLYWOOD_RIG_WORK": Root(env="SVFR_HOLLYWOOD_RIG_WORK", folder="temp/reef", marker="modules/config.json",
                               lib="90_attic/laptop-2026-10-12_review-2027-01/temp/reef",
                               about="the Hollywood 2026-09-19 rig processing root: section_recon windows "
                                     "(recon/site{3,4}/<window>/section.npz) and module_merge records "
                                     "(modules/site{3,4}/<module>/merge.json + fuse.json)"),
}


@dataclass
class Found:
    """Where a root resolved to. `path` is None when it is not on this machine; `looked` lists every place tried."""
    name: str
    path: Optional[Path]
    how: str
    looked: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.path is not None and (self.path / ROOTS[self.name].marker).exists()

    def reason(self) -> str:
        """One line for a pytest.skip / an error: what is missing and every place that was searched."""
        r = ROOTS[self.name]
        if self.path is not None and not self.ok:
            return (f"{self.name} is set by {self.how} to {self.path}, but {r.marker!r} is not inside it "
                    f"({r.about})")
        tried = "; ".join(self.looked) if self.looked else "nowhere"
        return f"{self.name} ({r.about}) is not on this machine; looked in: {tried}; set {r.env} to point at it"


_CACHE: Dict[tuple, Found] = {}


def shared_checkout(repo: Optional[Path] = None) -> Optional[Path]:
    """The MAIN checkout this repo belongs to (itself when it is the main one), read from `.git` without git.

    A linked worktree's `.git` is a FILE: `gitdir: <common>/.git/worktrees/<name>`; that directory's `commondir`
    file names the common dir relative to it, and the main checkout is the common dir's parent."""
    repo = Path(repo) if repo is not None else REPO
    dot = repo / ".git"
    try:
        if dot.is_dir():
            return repo
        if not dot.is_file():
            return None
        text = dot.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text.lower().startswith("gitdir:"):
        return None
    gitdir = Path(text.split(":", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (repo / gitdir).resolve()
    try:
        rel = (gitdir / "commondir").read_text(encoding="utf-8").strip()
        common = (gitdir / rel).resolve()
    except OSError:
        common = gitdir.parent.parent
    return common.parent if common.name == ".git" else None


def local_drive_roots() -> List[Path]:
    """Roots of the local FIXED and REMOVABLE drives on Windows, in letter order; [] elsewhere.

    Network (4), optical (5) and RAM (6) drives are skipped: a stat on a dropped share can block for tens of seconds,
    and neither kind ever holds this project's survey folders."""
    if os.name != "nt":
        return []
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        mask = k32.GetLogicalDrives()
        out = []
        for i, letter in enumerate(string.ascii_uppercase):
            if not mask & (1 << i):
                continue
            root = f"{letter}:\\"
            if k32.GetDriveTypeW(ctypes.c_wchar_p(root)) in (2, 3):    # DRIVE_REMOVABLE, DRIVE_FIXED
                out.append(Path(root))
        return out
    except Exception:                                                    # noqa: BLE001 -- never let discovery raise
        return []


def _has(path: Path, marker: str) -> bool:
    try:
        return (path / marker).exists()
    except OSError:
        return False


_LOCATED: Dict[tuple, Path] = {}


def locate(default, marker: Optional[str] = None, *, drives=None, refresh: bool = False) -> Path:
    """A DRIVE-LETTER-INDEPENDENT default (owner, 2026-10-05). `default` itself when it exists (holding `marker`, when
    one is given); else the SAME path on the first other local drive where it exists (`local_drive_roots`: fixed +
    removable, in letter order); else `default` unchanged, so a caller's "missing" message still names the place.

        UE = Path(os.environ.get("UE_ROOT") or locate(r"E:\\UE_5.3"))     # the variable still wins

    WHY. The laptop's data drive is E: there and F: on the lab computer, and every `E:\\svfr\\...` default -- texture
    assets, the worldgen work root, the UE toolchain, the octree archive -- went missing on the lab although the data
    was plugged in. This replaces only the hard-coded LETTER of a fallback; an environment variable keeps priority at
    each call site. A default with no drive letter (any POSIX path) is returned as given. Cached per process;
    `refresh=True` re-searches; `drives` is a seam for tests. tools/tests/test_no_hardcoded_drive_defaults.py fails
    on a new absolute `E:` default that does not go through here."""
    key = (str(default), marker)
    if not refresh and key in _LOCATED:
        return _LOCATED[key]
    p = Path(default)

    def ok(c: Path) -> bool:
        try:
            return c.exists() and (marker is None or (c / marker).exists())
        except OSError:
            return False
    found = p
    drive, tail = ntpath.splitdrive(str(default))
    if drive and tail.strip("\\/") and not ok(p):
        rel = tail.replace("\\", "/").strip("/")
        for root in (local_drive_roots() if drives is None else drives):
            if ok(Path(root) / rel):
                found = Path(root) / rel
                break
    _LOCATED[key] = found
    return found


def find(name: str, *, env: Optional[dict] = None, repo: Optional[Path] = None,
         drives: Optional[Iterable[Path]] = None, refresh: bool = False) -> Found:
    """Resolve root `name` (a key of ROOTS). `env`, `repo` and `drives` exist for tests; callers pass nothing."""
    if name not in ROOTS:
        raise KeyError(f"unknown data root {name!r}; known: {', '.join(sorted(ROOTS))}")
    environ = os.environ if env is None else env
    repo = Path(repo) if repo is not None else REPO
    chain, n = [], name
    while n:                                            # this root and every root it derives from
        chain.append(n)
        n = ROOTS[n].parent
    # The cache key carries the VALUES of every env var that can decide the answer, so setting one mid-process
    # (a test's monkeypatch, a tool's --flag exported to the environment) is never answered from a stale entry.
    key = (name, tuple(environ.get(ROOTS[c].env) or "" for c in chain), str(repo),
           tuple(str(d) for d in drives) if drives is not None else None)
    if not refresh and key in _CACHE:
        return _CACHE[key]
    r = ROOTS[name]
    looked: List[str] = []
    hit: Optional[Found] = None

    val = environ.get(r.env)
    if val:
        hit = Found(name, Path(val), f"${r.env}", [f"${r.env}={val}"])
    elif r.parent is not None:
        par = find(r.parent, env=env, repo=repo, drives=drives, refresh=refresh)
        looked += par.looked
        if par.path is not None:
            cand = par.path / r.sub
            looked.append(str(cand))
            if _has(cand, r.marker):
                hit = Found(name, cand, f"inside {r.parent} ({par.how})", looked)
    else:
        candidates: List[tuple] = [(repo.parent / r.folder, "beside this checkout")]
        main = shared_checkout(repo)
        if main is not None and main.resolve() != repo.resolve():
            candidates.append((main.parent / r.folder, "beside the shared checkout"))
        for d in (local_drive_roots() if drives is None else [Path(x) for x in drives]):
            candidates.append((d / r.folder, f"drive root {d}"))
        if r.lib:                                            # LAST: a machine's working copy wins (owner, 2026-10-12)
            try:
                from tools.library import locate as _lib   # imported here: tools.library imports this module
                lib_hit = _lib.find(env=environ, drives=drives)
            except Exception:                                # noqa: BLE001 -- discovery never raises
                lib_hit = None
            if lib_hit is not None and lib_hit.ok:
                candidates.append((lib_hit.path / r.lib, f"in the library ({lib_hit.how})"))
        seen = set()
        for cand, how in candidates:
            k = str(cand).lower()
            if k in seen:
                continue
            seen.add(k)
            looked.append(str(cand))
            if _has(cand, r.marker):
                hit = Found(name, cand, how, looked)
                break
    if hit is None:
        hit = Found(name, None, "not found", looked)
    _CACHE[key] = hit
    return hit


def path(name: str, **kw) -> Optional[Path]:
    """The root's folder when it is really there (marker present), else None."""
    hit = find(name, **kw)
    return hit.path if hit.ok else None


def token(name: str) -> str:
    return "${%s}" % name


def split_token(raw: str):
    """`${LAKE_SCENE}/x.ply` -> ("LAKE_SCENE", "x.ply"); anything else -> (None, raw)."""
    s = str(raw).replace("\\", "/")
    if s.startswith("${") and "}" in s:
        name, _, rest = s[2:].partition("}")
        return name, rest.lstrip("/")
    return None, raw


def expand(raw: str, **kw) -> Optional[Path]:
    """`${NAME}/rel` -> the real path (None when NAME is not found); a plain path is returned as a Path."""
    name, rest = split_token(raw)
    if name is None:
        return Path(raw)
    base = path(name, **kw)
    return None if base is None else base / rest


def tokenize(p, names: Iterable[str] = tuple(ROOTS), **kw) -> Optional[str]:
    """The `${NAME}/rel` form of `p` if it lies inside a FOUND root (most specific root first), else None."""
    p = Path(p).resolve()
    best = None
    for name in names:
        base = path(name, **kw)
        if base is None:
            continue
        try:
            rel = p.relative_to(base.resolve())
        except ValueError:
            continue
        if best is None or len(str(base)) > len(str(best[1])):
            best = (name, base, rel)
    if best is None:
        return None
    return token(best[0]) + "/" + best[2].as_posix()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Where are this project's external data folders on this machine?")
    ap.add_argument("names", nargs="*", help=f"roots to resolve (default: all of {', '.join(ROOTS)})")
    a = ap.parse_args(argv)
    missing = 0
    for name in (a.names or list(ROOTS)):
        hit = find(name)
        if hit.ok:
            print(f"{name:12s} {hit.path}   ({hit.how})")
        else:
            missing += 1
            print(f"{name:12s} NOT FOUND -- {hit.reason()}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
