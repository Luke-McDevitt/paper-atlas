"""Find the library root without naming a drive: `$SVFR_LIBRARY`, else the `_LIBRARY.toml` marker on a local drive.

Same idiom as tools/data_roots.py (whose `local_drive_roots` this reuses): an explicit variable always wins and is
returned even when wrong, so a typo surfaces as "missing" instead of being replaced by a disk found elsewhere; the scan
looks at each fixed/removable drive's root and ONE level below it (so `G:\\lab\\_LIBRARY.toml` works as well as
`G:\\_LIBRARY.toml`); network drives are never scanned (a dropped share can block for tens of seconds).
"""
from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:                                                          # pragma: no cover -- CI and both machines are >= 3.12
    tomllib = None

from tools import data_roots

ENV = "SVFR_LIBRARY"
MARKER = "_LIBRARY.toml"
LAYOUT_VERSION = 1

#: The numbered areas (report/LAB_LIBRARY_RESEARCH.md sec 4). `irreplaceable` areas are the ones `mirror` copies.
AREAS = {
    "00_meta": ("rules, the generated index, manifests, the paper text cache, ledgers", False),
    "01_inbox": ("FAST DUMP, to be sorted: YYYY-MM-DD_<who>_<what>/; indexed + backed up; sort within 14 days", True),
    "10_field": ("raw sensor data, IMMUTABLE: YYYY-MM-DD_site_platform/{raw,derived}/ + README.md", True),
    "20_papers": ("every paper PDF once, by collection", True),
    "30_sim": ("worlds, handoff kits, the octree archive, sim runs worth keeping", False),
    "40_deliverables": ("YYYY-MM-DD_slug/: decks, reels, share kits, proposals, submissions", True),
    "50_projects": ("other projects, one folder each", True),
    "60_software": ("installers, ISOs, Orin images and backups, toolchains", False),
    "80_machines": ("as-dumped machine snapshots awaiting triage", False),
    "90_attic": ("kept but probably dead; each name ends in a review date (..._review-YYYY-MM)", False),
}
IRREPLACEABLE = tuple(a for a, (_, keep) in AREAS.items() if keep) + ("00_meta",)

#: Folders a drive-root scan never descends into.
_SKIP = {"system volume information", "$recycle.bin", "windows", "program files", "program files (x86)",
         "programdata", "users", "found.000", "recovery", "$windows.~bt", "perflogs"}


@dataclass
class Found:
    path: Optional[Path]
    how: str
    looked: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.path is not None and (self.path / MARKER).is_file()


def read_marker(root: Path) -> dict:
    """The parsed `_LIBRARY.toml` ({} when absent or unreadable)."""
    try:
        with open(Path(root) / MARKER, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, ValueError, AttributeError):
        return {}


def _candidates(drive: Path) -> Iterable[Path]:
    yield drive
    try:
        kids = sorted(os.scandir(drive), key=lambda e: e.name.lower())
    except OSError:
        return
    for e in kids:
        try:
            if e.is_dir(follow_symlinks=False) and e.name.lower() not in _SKIP and not e.name.startswith("."):
                yield Path(e.path)
        except OSError:
            continue


def find(*, env: Optional[dict] = None, drives: Optional[Iterable[Path]] = None, role: Optional[str] = None) -> Found:
    """Resolve the library. `role` ("master" / "backup") restricts the drive scan to libraries of that role; the
    variable is never second-guessed. `env` and `drives` are test seams."""
    environ = os.environ if env is None else env
    val = environ.get(ENV)
    if val:
        return Found(Path(val), f"${ENV}", [f"${ENV}={val}"])
    looked: List[str] = []
    backup: Optional[Found] = None
    for d in (data_roots.local_drive_roots() if drives is None else [Path(x) for x in drives]):
        for cand in _candidates(Path(d)):
            looked.append(str(cand))
            try:
                if (cand / MARKER).is_file():
                    got = read_marker(cand).get("role", "master")
                    if role and got != role:
                        continue
                    if role is None and got == "backup":
                        # A MASTER anywhere wins: on the lab computer the backup (F:) sorts before the master (G:),
                        # and `index` / `ingest` must never write into the backup. With no master plugged in (the
                        # laptop, carrying only F:) the backup is the answer, so data_roots still finds the data.
                        backup = backup or Found(cand, f"BACKUP library marker on drive {d} (no master found)", [])
                        continue
                    return Found(cand, f"marker on drive {d}", looked)
            except OSError:
                continue
    if backup is not None:
        backup.looked = looked
        return backup
    return Found(None, "not found", looked)


def require(**kw) -> Path:
    hit = find(**kw)
    if not hit.ok:
        where = "; ".join(hit.looked[:12]) + (" ..." if len(hit.looked) > 12 else "")
        raise SystemExit(f"ERROR: no library found ({hit.how}); looked in: {where or 'nowhere'}. "
                         f"Set {ENV} or plug in the library disk (it carries {MARKER}).")
    return hit.path


def init(root: Path, name: str, role: str = "master") -> Path:
    """Create a library at `root`: the marker, the numbered areas, 00_meta/README.md and RULES.md. Refuses to overwrite."""
    root = Path(root)
    if (root / MARKER).exists():
        raise SystemExit(f"ERROR: {root / MARKER} already exists; a library is initialised once")
    if role not in ("master", "backup"):
        raise SystemExit("ERROR: role is 'master' or 'backup'")
    root.mkdir(parents=True, exist_ok=True)
    for a in AREAS:
        (root / a).mkdir(exist_ok=True)
    for sub in ("index", "manifests", "text"):
        (root / "00_meta" / sub).mkdir(exist_ok=True)
    rules = Path(__file__).with_name("RULES.md").read_text(encoding="utf-8")
    (root / "00_meta" / "RULES.md").write_text(rules, encoding="utf-8", newline="\n")
    readme = ["# The lab library -- start here", "",
              f"Library `{name}` ({role}). Read `RULES.md` before adding anything. To find a file, read",
              "`index/INDEX.md` first, then grep `index/catalog.tsv` / `index/papers.tsv`, or run",
              "`python -m tools.library find <words>` from the sonar-video-fusion-review repo. Never list the disk recursively.",
              "", "| area | holds |", "|---|---|"]
    readme += [f"| `{a}` | {about} |" for a, (about, _) in AREAS.items()]
    (root / "00_meta" / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8", newline="\n")
    marker = (f'name = "{name}"\nrole = "{role}"\nvolume_id = "{uuid.uuid4().hex}"\nlayout = {LAYOUT_VERSION}\n'
              f'# Found by tools/library/locate.py: $SVFR_LIBRARY, else this file at a drive root (or one level down).\n'
              f'# Every path recorded inside this library is RELATIVE to the folder holding this file.\n')
    (root / MARKER).write_text(marker, encoding="utf-8", newline="\n")
    return root
