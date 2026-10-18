"""tools.library -- the lab LIBRARY: one drive-letter-free master copy of every file worth keeping, with an index cheap
enough that "where is X" is one grep.

WHY (owner, 2026-10-12): files were spread over C:, D:, the laptop's travel drive and a new 5 TB disk, with several
copies of some data and no record of which was the real one, and the owner asked for organization rules, an
auto-updating grep-able index, no hard-coded paths ("in case we want to move to a different drive or a NAS"), and a
rule that keeps worktrees and temp files off C:. The design and the research behind it: report/LAB_LIBRARY_RESEARCH.md.
The rules that travel WITH the disk: tools/library/RULES.md, copied to `<library>/00_meta/RULES.md` by `init` and kept
current by `rules --sync` (rules.py; `doctor` says when the copy is stale).

    python -m tools.library where                     # which disk is the library on this machine, and how it was found
    python -m tools.library index                     # incremental: catalog.tsv, papers.tsv, dupes.tsv, INDEX.md, FTS
    python -m tools.library find "bellhop eigenray"   # full-text over names, READMEs and paper text
    python -m tools.library ingest <src> 10_field/2026-09-19_hollywood-beach_rig   # verified copy in (+ manifest)
    python -m tools.library file 01_inbox/<item> <area>/<topic>/<YYYY-MM-DD_slug>  # sort the inbox: same-disk move (filing.py)
    python -m tools.library export <library path> <fast-disk folder>               # verified copy OUT (a working copy)
    python -m tools.library inventory <folder> --out <dir> [--rules <toml>]        # census of a code/data drop
    python -m tools.library verify [<rel>]            # re-hash against the manifests
    python -m tools.library mirror <backup-root>      # verified copy of the irreplaceable areas to a backup library
    python -m tools.library doctor                    # is anything of ours growing on the system drive?
    python -m tools.library space [--census]          # will the system drive block a push soon, and what fills it?
    python -m tools.library rules [--sync]            # is the library's RULES.md this checkout's committed one?
    python -m tools.library gaps --plan <plan.toml>   # fast-disk folders the library does not hold yet (backup.py)
    python -m tools.library backup --plan <plan.toml> # ... a verified dated snapshot of each; `restore` puts one back
    python -m tools.library worktrees [--apply]       # landed + clean + idle worktrees: archive, then remove SAFELY
    python -m tools.library publish <xfer outbox>     # lab: catalog.tsv -> the outbox, for another machine (offload.py)
    python -m tools.library offload <folder> --pull outbox/catalog.tsv [--delete]  # free a machine of what G: holds

The library is FOUND, never named: `$SVFR_LIBRARY` when set (a NAS = a UNC path there), else the `_LIBRARY.toml` marker
at the root, or one folder below the root, of a local drive (`locate.py`). Everything recorded INSIDE the library is a
path relative to its root, so the disk can change letter, machine or medium.

HONEST:
  * The index covers the LIBRARY only. Filename search over the hot tier (D:) is Everything/es.exe's job.
  * A file is hashed when it is copied in, when its size collides with another file's (duplicate detection), or with
    `index --hash-all`; otherwise the catalog's sha256 column is empty, and "unchanged" means same size + mtime.
  * Paper metadata (title / authors / year / DOI) is read from the PDF's own metadata and first pages: a heuristic,
    good enough to find a paper, not a bibliography (references/ stays the verified one, CLAUDE.md rule 2).
"""
