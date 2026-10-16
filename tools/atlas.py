r"""atlas -- ONE command for a paper atlas: start a project, bring papers in (PDFs, a Zotero / BibTeX export, a list of
DOIs / arXiv ids / titles), and run the whole pipeline -- text, OCR, facts, embeddings, figures, citations, glossary,
lookup tables, the atlas page and the portable zip -- resumably, skipping every step whose inputs did not change.

    python tools/atlas.py new  <folder> [--title "My Paper Atlas"]         # a new project (atlas.toml + folders)
    python tools/atlas.py add  <pdf|folder|.bib|.json|.txt>... [--topic T] [--as FILE=DOI] [--offline] [--no-fetch]
    python tools/atlas.py update [--only a,b] [--skip a,b] [--force] [--gpu auto|yes|no] [--offline]
    python tools/atlas.py status                                            # what the project holds, what is stale
    python tools/atlas.py find "a task in plain words" [--term X] [--keys]  # tools/paper_find.py on this project
    python tools/atlas.py zip [--no-pdfs] [--no-figures]                    # just the portable .zip
    python tools/atlas.py open                                              # the atlas page (with figure images)
    python tools/atlas.py selftest                                          # an offline end-to-end run in a temp folder
    python tools/atlas.py export-kit <folder> [--zip] [--check]             # (this repo) the standalone lab-mate kit

Every command takes --project <folder> (default: $ATLAS_PROJECT, else the nearest folder holding atlas.toml from
here up; in this repository with neither, the BlueBoat world-model atlas). tools/atlas_project.py says what a project
is; tools/atlas_intake.py brings papers in; the stages call the existing tools unchanged, with ATLAS_PROJECT /
ATLAS_PAPERS set, so `--bwm` in each tool means THIS project.

WHY (owner, 2026-10-15): "can we further generalize the atlas backbone and make an easy onboarding process for new
papers that does the entire thing from mining to getting figures to updating the atlas to rebuilding the zip. Can you
make a generalized version I can hand off to lab mates so they can add their papers and customize it for their usage?"
-- as a STANDALONE kit (owner's choice: lab mates never see this repository), taking PDFs, Zotero / BibTeX exports and
plain paper lists. The pieces existed (tools/paper_text.py ... report/boat_world_model/build_bwm.py); what was missing
was one entry point, one notion of "project", and a pipeline that knows the order and what is already done.

THE PIPELINE (`update`; a stage runs only when its inputs changed since its last good run -- state and logs in
<pdfs>_atlas/ beside the project's PDF folder):
    scan       which PDFs the project holds              build_bwm.py scan          -> papers/_held.json
    text       page text + BM25 index (new PDFs only)    paper_text.py build        -> <pdfs>_text
    ocr        scanned pages (RapidOCR, CPU), then text   paper_ocr.py run           -> <pdfs>_ocr
    html       arXiv LaTeX HTML (15 s / page; off by default)  arxiv_html.py fetch + parse
    facts      code links, datasets, metrics, hardware    paper_facts.py --bwm       -> papers/_facts.json
    embed      dense vectors (GPU; minilm on a CPU)       paper_embed.py embed       -> <pdfs>_embed
    figures    every figure + caption (YOLO; heuristic)   figure_bank.py detect+build-> <pdfs>_figures
    meta       venue, dates, citations, references        build_bwm.py meta          -> papers/_meta.json  (network)
    citemap    the citation map's layout + figures        build_bwm.py citemap       -> papers/_citemap.json, figures/
    figindex   the Figures tab's digest                   build_bwm.py figures       -> index/atlas_figures.js
    terms      the automatic glossary (verbatim quotes)   term_mine.py glossary      -> lexicon/auto_glossary.json
    build      the atlas, ATLAS.xlsx, the Word copies     build_bwm.py build         -> index/
    lookup     each term's / task's full-text ranking     paper_find.py precompute   -> papers/_lookup.json
    rebuild    the atlas again, with the lookup tables    build_bwm.py build
    zip        the portable bundle (atlas + PDFs + images) build_bwm.py bundle       -> <short_name>_<date>.zip

HONEST SCOPE: the pipeline is as good as the tools it calls (their own docstrings and LIT_PAPER_LOOKUP.md /
LIT_PAPER_MINING.md say what each measured). A project's papers are whatever its owner adds -- intake checks that each
is a real, correctly identified paper, not that it is relevant. Network stages (add, meta, html) need the internet;
OpenAlex allows 1,000 keyless calls a day (a free key -- $OPENALEX_API_KEY -- gives ten times that).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas_project as AP  # noqa: E402
import atlas_setup as SETUP  # noqa: E402

ROOT = AP.ROOT
ENGINE = AP.ENGINE
BUILD = ENGINE / "build_bwm.py"
#: the kit this file runs from (`export-kit` writes KIT.json at the kit's root); None in the repository
KIT_ROOT: Optional[Path] = HERE.parent if (HERE.parent / "KIT.json").is_file() else None

TEMPLATE_TOML = '''# {title} -- the atlas project's settings (tools/atlas_project.py reads them; everything is optional).

[atlas]
title = "{title}"                  # the page's title
subtitle = ""
short_name = "{short}"             # names the zip: <short_name>_<date>.zip (letters, digits, _ - .)
bundle_title = "{title} -- portable bundle"

[paths]
papers = "pdfs"                    # the PDF folder, relative to this file; its caches go beside it (pdfs_text, ...)

# A TOPIC is the name of a papers/<topic>.json file: `atlas.py add --topic sonar ...`, or the subfolder of the inbox a
# PDF sits in. Give each a readable label here (else "Sonar" is made from "sonar").
# [topics.sonar]
# label = "Sonar navigation"

# The citation map colours papers by GROUP (at most 7; topics no group names fall in the last one). With none given,
# each topic is its own group (the six biggest; the rest share "Other topics").
# [[groups]]
# name = "Perception"
# topics = ["sonar", "vision"]

# Tasks the atlas's Find tab offers as one click, ranked by the full text when `update` runs.
# [[tasks]]
# id = "slam"
# title = "Underwater SLAM"
# query = "simultaneous localization and mapping for underwater vehicles with sonar and an IMU"

[pipeline]
gpu = "auto"                       # auto | yes | no -- embeddings, figure detection and the lookup's dense half
python_gpu = ""                    # a Python WITH torch for those stages ("" = the one that runs atlas.py)
embed_model = "auto"               # auto: nemotron-1b with a GPU, minilm (small, fine on a CPU) without
figures = "auto"                   # auto: yolo with a GPU, heuristic (no model) without; "off" to skip
ocr = true                         # OCR scanned pages (RapidOCR, CPU)
arxiv_html = false                 # arXiv's HTML (equations, sections) -- one page per 15 s, so off by default
facts = true                       # code links, datasets, metrics, hardware -- each with its sentence
meta = true                        # OpenAlex: venue, dates, citation counts, reference lists (citation map)
github_stars = false               # GitHub stars of code links (needs `gh auth login`)
semantic_scholar = false           # Semantic Scholar counts (rate-limited without a key)
terms = true                       # the automatic glossary of abbreviations the papers define
bundle = true                      # rebuild the zip at the end of `update`
bundle_pdfs = true
bundle_figures = true
'''

INBOX_README = """Drop papers here, then run `atlas add inbox` (or `atlas.cmd add inbox`) from the project folder.

  * PDFs: in a SUBFOLDER to give them a topic (inbox/sonar/x.pdf -> topic "sonar"), else they land in "unsorted".
  * A Zotero / BibTeX export (.bib, with "Export Files" ticked to bring the PDFs) or Zotero's CSL JSON (.json).
  * A list (.txt / .csv): one DOI, arXiv id or title per line (a second column after a tab or " | " names the topic).

Nothing here is moved or changed: `add` copies each identified PDF into the project's PDF folder as
Surname_Year_Title.pdf and prints what it could not identify (NEEDS A LOOK) with what it found.
"""


# ---------------------------------------------------------------------------------------------- the project
@dataclass
class Project:
    dir: Path
    papers: Path

    @property
    def state(self) -> Path:
        return AP.beside(self.papers, "atlas")

    def env(self) -> Dict[str, str]:
        e = dict(os.environ)
        e.update({AP.ENV_PROJECT: str(self.dir), AP.ENV_PAPERS: str(self.papers), "PYTHONIOENCODING": "utf-8",
                  "PYTHONUTF8": "1"})
        return e


def project_of(arg: Optional[str]) -> Project:
    if arg:
        d = Path(arg).expanduser().resolve()
        if not (d / AP.CONFIG).is_file() and not AP.is_blueboat(d):
            raise SystemExit(f"{d} is not an atlas project (no {AP.CONFIG}): `atlas.py new {d}` makes one")
    else:
        d = AP.project_dir()
        if not d.is_dir():
            raise SystemExit("no atlas project here: run inside a project folder, pass --project <folder>, or "
                             "make one with `atlas.py new <folder>`")
    os.environ[AP.ENV_PROJECT] = str(d)              # this process's tools (intake, paper_find) see it too
    return Project(d, Path(AP.papers_dir(d, legacy=_blueboat_pdfs)).resolve())


def _blueboat_pdfs() -> Path:
    import paper_text as PT
    return PT.DEFAULT_PDFS


def run(cmd: Sequence[str], P: Project, log: Optional[Path] = None, say: Callable = print) -> int:
    """One tool run with the project's environment; its output to `log` (and its last lines to the screen on failure)."""
    if log is None:
        return subprocess.call(list(cmd), env=P.env(), cwd=str(P.dir))
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        fh.write(("\n$ " + " ".join(str(c) for c in cmd) + "\n").encode("utf-8"))
        fh.flush()
        rc = subprocess.call(list(cmd), env=P.env(), cwd=str(P.dir), stdout=fh, stderr=subprocess.STDOUT)
    if rc:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        for ln in tail:
            say(f"      | {ln}")
    return rc


def py(*args) -> List[str]:
    return [sys.executable, *[str(a) for a in args]]


#: what a reader types to run this tool: `atlas` in a kit (its launcher), `python tools/atlas.py` in this repository
ME = "python tools/atlas.py" if AP.BLUEBOAT.is_dir() else "atlas"


def _q(p) -> str:
    """A path as a reader pastes it into a shell: quoted when it holds a space."""
    s = str(p)
    return f'"{s}"' if " " in s else s


LAUNCHER_CMD = """@echo off
rem Paper Atlas: runs the kit at {kit} on THIS project folder (written by `atlas new`).
rem Moved the kit? Run `atlas new <this folder>` from its new place: this file is rewritten, atlas.toml never touched.
call "{kit}\\atlas.cmd" %* --project "%~dp0."
"""
LAUNCHER_SH = """#!/usr/bin/env sh
# Paper Atlas: runs the kit at {kit} on THIS project folder (written by `atlas new`).
# Moved the kit? Run `atlas new <this folder>` from its new place: this file is rewritten, atlas.toml never touched.
exec "{kit}/atlas.sh" "$@" --project "$(cd "$(dirname "$0")" && pwd)"
"""


def write_launchers(d: Path) -> List[Path]:
    """In a kit, a project gets `atlas.cmd` + `atlas.sh` that run the kit ON IT, so `atlas add inbox` works inside the
    project folder (2026-10-15: the guide's quick start cd'd into the project, where no `atlas` existed). The kit's
    path is machine-specific, so the project's .gitignore leaves them out."""
    if KIT_ROOT is None:
        return []
    out = []
    for name, text, kit in (("atlas.cmd", LAUNCHER_CMD, str(KIT_ROOT)), ("atlas.sh", LAUNCHER_SH, KIT_ROOT.as_posix())):
        f = d / name
        f.write_text(text.format(kit=kit), encoding="utf-8", newline="\n")
        out.append(f)
    try:
        (d / "atlas.sh").chmod(0o755)
    except OSError:                                  # a file system without modes: run it with `sh atlas.sh`
        pass
    return out


# ------------------------------------------------------------------------------------------------------ new
def cmd_new(folder: str, title: str = "") -> int:
    d = Path(folder).expanduser().resolve()
    if (d / AP.CONFIG).exists():
        if write_launchers(d):                       # the kit moved, or the project came from someone else
            print(f"{d} already has {AP.CONFIG} (kept); its atlas launcher now runs the kit at {KIT_ROOT}")
        else:
            print(f"{d} already has {AP.CONFIG}: nothing to do")
        return 0
    title = title or d.name.replace("_", " ").replace("-", " ").strip().title() + " Atlas"
    short = re.sub(r"[^A-Za-z0-9_.-]+", "_", d.name).strip("_.-")[:40] or "atlas"
    for sub in ("papers", "refs", "lexicon", "inbox", "pdfs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / AP.CONFIG).write_text(TEMPLATE_TOML.format(title=title.replace('"', "'"), short=short),
                               encoding="utf-8", newline="\n")
    (d / "inbox" / "README.txt").write_text(INBOX_README, encoding="utf-8", newline="\n")
    (d / ".gitignore").write_text("pdfs/\npdfs_*/\n*.zip\ninbox/\n# the kit's launchers: machine-specific (`atlas new <this "
                                  "folder>` from a kit writes them)\natlas.cmd\natlas.sh\n",
                                  encoding="utf-8", newline="\n")
    AP.config(d)                                     # parses the template (a typo here would be ours)
    if write_launchers(d):
        print(f"new atlas project: {d}\n  next: put PDFs (or a .bib / a list) in {d / 'inbox'}, then open a terminal "
              f"in {d} and run\n        atlas add inbox        (PowerShell: .\\atlas add inbox)\n"
              f"        atlas update\n        atlas open")
    else:
        print(f"new atlas project: {d}\n  next: put PDFs (or a .bib / a list) in {d / 'inbox'}, then\n"
              f"        {ME} add inbox --project {_q(d)}\n        {ME} update --project {_q(d)}")
    return 0


# ------------------------------------------------------------------------------------------------------ add
def cmd_add(P: Project, inputs: Sequence[str], topic: str = "", vouch: Sequence[str] = (), offline: bool = False,
            fetch: bool = True, build: bool = True, say: Callable = print) -> int:
    import atlas_intake as I
    today = I.today_str()
    paths = [Path(x) if Path(x).is_absolute() else (P.dir / x if (P.dir / x).exists() else Path(x).resolve())
             for x in inputs]
    items = I.collect(paths, I.slug(topic) if topic else "", say=say)
    if not items:
        say("add: nothing to add (PDFs, .bib, CSL .json or .txt / .csv lists)")
        return 2
    by_name = {it.pdf.name: it for it in items if it.pdf is not None}
    for spec in vouch:
        name, sep, ident = spec.rpartition("=")
        if not sep or name not in by_name:
            say(f"add: --as {spec!r} -- give FILE=DOI or FILE=arXiv-id for one of the PDFs being added")
            return 2
        by_name[name].vouched = ident.strip()
    reg = None if offline else I.Registry(say=say)
    say(f"add: {len(items)} item(s) -> {P.dir}" + (" (OFFLINE: bibliography entries only)" if offline else ""))
    outcomes = []
    for n, it in enumerate(items, 1):
        label = it.pdf.name if it.pdf is not None and it.source == "pdf" else (it.title or it.doi or it.arxiv_id
                                                                              or it.origin)
        try:
            if it.source == "pdf":
                o = (I.Outcome(it, "look", note="offline: a PDF needs the registry to be identified")
                     if reg is None else I.identify_pdf(it, reg))
            else:
                o = I.resolve_entry(it, reg)
        except Exception as e:                       # noqa: BLE001 -- one bad item never stops the batch
            o = I.Outcome(it, "error", note=f"{type(e).__name__}: {str(e)[:160]}")
        outcomes.append(o)
        say(f"  [{n}/{len(items)}] {o.verdict:10} {str(label)[:90]}")
    if fetch and reg is not None:
        I.fetch_open(outcomes, P.state / "fetched", say=say)
    added = I.write(P.dir, outcomes, today, verified=reg is not None, say=say)
    held_keys = {r.get("key") for r in (json.loads((P.dir / "papers" / "_held.json").read_text(encoding="utf-8"))
                                        .get("files") or [])} if (P.dir / "papers" / "_held.json").is_file() else set()
    stage_dir = P.state / "staging" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs = I.stage(outcomes, stage_dir, held_keys)
    rc = 0
    filed = 0
    if build and pairs:
        P.papers.mkdir(parents=True, exist_ok=True)
        cmd = py(BUILD, "add", stage_dir, "--base", P.dir, "--pdfs", P.papers)
        for pr in pairs:
            cmd += ["--as", pr]
        rc = run(cmd, P)
        now = {r.get("key") for r in (json.loads((P.dir / "papers" / "_held.json").read_text(encoding="utf-8"))
                                      .get("files") or [])} if (P.dir / "papers" / "_held.json").is_file() else set()
        filed = len({pr.rsplit("=", 1)[1] for pr in pairs} & (now - held_keys))
        if rc == 0:
            shutil.rmtree(stage_dir, ignore_errors=True)
    elif build and sum(added.values()):
        rc = run(py(BUILD, "build", "--base", P.dir), P)
    rep = P.state / "intake" / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(I.report_md(outcomes, today), encoding="utf-8", newline="\n")
    look = [o for o in outcomes if o.verdict in ("look", "unresolved", "error")]
    say(f"add: {sum(added.values())} new paper(s), {filed} of {len(pairs)} PDF(s) filed, "
        f"{sum(o.verdict == 'present' for o in outcomes)} already in the atlas, {len(look)} need a look -> {rep}")
    for o in look[:15]:
        say(f"  ? {(o.item.pdf.name if o.item.pdf else o.item.origin)[:70]}: {o.note[:150]}")
    if sum(added.values()):
        say(f"  next: {ME} update --project {_q(P.dir)}   (text, figures, citations, the atlas, the zip)")
    return rc or (1 if look else 0)


# ---------------------------------------------------------------------------------------------- the pipeline
@dataclass
class Stage:
    name: str
    about: str
    cmds: Callable[["Ctx"], List[List[str]]]
    inputs: Callable[["Ctx"], str]
    enabled: Callable[["Ctx"], str]                  # "" = run; else the reason it does not
    required: bool = False
    gpu: bool = False                                 # its commands run on ctx.py_gpu


@dataclass
class Ctx:
    P: Project
    cfg: dict
    gpu: bool
    py_gpu: str
    offline: bool
    model: str
    arm: str
    notes: List[str]
    ocr_ok: bool = True


def _digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _read(p: Path) -> bytes:
    try:
        return p.read_bytes()
    except OSError:
        return b""


def _held_sig(c: Ctx) -> str:
    doc = json.loads(_read(c.P.dir / "papers" / "_held.json") or b"{}")
    return _digest(*sorted(f"{r.get('key')}:{r.get('sha256')}" for r in doc.get("files") or []))


def _folder_sig(c: Ctx) -> str:
    rows = sorted(f"{f.name}:{f.stat().st_size}" for f in c.P.papers.glob("*.pdf")) if c.P.papers.is_dir() else []
    return _digest(*rows)


def _files_sig(folder: Path, pattern: str = "*.json", skip: Sequence[str] = ()) -> str:
    return _digest(*[f.name.encode() + b"=" + _read(f) for f in sorted(folder.glob(pattern)) if f.name not in skip])


def _figstore_sig(c: "Ctx") -> str:
    """The figure bank as it stands: every paper's figures.json (path + size) -- <pdfs>_figures/<sha2>/<sha>/."""
    store = AP.beside(c.P.papers, "figures")
    if not store.is_dir():
        return ""
    return _digest(*sorted(f"{f.parent.name}:{f.stat().st_size}" for f in store.glob("*/*/figures.json")))


def _onoff(key: str, needs_net: bool = False, needs_gpu: bool = False) -> Callable[[Ctx], str]:
    def f(c: Ctx) -> str:
        if not c.cfg["pipeline"].get(key, True):
            return f"off in atlas.toml ([pipeline] {key} = false)"
        if needs_net and c.offline:
            return "--offline"
        if needs_gpu and not c.gpu:
            return "no GPU"
        return ""
    return f


def _always(c: Ctx) -> str:
    return ""


def _base_cmd(c: Ctx, *args) -> List[str]:
    return py(BUILD, *args, "--base", c.P.dir)


STAGES: List[Stage] = [
    Stage("scan", "which PDFs the project holds", lambda c: [_base_cmd(c, "scan", "--pdfs", c.P.papers)],
          lambda c: _folder_sig(c) + _files_sig(c.P.dir / "papers", "*.json", ("_held.json",)), _always, True),
    Stage("text", "page text + search index", lambda c: [py(HERE / "paper_text.py", "build", "--pdfs", c.P.papers)],
          _held_sig, _always, True),
    Stage("ocr", "scanned pages (RapidOCR), then their text",
          lambda c: [py(HERE / "paper_ocr.py", "run", "--bwm", "--engine", "rapid", "--pdfs", c.P.papers),
                     py(HERE / "paper_text.py", "build", "--pdfs", c.P.papers)], _held_sig,
          lambda c: _onoff("ocr")(c) or ("" if c.ocr_ok else "RapidOCR is not installed (pip install rapidocr "
                                                              "onnxruntime)")),
    Stage("html", "arXiv LaTeX HTML (one page per 15 s)",
          lambda c: [py(HERE / "arxiv_html.py", "fetch", "--bwm", "--pdfs", c.P.papers),
                     py(HERE / "arxiv_html.py", "parse", "--bwm", "--pdfs", c.P.papers)],
          _held_sig, _onoff("arxiv_html", needs_net=True)),
    Stage("facts", "code links, datasets, metrics, hardware",
          lambda c: [py(HERE / "paper_facts.py", "--bwm", "--pdfs", c.P.papers)], _held_sig, _onoff("facts")),
    Stage("embed", "dense vectors for search", lambda c: [[c.py_gpu, str(HERE / "paper_embed.py"), "embed", "--model",
                                                          c.model, "--bwm", "--pdfs", str(c.P.papers)]],
          lambda c: _held_sig(c) + c.model, lambda c: "" if c.model else "no embedding model", gpu=True),
    Stage("figures", "every figure and its caption",
          lambda c: [[c.py_gpu if (c.arm != "heuristic" and cmd == "detect") else sys.executable,
                      str(HERE / "figure_bank.py"), cmd, "--bwm", "--arm", c.arm, "--pdfs", str(c.P.papers)]
                     for cmd in (("build",) if c.arm == "heuristic" else ("detect", "build"))],
          lambda c: _held_sig(c) + c.arm, lambda c: "" if c.arm != "off" else "off in atlas.toml ([pipeline] figures)",
          gpu=True),
    Stage("meta", "venue, dates, citations, reference lists (OpenAlex)",
          lambda c: [_base_cmd(c, "meta", *([] if c.cfg["pipeline"]["github_stars"] else ["--no-stars"]),
                               *([] if c.cfg["pipeline"]["semantic_scholar"] else ["--no-s2"]))],
          lambda c: _files_sig(c.P.dir / "papers", "*.json", ("_meta.json", "_lookup.json", "_citemap.json",
                                                              "_facts.json", "_held.json")),
          _onoff("meta", needs_net=True)),
    Stage("citemap", "the citation map's layout and figures", lambda c: [_base_cmd(c, "citemap", "--if-changed")],
          lambda c: _digest(_read(c.P.dir / "papers" / "_meta.json")), _always),
    Stage("figindex", "the Figures tab's digest", lambda c: [_base_cmd(c, "figures", "--pdfs", c.P.papers)],
          lambda c: _held_sig(c) + c.arm + _figstore_sig(c),
          lambda c: ("figures are off" if c.arm == "off" else
                     "" if AP.beside(c.P.papers, "figures").is_dir() else "no figure bank yet (the figures stage)")),
    Stage("terms", "the automatic glossary",
          lambda c: [py(HERE / "term_mine.py", "glossary", "--pdfs", c.P.papers, "--min-papers",
                        3 if not AP.is_blueboat(c.P.dir) else 4)],
          lambda c: _held_sig(c) + _files_sig(c.P.dir / "lexicon", "*.json", ("auto_glossary.json",)),
          lambda c: _onoff("terms")(c) or ("" if (c.P.dir / "index" / "papers.json").is_file() else "no index yet")),
    Stage("build", "the atlas, ATLAS.xlsx and the Word copies", lambda c: [_base_cmd(c, "build")],
          lambda c: "always", _always, True),
    Stage("lookup", "each term's and task's full-text ranking",
          lambda c: [[c.py_gpu, str(HERE / "paper_find.py"), "precompute", "--model", c.model, "--pdfs",
                      str(c.P.papers)]],
          lambda c: _held_sig(c) + c.model + _digest(_read(c.P.dir / "index" / "lexicon.json"),
                                                    json.dumps(c.cfg["tasks"], sort_keys=True)),
          lambda c: "" if c.model else "no embedding model", gpu=True),
    Stage("rebuild", "the atlas again, with the lookup tables", lambda c: [_base_cmd(c, "build")],
          lambda c: _digest(_read(c.P.dir / "papers" / "_lookup.json")), _always, True),
    Stage("zip", "the portable bundle",
          lambda c: [_base_cmd(c, "bundle", "--pdfs", c.P.papers,
                               *([] if c.cfg["pipeline"]["bundle_pdfs"] else ["--no-pdfs"]),
                               *([] if c.cfg["pipeline"]["bundle_figures"] else ["--no-figures"]))],
          lambda c: _held_sig(c) + _files_sig(c.P.dir / "index", "*"), _onoff("bundle")),
]
STAGE_NAMES = [s.name for s in STAGES]


def gpu_available(python: str) -> bool:
    try:
        r = subprocess.run([python, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
                           capture_output=True, timeout=180)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def has_module(python: str, mod: str) -> bool:
    try:
        return subprocess.run([python, "-c", f"import {mod}"], capture_output=True, timeout=180).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def context(P: Project, gpu: str = "", offline: bool = False) -> Ctx:
    cfg = AP.config(P.dir)
    pl = cfg["pipeline"]
    py_gpu = os.path.expandvars(pl.get("python_gpu") or "").strip()
    if not py_gpu or "$" in py_gpu or not Path(py_gpu).is_file():   # unset, an unset ${VAR}, or a missing file
        if py_gpu:
            print(f"  note: [pipeline] python_gpu = {pl.get('python_gpu')!r} is not a Python here -- using {sys.executable}")
        py_gpu = sys.executable
    want = (gpu or pl.get("gpu") or "auto").lower()
    if want == "no":
        has = False
    else:
        has = gpu_available(py_gpu)
        if want == "yes" and not has:
            raise SystemExit(f"[pipeline] gpu = yes, but {py_gpu} sees no CUDA GPU (torch missing or no driver): "
                             f"set python_gpu to a Python with CUDA torch, or gpu = \"auto\"")
    torch_ok = has or has_module(py_gpu, "torch")
    model = pl.get("embed_model") or "auto"
    if model == "auto":
        model = ("nemotron-1b" if has else "minilm") if torch_ok else ""
    notes = []
    drv = SETUP.driver_cuda() if want == "auto" and not has else None
    if drv:                                          # 2026-10-15: `setup gpu` had swapped the kit's CUDA torch for a CPU
        notes.append(f"this machine has an NVIDIA GPU (its driver runs CUDA {drv[0]}.{drv[1]}), but {py_gpu} has no "
                     "CUDA torch, so the GPU stages run on the CPU -- " +   # build, and "GPU no" read as normal
                     ("run `setup gpu` again in the kit folder: it installs the CUDA build" if KIT_ROOT is not None
                      else "set [pipeline] python_gpu to a Python with CUDA torch"))
    arm = (pl.get("figures") or "auto").lower()
    if arm == "auto":
        arm = "yolo" if has else "heuristic"
    if arm in ("yolo", "ppdl3"):
        need = "doclayout_yolo" if arm == "yolo" else "transformers"
        if not has_module(py_gpu, need):
            notes.append(f"figures: {need} is not installed in {py_gpu} -- using the model-free heuristic "
                         f"(measured as good, figure_bank.py)")
            arm = "heuristic"
    ocr_ok = has_module(sys.executable, "rapidocr")
    return Ctx(P, cfg, has, py_gpu, offline, model, arm, notes, ocr_ok)


def _state(P: Project) -> dict:
    try:
        return json.loads((P.state / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(P: Project, st: dict) -> None:
    P.state.mkdir(parents=True, exist_ok=True)
    tmp = P.state / "state.json.tmp"
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, P.state / "state.json")


def _sig(stage: Stage, c: Ctx) -> str:
    return _digest(stage.inputs(c), json.dumps(stage.cmds(c), default=str))


def decide(c: Ctx, s: Stage, st: dict, only: Sequence[str] = (), skip: Sequence[str] = (),
           force: bool = False) -> tuple:
    """(action, why) for one stage NOW -- run / skip / done (its inputs are as at its last good run). Decided when the
    stage is reached: an earlier stage of the same update may have changed its inputs."""
    if only and s.name not in only:
        return "skip", "not in --only"
    if s.name in skip:
        return "skip", "--skip"
    why = s.enabled(c)
    if why:
        return "skip", why
    prev = st.get(s.name) or {}
    if not force and prev.get("ok") and prev.get("sig") == _sig(s, c) and s.name not in ("build", "rebuild"):
        return "done", f"up to date since {prev.get('at', '?')}"
    return "run", ""


def plan(c: Ctx, only: Sequence[str] = (), skip: Sequence[str] = (), force: bool = False) -> List[tuple]:
    """[(stage, action, why)] for every stage, as things stand (`status`)."""
    st = _state(c.P)
    return [(s, *decide(c, s, st, only, skip, force)) for s in STAGES]


def cmd_update(P: Project, only: Sequence[str] = (), skip: Sequence[str] = (), force: bool = False, gpu: str = "",
               offline: bool = False, say: Callable = print) -> int:
    c = context(P, gpu, offline)
    say(f"atlas update: {P.dir}\n  PDFs {P.papers}; GPU {'yes' if c.gpu else 'no'}; embeddings "
        f"{c.model or 'none (no torch)'}; figures {c.arm}{'; OFFLINE' if offline else ''}")
    for n in c.notes:
        say(f"  note: {n}")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logs = P.state / "logs"
    st = _state(P)
    failed = []
    for i, s in enumerate(STAGES, 1):
        action, why = decide(c, s, st, only, skip, force)     # NOW: an earlier stage may have changed its inputs
        if action != "run":
            say(f"  [{i:2}/{len(STAGES)}] {s.name:9} {'-' if action == 'skip' else '='} {why}")
            continue
        sig = _sig(s, c)
        t0 = time.time()
        say(f"  [{i:2}/{len(STAGES)}] {s.name:9} ... {s.about}")
        log = logs / f"{stamp}_{s.name}.log"
        rc = 0
        for cmd in s.cmds(c):
            rc = run(cmd, P, log, say)
            if rc:
                break
        dt = time.time() - t0
        if rc:
            say(f"  [{i:2}/{len(STAGES)}] {s.name:9} FAILED (exit {rc}, {dt:.0f} s) -- log: {log}")
            failed.append(s.name)
            st[s.name] = {"ok": False, "sig": sig, "at": stamp, "seconds": round(dt, 1), "log": str(log)}
            _save_state(P, st)
            if s.required:
                say("  stopped: a required stage failed (fix it, then run update again -- done stages are kept)")
                return 1
            continue
        st[s.name] = {"ok": True, "sig": sig, "at": stamp, "seconds": round(dt, 1), "log": str(log)}
        _save_state(P, st)
        say(f"  [{i:2}/{len(STAGES)}] {s.name:9} done in {dt:.0f} s")
    page = P.dir / "index" / "browser.html"
    say(f"atlas update: {'done' if not failed else 'done, with failures: ' + ', '.join(failed)}\n"
        f"  the atlas: {page}  ({ME} open --project {_q(P.dir)} -- with figure images)")
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------ status, find, ...
def cmd_status(P: Project, gpu: str = "", say: Callable = print) -> int:
    c = context(P, gpu)
    rows = sum(len(json.loads(_read(f) or b"[]")) for f in (P.dir / "papers").glob("*.json")
               if not f.name.startswith("_"))
    held = json.loads(_read(P.dir / "papers" / "_held.json") or b"{}").get("files") or []
    cfg = c.cfg
    say(f"project  {P.dir}\n  title  {cfg['atlas']['title']}\n  PDFs   {P.papers}\n  state  {P.state}")
    say(f"  papers {rows} rows in {len(AP.topic_keys(P.dir))} topics; {len(held)} PDFs held; "
        f"{len(cfg['tasks'])} tasks; GPU {'yes' if c.gpu else 'no'}; embeddings {c.model or 'none'}; figures {c.arm}")
    for n in c.notes:
        say(f"  note: {n}")
    st = _state(P)
    for s, action, why in plan(c):
        prev = st.get(s.name) or {}
        last = f"last {'ok' if prev.get('ok') else 'FAILED'} {prev.get('at', '')} ({prev.get('seconds', 0)} s)" \
            if prev else "never run"
        say(f"  {s.name:9} {action:5} {why[:60]:60} {last}")
    return 0


def cmd_find(P: Project, rest: Sequence[str]) -> int:
    return run(py(HERE / "paper_find.py", *rest), P)


def cmd_zip(P: Project, no_pdfs: bool = False, no_figures: bool = False) -> int:
    return run(py(BUILD, "bundle", "--base", P.dir, "--pdfs", P.papers, *(["--no-pdfs"] if no_pdfs else []),
                  *(["--no-figures"] if no_figures else [])), P)


def cmd_open(P: Project, launch: bool = True, say: Callable = print) -> int:
    """A viewing folder <pdfs>_atlas/view: the page as a standalone atlas.html with its two sidecars, and a link to the
    figure bank as figures/ (a junction on Windows, a symlink elsewhere) -- so the Figures tab shows images."""
    idx = P.dir / "index"
    if not (idx / "browser.html").is_file():
        say(f"open: no atlas yet -- run `{ME} update`")
        return 2
    import importlib.util
    spec = importlib.util.spec_from_file_location("_atlas_open_build_bwm", BUILD)
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    view = P.state / "view"
    view.mkdir(parents=True, exist_ok=True)
    (view / "atlas.html").write_text(B.standalone_page((idx / "browser.html").read_text(encoding="utf-8")),
                                     encoding="utf-8", newline="\n")
    for side in ("atlas_citations.js", "atlas_figures.js"):
        if (idx / side).is_file():
            shutil.copy2(idx / side, view / side)
    store = AP.beside(P.papers, "figures")
    link = view / "figures"
    if store.is_dir() and not link.exists():
        try:
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(store)], capture_output=True, check=True)
            else:
                link.symlink_to(store, target_is_directory=True)
        except (OSError, subprocess.CalledProcessError):
            say("  (no figure images: could not link the figure bank into the view folder)")
    page = view / "atlas.html"
    say(f"the atlas: {page}")
    if launch:
        try:
            if os.name == "nt":
                os.startfile(str(page))              # noqa: S606 -- the user's own file, their default browser
            else:
                subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", str(page)])
        except OSError:
            pass
    return 0


# -------------------------------------------------------------------------------------------------- selftest
def _sample_pdf(path: Path, title: str, authors: str, body: str) -> None:
    """A three-page PDF with real text (matplotlib's PDF backend): a title block, authors and an abstract on page 1, a
    captioned plot on page 2, a references page -- over the 20 kB a real paper is (paper_links.pdf_problem)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.92, title, fontsize=15)
        fig.text(0.1, 0.885, authors, fontsize=10)
        fig.text(0.1, 0.84, "Abstract", fontsize=11)
        lines = body.split("\n")
        for i, ln in enumerate(lines + ["1 Introduction"] + [ln + " This sentence continues the method section."
                                                             for ln in lines] * 6):
            fig.text(0.1, 0.81 - 0.018 * i, ln, fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.15, 0.55, 0.7, 0.3])
        xs = [k / 10 for k in range(200)]
        ax.plot(xs, [((k * 37) % 23) / 23 + k / 100 for k in range(200)], lw=1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("drift (m)")
        fig.text(0.15, 0.5, "Figure 1: Drift of the estimate over a survey line.", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.92, "References", fontsize=11)
        for i in range(25):
            fig.text(0.1, 0.89 - 0.02 * i, f"[{i + 1}] A. Author, A cited paper number {i + 1}, Journal of Tests, 20{i % 25:02d}.",
                     fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)


SAMPLE = [
    ("Sonar odometry for small surface vehicles", "Ada Lovelace and Alan Turing", "lovelace2024sonar", 2024,
     "We estimate the motion of an unmanned surface vehicle (USV) from side-scan sonar (SSS).\n"
     "The unmanned surface vehicle (USV) carries a side-scan sonar (SSS) and a low-cost inertial unit.\n"
     "Results on a lake survey show sonar odometry reduces drift without a Doppler velocity log (DVL)."),
    ("Learning world models of boats from logged data", "Grace Hopper", "hopper2025learning", 2025,
     "A world model of an unmanned surface vehicle (USV) is trained from autopilot logs.\n"
     "We compare model-based planning against a side-scan sonar (SSS) survey baseline.\n"
     "The learned model predicts the boat's motion for ten seconds."),
    ("Figures of merit for reef mapping with side-scan sonar", "Katherine Johnson and Dorothy Vaughan",
     "johnson2023figures", 2023,
     "Reef maps from side-scan sonar (SSS) are scored against a reference bathymetry.\n"
     "An unmanned surface vehicle (USV) ran the survey lines at two speeds.\n"
     "We report coverage and resolution for each pass."),
]


def cmd_selftest(keep: bool = False, say: Callable = print) -> int:
    """An OFFLINE end-to-end run: a new project in a temp folder, three generated PDFs brought in through a BibTeX file
    with Zotero-style file links, then update (scan, text, facts, terms, figures + their tab when PyMuPDF is installed,
    build, rebuild, zip -- the stages that need no network or GPU); checks the atlas holds three papers with their PDFs,
    each paper's captioned plot is found, and the zip packs them (2026-10-15: a figure stage that built NOTHING had
    passed this self-test, which never ran it)."""
    tmp = Path(tempfile.mkdtemp(prefix="atlas_selftest_", dir=os.environ.get("SVFR_TMPDIR") or None))
    try:
        proj = tmp / "demo"
        assert cmd_new(str(proj), "Self-test Atlas") == 0
        inbox = proj / "inbox"
        (inbox / "files").mkdir(parents=True, exist_ok=True)
        bib = []
        for title, authors, key, year, body in SAMPLE:
            _sample_pdf(inbox / "files" / f"{key}.pdf", title, authors.replace(" and ", ", "), body)
            bib.append(f"@article{{{key},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{year}}},\n"
                       f"  journal = {{Journal of Tests}},\n  file = {{Full Text PDF:files/{key}.pdf:application/pdf}}\n}}\n")
        (inbox / "library.bib").write_text("\n".join(bib), encoding="utf-8")
        P = project_of(str(proj))
        rc = cmd_add(P, [str(inbox / "library.bib")], topic="tests", offline=True, say=say)
        assert rc == 0, f"add returned {rc}"
        figs = has_module(sys.executable, "fitz") and has_module(sys.executable, "PIL")   # what the stage renders with
        rc = cmd_update(P, only=("scan", "text", "facts", "terms", *(("figures", "figindex") if figs else ()), "build",
                                 "rebuild", "zip"), gpu="no", offline=True, say=say)
        assert rc == 0, f"update returned {rc}"
        papers = json.loads((proj / "index" / "papers.json").read_text(encoding="utf-8"))
        assert len(papers["papers"]) == 3, papers["stats"]
        held = json.loads((proj / "papers" / "_held.json").read_text(encoding="utf-8"))["files"]
        assert len(held) == 3, held
        zips = sorted(P.papers.parent.glob("*.zip"))
        assert zips, "no zip"
        import zipfile
        with zipfile.ZipFile(zips[-1]) as z:
            names = z.namelist()
        assert sum(n.endswith(".pdf") for n in names) == 3, names[:10]
        n_img = sum(n.endswith(".webp") for n in names)
        if figs:                                     # one captioned plot per paper: found, rendered, in the tab + zip
            assert n_img == 3, f"{n_img} figure images in the zip, not 3"
            assert '"figures":3' in (proj / "index" / "atlas_figures.js").read_text(encoding="utf-8")
        page = (proj / "index" / "browser.html").read_text(encoding="utf-8")
        assert "Self-test Atlas" in page and "Sonar odometry" in page
        if KIT_ROOT is not None:                     # the project's launcher reaches the kit and names this project --
            sh = ["cmd", "/c", str(proj / "atlas.cmd")] if os.name == "nt" else ["sh", str(proj / "atlas.sh")]
            env = {k: v for k, v in os.environ.items() if k not in (AP.ENV_PROJECT, AP.ENV_PAPERS)}   # by itself
            r = subprocess.run(sh + ["status", "--gpu", "no"], cwd=str(tmp), env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=600)   # (project_of set the env above)
            assert r.returncode == 0 and str(proj) in r.stdout, f"the project launcher: rc {r.returncode}\n" \
                                                                 f"{r.stdout[-800:]}\n{r.stderr[-800:]}"
        say(f"selftest: OK -- 3 papers in, 3 PDFs held, "
            + (f"{n_img} figures found and rendered, " if figs else "figures NOT checked (PyMuPDF / Pillow missing), ")
            + f"atlas page + zip ({zips[-1].name}) in {proj}")
        return 0
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ export-kit
def cmd_export_kit(out: str, make_zip: bool = False, check: bool = False, say: Callable = print) -> int:
    import atlas_kit as K
    return K.export(Path(out), make_zip=make_zip, check=check, say=say)


# ------------------------------------------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["new", "add", "update", "status", "find", "zip", "open", "selftest",
                                    "export-kit"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--project", default="", help="the project folder (default: $ATLAS_PROJECT, or the nearest "
                                                  "atlas.toml from here up)")
    ap.add_argument("--title", default="", help="new: the atlas's title")
    ap.add_argument("--topic", default="", help="add: the topic for these papers (else the inbox subfolder)")
    ap.add_argument("--as", dest="vouch", action="append", default=[], help="add: FILE=DOI (or arXiv id) -- this PDF "
                                                                            "IS that paper")
    ap.add_argument("--offline", action="store_true", help="add / update: no network")
    ap.add_argument("--no-fetch", action="store_true", help="add: do not download open copies for entries w/o a PDF")
    ap.add_argument("--no-build", action="store_true", help="add: write the rows, file nothing, build nothing")
    ap.add_argument("--only", default="", help=f"update: only these stages ({','.join(STAGE_NAMES)})")
    ap.add_argument("--skip", default="", help="update: skip these stages")
    ap.add_argument("--force", action="store_true", help="update: run the stages even when nothing changed")
    ap.add_argument("--gpu", default="", choices=["", "auto", "yes", "no"], help="update: override [pipeline] gpu")
    ap.add_argument("--no-pdfs", action="store_true", help="zip: the atlas and index only")
    ap.add_argument("--no-figures", action="store_true", help="zip: no figure images")
    ap.add_argument("--no-launch", action="store_true", help="open: print the path only")
    ap.add_argument("--zip", action="store_true", help="export-kit: also zip the kit")
    ap.add_argument("--check", action="store_true", help="export-kit: run the kit's selftest from the exported copy")
    ap.add_argument("--keep", action="store_true", help="selftest: keep the temp project")
    a, rest = ap.parse_known_args(argv)
    if a.cmd == "new":
        if len(a.args) != 1:
            ap.error("new takes one folder")
        return cmd_new(a.args[0], a.title)
    if a.cmd == "selftest":
        return cmd_selftest(keep=a.keep)
    if a.cmd == "export-kit":
        if len(a.args) != 1:
            ap.error("export-kit takes the folder to write the kit to")
        return cmd_export_kit(a.args[0], make_zip=a.zip, check=a.check)
    P = project_of(a.project or None)
    if a.cmd == "add":
        if not a.args:
            ap.error("add takes PDFs, folders, .bib / .json exports or .txt / .csv lists")
        return cmd_add(P, a.args, topic=a.topic, vouch=a.vouch, offline=a.offline, fetch=not a.no_fetch,
                       build=not a.no_build)
    if a.cmd == "update":
        bad = [x for x in (a.only + "," + a.skip).split(",") if x and x not in STAGE_NAMES]
        if bad:
            ap.error(f"unknown stage(s) {bad}; stages: {', '.join(STAGE_NAMES)}")
        return cmd_update(P, [x for x in a.only.split(",") if x], [x for x in a.skip.split(",") if x], a.force,
                          a.gpu, a.offline)
    if a.cmd == "status":
        return cmd_status(P, a.gpu)
    if a.cmd == "find":
        return cmd_find(P, list(a.args) + list(rest) + (["--topic", a.topic] if a.topic else []))
    if a.cmd == "zip":
        return cmd_zip(P, a.no_pdfs, a.no_figures)
    if a.cmd == "open":
        return cmd_open(P, launch=not a.no_launch)
    return 2


if __name__ == "__main__":
    sys.exit(main())
