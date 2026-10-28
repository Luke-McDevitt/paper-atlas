r"""atlas_app -- the Paper Atlas APP: double-click it and the atlas opens in your browser with a toolbar -- Add papers,
Update, Status, Discover, Make the zip, Settings -- so a lab mate never needs a terminal. (2026-10-27)

    double-click atlas-app.cmd (Windows) / ./atlas-app.sh (macOS, Linux) in the kit folder
    atlas app                                         (in a project folder: the app on THAT project)
    python tools/atlas_app.py [--project DIR] [--port N] [--no-browser] [--idle-exit S]

WHY (owner, 2026-10-27): "can you work on making this launcher into a real release that we can add to our atlas
github repo (with buttons to ingress papers or to add api keys to openalex, etc?)". The kit (tools/atlas.py) does
everything from a terminal; this is the same commands behind buttons, for people who never open one.

WHAT IT IS: a stdlib-only server on 127.0.0.1 (the pattern of tools/library/atlas_browser.py, written out again here
rather than imported: that file is copied ALONE onto the lab library disk and must stay self-contained, and the kit
must not carry the lab's library tools) serving
  * `/`            the app page: the toolbar, the atlas below it in a frame, an output panel for the running job;
  * `/view/...`    the project's atlas view (`atlas open`'s folder <pdfs>_atlas/view: atlas.html, its sidecars, the
                   figure bank linked in as figures/), read-only; no directory listings;
  * `/api/...`     JSON. Every button is a JOB: `tools/atlas.py <cmd> ... --project <project>` run by the Python that
                   runs this app (python.exe beside a pythonw.exe), one job at a time, its output streamed to the page
                   (`GET /api/job?since=N`) and appended to the app's log; Cancel kills the job's whole process tree.
                   Jobs that change the atlas end by re-making the view (`atlas open --no-launch`), and the frame
                   reloads.

THE BUTTONS -> the commands they run (argv lists, never a shell):
    Add papers   add <folder | files | a pasted list saved as inbox/list_<time>.txt> [--topic=T] [--offline]
    Update       update [--force] [--offline] [--gpu yes|no]
    Status       status
    Discover     discover "<task>" [--title=T];  then, per discovery:  discover <slug> --add [--section]
    Make zip     zip [--no-pdfs]
    New / Open   new <folder> [--title=T]; or switch to a folder holding atlas.toml (remembered: Recent)

SETTINGS (per user, OUTSIDE every project): `settings.json` in $ATLAS_APP_HOME, else %APPDATA%\paper-atlas (Windows),
~/Library/Application Support/paper-atlas (macOS), $XDG_CONFIG_HOME or ~/.config /paper-atlas (elsewhere); created
owner-only (0600) where the file system has modes. It holds the OpenAlex API key, the contact e-mail, the GPU choice
and the recent projects. The key and the address reach a job ONLY as its environment ($OPENALEX_API_KEY,
$ATLAS_CONTACT_EMAIL) -- never in a command line, a URL or the page: the page is told "set, ends ...abcd" and a masked
address; every output line is scrubbed of both before it is shown or logged.
  HONEST: on Windows tools/openalex_key.py prefers a key STORED with `setx` over the process's, so a setx key wins over
  the app's; the Settings dialog says when one is stored (never its value).

FILE PICKER: a helper process (`atlas_app.py --pick folder|files`) shows tkinter's native dialog and prints the chosen
paths as JSON. Chosen because tkinter ships with the python.org installers (Windows, macOS) -- no extra GUI toolkit for
the kit -- and a helper process keeps Tk on its own main thread, away from the threaded server. Without tkinter (some
Linux Pythons: `apt install python3-tk`) the page says so and takes a pasted path instead.

SECURITY: binds 127.0.0.1 only; a request whose Host is not loopback is refused (DNS rebinding), and so is a POST from
another origin, without the per-run token the app page carries (`X-Atlas-Token`; a page elsewhere cannot read it), or
that is not JSON. Paths from the page must exist (a paper input: a folder or a .pdf/.bib/.json/.txt/.csv file; a
project: a folder holding atlas.toml) and are passed resolved and absolute, so none reads as an option; a topic, title
or task that starts with '-' is refused.

THE WINDOW: Edge or Chrome with `--app=` (a window of its own, as tools/farm/land_board_launch.py opens the landing
board) when one is installed -- Edge ships with Windows 10 / 11 -- else the default browser; `--tab` asks for a tab.

ONE APP PER USER AND KIT: `app.json` beside the settings holds the port, pid and token; a second launch that finds the
app answering /api/ping for the same kit opens the browser on it (switching it to --project when given). The app
exits by itself once no page has polled it for --idle-exit seconds (default 900) and no job or picker is open.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hmac
import http.client
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas_project as AP  # noqa: E402
if str(HERE.parent) not in sys.path:                          # run by path: the repo / kit root, for tools/
    sys.path.insert(0, str(HERE.parent))
from tools.atomic_replace import replace_retry  # noqa: E402

APP = "paper-atlas-app"
ATLAS = HERE / "atlas.py"
KIT = HERE.parent
DEFAULT_PORT = 8770
IDLE_EXIT_S = 900.0
LOOPBACK = {"127.0.0.1", "localhost", "[::1]"}
KEY_SHAPE = re.compile(r"[A-Za-z0-9._~+/=-]{8,256}")         # tools/openalex_key.py's: one plain token
EMAIL_SHAPE = re.compile(r"[^\s@<>\"']{1,64}@[^\s@<>\"']{1,190}\.[A-Za-z]{2,24}")
PAPER_SUFFIXES = (".pdf", ".bib", ".json", ".txt", ".csv")
GPU_CHOICES = ("", "yes", "no")                                # "" = as the project's atlas.toml says
MAX_BODY = 1_000_000
MAX_LINES = 3000


# ------------------------------------------------------------------------------------------ settings
def settings_dir(env: Optional[dict] = None, platform: str = sys.platform) -> Path:
    environ = os.environ if env is None else env
    if environ.get("ATLAS_APP_HOME"):
        return Path(environ["ATLAS_APP_HOME"])
    if platform.startswith("win") and environ.get("APPDATA"):
        return Path(environ["APPDATA"]) / "paper-atlas"
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "paper-atlas"
    base = environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "paper-atlas"


def mask_key(key: str) -> str:
    return "" if not key else (f"set (ends ...{key[-4:]})" if len(key) >= 12 else "set")


def mask_email(addr: str) -> str:
    if "@" not in addr:
        return ""
    user, _, domain = addr.rpartition("@")
    return f"{user[:1]}***@{domain}"


def _setx_has(name: str) -> bool:
    """Whether Windows stores a user variable `name` (setx) -- its presence only, never its value."""
    if os.name != "nt" or os.environ.get("PYTEST_VERSION"):
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return bool(str(winreg.QueryValueEx(k, name)[0] or "").strip())
    except OSError:
        return False


class Settings:
    """settings.json: {openalex_api_key, contact_email, gpu, recent: [project folders]} -- read per call."""

    def __init__(self, folder: Path, setx_has: Callable[[str], bool] = _setx_has):
        self.folder, self.path, self.setx_has = Path(folder), Path(folder) / "settings.json", setx_has
        self.lock = threading.Lock()

    def load(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        d = d if isinstance(d, dict) else {}
        return {"openalex_api_key": str(d.get("openalex_api_key") or ""), "contact_email": str(d.get("contact_email")
                                                                                                or ""),
                "gpu": d.get("gpu") if d.get("gpu") in GPU_CHOICES else "",
                "recent": [str(p) for p in d.get("recent") or [] if isinstance(p, str)][:8]}

    def _save(self, d: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".settings.{os.getpid()}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # owner-only before a byte is written
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(d, indent=1) + "\n")
        replace_retry(tmp, self.path)

    def public(self) -> dict:
        """What the page may see: whether each secret is set, masked; never a value."""
        d = self.load()
        return {"openalex_api_key": mask_key(d["openalex_api_key"]), "contact_email": mask_email(d["contact_email"]),
                "gpu": d["gpu"], "setx_key": self.setx_has("OPENALEX_API_KEY"), "file": str(self.path)}

    def update(self, body: dict) -> Tuple[int, str]:
        with self.lock:
            d = self.load()
            if "openalex_api_key" in body:
                k = str(body.get("openalex_api_key") or "").strip()
                if k and not KEY_SHAPE.fullmatch(k):
                    return 400, "that is not an OpenAlex key (one token: letters, digits and ._~+/=- only)"
                d["openalex_api_key"] = k
            if "contact_email" in body:
                e = str(body.get("contact_email") or "").strip()
                if e and not EMAIL_SHAPE.fullmatch(e):
                    return 400, "that is not an e-mail address"
                d["contact_email"] = e
            if "gpu" in body:
                if body.get("gpu") not in GPU_CHOICES:
                    return 400, "gpu is one of: (the project's setting), yes, no"
                d["gpu"] = body["gpu"]
            self._save(d)
        return 200, "saved"

    def remember(self, project: Path) -> None:
        with self.lock:
            d = self.load()
            p = str(project)
            d["recent"] = [p] + [r for r in d["recent"] if os.path.normcase(r) != os.path.normcase(p)][:7]
            try:
                self._save(d)
            except OSError:
                pass

    def secrets(self) -> List[str]:
        d = self.load()
        return [s for s in (d["openalex_api_key"], d["contact_email"]) if len(s) >= 4]

    def child_env(self, base: Optional[dict] = None) -> Dict[str, str]:
        env = {k: v for k, v in (os.environ if base is None else base).items() if not k.startswith("GIT_")}
        for k in ("ATLAS_PROJECT", "ATLAS_PAPERS"):                       # the job names its project itself
            env.pop(k, None)
        d = self.load()
        if d["openalex_api_key"]:
            env["OPENALEX_API_KEY"] = d["openalex_api_key"]
        if d["contact_email"]:
            env["ATLAS_CONTACT_EMAIL"] = d["contact_email"]
        env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        return env


def redact(line: str, hidden: Sequence[str]) -> str:
    for s in hidden:
        if s:
            line = line.replace(s, "[hidden]")
    return line


# ------------------------------------------------------------------------------------------ jobs
class Step(NamedTuple):
    argv: List[str]
    show: str                       # what the output panel prints for it: "atlas update --force"
    always: bool = False            # run even after an earlier step failed (re-making the view)
    ok: Tuple[int, ...] = (0,)


def console_python(exe: str) -> str:
    """python.exe beside a pythonw.exe (a job's output is read through a pipe; pythonw has no stdout)."""
    p = Path(exe)
    if p.name.lower() == "pythonw.exe" and (p.parent / "python.exe").is_file():
        return str(p.parent / "python.exe")
    return exe


def _kill_tree(proc) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass


class Jobs:
    """One job at a time: its steps run in a thread, their output kept (the last MAX_LINES lines, numbered so the page
    asks only for new ones), scrubbed of the settings' secrets and appended to `log`."""

    def __init__(self, hidden: Callable[[], Sequence[str]] = lambda: (), log: Optional[Path] = None,
                 popen: Callable = subprocess.Popen, kill: Callable = _kill_tree):
        self.hidden, self.log, self.popen, self.kill = hidden, log, popen, kill
        self.lock = threading.Lock()
        self.name = ""
        self.running = self.cancelled = False
        self.started = self.finished = None
        self.code: Optional[int] = None
        self.lines: List[str] = []
        self.first = 0
        self.proc = None
        self.thread: Optional[threading.Thread] = None

    def status(self, since: int = 0) -> dict:
        with self.lock:
            lo = max(since - self.first, 0)
            return {"name": self.name, "running": self.running, "cancelled": self.cancelled, "code": self.code,
                    "started": self.started, "finished": self.finished, "first": self.first,
                    "next": self.first + len(self.lines), "lines": self.lines[lo:]}

    def start(self, name: str, steps: Sequence[Step], env: Dict[str, str], cwd: Path,
              on_done: Optional[Callable[[int], None]] = None) -> Tuple[int, str]:
        with self.lock:
            if self.running:
                return 409, f"'{self.name}' is still running -- wait for it, or cancel it"
            self.name, self.running, self.cancelled, self.code, self.finished = name, True, False, None, None
            self.started = _now()
            self.first += len(self.lines)
            self.lines = []
        self.thread = threading.Thread(target=self._run, args=(list(steps), env, cwd, on_done), daemon=True)
        self.thread.start()
        return 202, f"{name} started"

    def cancel(self) -> Tuple[int, str]:
        with self.lock:
            if not self.running:
                return 409, "nothing is running"
            self.cancelled = True
            proc = self.proc
        if proc is not None:
            self.kill(proc)
        return 200, "cancelling"

    def say(self, line: str) -> None:
        line = redact(line.rstrip("\r\n"), self.hidden())
        with self.lock:
            self.lines.append(line)
            if len(self.lines) > MAX_LINES:
                drop = len(self.lines) - MAX_LINES
                del self.lines[:drop]
                self.first += drop
        if self.log is not None:
            try:
                self.log.parent.mkdir(parents=True, exist_ok=True)
                if self.log.is_file() and self.log.stat().st_size > 5_000_000:
                    replace_retry(self.log, self.log.with_suffix(".log.1"), may_drop=True)   # a held log: start a new one
                with open(self.log, "a", encoding="utf-8") as fh:
                    fh.write(f"{_now()} {line}\n")
            except OSError:
                pass

    def _run(self, steps: List[Step], env: Dict[str, str], cwd: Path, on_done) -> None:
        failed: Optional[int] = None
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        extra = {} if os.name == "nt" else {"start_new_session": True}     # a process group Cancel can end
        try:
            for st in steps:
                if self.cancelled or (failed is not None and not st.always):
                    continue
                self.say("$ " + st.show)
                try:
                    p = self.popen(list(st.argv), cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                                   errors="replace", creationflags=flags, **extra)
                except OSError as e:
                    self.say(f"could not start: {e}")
                    failed = 2 if failed is None else failed
                    continue
                with self.lock:
                    self.proc = p
                for line in p.stdout:
                    self.say(line)
                rc = p.wait()
                with self.lock:
                    self.proc = None
                if self.cancelled:
                    self.say("CANCELLED")
                elif rc not in st.ok:
                    if st.always:
                        self.say(f"(exit {rc})")
                    else:
                        self.say(f"FAILED: exit {rc}")
                        failed = rc
        finally:
            code = -1 if self.cancelled else (failed or 0)
            with self.lock:
                self.running, self.code, self.finished = False, code, _now()
            if on_done is not None:
                try:
                    on_done(code)
                except Exception as e:                       # noqa: BLE001 -- a hook must not kill the thread
                    self.say(f"(after the job: {e})")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------ checking what the page sends
class Refused(ValueError):
    pass


def _plain(text: str, what: str, limit: int, empty_ok: bool = True) -> str:
    t = " ".join(str(text or "").split())
    if not t and not empty_ok:
        raise Refused(f"give the {what}")
    if len(t) > limit:
        raise Refused(f"the {what} is longer than {limit} characters")
    if t.startswith("-"):
        raise Refused(f"the {what} cannot start with '-'")
    return t


def check_topic(t: str) -> str:
    t = _plain(t, "topic", 60)
    if t and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", t):
        raise Refused("a topic is letters, digits, spaces, _ and - (it names papers/<topic>.json)")
    return t


def check_inputs(paths: Sequence[str]) -> List[Path]:
    if not isinstance(paths, list) or not paths:
        raise Refused("choose a folder or files to add (or paste a list)")
    if len(paths) > 500:
        raise Refused("more than 500 inputs at once -- choose their folder instead")
    out = []
    for raw in paths:
        s = str(raw or "").strip().strip('"')
        if not s or "\0" in s:
            raise Refused("an empty path")
        p = Path(s).expanduser()
        if not p.is_absolute():
            raise Refused(f"not a full path: {s}")
        p = p.resolve()
        if p.is_dir() or (p.is_file() and p.suffix.lower() in PAPER_SUFFIXES):
            out.append(p)
        elif p.is_file():
            raise Refused(f"{p.name}: the kit reads PDFs, .bib / .json exports and .txt / .csv lists")
        else:
            raise Refused(f"no such file or folder: {p}")
    return out


def check_project(path: str) -> Path:
    s = str(path or "").strip().strip('"')
    if not s or "\0" in s or not Path(s).expanduser().is_absolute():
        raise Refused("give the full path of an atlas project folder")
    p = Path(s).expanduser().resolve()
    if not (p / AP.CONFIG).is_file():
        raise Refused(f"{p} holds no {AP.CONFIG} -- not an atlas project (New makes one)")
    return p


def check_new_folder(path: str) -> Path:
    s = str(path or "").strip().strip('"')
    if not s or "\0" in s or not Path(s).expanduser().is_absolute():
        raise Refused("give the full path of the new project's folder")
    p = Path(s).expanduser().resolve()
    if not p.parent.is_dir():
        raise Refused(f"its parent folder does not exist: {p.parent}")
    if p.exists() and not p.is_dir():
        raise Refused(f"{p} is a file")
    return p


# ------------------------------------------------------------------------------------------ the app
class App:
    """The current project, the settings and the job runner -- every POST /api/<name> is `post(name, body)`."""

    def __init__(self, settings: Settings, jobs: Jobs, python: str = "", project: Optional[Path] = None,
                 picker: Optional[Callable[[str, Optional[Path]], Tuple[int, dict]]] = None,
                 opener: Optional[Callable[[Path], None]] = None):
        self.settings, self.jobs = settings, jobs
        self.python = python or console_python(sys.executable)
        self.picker = picker or run_picker
        self.opener = opener or open_folder
        self.project: Optional[Path] = None
        self.view_stamp = 0.0
        self.picking = threading.Lock()
        if project is not None:
            self.set_project(project)

    # -- the project
    def set_project(self, p: Path) -> None:
        self.project = Path(p).resolve()
        self.settings.remember(self.project)

    def papers(self) -> Path:
        return Path(AP.papers_dir(self.project)).resolve()

    def view_dir(self) -> Optional[Path]:
        return None if self.project is None else AP.beside(self.papers(), "atlas") / "view"

    def view_ready(self) -> bool:
        v, page = self.view_dir(), self.project and self.project / "index" / "browser.html"
        return bool(v and (v / "atlas.html").is_file() and page and page.is_file()
                    and (v / "atlas.html").stat().st_mtime >= page.stat().st_mtime)

    def _cmd(self, *args: str, show: str = "", always: bool = False, ok: Tuple[int, ...] = (0,)) -> Step:
        argv = [self.python, str(ATLAS), *args, "--project", str(self.project)]
        return Step(argv, "atlas " + (show or " ".join(args)), always, ok)

    def _view_step(self) -> Step:
        return self._cmd("open", "--no-launch", show="open (the atlas view)", always=True, ok=(0, 2))

    def _gpu(self) -> List[str]:
        g = self.settings.load()["gpu"]
        return ["--gpu", g] if g else []

    def state(self) -> dict:
        out = {"app": APP, "kit": str(KIT), "settings": self.settings.public(), "job": self.jobs.status(10 ** 9),
               "recent": [r for r in self.settings.load()["recent"] if Path(r, AP.CONFIG).is_file()],
               "picker": tk_available(), "project": None}
        if self.project is None:
            return out
        P = self.project
        try:
            title = AP.names(P)["title"]
            problem = ""
        except Exception as e:                                 # noqa: BLE001 -- a broken atlas.toml: say what
            title, problem = P.name, f"{AP.CONFIG}: {e}"
        topics = AP.topic_keys(P)
        disc = []
        for d in sorted((P / "discover").glob("*/DISCOVER.md"), key=lambda f: -f.stat().st_mtime)[:20]:
            try:
                plan = json.loads((d.parent / "plan.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                plan = {}
            disc.append({"slug": d.parent.name, "title": str(plan.get("title") or plan.get("task") or d.parent.name),
                         "add_list": (d.parent / "add_list.txt").is_file()})
        zips = []
        try:
            short = AP.names(P)["short_name"]
            zips = sorted(self.papers().parent.glob(f"{short}_*.zip"), key=lambda f: -f.stat().st_mtime)
        except Exception:                                      # noqa: BLE001
            pass
        out["project"] = {"path": str(P), "title": title, "problem": problem, "topics": topics,
                          "has_atlas": (P / "index" / "browser.html").is_file(), "view_ready": self.view_ready(),
                          "view_stamp": self.view_stamp, "discoveries": disc,
                          "zip": str(zips[0]) if zips else "", "inbox": str(P / "inbox")}
        return out

    # -- POST /api/<name>
    def post(self, name: str, body: dict) -> Tuple[int, dict]:
        try:
            fn = getattr(self, "api_" + name.replace("-", "_"), None)
            if fn is None:
                return 404, {"error": "not found"}
            return fn(body)
        except Refused as e:
            return 400, {"error": str(e)}

    def _need_project(self) -> Path:
        if self.project is None:
            raise Refused("open or make a project first")
        return self.project

    def _start(self, name: str, steps: Sequence[Step], on_done=None) -> Tuple[int, dict]:
        def done(code: int) -> None:
            self.view_stamp = time.time()
            if on_done is not None:
                on_done(code)
        code, msg = self.jobs.start(name, steps, self.settings.child_env(), self.project or KIT, done)
        return code, {"message": msg} if code < 400 else {"error": msg}

    def api_project(self, body: dict) -> Tuple[int, dict]:
        if self.jobs.status()["running"]:
            raise Refused("a job is running -- switch projects when it ends")
        self.set_project(check_project(body.get("path")))
        if not self.view_ready() and (self.project / "index" / "browser.html").is_file():
            self._start("view", [self._view_step()])
        return 200, {"message": f"opened {self.project}"}

    def api_new(self, body: dict) -> Tuple[int, dict]:
        folder = check_new_folder(body.get("folder"))
        if (folder / AP.CONFIG).is_file():
            return self.api_project({"path": str(folder)})
        title = _plain(body.get("title"), "title", 120)
        argv = [self.python, str(ATLAS), "new", str(folder)] + ([f"--title={title}"] if title else [])
        step = Step(argv, "atlas new " + folder.name + (f' --title "{title}"' if title else ""))

        def opened(code: int) -> None:
            if code == 0 and (folder / AP.CONFIG).is_file():
                self.set_project(folder)
        return self._start("new project", [step], opened)

    def api_add(self, body: dict) -> Tuple[int, dict]:
        P = self._need_project()
        topic = check_topic(body.get("topic"))
        listed = str(body.get("list_text") or "")
        inputs = [x for x in body.get("inputs") or [] if str(x or "").strip()] if isinstance(body.get("inputs"),
                                                                                            list) else []
        if listed.strip():
            if len(listed) > 200_000 or "\0" in listed or listed.count("\n") > 5000:
                raise Refused("the pasted list is too long (5,000 lines at most)")
            f = P / "inbox" / f"list_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("\n".join(ln.strip() for ln in listed.splitlines() if ln.strip()) + "\n",
                         encoding="utf-8", newline="\n")
            inputs.append(str(f))
        paths = check_inputs(inputs)
        args = ["add", *map(str, paths)] + ([f"--topic={topic}"] if topic else []) + (
            ["--offline"] if body.get("offline") else [])
        show = "add " + " ".join(p.name for p in paths[:4]) + (f" (+{len(paths) - 4} more)" if len(paths) > 4 else "")
        show += (f" --topic {topic}" if topic else "") + (" --offline" if body.get("offline") else "")
        return self._start("add papers", [self._cmd(*args, show=show), self._view_step()])

    def api_update(self, body: dict) -> Tuple[int, dict]:
        self._need_project()
        args = ["update", *(["--force"] if body.get("force") else []), *(["--offline"] if body.get("offline") else []),
                *self._gpu()]
        return self._start("update", [self._cmd(*args), self._view_step()])

    def api_status(self, body: dict) -> Tuple[int, dict]:
        self._need_project()
        return self._start("status", [self._cmd("status", *self._gpu())])

    def api_view(self, body: dict) -> Tuple[int, dict]:
        self._need_project()
        return self._start("view", [self._view_step()])

    def api_discover(self, body: dict) -> Tuple[int, dict]:
        self._need_project()
        task = _plain(body.get("task"), "task", 2000, empty_ok=False)
        if len(task) < 12:
            raise Refused("describe the task in a sentence or two")
        title = _plain(body.get("title"), "title", 120)
        args = ["discover", task, *([f"--title={title}"] if title else []), *self._gpu()]
        return self._start("discover", [self._cmd(*args, show=f'discover "{task[:60]}..."')])

    def api_discover_add(self, body: dict) -> Tuple[int, dict]:
        P = self._need_project()
        slug = str(body.get("slug") or "")
        if not re.fullmatch(r"[a-z0-9_]{1,80}", slug) or not (P / "discover" / slug / "plan.json").is_file():
            raise Refused("no such discovery in this project")
        args = ["discover", slug, "--add", *(["--section"] if body.get("section") else [])]
        return self._start("add discovered papers", [self._cmd(*args), self._view_step()])

    def api_zip(self, body: dict) -> Tuple[int, dict]:
        self._need_project()
        return self._start("zip", [self._cmd("zip", *(["--no-pdfs"] if body.get("no_pdfs") else []))])

    def api_cancel(self, body: dict) -> Tuple[int, dict]:
        code, msg = self.jobs.cancel()
        return code, {"message": msg} if code < 400 else {"error": msg}

    def api_settings(self, body: dict) -> Tuple[int, dict]:
        code, msg = self.settings.update(body if isinstance(body, dict) else {})
        return code, ({"message": msg, "settings": self.settings.public()} if code < 400 else {"error": msg})

    def api_pick(self, body: dict) -> Tuple[int, dict]:
        kind = body.get("kind")
        if kind not in ("folder", "files"):
            raise Refused("kind is folder or files")
        if not self.picking.acquire(blocking=False):
            return 409, {"error": "a file dialog is already open (it may be behind this window)"}
        try:
            start = self.project / "inbox" if self.project and (self.project / "inbox").is_dir() else self.project
            return self.picker(kind, start)
        finally:
            self.picking.release()

    def api_reveal(self, body: dict) -> Tuple[int, dict]:
        P = self._need_project()
        what = body.get("what")
        st = self.state()["project"]
        target = {"project": P, "inbox": P / "inbox", "zip": Path(st["zip"]).parent if st["zip"] else None}.get(what)
        if target is None and str(what or "").startswith("discover:"):
            slug = str(what)[9:]
            if re.fullmatch(r"[a-z0-9_]{1,80}", slug):
                target = P / "discover" / slug
        if target is None or not target.is_dir():
            raise Refused("nothing to show")
        self.opener(target)
        return 200, {"message": f"opened {target}"}

    def discovery_text(self, slug: str, name: str) -> Optional[str]:
        if self.project is None or not re.fullmatch(r"[a-z0-9_]{1,80}", slug or "") or name not in (
                "DISCOVER.md", "GET_THESE.md", "add_list.txt"):
            return None
        f = self.project / "discover" / slug / name
        return f.read_text(encoding="utf-8", errors="replace")[:400_000] if f.is_file() else None


# ------------------------------------------------------------------------------------------ the file picker
def tk_available() -> bool:
    return importlib.util.find_spec("tkinter") is not None


def run_picker(kind: str, start: Optional[Path]) -> Tuple[int, dict]:
    """The native dialog in a helper process (Tk on its own main thread) -> (200, {"paths": [...]}) or an error."""
    if not tk_available():
        return 501, {"error": "no file dialog here (this Python has no tkinter) -- paste the path instead"}
    argv = [console_python(sys.executable), str(Path(__file__).resolve()), "--pick", kind]
    if start is not None and Path(start).is_dir():
        argv += ["--initial", str(start)]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=900,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        got = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        return 500, {"error": f"the file dialog failed: {e}"}
    paths = [p for p in got.get("paths") or [] if isinstance(p, str) and p]
    return 200, {"paths": paths}


def pick_main(kind: str, initial: str = "") -> int:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    opts = {"parent": root, "initialdir": initial or None}
    if kind == "folder":
        got = filedialog.askdirectory(title="Paper Atlas -- choose a folder", mustexist=False, **opts)
        paths = [got] if got else []
    else:
        got = filedialog.askopenfilenames(title="Paper Atlas -- choose papers or a paper list", filetypes=[
            ("PDFs, Zotero / BibTeX exports, lists", " ".join("*" + s for s in PAPER_SUFFIXES)), ("All files", "*")],
            **opts)
        paths = list(got or [])
    root.destroy()
    print(json.dumps({"paths": [str(Path(p)) for p in paths]}))
    return 0


#: Chromium-family browsers with `--app=` (a window of its own: no tabs, no address bar), as tools/farm/land_board_launch
_APP_BROWSERS = ((("Microsoft", "Edge", "Application", "msedge.exe"), "msedge"),
                 (("Google", "Chrome", "Application", "chrome.exe"), "chrome"))
_MAC_BROWSERS = ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                 "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
_PATH_BROWSERS = ("msedge", "chrome", "microsoft-edge", "google-chrome", "chromium", "chromium-browser")


def find_app_browser(env: Optional[dict] = None, which: Callable = shutil.which,
                     platform: str = sys.platform) -> Optional[str]:
    environ = os.environ if env is None else env
    upper = {str(k).upper(): v for k, v in environ.items()}
    if platform.startswith("win"):
        for parts, _ in _APP_BROWSERS:
            for var in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
                if upper.get(var) and Path(upper[var]).joinpath(*parts).is_file():
                    return str(Path(upper[var]).joinpath(*parts))
    if platform == "darwin":
        for exe in _MAC_BROWSERS:
            if Path(exe).is_file():
                return exe
    for name in _PATH_BROWSERS:
        if which(name):
            return which(name)
    return None


def open_page(url: str, tab: bool = False, popen: Callable = subprocess.Popen,
              fallback: Callable = webbrowser.open) -> str:
    """The app in a window of its own ("app": Edge / Chrome --app), else -- or with `tab` -- a browser tab ("tab")."""
    exe = None if tab else find_app_browser()
    if exe:
        kw = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
              | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        try:
            popen([exe, f"--app={url}", "--window-size=1360,900"], stdin=subprocess.DEVNULL,
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, **kw)
            return "app"
        except OSError:
            pass
    fallback(url)
    return "tab"


def open_folder(p: Path) -> None:
    if os.name == "nt":
        os.startfile(str(p))                               # noqa: S606 -- a folder of the user's own project
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])


# ------------------------------------------------------------------------------------------ the server
class Handler(SimpleHTTPRequestHandler):
    app: App = None
    token: str = ""
    touch: Callable[[], None] = staticmethod(lambda: None)

    def log_message(self, fmt, *args):     # pythonw has no stderr; requests are not worth a log
        pass

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "").rsplit(":", 1)[0].lower() in LOOPBACK

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def end_headers(self):
        if getattr(self, "_view", False):
            self.send_header("Cache-Control", "no-cache")      # an updated atlas is seen on reload
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404, "no listings")
        return None

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "loopback only"})
        u = urlsplit(self.path)
        q = parse_qs(u.query)
        self.touch()
        if u.path == "/api/ping":
            return self._json(200, {"app": APP, "pid": os.getpid(), "kit": str(KIT)})
        if u.path == "/api/state":
            return self._json(200, self.app.state())
        if u.path == "/api/job":
            try:
                since = int((q.get("since") or ["0"])[0])
            except ValueError:
                since = 0
            return self._json(200, self.app.jobs.status(since))
        if u.path == "/api/discovery":
            text = self.app.discovery_text((q.get("slug") or [""])[0], (q.get("file") or [""])[0])
            if text is None:
                return self._json(404, {"error": "not found"})
            return self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE.replace("__TOKEN__", self.token).encode("utf-8"), "text/html; charset=utf-8")
        if u.path.startswith("/view/"):
            view = self.app.view_dir()
            if view is None or not view.is_dir():
                return self._json(404, {"error": "no atlas view yet"})
            self.directory, self._view = str(view), True
            self.path = self.path[len("/view"):]
            return super().do_GET()
        return self._json(404, {"error": "not found"})

    def do_HEAD(self):
        return self._json(405, {"error": "GET only"})

    def _read_body(self) -> Optional[bytes]:
        """The whole body, read before any answer (an unread body makes Windows reset the connection, so the client
        sees WinError 10053 instead of the refusal); None when it is over MAX_BODY (drained, then dropped)."""
        try:
            n = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            n = 0
        if n <= MAX_BODY:
            return self.rfile.read(n) if n else b""
        if n > 64 * MAX_BODY:                                  # too big even to drain: answer, then hang up
            self.close_connection = True
            return None
        while n > 0:
            got = self.rfile.read(min(n, 65536))
            if not got:
                break
            n -= len(got)
        return None

    def do_POST(self):
        raw = self._read_body()
        if not self._host_ok():
            return self._json(403, {"error": "loopback only"})
        origin = self.headers.get("Origin")
        if origin is not None and (urlsplit(origin).hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
            return self._json(403, {"error": "cross-origin request refused"})
        if not hmac.compare_digest(self.headers.get("X-Atlas-Token") or "", self.token):
            return self._json(403, {"error": "no app token (open the app from its own page)"})
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return self._json(415, {"error": "JSON only"})
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            return self._json(404, {"error": "not found"})
        if raw is None:
            return self._json(413, {"error": "too large"})
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return self._json(400, {"error": "bad JSON"})
        self.touch()
        code, obj = self.app.post(path[5:], body if isinstance(body, dict) else {})
        return self._json(code, obj)


class Idle:
    """When a page last asked anything: the app exits once that is `idle_s` ago and no job or dialog is open."""

    def __init__(self, idle_s: float, clock: Callable[[], float] = time.monotonic):
        self.idle_s, self.clock, self.last = idle_s, clock, clock()

    def touch(self) -> None:
        self.last = self.clock()

    def expired(self, app: App) -> bool:
        return (not app.jobs.status()["running"] and not app.picking.locked()
                and self.clock() - self.last > self.idle_s)


def make_server(app: App, port: int, idle: Idle, token: str) -> ThreadingHTTPServer:
    handler = type("AtlasAppHandler", (Handler,), {"app": app, "token": token, "touch": staticmethod(idle.touch)})
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    return srv


def _call(port: int, method: str, path: str, body=None, token: str = "", timeout: float = 3.0) -> Optional[dict]:
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        h = {"Content-Type": "application/json", "X-Atlas-Token": token} if method == "POST" else {}
        c.request(method, path, body=None if body is None else json.dumps(body), headers=h)
        r = c.getresponse()
        d = json.loads(r.read() or b"{}")
        c.close()
        return d if isinstance(d, dict) else None
    except (OSError, ValueError, http.client.HTTPException):
        return None


def _same(a: str, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(str(b)))


def first_project(arg: Optional[str], settings: Settings) -> Optional[Path]:
    """--project, else $ATLAS_PROJECT, else the nearest atlas.toml from here up, else the most recent project."""
    for cand in (arg, os.environ.get("ATLAS_PROJECT")):
        if cand and (Path(cand).expanduser() / AP.CONFIG).is_file():
            return Path(cand).expanduser().resolve()
    here = AP.find_project(Path.cwd())
    if here is not None and not AP.is_blueboat(here):
        return here
    for r in settings.load()["recent"]:
        if (Path(r) / AP.CONFIG).is_file():
            return Path(r)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", default="", help="the atlas project to open (default: the last one)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--idle-exit", type=float, default=IDLE_EXIT_S)
    ap.add_argument("--tab", action="store_true", help="open in a browser tab, not a window of its own")
    ap.add_argument("--pick", choices=["folder", "files"], help=argparse.SUPPRESS)
    ap.add_argument("--initial", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.pick:
        return pick_main(a.pick, a.initial)
    home = settings_dir()
    settings = Settings(home)
    project = first_project(a.project or None, settings)
    state = home / "app.json"
    try:
        old = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    for port in [p for p in (old.get("port"), a.port) if isinstance(p, int)]:
        got = _call(port, "GET", "/api/ping")
        if got and got.get("app") == APP and _same(got.get("kit", ""), KIT):     # this kit's app is up: reuse it
            if a.project and project is not None and old.get("token"):
                _call(port, "POST", "/api/project", {"path": str(project)}, token=str(old["token"]))
            if not a.no_browser:
                open_page(f"http://127.0.0.1:{port}/", a.tab)
            print(f"the Paper Atlas app is already running: http://127.0.0.1:{port}/")
            return 0
    jobs = Jobs(settings.secrets, log=home / "logs" / "app.log")
    app = App(settings, jobs, project=project)
    if project is not None and not app.view_ready() and (project / "index" / "browser.html").is_file():
        app.post("view", {})
    idle = Idle(a.idle_exit)
    token = secrets.token_urlsafe(24)
    srv = make_server(app, a.port, idle, token)
    port = srv.server_address[1]
    try:
        home.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(state), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"port": port, "pid": os.getpid(), "kit": str(KIT), "token": token}))
    except OSError:
        pass
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    if sys.stdout is not None:
        print(f"Paper Atlas app: {url}  (it stops by itself {int(a.idle_exit)} s after its page is closed)")
    if not a.no_browser:
        open_page(url, a.tab)
    try:
        while not idle.expired(app):
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    srv.shutdown()
    return 0


# ------------------------------------------------------------------------------------------ the page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Atlas</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2430;--mute:#5d6878;--line:#d9dee6;--bar:#1f2a37;--barink:#e9eef5;
--acc:#2b6a86;--acc2:#e7f1f5;--bad:#a8322d;--ok:#2f7a45}
@media (prefers-color-scheme:dark){:root{--bg:#14181e;--panel:#1c222a;--ink:#e3e8ef;--mute:#9aa6b6;--line:#323b47;
--bar:#0f1318;--barink:#e3e8ef;--acc:#5fa8c6;--acc2:#1d3540}}
*{box-sizing:border-box}html,body{height:100%;margin:0}[hidden]{display:none!important}
body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink);
display:flex;flex-direction:column}
header{background:var(--bar);color:var(--barink);display:flex;flex-wrap:wrap;align-items:center;gap:6px;padding:7px 12px}
header .brand{font-weight:650;margin-right:6px;letter-spacing:.2px}
header .proj{max-width:28em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:.85;margin-right:8px;
cursor:pointer;border-bottom:1px dotted currentColor}
header button{background:transparent;color:var(--barink);border:1px solid #ffffff33;border-radius:6px;padding:5px 11px;
font:inherit;cursor:pointer}header button:hover{background:#ffffff18}header button:disabled{opacity:.4;cursor:default}
header .sp{flex:1}#jobchip{font-size:12.5px;padding:3px 9px;border-radius:10px;background:#ffffff18;cursor:pointer}
main{flex:1;min-height:0;display:flex;flex-direction:column}
#frame{flex:1;border:0;width:100%;background:#fff}
#empty{flex:1;display:flex;align-items:center;justify-content:center;padding:24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:22px 26px;max-width:640px}
.card h2{margin:0 0 8px;font-size:19px}.card p{color:var(--mute);margin:6px 0}
#out{height:34%;min-height:140px;display:none;flex-direction:column;border-top:1px solid var(--line);background:var(--panel)}
#out.show{display:flex}#outbar{display:flex;align-items:center;gap:8px;padding:5px 10px;border-bottom:1px solid var(--line)}
#outbar b{font-weight:600}#outbar .sp{flex:1}
#log{flex:1;overflow:auto;margin:0;padding:8px 12px;font:12.5px/1.4 ui-monospace,Consolas,Menlo,monospace;white-space:pre-wrap}
#log .cmd{color:var(--acc);font-weight:600}#log .bad{color:var(--bad);font-weight:600}
button.b{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:7px 14px;font:inherit;cursor:pointer}
button.g{background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 12px;
font:inherit;cursor:pointer}button.g:hover{background:var(--acc2)}button:disabled{opacity:.45;cursor:default}
dialog{border:1px solid var(--line);border-radius:12px;padding:0;background:var(--panel);color:var(--ink);
width:min(640px,calc(100vw - 32px));max-height:calc(100vh - 40px)}
dialog::backdrop{background:#0008}dialog form{padding:18px 22px;display:flex;flex-direction:column;gap:10px}
dialog h3{margin:0 0 2px;font-size:17px}.hint{color:var(--mute);font-size:12.8px;margin:0}
label{font-weight:550}input[type=text],input[type=password],textarea{width:100%;font:inherit;color:var(--ink);
background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:7px 9px}
textarea{min-height:84px;resize:vertical;font:12.5px/1.4 ui-monospace,Consolas,Menlo,monospace}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.row input[type=text],.row input[type=password]{flex:1;
width:auto;min-width:14em}.row.end{justify-content:flex-end}
.chk{font-weight:400;display:flex;gap:6px;align-items:center}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line)}.tabs button{border:0;background:none;color:var(--mute);
padding:6px 10px;font:inherit;cursor:pointer;border-bottom:2px solid transparent}
.tabs button.on{color:var(--ink);border-bottom-color:var(--acc)}
.list{border:1px solid var(--line);border-radius:6px;max-height:220px;overflow:auto}
.list div{padding:6px 9px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
.list div:last-child{border-bottom:0}.list .t{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.status{font-size:12.8px}.status.ok{color:var(--ok)}.status.bad{color:var(--bad)}
#toast{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);background:var(--bar);color:var(--barink);
padding:8px 14px;border-radius:8px;display:none;max-width:90vw;z-index:9}
pre.doc{white-space:pre-wrap;font:12.5px/1.45 ui-monospace,Consolas,Menlo,monospace;max-height:60vh;overflow:auto;margin:0}
</style></head><body>
<header>
 <span class="brand">Paper Atlas</span><span class="proj" id="proj" title="Open or make a project">no project</span>
 <button id="bAdd">Add papers</button><button id="bUpd">Update</button><button id="bStat">Status</button>
 <button id="bDisc">Discover</button><button id="bZip">Make zip</button>
 <span class="sp"></span><span id="jobchip" title="Show the output">idle</span><button id="bSet">Settings</button>
</header>
<main>
 <iframe id="frame" title="The atlas" hidden></iframe>
 <div id="empty"><div class="card" id="emptycard"></div></div>
 <section id="out"><div id="outbar"><b id="outname">Output</b><span id="outstate" class="status"></span>
  <span class="sp"></span><button class="g" id="bCancel">Cancel</button><button class="g" id="bHide">Hide</button></div>
  <pre id="log"><div class="hint" id="logEmpty">Nothing has run yet. A button's job shows its output here.</div></pre></section>
</main>
<div id="toast"></div>

<dialog id="dProj"><form method="dialog"><h3>Projects</h3>
 <p class="hint">An atlas project is a folder holding <code>atlas.toml</code>; your papers, the atlas and its zip live in it.</p>
 <div class="list" id="recent"></div>
 <label>Open a project folder</label><div class="row"><input type="text" id="openPath" placeholder="full path of a project folder">
  <button type="button" class="g pick" data-kind="folder" data-into="openPath">Browse...</button>
  <button type="button" class="b" id="doOpen">Open</button></div>
 <label>Make a new project</label><div class="row"><input type="text" id="newPath" placeholder="full path of a new folder">
  <button type="button" class="g pick" data-kind="folder" data-into="newPath">Browse...</button></div>
 <div class="row"><input type="text" id="newTitle" placeholder="its title, e.g. Reef Mapping Atlas (optional)">
  <button type="button" class="b" id="doNew">Make it</button></div>
 <div class="row end"><button class="g" value="x">Close</button></div></form></dialog>

<dialog id="dAdd"><form method="dialog"><h3>Add papers</h3>
 <div class="tabs" id="addTabs"><button type="button" data-t="folder" class="on">A folder of PDFs</button>
  <button type="button" data-t="files">Files or an export</button><button type="button" data-t="list">A list</button></div>
 <div data-p="folder"><p class="hint">PDFs in subfolders get the subfolder's name as their topic. The project's
  <code>inbox</code> folder is the usual place to drop them.</p>
  <div class="row"><input type="text" id="addFolder"><button type="button" class="g pick" data-kind="folder"
  data-into="addFolder">Browse...</button><button type="button" class="g" id="useInbox">The inbox</button></div></div>
 <div data-p="files" hidden><p class="hint">PDFs, a Zotero / BibTeX export (<code>.bib</code>, with its
  <code>files</code> folder beside it) or Zotero's CSL JSON, or a <code>.txt</code> / <code>.csv</code> list. One per line.</p>
  <textarea id="addFiles" placeholder="full paths, one per line"></textarea>
  <div class="row"><button type="button" class="g pick" data-kind="files" data-into="addFiles" data-append="1">Choose files...</button></div></div>
 <div data-p="list" hidden><p class="hint">One paper per line: a DOI (10.1109/...), an arXiv id (2301.04104) or a title.
  Add <code> | topic</code> after one to set its topic. Open copies are downloaded and checked.</p>
  <textarea id="addList" placeholder="10.1016/j.oceaneng.2020.107000&#10;2301.04104&#10;Side-scan sonar odometry for AUVs | sonar"></textarea></div>
 <label for="addTopic">Topic <span class="hint">(optional: blank = the subfolder's name, else "unsorted")</span></label>
 <input type="text" id="addTopic" list="topics" placeholder="e.g. sonar"><datalist id="topics"></datalist>
 <label class="chk"><input type="checkbox" id="addOffline"> Offline (no registry checks: bibliography entries only)</label>
 <p class="hint">Each paper is checked against arXiv / Crossref / OpenAlex and filed only when its PDF really is that
  paper; the rest are listed as NEEDS A LOOK. Then run <b>Update</b>.</p>
 <div class="row end"><button class="g" value="x">Cancel</button><button type="button" class="b" id="doAdd">Add</button></div></form></dialog>

<dialog id="dUpd"><form method="dialog"><h3>Update the atlas</h3>
 <p class="hint">Runs every stage whose inputs changed: text, OCR, facts, embeddings, figures, citations, glossary,
  the page and the zip. The first run on a few hundred papers can take an hour; later runs only do what changed.</p>
 <label class="chk"><input type="checkbox" id="updForce"> Re-run every stage (force)</label>
 <label class="chk"><input type="checkbox" id="updOffline"> Offline (skip the stages that need the internet)</label>
 <p class="hint" id="updGpu"></p>
 <div class="row end"><button class="g" value="x">Cancel</button><button type="button" class="b" id="doUpd">Update</button></div></form></dialog>

<dialog id="dDisc"><form method="dialog"><h3>Discover papers for a task</h3>
 <p class="hint">Describe what you are trying to do. The papers you have for it are listed, and the ones out there
  (OpenAlex, Semantic Scholar, arXiv), each with why. Nothing is added until you say so.</p>
 <textarea id="discTask" placeholder="e.g. follow a pipeline with a sonar on a small AUV and map its condition"></textarea>
 <input type="text" id="discTitle" placeholder="a short title (optional)">
 <div class="row end"><button type="button" class="b" id="doDisc">Discover</button></div>
 <label>Earlier discoveries</label><div class="list" id="discList"></div>
 <label class="chk"><input type="checkbox" id="discSection"> When adding, also make the task a section of the atlas</label>
 <div class="row end"><button class="g" value="x">Close</button></div></form></dialog>

<dialog id="dDoc"><form method="dialog"><h3 id="docTitle"></h3><pre class="doc" id="docText"></pre>
 <div class="row end"><button class="g" value="x">Close</button></div></form></dialog>

<dialog id="dZip"><form method="dialog"><h3>Make the zip</h3>
 <p class="hint">A portable copy: unzip anywhere and open <code>atlas.html</code>. PDFs are often copyrighted --
  share the full zip inside your group only.</p>
 <label class="chk"><input type="checkbox" id="zipNoPdfs"> Without the PDFs (a small zip)</label>
 <p class="hint" id="zipLast"></p>
 <div class="row end"><button type="button" class="g" id="zipShow">Show the folder</button><button class="g" value="x">Cancel</button>
  <button type="button" class="b" id="doZip">Make it</button></div></form></dialog>

<dialog id="dSet"><form method="dialog"><h3>Settings</h3>
 <p class="hint">Kept for you on this computer, outside every project (<span id="setFile"></span>), and handed to
  the kit's tools only while they run. Never shown again in full, never written to a log.</p>
 <label for="setKey">OpenAlex API key <span class="status" id="keyState"></span></label>
 <p class="hint">Free: openalex.org, sign in, Settings, API key. Without one OpenAlex answers about 1,000 requests a
  day (a few hundred papers); with one, ten times that.</p>
 <div class="row"><input type="password" id="setKey" autocomplete="off" placeholder="paste the key to set or replace it">
  <button type="button" class="g" id="clearKey">Remove</button></div>
 <p class="hint status bad" id="setxNote" hidden>Windows also stores an OPENALEX_API_KEY for you (setx). The kit's
  tools use THAT one; remove it to use this one: <code>reg delete HKCU\Environment /v OPENALEX_API_KEY</code></p>
 <label for="setMail">Contact e-mail <span class="status" id="mailState"></span></label>
 <p class="hint">Sent to Crossref and OpenAlex with each request (their "polite pool": faster, fewer refusals).
  Nothing is sent without one.</p>
 <div class="row"><input type="text" id="setMail" autocomplete="off" placeholder="your address, to set or replace it">
  <button type="button" class="g" id="clearMail">Remove</button></div>
 <label>GPU</label><div class="row">
  <label class="chk"><input type="radio" name="gpu" value=""> as the project says</label>
  <label class="chk"><input type="radio" name="gpu" value="yes"> on</label>
  <label class="chk"><input type="radio" name="gpu" value="no"> off</label></div>
 <div class="row end"><button class="g" value="x">Close</button><button type="button" class="b" id="doSet">Save</button></div></form></dialog>

<script>
"use strict";
const TOKEN = "__TOKEN__";
const $ = id => document.getElementById(id);
let S = null, since = 0, lastName = "", lastStamp = -1, wasRunning = false, pollT = null;

async function api(path, body) {
  const opt = body === undefined ? {} : {method: "POST", body: JSON.stringify(body),
    headers: {"Content-Type": "application/json", "X-Atlas-Token": TOKEN}};
  const r = await fetch(path, opt);
  let d = {};
  try { d = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
  return d;
}
function toast(msg, ms) {
  const t = $("toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", ms || 4000);
}
async function act(path, body, closeDlg) {
  try {
    const d = await api(path, body);
    if (closeDlg) closeDlg.close();
    if (d.message) toast(d.message);
    showOut(); await refresh(); poll();
    return d;
  } catch (e) { toast(e.message, 7000); return null; }
}
function showOut() { $("out").classList.add("show"); }

function render() {
  const p = S.project, busy = S.job.running;
  $("proj").textContent = p ? p.title + "  -  " + p.path : "no project: open or make one";
  for (const id of ["bAdd", "bUpd", "bStat", "bDisc", "bZip"]) $(id).disabled = !p || busy;
  $("jobchip").textContent = busy ? ("running: " + S.job.name) : (S.job.name ? (S.job.name + ": " +
    (S.job.cancelled ? "cancelled" : S.job.code === 0 ? "done" : "failed (" + S.job.code + ")")) : "idle");
  $("bCancel").disabled = !busy;
  const card = $("emptycard");
  if (p && p.view_ready) {
    $("empty").hidden = true; $("frame").hidden = false;
    if (lastStamp !== p.view_stamp || !$("frame").src) {
      lastStamp = p.view_stamp; $("frame").src = "/view/atlas.html?v=" + Date.now();
    }
  } else {
    $("frame").hidden = true; $("empty").hidden = false;
    card.innerHTML = !p ? "<h2>Welcome</h2><p>Open an atlas project, or make a new one: a folder for your papers and " +
      "their atlas.</p><p><button class='b' onclick='openProj()'>Open or make a project</button></p>"
      : p.problem ? "<h2>This project's settings cannot be read</h2><p></p>"
      : p.has_atlas ? "<h2>Preparing the atlas view...</h2><p>One moment.</p>"
      : "<h2>No atlas yet</h2><p>1. <b>Add papers</b>: a folder of PDFs, a Zotero / BibTeX export or a list of DOIs." +
        "</p><p>2. <b>Update</b>: reads them, finds figures and citations, and builds the atlas.</p>";
    if (p && p.problem) card.querySelector("p").textContent = p.problem;
  }
  $("topics").innerHTML = "";
  for (const t of (p ? p.topics : [])) { const o = document.createElement("option"); o.value = t; $("topics").append(o); }
}
async function refresh() { S = await api("/api/state"); render(); return S; }

async function poll() {
  clearTimeout(pollT);
  try {
    const j = await api("/api/job?since=" + since);
    const log = $("log");
    if (j.lines.length && $("logEmpty")) $("logEmpty").remove();
    if (j.first > since) since = j.first;
    for (const ln of j.lines) {
      const div = document.createElement("div");
      div.textContent = ln;
      if (ln.startsWith("$ ")) div.className = "cmd";
      else if (/FAILED|NEEDS A LOOK|CANCELLED|Traceback/.test(ln)) div.className = "bad";
      log.append(div);
    }
    if (j.lines.length) log.scrollTop = log.scrollHeight;
    since = j.next; lastName = j.name;
    $("outname").textContent = j.name ? "Output: " + j.name : "Output";
    const st = $("outstate");
    st.textContent = j.running ? "running..." : j.name ? (j.cancelled ? "cancelled" : j.code === 0 ? "done" :
      "failed (exit " + j.code + ")") : "";
    st.className = "status " + (j.running ? "" : j.code === 0 ? "ok" : j.name ? "bad" : "");
    if (wasRunning && !j.running) { await refresh(); if (j.name === "zip" && S.project && S.project.zip)
      toast("the zip: " + S.project.zip, 9000); }
    if (j.running !== S.job.running) await refresh();
    wasRunning = j.running;
    pollT = setTimeout(poll, j.running ? 1000 : 15000);
  } catch (e) { pollT = setTimeout(poll, 5000); }
}

async function pick(btn) {
  try {
    btn.disabled = true;
    const d = await api("/api/pick", {kind: btn.dataset.kind});
    if (!d.paths.length) return;
    const into = $(btn.dataset.into);
    into.value = btn.dataset.append && into.value.trim() ? into.value.trim() + "\n" + d.paths.join("\n")
      : d.paths.join("\n");
  } catch (e) { toast(e.message, 7000); } finally { btn.disabled = false; }
}
document.querySelectorAll("button.pick").forEach(b => b.onclick = () => pick(b));
function noPicker() { if (S && !S.picker) document.querySelectorAll("button.pick").forEach(b => {
  b.disabled = true; b.title = "no file dialog with this Python: paste the path"; }); }

function openProj() {
  const box = $("recent"); box.innerHTML = "";
  if (!S.recent.length) box.innerHTML = "<div><span class='t hint'>no recent projects</span></div>";
  for (const r of S.recent) {
    const d = document.createElement("div"), t = document.createElement("span"), b = document.createElement("button");
    t.className = "t"; t.textContent = r; t.title = r; b.className = "g"; b.type = "button"; b.textContent = "Open";
    b.onclick = () => act("/api/project", {path: r}, $("dProj"));
    d.append(t, b); box.append(d);
  }
  noPicker(); $("dProj").showModal();
}
$("proj").onclick = openProj;
$("doOpen").onclick = () => act("/api/project", {path: $("openPath").value}, $("dProj"));
$("doNew").onclick = () => act("/api/new", {folder: $("newPath").value, title: $("newTitle").value}, $("dProj"));

let addTab = "folder";
document.querySelectorAll("#addTabs button").forEach(b => b.onclick = () => {
  addTab = b.dataset.t;
  document.querySelectorAll("#addTabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll("#dAdd [data-p]").forEach(x => x.hidden = x.dataset.p !== addTab);
});
$("useInbox").onclick = () => { $("addFolder").value = S.project.inbox; };
$("bAdd").onclick = () => { noPicker(); $("dAdd").showModal(); };
$("doAdd").onclick = () => {
  const body = {topic: $("addTopic").value, offline: $("addOffline").checked, inputs: []};
  if (addTab === "folder") body.inputs = [$("addFolder").value];
  else if (addTab === "files") body.inputs = $("addFiles").value.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  else body.list_text = $("addList").value;
  act("/api/add", body, $("dAdd"));
};
$("bUpd").onclick = () => { const g = S.settings.gpu;
  $("updGpu").textContent = "GPU: " + (g === "yes" ? "on" : g === "no" ? "off" : "as the project says") + " (Settings)";
  $("dUpd").showModal(); };
$("doUpd").onclick = () => act("/api/update", {force: $("updForce").checked, offline: $("updOffline").checked}, $("dUpd"));
$("bStat").onclick = () => act("/api/status", {});

function discList() {
  const box = $("discList"); box.innerHTML = "";
  const ds = S.project ? S.project.discoveries : [];
  if (!ds.length) box.innerHTML = "<div><span class='t hint'>none yet</span></div>";
  for (const d of ds) {
    const row = document.createElement("div"), t = document.createElement("span");
    t.className = "t"; t.textContent = d.title; t.title = d.slug; row.append(t);
    const mk = (label, fn) => { const b = document.createElement("button"); b.type = "button"; b.className = "g";
      b.textContent = label; b.onclick = fn; row.append(b); };
    mk("Report", () => showDoc(d.slug, "DISCOVER.md"));
    mk("Closed papers", () => showDoc(d.slug, "GET_THESE.md"));
    if (d.add_list) mk("Add the new papers", () => act("/api/discover-add", {slug: d.slug,
      section: $("discSection").checked}, $("dDisc")));
    box.append(row);
  }
}
async function showDoc(slug, file) {
  try {
    const r = await fetch("/api/discovery?slug=" + encodeURIComponent(slug) + "&file=" + encodeURIComponent(file));
    if (!r.ok) throw new Error("not written yet");
    $("docTitle").textContent = slug + " / " + file; $("docText").textContent = await r.text(); $("dDoc").showModal();
  } catch (e) { toast(e.message); }
}
$("bDisc").onclick = () => { discList(); $("dDisc").showModal(); };
$("doDisc").onclick = () => act("/api/discover", {task: $("discTask").value, title: $("discTitle").value}, $("dDisc"));

$("bZip").onclick = () => { $("zipLast").textContent = S.project.zip ? "Last zip: " + S.project.zip : "";
  $("zipShow").hidden = !S.project.zip; $("dZip").showModal(); };
$("doZip").onclick = () => act("/api/zip", {no_pdfs: $("zipNoPdfs").checked}, $("dZip"));
$("zipShow").onclick = () => act("/api/reveal", {what: "zip"});

function setDlg() {
  const s = S.settings;
  $("setFile").textContent = s.file;
  $("keyState").textContent = s.openalex_api_key || "not set"; $("mailState").textContent = s.contact_email || "not set";
  $("setxNote").hidden = !s.setx_key; $("setKey").value = ""; $("setMail").value = "";
  document.querySelectorAll("input[name=gpu]").forEach(r => r.checked = r.value === s.gpu);
}
$("bSet").onclick = () => { setDlg(); $("dSet").showModal(); };
async function saveSet(body) {
  try { const d = await api("/api/settings", body); S.settings = d.settings; setDlg(); toast(d.message); }
  catch (e) { toast(e.message, 7000); }
}
$("doSet").onclick = () => {
  const body = {gpu: (document.querySelector("input[name=gpu]:checked") || {value: ""}).value};
  if ($("setKey").value.trim()) body.openalex_api_key = $("setKey").value.trim();
  if ($("setMail").value.trim()) body.contact_email = $("setMail").value.trim();
  saveSet(body);
};
$("clearKey").onclick = () => saveSet({openalex_api_key: ""});
$("clearMail").onclick = () => saveSet({contact_email: ""});

$("jobchip").onclick = () => $("out").classList.toggle("show");
$("bHide").onclick = () => $("out").classList.remove("show");
$("bCancel").onclick = () => act("/api/cancel", {});

// a link can open one dialog: http://127.0.0.1:<port>/#settings (add, update, discover, zip, projects, output)
const DEEP = {settings: "bSet", add: "bAdd", update: "bUpd", discover: "bDisc", zip: "bZip"};
refresh().then(async s => {
  await poll();
  const h = location.hash.slice(1);
  if (h === "output" || s.job.running) showOut();
  if (h === "projects" || !s.project) openProj();
  else if (DEEP[h] && !$(DEEP[h]).disabled) $(DEEP[h]).click();
});
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
