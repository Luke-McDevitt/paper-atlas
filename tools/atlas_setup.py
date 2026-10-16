r"""atlas_setup -- the Paper Atlas kit's `setup gpu`: PyTorch's CUDA build when this machine has an NVIDIA driver, then
the GPU extras (requirements-gpu.txt) installed so that none of them can swap that torch for another build, then a check
that a CUDA operation really runs.

    python tools/atlas_setup.py gpu <requirements-gpu.txt> [--cuda auto|keep|cpu|cu130|cu128|cu126|...]
    python tools/atlas_setup.py check                               # what torch this Python has, and its device

The kit's setup.cmd / setup.sh run `gpu` with the kit's own Python (`setup gpu`); run it again any time -- a CUDA torch
that works is kept as it is. `--cuda` (or $ATLAS_CUDA): `keep` = never touch the installed torch, `cpu` = the CPU
build, a tag = that PyTorch index (download.pytorch.org/whl/<tag>).

WHY (2026-10-15, the kit's first GPU run, on the lab box: RTX 4060, driver 581.04 = CUDA 13.0, Python 3.14.7). The
guide said "install the CUDA build of torch first (e.g. cu128), then setup gpu", and setup ran `pip install -r
requirements-gpu.txt`. cu128's newest torch was 2.11.0 while PyPI's was 2.14.1; doclayout-yolo needs torchvision, pip
took PyPI's newest (0.29.1), which pins torch==2.14.1 -- so pip UNINSTALLED torch 2.11.0+cu128 and installed
2.14.1+cpu (PyPI's Windows wheels are CPU-only). setup still printed "Done" and exited 0, and `atlas status` said
GPU no (minilm, the model-free figure finder) on a machine with a GPU.

HOW
1. The driver: `nvidia-smi` prints "CUDA Version: 13.0", the newest CUDA it runs. No nvidia-smi: no NVIDIA GPU.
2. torch. A CUDA build that runs a CUDA operation is KEPT (installed by hand, or by an earlier run), and so is any
   torch on a machine with no NVIDIA driver (there is nothing better to get); a missing torchvision comes from torch's
   own PyTorch index. Otherwise the PyTorch indexes the driver can run are tried, newest CUDA first, then the CPU
   index -- the old torch removed first, because pip keeps an installed torch whose version already satisfies the
   request -- until one installs and (for a CUDA index) runs a CUDA operation. On a Mac nothing is tried: PyPI's
   torch is the one with MPS.
3. The extras: `pip install -r requirements-gpu.txt -c <torch==X, torchvision==Y as installed>`.
4. The check, in a child process (this one never imports torch, so pip can replace its DLLs): versions, device, a
   CUDA operation, the extras importable. Exit 1 when pip failed or an extra does not import. A probe that gives no
   answer STOPS setup with nothing more changed: the first build read its silence as "no torch" and, on the lab box,
   uninstalled a working CUDA torch (its probe died on a name only this module defines; the faked tests never ran it).

HONEST SCOPE: run end to end on Windows only (the lab box above). The Linux / macOS branches and the index fallbacks
are exercised by tools/tests/test_atlas_setup.py with pip faked; its probe runs for real there, in a child Python,
against stand-in torch packages.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

INDEX = "https://download.pytorch.org/whl/{tag}"
#: PyTorch's CUDA wheel indexes, newest CUDA first, with the CUDA version each needs from the driver. In 2026-10, cu130
#: and cu126 carried torch 2.14.1 and cu128 stopped at 2.11.0 (for Python 3.14); an index with no wheel for this Python
#: or platform just fails to install, and the next is tried.
CUDA_TAGS: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6)), ("cu124", (12, 4)), ("cu121", (12, 1)),
    ("cu118", (11, 8)))
#: what the GPU stages import (tools/paper_embed.py, tools/figure_bank.py)
EXTRAS = ("sentence_transformers", "doclayout_yolo")

#: run by ANOTHER interpreter (the kit's .venv): it sees nothing of this module -- names it needs are filled in by
#: probe() (2026-10-15: it looped over EXTRAS, a NameError in the child, and setup read the silence as "no torch")
_PROBE = r"""
import json
out = {"extras": {}}
try:
    import torch
    out["torch"], out["cuda"] = torch.__version__, torch.version.cuda
    try:
        import torchvision
        out["torchvision"] = torchvision.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["cuda_op"] = float((torch.ones(4, device="cuda") * 2).sum().item()) == 8.0
except Exception as e:
    out["error"] = (type(e).__name__ + ": " + str(e))[:300]
for m in __EXTRAS__:
    try:
        __import__(m)
        out["extras"][m] = True
    except Exception:
        out["extras"][m] = False
print(json.dumps(out))
"""


def driver_cuda(run: Callable = subprocess.run) -> Optional[Tuple[int, int]]:
    """The newest CUDA this machine's NVIDIA driver runs -- (13, 0) from nvidia-smi's "CUDA Version: 13.0" -- or None
    (no nvidia-smi: no NVIDIA GPU, or no driver)."""
    try:
        r = run(["nvidia-smi"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version\s*:\s*(\d+)\.(\d+)", r.stdout or "")
    return (int(m.group(1)), int(m.group(2))) if r.returncode == 0 and m else None


def candidate_indexes(driver: Optional[Tuple[int, int]], want: str = "auto",
                      platform: str = sys.platform) -> List[str]:
    """The PyTorch index tags to try for torch, in order: an explicit tag alone; `keep` nothing; else the CUDA indexes
    the driver runs, newest CUDA first, then `cpu` -- on a Mac nothing (PyPI's torch is the MPS build)."""
    if want not in ("", "auto"):
        return [] if want == "keep" else [want]
    if platform == "darwin":
        return []
    return [tag for tag, need in CUDA_TAGS if driver is not None and need <= driver] + ["cpu"]


def index_of(torch_cuda: Optional[str]) -> str:
    """The PyTorch index tag of an installed torch: "12.8" -> cu128, a CPU build (None) -> cpu."""
    return "cu" + torch_cuda.replace(".", "") if torch_cuda else "cpu"


def probe(py: str, run: Callable = subprocess.run) -> Dict[str, object]:
    """What `py` has -- {torch, cuda (None: a CPU build), torchvision, device, cuda_op (a CUDA operation ran), extras,
    error (why torch did not import)} -- read in a CHILD process: this one must not hold torch's DLLs while pip
    replaces them. No answer is {"probe_error": ...}, never "no torch"."""
    try:
        r = run([py, "-c", _PROBE.replace("__EXTRAS__", repr(EXTRAS))], capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as e:
        return {"probe_error": f"{type(e).__name__}: {e}"[:300]}
    for ln in reversed((r.stdout or "").splitlines()):            # its JSON is the last line; warnings may precede it
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if isinstance(d, dict) and "extras" in d:
            return d
    tail = (r.stderr or r.stdout or "").strip().splitlines()
    return {"probe_error": f"exit {r.returncode}: {tail[-1] if tail else 'no output'}"[:300]}


def constraints(have: Dict[str, object]) -> str:
    """pip constraints holding torch (and torchvision) at the INSTALLED build, local label and all (torch==2.14.1+cu130):
    with them no requirement can pull another torch from PyPI."""
    return "".join(f"{name}=={have[name]}\n" for name in ("torch", "torchvision") if have.get(name))


def _keep(have: Dict[str, object], driver: Optional[Tuple[int, int]], want: str) -> bool:
    if not have.get("torch"):
        return False
    if want == "keep" or have.get("cuda_op"):
        return True
    return want in ("", "auto") and driver is None          # no NVIDIA GPU: a CPU torch is as good as it gets


def _pip_with(pip: Callable[..., int], cons: str, *args: str) -> int:
    if not cons:
        return pip(*args)
    fd, path = tempfile.mkstemp(prefix="atlas_torch_pins_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(cons)
        return pip(*args, "-c", path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def install_gpu(requirements: Path, want: str = "auto", py: str = sys.executable, run: Callable = subprocess.run,
                pip: Optional[Callable[..., int]] = None, platform: str = sys.platform,
                say: Callable = print) -> int:
    """Steps 1-4 of the module docstring. `pip(*args)` runs `py -m pip *args` (tests pass a recorder)."""
    if pip is None:
        def pip(*args: str) -> int:
            return subprocess.call([py, "-m", "pip", *args])
    drv = driver_cuda(run)
    say(f"setup gpu: NVIDIA driver: {'runs CUDA %d.%d' % drv if drv else 'none found (no nvidia-smi)'}")
    have = probe(py, run)
    if "probe_error" in have:                         # what is installed is unknown: change nothing
        return report(have, drv, 0, say)
    if _keep(have, drv, want):
        say(f"  torch {have['torch']} is kept ({'CUDA ' + str(have['cuda']) if have.get('cuda') else 'CPU build'})")
        if not have.get("torchvision"):
            tag = index_of(have.get("cuda"))                  # the matching torchvision is on torch's own index
            if _pip_with(pip, constraints(have), "install", "torchvision", "--index-url", INDEX.format(tag=tag)):
                say("  torchvision did not install from that index: the extras will try PyPI (held to this torch)")
    else:
        for tag in candidate_indexes(drv, want, platform):
            pip("uninstall", "-y", "torch", "torchvision")   # pip keeps an installed torch whose version satisfies
            say(f"  torch + torchvision from {INDEX.format(tag=tag)} ...")
            if pip("install", "torch", "torchvision", "--index-url", INDEX.format(tag=tag)) == 0:
                have = probe(py, run)                # silent now = THIS build is broken (the probe answered above)
                if have.get("torch") and (tag == "cpu" or have.get("cuda_op")):
                    break
                say(f"  {tag}: torch {have.get('torch', '?')} installed but "
                    + ("ran no CUDA operation" if have.get("torch") else "did not import") + " -- next index")
            else:
                say(f"  {tag}: no torch for this Python / platform there -- next index")
    have = probe(py, run)
    rc = _pip_with(pip, constraints(have), "install", "-r", str(requirements))
    have = probe(py, run)
    return report(have, drv, rc, say)


def report(have: Dict[str, object], drv: Optional[Tuple[int, int]], rc: int = 0, say: Callable = print) -> int:
    """One verdict line (and a WARNING when an NVIDIA GPU is here but torch cannot use it); 1 = the install failed."""
    extras = have.get("extras") or {}
    missing = [m for m in EXTRAS if not extras.get(m)]
    if "probe_error" in have:
        say(f"setup gpu: FAILED -- could not ask the kit's Python what it has ({have['probe_error']}); nothing more "
            f"was changed")
        return 1
    if not have.get("torch"):
        say(f"setup gpu: FAILED -- torch does not import ({have.get('error', 'no torch')})")
        return 1
    if have.get("cuda_op"):
        dev = f"CUDA {have.get('cuda')} on {have.get('device')} -- a CUDA operation ran"
    elif have.get("cuda"):
        dev = f"a CUDA {have.get('cuda')} build, but NO usable GPU"
    else:
        dev = "the CPU build"
    say(f"setup gpu: torch {have['torch']} ({dev}); torchvision {have.get('torchvision', 'none')}; "
        + ("extras " + ", ".join(EXTRAS) + " import" if not missing else "NOT importable: " + ", ".join(missing)))
    if drv and not have.get("cuda_op"):
        say(f"  WARNING: this machine has an NVIDIA driver (CUDA {drv[0]}.{drv[1]}) but torch cannot use the GPU, so "
            f"embeddings and figure detection run on the CPU. Update the driver (nvidia.com/drivers) and run "
            f"`setup gpu` again, or name an index: set ATLAS_CUDA=cu126 (any tag of download.pytorch.org/whl/).")
    if rc or missing:
        say(f"setup gpu: FAILED -- pip exit {rc}" + (f"; {', '.join(missing)} missing" if missing else ""))
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["gpu", "check"])
    ap.add_argument("requirements", nargs="?", default="", help="gpu: the kit's requirements-gpu.txt")
    ap.add_argument("--cuda", default=os.environ.get("ATLAS_CUDA", "auto"),
                    help="auto (default; $ATLAS_CUDA) | keep | cpu | a PyTorch index tag such as cu126")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return report(probe(sys.executable), driver_cuda())
    req = Path(a.requirements or Path(__file__).resolve().parent.parent / "requirements-gpu.txt")
    if not req.is_file():
        ap.error(f"no requirements file at {req}")
    return install_gpu(req, a.cuda.strip().lower())


if __name__ == "__main__":
    sys.exit(main())
