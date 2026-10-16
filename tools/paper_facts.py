"""paper_facts -- mine a review's FULL TEXTS for the facts a comparison needs, each with its evidence sentence.

    python tools/paper_facts.py --refs report/world_models/options/wm_options.json --dir report/world_models/papers \
        --out report/world_models/options/paper_facts.json
    python tools/paper_facts.py --bwm      # every held paper of the BlueBoat world-model index (2026-10-13)

WHAT IT IS. The PDFs in --dir are matched to the papers in --refs by `tools/paper_pdfs.py` (DOI / arXiv id / title in
the PDF's own metadata), their text is extracted ONCE with pypdf into a local cache (`<dir>/.text/`, gitignored with
the PDFs: full texts are the publishers', never committed), and fixed PROBES pull out, per paper:
  hardware   GPU / accelerator models named (RTX 4090, A100, Jetson Orin ...) + the sentences naming them
  training   sentences giving a training / run time
  code       URLs (GitHub / GitLab / Hugging Face / project pages), split into OWN (stated as the paper's code or
             release) vs CITED (someone else's repo the paper uses), + "code will be released" promises
  tested     where it was evaluated: sim / tank-or-pool / field (sea, lake, reef, harbour ...) / real data
  sensors    DVL, IMU, imaging / forward-looking sonar, multibeam, side-scan, camera, stereo, LiDAR, USBL, GNSS ...
  platform   named robots / vehicles (BlueROV2, BlueBoat, Girona 500, Aqua, Franka ...)
  data       sentences giving a data scale (hours of data, frames, episodes, trajectories ...)
  params     model sizes stated as a parameter count (38k parameters, 1.3B params ...)
  speed      rates stated where speed is the subject (fps / frames per second / Hz / ms per frame, in a sentence about
             inference, real time, latency or throughput) -- a SENSOR rate ("the sonar pings at 10 Hz") can still slip in
  datasets   names written as "<Name> dataset" / "<Name> benchmark" (KITTI dataset, AQUALOC dataset ...)
  metrics    which standard metrics the paper reports (PSNR, SSIM, LPIPS, FVD, FID, ATE, RPE, RMSE, success rate ...)
  simulators named simulators / engines (HoloOcean, Stonefish, Gazebo, VRX, Isaac Sim, MuJoCo, AirSim, BELLHOP ...)
  world_models  world-model families named (Dreamer / DreamerV3, PlaNet, TD-MPC, IRIS, DIAMOND, MuZero, JEPA, RSSM ...)
  planners   planners / controllers named (MPPI, MPC, CEM, iLQR, RRT*, PID, line-of-sight guidance, pure pursuit ...)
  rl         RL / imitation algorithms named (PPO, SAC, TD3, DQN, CQL, IQL, behaviour cloning, offline / model-based RL)
Every value carries <= 3 evidence snippets (<= 240 chars) so a person can CHECK it; the output is a facts JSON (committed:
short snippets, not the papers) plus a Markdown table. Re-run after adding PDFs; extraction is cached.

WHY. 2026-10-01: the owner collected 54 of 56 full texts for the world-model options so the decks can quote the papers.
Reading 54 PDFs by hand -- or into a model -- is the expensive way; most of what a comparison needs is a handful of
probe-able facts, and the snippets make each one auditable.

MEASURED on its first run (2026-10-01, 54 texts): a bare "pool" matched ML pooling and "candidate pools", and citation
addresses ("National Harbor") read as field trials, so pool needs a water context and the bibliography is cut at its
heading + first entry. The "sim" tag still fires on any mention of simulation: read the snippets.

BWM MODE (`--bwm`, 2026-10-13, ~1,000 full texts). Papers = `report/boat_world_model/index/papers.json`, files =
`papers/_held.json` (by sha256 -- no title matching), text = the shared, content-addressed cache of
`tools/paper_text.py` (`C:/svfr/papers/boat_wm_text`; missing texts are extracted first, in parallel). Evidence is
PAGE-NUMBERED ("p.4: ...") and pages after the bibliography are not mined. Output
`report/boat_world_model/papers/_facts.json`, one paper per line: every fact's values, and one evidence snippet for
the facts a person checks before trusting (own code, promise, hardware, platform, simulators). Grep the full text for
anything else: `python tools/paper_text.py grep "<regex>" --bwm`.

ATTRIBUTION (2026-10-14). A named value (simulator, world model, planner, RL algorithm, sensor, platform, hardware,
dataset, metric, tested-in) counts only when at least one sentence presents it as the paper's OWN: first person ("we",
"our", "this paper"), or outside the introduction / related work with no citation in the sentence ('tested' also needs
an experiment word, and no future tense or affiliation line). The rest go to `mentioned` (kept, not counted). Section
roles come from numbered headings, '1 Introduction', '1. Introduction' or 'I. INTRODUCTION' (an Introduction / Related
Work / Background / Preliminaries section ends at the next section number, any title; '1.4 X' is a subsection).

GOLD SET (`python tools/paper_facts.py --gold [--split dev|test|all] [--variant raw|attributed]`):
`papers/_gold_facts.json`, 40 papers read in full by two independent LLM labelers blind to this miner + an adjudicator
on every disagreement (857 items, every quote verified on its page; labeler agreement on 'used' items Dice 0.83). DEV
(24) tuned the attribution rules; TEST (16, drawn after) never did -- but it was SCORED four times (first run, after
attribution, after the code review, after the fix check of that review; every change came from code reading checked
on DEV). MEASURED 2026-10-14 after the fix check, micro over 12 types, each gold item credited at most once (another
mined name for an item already found is a `duplicate`, left out of precision):
  TEST  raw (every regex hit)  precision 0.68  recall 0.36 | attributed  precision 0.74  recall 0.30
  DEV   raw                    precision 0.60  recall 0.33 | attributed  precision 0.75  recall 0.32
attributed - raw, paired paper bootstrap: precision +0.06 (95% CI +0.00..+0.14) on TEST, +0.14 (+0.10..+0.20) on DEV;
recall -0.07 (-0.12..-0.03) on TEST, -0.01 (-0.03..-0.00) on DEV -- attribution TRADES recall for precision: with the
sections now bounded right, a fact a paper states only in its introduction counts as a mention (kept in `mentioned`).
`--variant html` (arXiv papers mined from their own section tree, mine_html; the rest from the PDF; 14 of 24 DEV and
10 of 16 TEST papers had HTML): DEV 0.74 / 0.32 (precision -0.01 vs attributed, CI -0.03..+0.02), TEST 0.77 / 0.32
(+0.03, CI +0.00..+0.06) -- a gain only the never-tuned split shows, so it is an option, not the default.
Recall is the open problem: the probes know fixed vocabularies (in-vocabulary recall 0.53 on TEST), and most gold
metrics / datasets are names no list holds.

HONEST SCOPE (do NOT overclaim): regex probes over extracted PDF text. A probe can MISS (a fact phrased oddly, text
lost in extraction, a table) and can MIS-ATTRIBUTE (a GPU named for a baseline, a field trial of another system), so
facts are CANDIDATES until a person reads the snippet -- the snippets are there for exactly that. OWN vs CITED code is a
sentence heuristic ("our code", "available at", "we release" ...). Exit 2 when pypdf is missing or a PDF cannot be read.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("paper_pdfs", ROOT / "tools" / "paper_pdfs.py")
PP = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("paper_pdfs", PP)
_spec.loader.exec_module(PP)



def _project() -> Path:
    """The atlas project `--bwm` means (tools/atlas_project.py: $ATLAS_PROJECT, the nearest atlas.toml, else the
    BlueBoat world-model atlas)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import atlas_project
    p = atlas_project.project_dir()                    # the BlueBoat atlas through this module's ROOT (a test seam)
    return ROOT / "report" / "boat_world_model" if atlas_project.is_blueboat(p) else p


SNIP = 240
MAX_EV = 3
GPU = re.compile(r"\b(?:(?:NVIDIA|GeForce)\s+)?(?:RTX\s?(?:A\d{4}|\d{4}(?:\s?Ti)?(?:\s?Super)?)|GTX\s?\d{3,4}(?:\s?Ti)?|"
                 r"TITAN\s?(?:X[p]?|V|RTX)|Quadro\s?[A-Z]+\d*|Tesla\s?[A-Z]\d+|A100|H100|H200|B200|V100|P100|(?<!Sony )(?<!Sony-)A6000|A40|"
                 r"A10G?|L40S?|L4|T4|RTX\s?PRO\s?6000|Jetson\s?(?:AGX\s?)?(?:Orin|Xavier|Nano|TX2)(?:\s?NX)?|TPU\s?v?\d+)\b")
URL = re.compile(r"(?:https?://)?(?:www\.)?((?:github\.com|gitlab\.com|huggingface\.co|bitbucket\.org)/[\w.-]+(?:/[\w.-]+)?|"
                 r"[\w-]+\.github\.io(?:/[\w./-]*)?)", re.I)
OWN_CODE = re.compile(r"\b(our|the)\s+(code|implementation|source code|dataset|models?)\b|available at|available on|"
                      r"is available|are available|will be (?:made )?(?:publicly )?(?:available|released)|we release|"
                      r"released at|open[- ]source|project page|project website|code:|supplementary", re.I)
PROMISE = re.compile(r"(code|implementation|dataset)[^.]{0,60}(will be|to be) (made )?(publicly )?(available|released)",
                     re.I)
TRAIN_T = re.compile(r"\b(train(?:ed|ing)?|converge[sd]?|took|takes|runtime|run time|inference)\b[^.]{0,120}?"
                     r"\b\d+(?:\.\d+)?\s?(?:GPU[- ]?)?(?:hours?|h|hrs|days?|minutes?|min|seconds?)\b", re.I)
DATA = re.compile(r"\b\d[\d,.]*\s?(?:k|K|M)?\s?(?:hours? of|h of|frames|images|episodes|trajectories|sequences|"
                  r"samples|transitions|scenes|runs|dives|missions)\b", re.I)
TESTED = {
    "sim": r"\b(?:simulat(?:ion|or|ed)|Gazebo|Stonefish|HoloOcean|Isaac\s?(?:Sim|Gym|Lab)|MuJoCo|Unreal|UUV\s?Sim|"
           r"CARLA|DM\s?Control|DeepMind Control|Habitat-?Sim|AI\s?Habitat|MarineGym|OceanSim)\b",
    "tank/pool": r"\b(?:test tank|water tank|towing tank|tank experiments?|tank tests?|in a tank|(?:test|swimming|diving|"
                 r"indoor|water|experimental) pool|pool (?:tests?|experiments?|trials?)|in a pool|wave basin|flume)\b",
    "field": r"\b(?:field (?:trials?|tests?|experiments?|deployments?)|sea trials?|open[- ]water|at sea|in the ocean|"
             r"offshore|harbou?r|lagoon|lake|river|reef|coastal waters|real[- ]world deployment|in the wild)\b",
    "real data": r"\b(?:real[- ]world (?:data|dataset|images|footage|experiments?)|real data|real images|real robot|"
                 r"real hardware|physical robot|hardware experiments?)\b",
}
SENSORS = {
    "DVL": r"\b(?:DVL|Doppler velocity log)\b", "IMU": r"\b(?:IMU|inertial measurement)\b",
    "imaging sonar": r"\b(?:imaging sonar|forward[- ]looking sonar|FLS|Oculus|BlueView|Gemini\s?\d|ARIS|DIDSON|"
                     r"acoustic camera)\b",
    "multibeam": r"\b(?:multibeam|MBES)\b", "side-scan": r"\b(?:side[- ]?scan(?: sonar)?|SSS)\b",
    "Ping360/echosounder": r"\b(?:Ping360|Ping2|echo ?sounder|altimeter)\b",
    "camera": r"\b(?:camera|RGB|monocular|stereo)\b", "LiDAR": r"\b(?:LiDAR|lidar)\b",
    "USBL": r"\b(?:USBL|acoustic positioning)\b", "GNSS": r"\b(?:GNSS|GPS|RTK)\b",
    "depth/pressure": r"\b(?:pressure sensor|depth sensor|Bar30|barometer)\b",
    "hydrophone": r"\b(?:hydrophones?|acoustic array)\b",
}
PARAMS = re.compile(r"\b(\d+(?:\.\d+)?\s?(?:[KkMB]|thousand|million|billion))\s?(?:trainable\s)?(?:parameters|params)\b",
                    re.I)
SPEED = re.compile(r"\b\d+(?:\.\d+)?\s?(?:fps|FPS|frames per second|Hz|kHz|ms(?:\s?(?:per|/)\s?(?:frame|step|image))?)\b")
SPEED_CONTEXT = re.compile(r"\b(?:real[- ]time|inference|runs? at|running at|latency|speed|throughput|per frame|per step|"
                           r"on (?:a|an|one|the) (?:single )?(?:GPU|CPU|Jetson|NVIDIA))\b", re.I)
DATASET = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:[- ][A-Z0-9][A-Za-z0-9]*)?)\s(?:dataset|benchmark|data set)\b")
_NOT_A_NAME = {"The", "This", "Our", "A", "An", "Each", "Every", "Real", "Synthetic", "Simulated", "Test", "Training",
               "Validation", "New", "Public", "Large", "Small", "Full", "Same", "Underwater", "Sonar", "That", "These",
               "Its", "Their", "One", "Two", "Three", "Both", "Any", "Such", "Another", "Proposed", "Collected", "Original",
               "Whole", "Entire", "Existing", "Standard", "Popular", "Benchmark", "Dataset", "In", "On", "For", "From", "We",
               "To", "With", "Using", "First", "Second", "Final", "Additional", "Custom", "Raw", "Video", "Image",
               "Experience", "Online", "Offline", "Replay", "Evaluation", "Table", "Figure", "Fig", "Comprehensive",
               "Ablation", "Further", "Parameter", "Five", "Initialize", "Simulation"}
METRICS = {
    "PSNR": r"\bPSNR\b", "SSIM": r"\bSSIM\b", "LPIPS": r"\bLPIPS\b", "FVD": r"\bFVD\b", "FID": r"\bFID\b",
    "ATE": r"\bATE\b|(?i:\babsolute trajectory error\b)", "RPE": r"\bRPE\b|(?i:\brelative pose error\b)",
    "RMSE": r"\bRMSE\b", "success rate": r"(?i:\bsuccess rates?\b)", "IoU": r"\bm?IoU\b", "Chamfer": r"\bChamfer\b",
    "episode return": r"(?i:\b(?:episode|episodic|average|mean|cumulative) (?:returns?|rewards?)\b)",
}  # case-SENSITIVE acronyms ("ate" is a verb); the spelled-out phrases carry their own (?i:...)
SIMULATORS = re.compile(r"\b(HoloOcean|Stonefish|Gazebo|VRX|UUV\s?Simulator|DAVE\s?simulator|Isaac\s?(?:Sim|Gym|Lab)|"
                        r"MuJoCo|Unreal\s?Engine|AirSim|OceanSim|MarineGym|Webots|CoppeliaSim|PyBullet|Habitat-?Sim|AI\s?Habitat|CARLA|"
                        r"BELLHOP|ArduPilot\s?SITL|MOOS-IvP|Blender)\b")
WORLD_MODELS = re.compile(r"\b(Dreamer(?:V[123])?|PlaNet|TD-MPC2?|IRIS|DIAMOND|MuZero|EfficientZero|V-?JEPA(?:\s?2)?|"
                          r"I-JEPA|JEPA|RSSM|Genie(?:\s?[23])?|GAIA-[12]|DINO-WM|world models?)\b")
PLANNERS = re.compile(r"\b(MPPI|MPC|CEM|iLQR|DDP|RRT\*?|PRM|line[- ]of[- ]sight guidance|LOS guidance|pure pursuit|"
                      r"Stanley controller|PID|LQR|sliding[- ]mode control(?:ler)?|backstepping)\b")
RL_ALGOS = re.compile(r"\b(PPO|SAC|TD3(?:\+BC)?|DDPG|DQN|CQL|IQL|AWAC|BCQ|IMPALA|A3C|behaviou?r cloning|offline RL|"
                      r"model-based RL|MBRL)\b")
PLATFORM = re.compile(r"\b(BlueROV\s?2?(?:\s?Heavy)?|BlueBoat|Girona\s?\d{3}|Sparus\s?II|Iver\s?\d|REMUS\s?\d*|HUGIN|"
                      r"Aqua\s?(?:robot|\d)?|Heron|WAM-V|Otter|Kingfisher|Jackal|Husky|Spot|Franka(?:\s?Panda)?|UR[35]e?|"
                      r"Unitree\s?\w+|Clearpath\s?\w+|LoCoBot|DJI\s?\w+|NemeSys|CavePI|Nemosens|Ocean ?One|Saab\s?\w+)\b")


def sentences(text):
    """Extracted PDF text -> sentences (hyphenated line breaks re-joined, whitespace collapsed). Pure."""
    t = re.sub(r"-\s*\n\s*", "", text or "")
    t = re.sub(r"\s+", " ", t)
    return _split(t)


def _split(t):
    # also before a numbered heading ("... 2004). 2 Related Work Woo et al."): a heading glued to the previous
    # sentence's tail was never seen as a section start (found by a mutation check, 2026-10-14)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[]|(?:\d{1,2}|[IVX]{1,4})\.?\s+[A-Z][A-Za-z])", t)
            if len(s.strip()) > 20]


def _snip(s):
    return s if len(s) <= SNIP else s[:SNIP - 3].rstrip() + "..."


_REFS = re.compile(r"\b(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Literature Cited|LITERATURE CITED)\b"
                   r"(?:\s+(?:AND|and)\s+(?:NOTES|Notes))?\s*"
                   # what a first ENTRY looks like (gold set + review + fix check 2026-10-14): numbered '[1]' / '1. '
                   # / '(1)'; a bracket label '[Abbeel and Ng 2004]'; a surname with initials 'Abdo DA', 'Abadi, M.',
                   # 'Amiri-Simkooei, A.'; a full name 'Martín Abadi,' / 'Pieter Abbeel and'; 'A. Author'. NOT any
                   # capitalised word: a table header 'References Sensors Key' and 'References 1 and 2' cut 9 papers
                   r"(?:\[\s?1\s?\]|1\.\s|\(1\)|\[[A-Z\u00c0-\u024f][^\]]{2,60}\d{4}[a-z]?\]|"
                   r"[A-Z\u00c0-\u024f][\w'\u00c0-\u024f\-]+,?\s(?:[A-Z]\.|[A-Z]{1,3}\b)|"
                   r"[A-Z\u00c0-\u024f][a-z\u00c0-\u024f]+\s[A-Z\u00c0-\u024f][\w'\u00c0-\u024f\-]+(?:,|\sand\s)|"
                   r"[A-Z]\.\s?[A-Z][a-z]+)")
#: the fallback: a References heading followed by ANY capitalised token or a number -- taken only when no heading of the
#: strict shape exists, and then the LAST one (a garbled first entry 'AAstrom, ¨ K.J.' or 'J. Á. Acosta' is still the
#: bibliography; an earlier 'References Sensors Key components' table header is not)
_REFS_LOOSE = re.compile(r"\b(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Literature Cited|LITERATURE CITED)\b"
                         r"(?:\s+(?:AND|and)\s+(?:NOTES|Notes))?\s*"
                         r"(?:\[\s?1\s?\]|1\.?\s|\(1\)|[A-Z\u00c0-\u024f][\w'\u00c0-\u024f\-]*[ ,.])")

#: a numbered section heading in extracted text -> the role its section plays for attribution
_HEAD = re.compile(r"(?:^|(?<=\s)|(?<=[^\d\s]\.))(?:(\d{1,2})\.?|([IVX]{1,4})\.)\s+(INTRODUCTION|Introduction|RELATED\s+WORKS?|"
                   r"Related\s+[Ww]orks?|LITERATURE\s+REVIEW|Literature\s+[Rr]eview|PRIOR\s+WORKS?|Prior\s+[Ww]orks?|"
                   r"STATE\s+OF\s+THE\s+ART|State\s+of\s+the\s+[Aa]rt|BACKGROUND|Background|PRELIMINARIES|Preliminaries|"
                   r"METHODS?|Methods?|METHODOLOGY|Methodology|APPROACH|Approach|PROPOSED|Proposed|SYSTEM|System|"
                   r"EXPERIMENTS?|Experiments?|EXPERIMENTAL|Experimental|EVALUATION|Evaluation|RESULTS|Results|"
                   r"IMPLEMENTATION|Implementation|DISCUSSION|Discussion|CONCLUSIONS?|Conclusions?)\b")
_ROLE = {"introduction": "intro", "related": "related", "literature": "related", "prior": "related",
         "state": "related", "background": "related", "preliminaries": "related"}
# Background / Preliminaries as related work: the HTML miner's rule, so both read a paper the same way. Measured on
# the gold DEV split 2026-10-14: related 23 spurious vs body 24 (PDF) / 24 vs 25 (HTML), recall unchanged -- one item
# either way, so this is for consistency, not a measured gain
#: a citation: [12], [3, 4], [1-5], (Smith et al., 2020), (Lekkas et al. 2012; Moreira 2007), Smith et al. (2019)
_CITE = re.compile(r"\[\s?\d{1,3}(?:\s?[,\u2013-]\s?\d{1,3})*\s?\]|\[[A-Z][A-Za-z'\-]+(?: et al\.?)?,? \d{4}[a-z]?[^\]]{0,80}\]|"
                   r"\([A-Z][A-Za-z'\-]+(?: et\.? al\.?| and [A-Z][A-Za-z'\-]+| & [A-Z][A-Za-z'\-]+)?,? \d{4}[a-z]?"
                   r"(?:[;,] [^)]{0,120})?\)|\b[A-Z][A-Za-z'\-]+ et\.? al\.?,? \(?\d{4}[a-z]?\)?")
_FIRST = re.compile(r"\b(?:we|our|ours|us|this (?:paper|work|study|article|letter))\b", re.I)
_EXPER = re.compile(r"\b(?:experiment|evaluat|test|trial|deploy|validat|demonstrat|conduct|collected|recorded|results?|"
                    r"perform)\w*", re.I)
_FUTURE = re.compile(r"\b(?:future work|in the future|will be|next steps?|we plan|plan to|remains? to be)\b", re.I)
_AFFIL = re.compile(r"\b(?:University|Institute|Department|Dept\.|Laborator(?:y|ies)|Center|Centre|School of|Faculty|"
                    r"Corporation|Inc\.)\b|@|(?i:\be-?mail\b)")      # proper nouns: 'a laboratory water tank' is a test


def own_mention(s, start, end, role):
    """Is the occurrence s[start:end] the paper's OWN use (True) or a mention of someone else's (False)? Gold set
    2026-10-14: 42 of 205 regex facts (first scoring; 40 of 194 one-to-one after the review) were things the paper
    only MENTIONS -- cited ("REMUS [27]", "(Woo et al. 2019)
    used DDPG") or in the introduction / related work. First person ("we", "our", "this paper") always reads as own."""
    if _FIRST.search(s):
        return True
    if role in ("intro", "related"):
        return False
    return not _CITE.search(s)


def body_text(text):
    """The text before the bibliography (probes ignore it: a cited GPU or repo is not this paper's). The cut is made on
    the RAW text, before sentence splitting: the heading fragment ('References [1] A.') is shorter than the 20-character
    sentence floor and was being dropped before it could be found (caught by a test, 2026-10-01). Pure."""
    t = re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", text or ""))
    late = [m for m in _REFS.finditer(t) if m.start() > len(t) * 0.4]
    return t[: late[0].start()] if late else t


def mine(text, attribution=True):
    """Full text -> {fact: values, ...} with evidence snippets. Pure."""
    return _mine(_with_roles([(None, body_text(text))]), attribution)


_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


#: a caption / cross-reference label right before a number: 'Fig. 1. ', 'Table 2. ', 'Section 3 '
_CAPTION = re.compile(r"(?:\bFigs?\.|\bFigure|\bFIGURE|\bTable|\bTABLE|\bTab\.|\bSection|\bSec\.|\bEq\.|"
                      r"\bAlgorithm|\bAlg\.|\bChapter)\s*$")


def _starts_sentence(n, h):
    """Does position h begin a sentence -- the text before it (spaces aside) is empty or ends a sentence?"""
    before = n[:h].rstrip()
    return not before or before[-1] in ".!?:)]\"'"


def _next_section(m):
    """The heading that ENDS the section m opened -- section k + 1 under any title ("4 RECON: ...", "III. SYSTEM")."""
    if m.group(1):
        return re.compile(r"(?:^|(?<=\s)|(?<=[^\d\s]\.))" + str(int(m.group(1)) + 1) + r"\.?\s+[A-Z][A-Za-z]")
    i = _ROMAN.index(m.group(2)) if m.group(2) in _ROMAN else -1
    return re.compile(r"(?:^|(?<=\s)|(?<=[^\d\s]\.))" + _ROMAN[i + 1] + r"\.\s+[A-Z]") if 0 <= i < len(_ROMAN) - 1 \
        else None


def _with_roles(pages):
    """[(page, normalised text)] -> [(page, sentence, section role)]: intro / related from a numbered Introduction /
    Related Work / Background ... heading until the NEXT section number (any title); body after any other recognised
    heading; "front" before the first one. Pure."""
    out, role, nxt = [], "front", None
    for pg, n in pages:
        heads = [(m.start(), m) for m in _HEAD.finditer(n)]
        pos = prev = 0
        for sent in _split(n):
            at = n.find(sent, pos)
            at = pos if at < 0 else at
            pos = at + len(sent)
            # the windows TILE the page: a heading anywhere between the last sentence's middle and this one's counts
            # (review 2026-10-14: 'I. INTRODUCTION' left the numeral 3 characters before the next sentence, and a
            # heading inside a dropped short fragment fell in no window at all)
            lo, hi = prev, at + max(1, len(sent) // 2)
            prev = hi
            for h, m in heads:
                if lo <= h < hi and not _CAPTION.search(n, max(0, h - 12), h):   # 'Table 2. Background noise' is a
                    role = _ROLE.get(m.group(3).split()[0].lower(), "body")      # caption, not a section (fix check)
                    nxt = _next_section(m) if role in ("intro", "related") else None
            # the NEXT section starts a sentence: searched from this sentence's start, not in the tail of the last one,
            # and never after a caption label ('Fig. 1. System') or inside a sentence ('labs field 2 AUVs at once')
            if nxt is not None and role in ("intro", "related") and any(
                    not _CAPTION.search(n, max(0, mm.start() - 12), mm.start()) and _starts_sentence(n, mm.start())
                    for mm in nxt.finditer(n, max(0, at - 1), hi)):
                role, nxt = "body", None
            out.append((pg, sent, role))
    return out


def _bib_cut(norm):
    """(page index, offset) where the bibliography starts, after the first 40% of the text: the FIRST heading whose next
    words look like a first entry (_REFS), else the LAST heading of the loose shape (_REFS_LOOSE), else None (fix check
    2026-10-14: the loose shape alone cut 9 papers at a 'References' table column; the strict shape alone left 46 real
    bibliographies uncut)."""
    total = sum(len(n) for n in norm) or 1
    acc, loose = 0, None
    for i, n in enumerate(norm):
        hit = next((m for m in _REFS.finditer(n) if acc + m.start() > 0.4 * total), None)
        if hit:
            return i, hit.start()
        for m in _REFS_LOOSE.finditer(n):
            if acc + m.start() > 0.4 * total:
                loose = (i, m.start())
        acc += len(n)
    return loose


def mine_pages(pages, attribution=True):
    """[page text, ...] -> the same as mine(), with every evidence snippet prefixed by its page ("p.4: ..."); the
    bibliography (its heading after the first 40% of the text) and every page after it are not mined. Pure."""
    norm = [re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", p or "")) for p in pages]
    cut = _bib_cut(norm)
    if cut:
        ci, cpos = cut
        app = None                                     # mining resumes at an APPENDIX after the bibliography (fix
        for i in range(ci, len(norm)):                 # check 2026-10-14: an A100 and SSIM / LPIPS stated only in an
            m = _APPENDIX.search(norm[i], cpos + 1 if i == ci else 0)   # appendix were lost); the list stays out
            if m:
                app = (i, m.start())
                break
        kept = norm[:ci] + [norm[ci][:cpos]]
        if app is not None:                            # skipped pages stay as empty strings: page numbers hold
            ai, apos = app
            if ai == ci:
                kept[-1] += " " + norm[ci][apos:]
                kept += norm[ci + 1:]
            else:
                kept += [""] * (ai - ci - 1) + [norm[ai][apos:]] + norm[ai + 1:]
        norm = kept
    return _mine(_with_roles([(i + 1, n) for i, n in enumerate(norm)]), attribution)


_HTML_ROLE = {"introduction": "intro", "related_work": "related"}
#: an appendix heading (not 'Supplementary', which reference titles carry)
_APPENDIX = re.compile(r"(?:^|(?<=[\s.]))(?:Appendix|APPENDIX|Appendices|APPENDICES)\b")


def mine_html(d, attribution=True):
    """A paper's arXiv HTML (tools/arxiv_html.py parsed JSON) -> the same facts as mine_pages, with section roles from
    the AUTHORS' section tree instead of guessed headings; references and acknowledgments are not mined; figure and
    table captions and table cells count as body. Evidence carries the section id ("S4.SS2: ...") instead of a page."""
    body = [("abstract", s, "front") for s in _split(re.sub(r"\s+", " ", d.get("abstract") or ""))]
    for sec in d.get("sections") or []:
        if sec.get("role") in ("references", "acknowledgments"):
            continue
        for s in _split(re.sub(r"\s+", " ", sec.get("text") or "")):
            body.append((sec["id"], s, _HTML_ROLE.get(sec.get("role"), "body")))
    for f in (d.get("figures") or []) + (d.get("tables") or []):
        cap = re.sub(r"\s+", " ", f.get("caption") or "")
        if len(cap) > 20:
            body.append((f.get("section") or f.get("id"), cap, "body"))
        for row in f.get("rows") or []:
            cells = " | ".join(c for c in row if c)
            if len(cells) > 20:
                body.append((f.get("section") or f.get("id"), cells + ".", "body"))
    return _mine(body, attribution)


#: the named fact types whose values are attributed (own use vs mention); the others are evidence lists
ATTRIBUTED = ("simulators", "world_models", "planners", "rl", "sensors", "platform", "hardware", "datasets", "metrics",
              "tested")


def _mine(body, attribution=True):
    """[(page, sentence, role)] -> facts. With attribution, a value of an ATTRIBUTED type is kept only when at least
    one sentence uses it as the paper's own (own_mention; 'tested' also needs an experiment word or first person, no
    future tense and no affiliation line); the others go to out["mentioned"][type] with their own evidence."""
    raw = _mine_raw(body)
    if not attribution:
        raw.pop("occ")
        return raw
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in raw.items() if k not in ("occ",)}
    out["mentioned"] = {}
    for t in ATTRIBUTED:
        keep, ment, pages = [], [], {}
        for v in raw[t]:
            occ = raw["occ"].get((t, v), [])
            own = [o for o in occ if o[2]]
            (keep if own else ment).append(v)
            pages[v] = (own or occ or [(None,)])[0][0]
        out[t] = keep
        if ment:
            out["mentioned"][t] = ment
        out["pages"][t] = {v: pages[v] for v in keep} if keep else {}
        ev = []
        for v in keep:
            for pg, s, _ in [o for o in raw["occ"].get((t, v), []) if o[2]][:1]:
                e = _snip(s) if pg is None else (f"p.{pg}: " if isinstance(pg, int) else f"{pg}: ") + _snip(s)
                if e not in ev:
                    ev.append(e)
        if t == "tested":
            for v in keep:
                out["evidence"][f"tested:{v}"] = [(_snip(s) if pg is None else (f"p.{pg}: " if isinstance(pg, int)
                                                                                 else f"{pg}: ") + _snip(s))
                                                  for pg, s, o in raw["occ"].get((t, v), []) if o][:MAX_EV]
            for v in ment:
                out["evidence"].pop(f"tested:{v}", None)
        elif ev:
            out["evidence"][t] = ev[:MAX_EV]
        else:
            out["evidence"].pop(t, None)
    out["pages"] = {k: v for k, v in out["pages"].items() if v}
    return out


def _tested_own(s, role):
    if _AFFIL.search(s) or _FUTURE.search(s):
        return False
    if _FIRST.search(s):
        return True
    # the abstract / front matter counts (dev split 2026-10-14: F1 0.72 vs 0.70 without it); affiliations are out above
    return role not in ("intro", "related") and bool(_EXPER.search(s)) and not _CITE.search(s)


def _mine_raw(body):
    out = {"hardware": [], "training": [], "code_own": [], "code_cited": [], "code_promise": [], "tested": [],
           "sensors": [], "platform": [], "data": [], "params": [], "speed": [], "datasets": [], "metrics": [],
           "simulators": [], "world_models": [], "planners": [], "rl": [], "evidence": {}}
    occ = {}                                           # (type, value) -> [(page, sentence, own?)]
    page = [None]
    role = ["body"]

    def seen(t, v, s, own=None):
        occ.setdefault((t, v), []).append((page[0], s, own_mention(s, 0, 0, role[0]) if own is None else own))

    def ev(key, s):
        lst = out["evidence"].setdefault(key, [])
        s = _snip(s) if page[0] is None else (f"p.{page[0]}: " if isinstance(page[0], int) else f"{page[0]}: ") + _snip(s)
        if len(lst) < MAX_EV and s not in lst:
            lst.append(s)

    def named(key, rx, s):
        for m in rx.findall(s):
            v = re.sub(r"\s+", " ", m).strip()
            if v:
                seen(key, v, s)
            if v and v not in out[key]:
                out[key].append(v)
                ev(key, s)

    first_page = {}                                    # type -> {value: the page it was first found on}
    for page[0], s, role[0] in body:
        before = {k: len(v) for k, v in out.items() if isinstance(v, list)}
        named("simulators", SIMULATORS, s)
        named("world_models", WORLD_MODELS, s)
        named("planners", PLANNERS, s)
        named("rl", RL_ALGOS, s)
        for g in GPU.findall(s):
            g = re.sub(r"\s+", " ", g.replace("NVIDIA", "").replace("GeForce", "")).strip()
            seen("hardware", g, s)
            if g not in out["hardware"]:
                out["hardware"].append(g)
            ev("hardware", s)
        if TRAIN_T.search(s):
            ev("training", s)
        for u in URL.findall(s):
            u = u.rstrip(".,;)")
            if "github.io" not in u and u.count("/") < 2:
                continue                                      # a bare 'github.com/user' is not a repo
            key = "code_own" if OWN_CODE.search(s) else "code_cited"
            if u not in out[key]:
                out[key].append(u)
            ev(key, s)
        if PROMISE.search(s):
            out["code_promise"] = ["promised"]
            ev("code_promise", s)
        for tag, rx in TESTED.items():
            if re.search(rx, s, re.I):
                seen("tested", tag, s, _tested_own(s, role[0]))
            if re.search(rx, s, re.I) and tag not in out["tested"]:
                out["tested"].append(tag)
            if re.search(rx, s, re.I):
                ev(f"tested:{tag}", s)
        for name, rx in SENSORS.items():
            if re.search(rx, s):
                seen("sensors", name, s)
            if re.search(rx, s) and name not in out["sensors"]:
                out["sensors"].append(name)
                ev("sensors", s)
        for p in PLATFORM.findall(s):
            p = p.strip()
            seen("platform", p, s)
            if p not in out["platform"]:
                out["platform"].append(p)
                ev("platform", s)
        if DATA.search(s):
            ev("data", s)
        for m in PARAMS.findall(s):
            v = re.sub(r"\s+", "", m) + " parameters"
            if v not in out["params"]:
                out["params"].append(v)
            ev("params", s)
        if SPEED_CONTEXT.search(s):
            for m in SPEED.findall(s):
                v = re.sub(r"\s+", " ", m).strip()
                if v not in out["speed"]:
                    out["speed"].append(v)
                ev("speed", s)
        for m in DATASET.findall(s):
            w = m.split(" ")
            if len(w) == 2 and w[1] in _NOT_A_NAME:
                m = w[0]                                       # 'ProcGen This benchmark' (a table header) -> ProcGen
            if m.split()[0].split("-")[0] in _NOT_A_NAME or len(m) < 3:
                continue
            seen("datasets", m, s)
            if m not in out["datasets"]:
                out["datasets"].append(m)
            ev("datasets", s)
        for name, rx in METRICS.items():
            if re.search(rx, s):
                seen("metrics", name, s)
            if re.search(rx, s) and name not in out["metrics"]:
                out["metrics"].append(name)
                ev("metrics", s)
        for k, n in before.items():
            for v in out[k][n:]:
                first_page.setdefault(k, {}).setdefault(v, page[0])
    out["pages"] = first_page
    out["occ"] = occ
    return out


def extract_text(pdf, cache_dir):
    """PDF -> text via pypdf, cached as <cache>/<stem>.txt. Raises RuntimeError when pypdf is missing / unreadable."""
    cache = Path(cache_dir) / (Path(pdf).stem + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    try:
        import pypdf
    except ImportError as e:
        raise RuntimeError("pypdf is not installed: python -m pip install pypdf") from e
    try:
        reader = pypdf.PdfReader(str(pdf))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        raise RuntimeError(f"could not read {Path(pdf).name}: {str(e)[:100]}") from e
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8", newline="\n")
    return text


def facts_md(rows):
    lines = ["<!-- generated by tools/paper_facts.py from the full-text PDFs; do not hand-edit -->", "",
             "# Facts mined from the full texts", "",
             "Candidates until a person reads the snippet (paper_facts.json carries them). OWN code = a URL in a sentence "
             "that presents it as the paper's code / release.", "",
             "| paper | hardware | own code (stated in text) | tested | sensors | platform |", "|---|---|---|---|---|---|"]
    for r in rows:
        f = r.get("facts") or {}
        cell = lambda k: ", ".join(f.get(k) or []) or "-"
        own = cell("code_own") + (" (promised)" if f.get("code_promise") else "")
        lines.append(f"| {r['name']} | {cell('hardware')} | {own} | {cell('tested')} | {cell('sensors')} | "
                     f"{cell('platform')} |")
    return "\n".join(lines) + "\n"


#: the facts whose evidence `--bwm` keeps (one snippet each): the ones a person checks before trusting a value
BWM_EVIDENCE = ("code_own", "code_promise", "hardware", "platform", "simulators")


def why(key, pdfs, cache=None, say=print):
    """Every fact of one held paper with the sentences behind it (re-mined from the text cache): the audit view the
    committed _facts.json is too small to carry (it keeps sentences for 5 types, pages for all)."""
    import paper_text as PT
    bwm = _project()
    cache = Path(cache) if cache else PT.default_cache(Path(pdfs))
    held = json.loads((bwm / "papers" / "_held.json").read_text(encoding="utf-8"))
    row = next((r for r in held.get("files") or [] if r["key"] == key), None)
    if not row or not PT.cache_file(cache, row["sha256"]).is_file():
        say(f"{key}: not held, or its text is not cached (tools/paper_text.py build)")
        return 1
    f = mine_pages(PT.cache_file(cache, row["sha256"]).read_text(encoding="utf-8").split(PT.FF))
    for k, vals in f.items():
        if k in ("evidence", "pages", "mentioned") or not vals:
            continue
        say(f"{k}: " + ", ".join(f"{v} (p.{f['pages'].get(k, {}).get(v, '?')})" for v in vals))
        for e in f["evidence"].get(k, []) + [x for t, xs in f["evidence"].items() if t.startswith(k + ":") for x in xs]:
            say(f"    {e}")
    for k, vals in (f.get("mentioned") or {}).items():
        say(f"{k} -- only MENTIONED (cited / related work), not counted: " + ", ".join(vals))
    return 0


def main_bwm(a):
    """Every held paper of the BlueBoat world-model index, through the shared paper_text cache."""
    import paper_text as PT
    bwm = _project()
    pdfs = Path(a.pdfs)
    cache = Path(a.cache) if a.cache else PT.default_cache(pdfs)
    if pdfs.is_dir():
        res = PT.build(pdfs, cache, say=lambda *_: None)
        if res["new_texts"]:
            print(f"text: extracted {res['new_texts']} new PDFs into {cache}")
    held = json.loads((bwm / "papers" / "_held.json").read_text(encoding="utf-8"))
    rows, missing = [], []
    for r in sorted(held.get("files") or [], key=lambda r: r["key"]):
        cf = PT.cache_file(cache, r["sha256"])
        if not cf.is_file():
            missing.append(r["key"])
            continue
        f = mine_pages(cf.read_text(encoding="utf-8").split(PT.FF))
        ev = {k: f["evidence"][k][:1] for k in BWM_EVIDENCE if f["evidence"].get(k)}
        facts = {k: v for k, v in f.items() if k not in ("evidence", "pages", "mentioned") and v}
        pages = {k: {v: pg for v, pg in vals.items() if v in facts.get(k, [])} for k, vals in f["pages"].items()
                 if k in facts}                     # every fact's first page: `--why <key>` prints the sentences
        row = {"key": r["key"], "sha256": r["sha256"], "facts": facts, "evidence": ev, "pages": pages}   # the
        #   file mined: `build` drops a row whose file is rejected or no longer held (third fix check 2026-10-15)
        if f.get("mentioned"):                      # named, but only as someone else's (cited / related work)
            row["mentioned"] = f["mentioned"]
        rows.append(row)
    facts = ("hardware", "code_own", "code_promise", "tested", "sensors", "platform", "datasets", "metrics",
             "simulators", "world_models", "planners", "rl")
    summary = {"papers_held": len(held.get("files") or []), "mined": len(rows), "no_text_cached": missing,
               "with": {k: sum(1 for r in rows if r["facts"].get(k)) for k in facts}}
    one = lambda o: json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    out = Path(a.out) if a.out else bwm / "papers" / "_facts.json"
    out.write_text(one({"schema": "svfr.bwm.facts/1", "summary": summary})[:-1] + ',"papers":[\n'
                   + ",\n".join(one(r) for r in rows) + "\n]}\n", encoding="utf-8", newline="\n")
    print(f"facts: {len(rows)} of {summary['papers_held']} held papers mined -> {out}; " +
          ", ".join(f"{k} {v}" for k, v in summary["with"].items()))
    if missing:
        print(f"  no cached text for {len(missing)}: run tools/paper_text.py build --pdfs <folder>")
    return 0


# ------------------------------------------------------------------------------------------------ gold set
#: the fact types the gold set labels (papers/_gold_facts.json), in report order
GOLD_TYPES = ("simulators", "world_models", "planners", "rl", "sensors", "platform", "hardware", "datasets", "metrics",
              "tested", "code_own", "code_promise")


def _toks(v):
    v = re.sub(r"^other:\s*", "", str(v or "").lower()).replace("behaviour", "behavior")
    v = re.sub(r"https?://|www\.", "", v)
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in re.findall(r"[a-z0-9]+", v)]


def same_value(mined, gold, typ=""):
    """Does a mined value name the gold item? Same tokens; an acronym given in parentheses ("model predictive control
    (MPC)" ~ "MPC"); or the shorter's tokens (2+, or one model-numbered token for hardware / platform) inside the
    longer's -- so "TD3" is NOT "TD3+BC" and "camera" is NOT "thermal camera". URLs: one a prefix of the other."""
    if typ == "code_own":
        a, b = ["/".join(_toks(x)) for x in (mined, gold)]
        return bool(a and b) and (a.startswith(b) or b.startswith(a))
    a, b = _toks(mined), _toks(gold)
    if not a or not b:
        return False
    if "".join(a) == "".join(b):
        return True
    for p in re.findall(r"\(([^)]*)\)", f"{mined} {gold}"):
        if "".join(_toks(p)) in ("".join(a), "".join(b)):
            return True
    # the subsequence rule sees the names WITHOUT their parentheticals (review 2026-10-14: the generic 'world model'
    # matched 'NWM (Navigation World Models)' through its expansion)
    a, b = _toks(re.sub(r"\([^)]*\)", " ", str(mined))), _toks(re.sub(r"\([^)]*\)", " ", str(gold)))
    if not a or not b:
        return False
    if a == b:                                         # 'NWM' ~ 'NWM (Navigation World Models)'
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) == 1 and not (typ in ("hardware", "platform") and (re.search(r"\d", short[0]) or len(short[0]) >= 5)):
        return False
    return any(long_[i:i + len(short)] == short for i in range(len(long_) - len(short) + 1))


def _in_vocab(typ, value):
    """Could the miner's probe for this type name this gold value at all (the probe run on the value itself)?"""
    v = re.sub(r"^other:\s*", "", value)
    rx = {"simulators": SIMULATORS, "world_models": WORLD_MODELS, "planners": PLANNERS, "rl": RL_ALGOS,
          "platform": PLATFORM, "hardware": GPU}.get(typ)
    if rx is not None:
        return bool(rx.search(v))
    if typ == "sensors":
        return v in SENSORS or any(re.search(r, v) for r in SENSORS.values())
    if typ == "metrics":
        return v in METRICS or any(re.search(r, v) for r in METRICS.values())
    if typ == "code_own":
        return bool(URL.search(v))
    return True                                       # tested / code_promise: fixed vocabularies; datasets: open


def mined_values(f):
    """mine_pages output -> {gold type: [values]}."""
    out = {t: list(f.get(t) or []) for t in GOLD_TYPES if t not in ("code_promise",)}
    out["code_promise"] = ["promised"] if f.get("code_promise") else []
    return out


GOLD_COUNTS = ("tp", "attributed", "spurious", "duplicate", "gold_used", "found", "gold_used_in_vocab",
               "found_in_vocab")


def gold_score(gold, mined):
    """gold: {key: [items]}, mined: {key: {type: [values]}} -> per type: tp, attributed (named, but the paper only
    MENTIONS it), spurious (not in the gold at all), duplicate (another name for a gold item already credited --
    neither right nor wrong, left out of precision), gold_used, found, gold_used_in_vocab, found_in_vocab, and
    precision / recall / recall_in_vocab. One-to-one: each gold item is credited at most once. Pure."""
    rows = {t: dict.fromkeys(GOLD_COUNTS, 0) for t in GOLD_TYPES}
    for key, items in gold.items():
        m = mined.get(key) or {}
        for t in GOLD_TYPES:
            used = [it for it in items if it["type"] == t and it["role"] == "used"]
            ment = [it for it in items if it["type"] == t and it["role"] == "mentioned"]
            r = rows[t]
            taken = set()                              # ONE-TO-ONE: a gold item is credited once; another mined alias
            for v in m.get(t, []):                     # of it ('D4RL' + 'D4Rl', 'Jackal' + 'Clearpath Jackal') is a
                hits = [i for i, it in enumerate(used) if same_value(v, it["value"], t)]   # duplicate, not a second
                free = [i for i in hits if ("u", i) not in taken]                           # hit (review 2026-10-14)
                if free:
                    taken.add(("u", free[0]))
                    r["tp"] += 1
                    continue
                if hits:
                    r["duplicate"] += 1
                    continue
                mh = [i for i, it in enumerate(ment) if same_value(v, it["value"], t)]
                mfree = [i for i in mh if ("m", i) not in taken]
                if mfree:                              # one-to-one for MENTIONED items too (fix check: two aliases of
                    taken.add(("m", mfree[0]))         # one mentioned item were charged twice)
                    r["attributed"] += 1
                elif mh:
                    r["duplicate"] += 1
                else:
                    r["spurious"] += 1
            for i, it in enumerate(used):
                hit = ("u", i) in taken
                r["gold_used"] += 1
                r["found"] += hit
                if _in_vocab(t, it["value"]):
                    r["gold_used_in_vocab"] += 1
                    r["found_in_vocab"] += hit
    for r in rows.values():
        n = r["tp"] + r["attributed"] + r["spurious"]
        r["precision"] = round(r["tp"] / n, 3) if n else None
        r["recall"] = round(r["found"] / r["gold_used"], 3) if r["gold_used"] else None
        r["recall_in_vocab"] = round(r["found_in_vocab"] / r["gold_used_in_vocab"], 3) if r["gold_used_in_vocab"] else None
    return rows


def gold_eval(gold_path, pdfs, cache=None, variant="attributed", split="all", say=print):
    """Score the miner against the gold set (papers/_gold_facts.json): every gold paper re-mined from the text cache."""
    import paper_text as PT
    bwm = _project()
    cache = Path(cache) if cache else PT.default_cache(Path(pdfs))
    gold = {p["key"]: p["items"] for p in json.loads(Path(gold_path).read_text(encoding="utf-8"))["papers"]
            if split == "all" or p.get("split") == split}
    held = {r["key"]: r["sha256"] for r in json.loads((bwm / "papers" / "_held.json").read_text(encoding="utf-8"))
            .get("files") or []}
    html = {}
    if variant == "html":                              # the arXiv HTML where it is parsed, the PDF text elsewhere
        import arxiv_html as AH
        hc = AH.default_cache(Path(pdfs))
        for aid, r in AH._load_manifest(hc).items():
            if r.get("status") == "ok" and r.get("key") in gold:
                html[r["key"]] = AH.load_parsed(hc, aid)
    mined = {}
    for key in gold:
        f = PT.cache_file(cache, held.get(key, "-"))
        if not f.is_file():
            say(f"  {key}: no cached text -- skipped")
            continue
        if html.get(key):
            mined[key] = mined_values(mine_html(html[key]))
            continue
        mined[key] = mined_values(mine_pages(f.read_text(encoding="utf-8").split(PT.FF),
                                             attribution=variant != "raw"))
    if variant == "html":
        say(f"  {sum(1 for k in mined if html.get(k))} of {len(mined)} papers mined from their arXiv HTML")
    gold = {k: v for k, v in gold.items() if k in mined}
    rows = gold_score(gold, mined)
    tot = {k: sum(r[k] for r in rows.values()) for k in GOLD_COUNTS}
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    say(f"gold set ({split}): {len(gold)} papers, {sum(len(v) for v in gold.values())} items ({tot['gold_used']} "
        f"used) -- variant {variant}")
    say(f"{'type':13} {'precision':>9} {'tp':>4} {'attrib':>6} {'spur':>5} {'dup':>4} | {'recall':>6} "
        f"{'found/used':>10} | in-vocab recall")
    for t, r in list(rows.items()) + [("ALL (micro)", None)]:
        if r is None:
            n = tot["tp"] + tot["attributed"] + tot["spurious"]
            r = dict(tot, precision=tot["tp"] / max(1, n), recall=tot["found"] / max(1, tot["gold_used"]),
                     recall_in_vocab=tot["found_in_vocab"] / max(1, tot["gold_used_in_vocab"]))
        say(f"{t:13} {fmt(r['precision']):>9} {r['tp']:>4} {r['attributed']:>6} {r['spurious']:>5} "
            f"{r['duplicate']:>4} | {fmt(r['recall']):>6} {str(r['found']) + '/' + str(r['gold_used']):>10} | "
            f"{fmt(r['recall_in_vocab'])} ({r['found_in_vocab']}/{r['gold_used_in_vocab']})")
    return {"variant": variant, "split": split, "papers": len(gold), "types": rows, "total": tot}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refs", help="wm_options.json or a verify_refs.py JSON")
    ap.add_argument("--dir", help="folder of full-text PDFs (gitignored)")
    ap.add_argument("--out", default="", help="facts JSON to write (a .md table is written beside it)")
    ap.add_argument("--cache", default="", help="extracted-text cache (default <dir>/.text; --bwm: <pdfs>_text)")
    ap.add_argument("--bwm", action="store_true", help="every held paper of the BlueBoat world-model index")
    ap.add_argument("--pdfs", default="", help="--bwm: the papers folder (default: paper_text.DEFAULT_PDFS -- "
                                               "$SVFR_BWM_PAPERS, else found by data_roots)")
    ap.add_argument("--why", default="", metavar="KEY", help="--bwm: one paper's facts with the sentences behind them")
    ap.add_argument("--gold", nargs="?", const=str(_project() / "papers" / "_gold_facts.json"),
                    default="", metavar="FILE", help="score the miner on the gold set (default papers/_gold_facts.json)")
    ap.add_argument("--variant", default="attributed", choices=["attributed", "raw", "html"],
                    help="--gold: the miner as shipped (attributed: own use vs mention) or every regex hit (raw)")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"],
                    help="--gold: DEV papers tuned the miner; TEST papers (default) were labeled after and never tuned on")
    a = ap.parse_args(argv)
    if not a.pdfs:
        import paper_text as PT
        a.pdfs = str(PT.DEFAULT_PDFS)
    if a.why:
        return why(a.why, a.pdfs, a.cache or None)
    if a.gold:
        gold_eval(a.gold, a.pdfs, a.cache or None, variant=a.variant, split=a.split)
        return 0
    if a.bwm:
        return main_bwm(a)
    if not (a.refs and a.dir and a.out):
        ap.error("--refs, --dir and --out are required (or --bwm)")
    papers = PP.paper_rows(json.loads(Path(a.refs).read_text(encoding="utf-8")))
    folder = Path(a.dir)
    pdfs = [PP.pdf_identity(p) for p in sorted(folder.glob("*.pdf"))]
    found = PP.match(papers, pdfs)
    cache = Path(a.cache) if a.cache else folder / ".text"
    rows, errors = [], []
    for p in papers:
        key = p["key"] or p["title"]
        row = {"key": p["key"], "name": p.get("name") or p["title"], "file": found.get(key, ""), "facts": None}
        if row["file"]:
            try:
                text = extract_text(folder / row["file"], cache)
                row["chars"] = len(text)
                row["facts"] = mine(text)
            except RuntimeError as e:
                errors.append(f"{key}: {e}")
                row["error"] = str(e)
        rows.append(row)
    out = Path(a.out)
    have = [r for r in rows if r["facts"]]
    summary = {"papers": len(rows), "mined": len(have),
               "with_hardware": sum(bool(r["facts"]["hardware"]) for r in have),
               "with_own_code_url": sum(bool(r["facts"]["code_own"]) for r in have),
               "with_code_promise": sum(bool(r["facts"]["code_promise"]) for r in have),
               "tested": {t: sum(t in r["facts"]["tested"] for r in have) for t in TESTED}}
    out.write_text(json.dumps({"summary": summary, "papers": rows}, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    out.with_suffix(".md").write_text(facts_md(rows), encoding="utf-8", newline="\n")
    print(f"mined {len(have)} of {len(rows)} papers -> {out}; hardware named in {summary['with_hardware']}, own code URL in "
          f"{summary['with_own_code_url']}, tested {summary['tested']}")
    for e in errors:
        print("ERROR: " + e.encode("ascii", "replace").decode())
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
