"""report/boat_world_model/citation_map.py -- how the BlueBoat world-model papers cite EACH OTHER: the in-index citation
graph (OpenAlex reference lists, papers/_meta.json), a FIXED layout of it, its topic-to-topic flows, its most-cited
backbone and its citation ages -- as committed figures and as the data of the atlas's Citations tab.

    python report/boat_world_model/build_bwm.py citemap        # layout -> papers/_citemap.json, then the figures
    python report/boat_world_model/citation_map.py render      # the figures only, from the committed layout

WHY (owner, 2026-10-15): "make maps of how the papers interconnect now that we have the citation information". The
2026-10-15 `meta` run (owner's OpenAlex key) holds every index paper's OpenAlex record with its reference list, and
`meta_links.py` already resolves which index papers cite which (6,268 links). This turns that into maps a person can
read: where each topic's papers sit in the network, which topics build on which, which papers everything rests on,
and how old the cited work is. The atlas draws the same layout interactively (build_bwm.py BROWSER_TEMPLATE).

LAYOUT. Force-directed in the ForceAtlas2 style (Jacomy et al. 2014): every pair of papers repels in proportion to
the product of their degrees (+1), every citation pulls its two ends together linearly, and a gravity proportional to
degree holds the components together; the start is seeded by topic group, so the groups begin apart. numpy, seeded
-- but floating-point SIMD paths differ between machines, so the positions are an INPUT: `citemap` writes them to
papers/_citemap.json (committed, with the signature of the graph they lay out), and `build` only reads them; a paper
linked after the last `citemap` is placed on an outer ring and `build` says so (run `citemap` after `meta`).

RELATIONS (integers from committed data, recomputed by every build): cites / cited by (in the index); BIBLIOGRAPHIC
COUPLING -- two index papers whose OpenAlex reference lists share >= 3 works (a work cited by more than 200 index
papers is too generic to count); CO-CITATION -- two index papers that other index papers cite together (>= 2 times).

HONEST SCOPE (do NOT overclaim): a link is a citation OpenAlex records between two papers that are BOTH in this index;
the index is a curated selection (2,813 papers), so the maps show how THIS literature base hangs together, not the
field. OpenAlex's reference lists are incomplete for some venues (conference papers, preprints, theses): 1,916 of the
2,813 papers have one; a paper with no list can only be cited, never cite. 989 papers have no link in the index at all
and are not drawn. Topic groups are this program's 22 review topics folded into 7 for colour (atlas.toml [[groups]];
another project's own, tools/atlas_project.py), a paper with several topics takes its majority group.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CITEMAP = "_citemap.json"                                 # papers/_citemap.json
FIG_DIR = HERE / "figures"
FIGS = ("citation_network", "citation_topic_flows", "citation_backbone", "citation_age")
METRICS = "citation_map_metrics.json"                     # figures/citation_map_metrics.json (figure_explain reads it)
MIN_SHARED_REFS = 3                                       # bibliographic coupling: shared OpenAlex references
MAX_REF_FANOUT = 200                                      # a reference cited by more index papers is too generic
MIN_COCITED = 2                                           # co-citation: cited together by >= this many index papers
TOP_RELATED = 5                                           # coupled / co-cited partners kept per paper

#: the topic groups (one colour panel each) and topic labels of the project being drawn -- its atlas.toml, through
#: tools/atlas_project.py (the BlueBoat ones: report/boat_world_model/atlas.toml); `use_project(base)` re-reads them
TOPIC_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
TOPIC_LABEL: Dict[str, str] = {}


def _ap():
    """tools/atlas_project.py, loaded once by path (this folder is not a package)."""
    name = "_citation_map_atlas_project"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "atlas_project.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def use_project(base: Optional[Path] = None) -> None:
    """Draw with THIS project's topic groups and labels (its atlas.toml; generic ones when it has none)."""
    global TOPIC_GROUPS, TOPIC_LABEL
    AP = _ap()
    proj = Path(base) if base else AP.project_dir()
    TOPIC_GROUPS, TOPIC_LABEL = AP.topic_groups(proj), AP.topic_labels(proj)


use_project()


# ------------------------------------------------------------------------------------------------ the graph (pure)
def group_of(topics: Sequence[str]) -> int:
    """The topic group a paper is drawn in: the group holding most of its topics, the earlier group on a tie."""
    counts = [sum(1 for t in topics or () if t in ts) for _, ts in TOPIC_GROUPS]
    best = max(counts) if counts else 0
    return counts.index(best) if best else len(TOPIC_GROUPS) - 1


def edges_of(meta: Dict[str, dict], keys: Sequence[str]) -> List[Tuple[str, str]]:
    """(citing, cited) for every in-index reference, both ends index papers, no self-citation; sorted, unique."""
    known = set(keys)
    out = set()
    for k, r in meta.items():
        if k not in known:
            continue
        for x in r.get("refs_in_corpus") or []:
            if x in known and x != k:
                out.add((k, x))
    return sorted(out)


def signature(edges: Sequence[Tuple[str, str]]) -> str:
    """What a layout was made for: a hash of the sorted link list (the papers it holds follow from it)."""
    return hashlib.sha256("\n".join(f"{a} {b}" for a, b in sorted(edges)).encode()).hexdigest()[:16]


def components(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> List[List[str]]:
    """Connected components of the undirected graph, largest first (ties: the smallest key first)."""
    nb: Dict[str, set] = {n: set() for n in nodes}
    for a, b in edges:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for n in sorted(nb):
        if n in seen:
            continue
        comp, stack = [], [n]
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in nb[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def relations(meta: Dict[str, dict], keys: Sequence[str]) -> Dict[str, Dict[str, List]]:
    """Per paper: 'out' (cites, in the index), 'in' (cited by), 'couple' [(other, shared refs)] and 'cocite'
    [(other, times cited together)], the last two the TOP_RELATED strongest (ties by key). Integers only."""
    known = set(keys)
    edges = edges_of(meta, keys)
    out: Dict[str, List[str]] = {}
    inn: Dict[str, List[str]] = {}
    for a, b in edges:
        out.setdefault(a, []).append(b)
        inn.setdefault(b, []).append(a)
    refs = {k: sorted(set((meta[k].get("openalex") or {}).get("refs") or [])) for k in sorted(meta) if k in known}
    inv: Dict[str, List[str]] = {}
    for k, rl in refs.items():
        for w in rl:
            inv.setdefault(w, []).append(k)
    shared: Dict[Tuple[str, str], int] = {}
    for w in sorted(inv):
        ks = inv[w]
        if 2 <= len(ks) <= MAX_REF_FANOUT:
            for i, a in enumerate(ks):
                for b in ks[i + 1:]:
                    shared[(a, b)] = shared.get((a, b), 0) + 1
    co: Dict[Tuple[str, str], int] = {}
    for c in sorted(out):
        rs = sorted(out[c])
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                co[(a, b)] = co.get((a, b), 0) + 1

    def top(pairs: Dict[Tuple[str, str], int], floor: int) -> Dict[str, List[Tuple[str, int]]]:
        per: Dict[str, List[Tuple[str, int]]] = {}
        for (a, b), n in pairs.items():
            if n >= floor:
                per.setdefault(a, []).append((b, n))
                per.setdefault(b, []).append((a, n))
        return {k: sorted(v, key=lambda x: (-x[1], x[0]))[:TOP_RELATED] for k, v in per.items()}
    return {"out": {k: sorted(v) for k, v in out.items()}, "in": {k: sorted(v) for k, v in inn.items()},
            "couple": top(shared, MIN_SHARED_REFS), "cocite": top(co, MIN_COCITED)}


def topic_flows(papers: Dict[str, dict], edges: Sequence[Tuple[str, str]], topics: Sequence[str]) -> List[List[float]]:
    """flows[i][j] = in-index citations FROM papers of topic i TO papers of topic j; a link between multi-topic papers
    is shared out evenly (each link adds 1 in total), so row sums are the topic's outgoing links."""
    ix = {t: i for i, t in enumerate(topics)}
    m = [[0.0] * len(topics) for _ in topics]
    for a, b in edges:
        ta = [t for t in papers[a].get("topics") or [] if t in ix]
        tb = [t for t in papers[b].get("topics") or [] if t in ix]
        if not ta or not tb:
            continue
        w = 1.0 / (len(ta) * len(tb))
        for x in ta:
            for y in tb:
                m[ix[x]][ix[y]] += w
    return [[round(v, 4) for v in row] for row in m]


# ------------------------------------------------------------------------------------------------- layout (numpy)
def layout(keys: Sequence[str], edges: Sequence[Tuple[str, str]], groups: Dict[str, int], iters: int = 700,
           seed: int = 13, box: Tuple[float, float] = (1000.0, 760.0)) -> Dict[str, Tuple[float, float]]:
    """ForceAtlas2-style positions for the papers that have a link, scaled into `box` (x right, y down), rounded to
    0.1. Degree-weighted repulsion between every pair, linear attraction along each link (a mutual citation twice),
    degree-weighted gravity to the centre; a cooling step cap; start seeded by topic group."""
    import numpy as np
    keys = sorted(keys)
    n = len(keys)
    if n == 0:
        return {}
    ix = {k: i for i, k in enumerate(keys)}
    A = np.zeros((n, n))
    for a, b in edges:
        if a in ix and b in ix:
            A[ix[a], ix[b]] += 1.0
            A[ix[b], ix[a]] += 1.0
    deg = (A > 0).sum(axis=1)
    mass = deg + 1.0
    rng = np.random.default_rng(seed)
    g = np.array([groups.get(k, 0) for k in keys], dtype=float)
    ng = max(1, len(TOPIC_GROUPS))
    ang = 2 * math.pi * (g + 0.5) / ng + rng.normal(0.0, 0.35, n)
    rad = 30.0 * (1.0 + rng.random(n))
    pos = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    mm = np.outer(mass, mass)
    kr, kg = 6.0, 0.08
    for it in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist2 = (d * d).sum(-1) + 1e-6
        rep = (d * (kr * mm / dist2)[..., None]).sum(axis=1)
        att = -(A[..., None] * d).sum(axis=1)
        r = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-9
        grav = -kg * mass[:, None] * pos / r * np.sqrt(r)
        f = rep + att + grav
        step = 12.0 * (1.0 - it / iters) + 0.3
        disp = f / mass[:, None]
        length = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
        pos = pos + disp * np.minimum(1.0, step / length)
    # a few tiny components drift far out and would squash the main network into the middle of the box: pull every
    # paper beyond the 98.5th-percentile radius back onto it (their exact place carries no meaning)
    c = np.median(pos, axis=0)
    r = np.linalg.norm(pos - c, axis=1)
    rmax = float(np.quantile(r, 0.985))
    far = r > rmax
    pos[far] = c + (pos[far] - c) / r[far, None] * rmax
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    s = min((box[0] - 40) / span[0], (box[1] - 40) / span[1])
    off = (np.array(box) - span * s) / 2
    xy = (pos - lo) * s + off
    return {k: (round(float(xy[i, 0]), 1), round(float(xy[i, 1]), 1)) for k, i in ix.items()}


def fallback_xy(keys: Sequence[str], box: Tuple[float, float] = (1000.0, 760.0)) -> Dict[str, Tuple[float, float]]:
    """Papers linked after the last `citemap`: on a ring just inside the box, in key order (deterministic)."""
    keys = sorted(keys)
    cx, cy, r = box[0] / 2, box[1] / 2, min(box) / 2 - 8
    return {k: (round(cx + r * math.cos(2 * math.pi * i / max(1, len(keys))), 1),
                round(cy + r * math.sin(2 * math.pi * i / max(1, len(keys))), 1)) for i, k in enumerate(keys)}


# ----------------------------------------------------------------------------------------------- inputs from build
def _bwm():
    name = "_citation_map_build_bwm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "build_bwm.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def inputs(base: Path = HERE) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """(index papers by key, meta rows by key, aliases folded) -- read the way build_bwm reads them."""
    use_project(base)
    B = _bwm()
    papers, alias, _ = B.build_papers(base)
    return {p["key"]: p for p in papers}, B.meta_rows(base, alias)


def load_citemap(base: Path = HERE) -> dict:
    try:
        return json.loads((Path(base) / "papers" / CITEMAP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def positions(base: Path, linked: Sequence[str]) -> Tuple[Dict[str, Tuple[float, float]], List[str], dict]:
    """(positions for every linked paper, papers that took a fallback position, the citemap header)."""
    cm = load_citemap(base)
    have = {k: tuple(v) for k, v in (cm.get("xy") or {}).items()}
    missing = sorted(k for k in linked if k not in have)
    xy = {k: have[k] for k in linked if k in have}
    xy.update(fallback_xy(missing, tuple(cm.get("box") or (1000.0, 760.0))))
    return xy, missing, {k: v for k, v in cm.items() if k != "xy"}


def write_citemap(base: Path = HERE, today: str = "", iters: int = 700, say=print) -> dict:
    """`build_bwm.py citemap`: lay out the current in-index citation graph and write papers/_citemap.json."""
    papers, meta = inputs(base)
    edges = edges_of(meta, list(papers))
    linked = sorted({x for e in edges for x in e})
    groups = {k: group_of(papers[k].get("topics") or []) for k in linked}
    say(f"laying out {len(linked)} papers, {len(edges)} links ({iters} iterations) ...")
    xy = layout(linked, edges, groups, iters=iters)
    doc = {"schema": "svfr.bwm.citemap/1", "made": today, "graph": signature(edges), "papers": len(linked),
           "links": len(edges), "box": [1000.0, 760.0], "iters": iters, "seed": 13,
           "method": "ForceAtlas2-style: degree-weighted repulsion, linear attraction, degree gravity (citation_map.py)",
           "xy": {k: list(xy[k]) for k in sorted(xy)}}
    head = json.dumps({k: v for k, v in doc.items() if k != "xy"}, sort_keys=True, separators=(",", ":"))
    rows = ",\n".join(f"{json.dumps(k)}:{json.dumps(v)}" for k, v in doc["xy"].items())
    text = head[:-1] + ',"xy":{\n' + rows + "\n}}\n"
    (Path(base) / "papers" / CITEMAP).write_text(text, encoding="utf-8", newline="\n")
    say(f"wrote papers/{CITEMAP}: {len(xy)} positions, graph {doc['graph']}")
    return doc


# --------------------------------------------------------------------------------------------- metrics + atlas data
def metrics(papers: Dict[str, dict], meta: Dict[str, dict]) -> dict:
    """The numbers the figures, their captions (figure_explain) and the atlas state: counts, components, the
    backbone, topic flows, citation ages, bridges."""
    keys = sorted(papers)
    edges = edges_of(meta, keys)
    linked = sorted({x for e in edges for x in e})
    comps = components(linked, edges)
    indeg: Dict[str, int] = {}
    for _, b in edges:
        indeg[b] = indeg.get(b, 0) + 1
    topics = sorted({t for p in papers.values() for t in p.get("topics") or []})
    flows = topic_flows(papers, edges, topics)
    within = sum(flows[i][i] for i in range(len(topics)))
    ages = [papers[a]["year"] - papers[b]["year"] for a, b in edges if papers[a].get("year") and papers[b].get("year")]
    ages.sort()
    backbone = sorted(indeg, key=lambda k: (-indeg[k], k))[:25]
    grp = {k: group_of(papers[k].get("topics") or []) for k in keys}
    cited_from_groups: Dict[str, set] = {}
    for a, b in edges:
        cited_from_groups.setdefault(b, set()).add(grp[a])
    bridges = sorted((k for k in cited_from_groups if len(cited_from_groups[k]) >= 4),
                     key=lambda k: (-len(cited_from_groups[k]), -indeg.get(k, 0), k))[:15]
    with_refs = sum(1 for k in keys if ((meta.get(k) or {}).get("openalex") or {}).get("refs"))
    cross_group = sum(1 for a, b in edges if grp[a] != grp[b])
    return {
        "papers": len(keys), "with_reference_list": with_refs, "links": len(edges), "linked_papers": len(linked),
        "isolated_papers": len(keys) - len(linked), "giant_component": len(comps[0]) if comps else 0,
        "components": len(comps), "mean_links_per_linked_paper": round(2 * len(edges) / max(1, len(linked)), 2),
        "within_topic_share": round(within / max(1e-9, sum(map(sum, flows))), 3),
        "cross_group_share": round(cross_group / max(1, len(edges)), 3),
        "age_median": ages[len(ages) // 2] if ages else None,
        "age_p90": ages[int(0.9 * len(ages))] if ages else None,
        "age_share_le3": round(sum(1 for a in ages if a <= 3) / max(1, len(ages)), 3),
        "backbone": [{"key": k, "cited_here": indeg[k], "year": papers[k].get("year"), "title": papers[k]["title"],
                      "group": TOPIC_GROUPS[grp[k]][0]} for k in backbone],
        "bridges": [{"key": k, "groups": len(cited_from_groups[k]), "cited_here": indeg.get(k, 0)} for k in bridges],
        "topics": topics, "flows": flows,
        "groups": [{"name": n, "papers": sum(1 for k in keys if grp[k] == i),
                    "linked": sum(1 for k in linked if grp[k] == i)} for i, (n, _) in enumerate(TOPIC_GROUPS)],
    }


def atlas_data(papers: Dict[str, dict], meta: Dict[str, dict], base: Path, order: Sequence[str]) -> dict:
    """The Citations tab's data, indexed by the atlas's paper order (`order` = the keys of D.papers): nodes [paper
    index, x, y, group], links [i, j] (node indices, citing -> cited), per-node coupled / co-cited partners, the
    topic flows and the backbone. Compact on purpose (it is part of the page)."""
    use_project(base)
    keys = sorted(papers)
    rel = relations(meta, keys)
    edges = edges_of(meta, keys)
    linked = sorted({x for e in edges for x in e})
    xy, missing, head = positions(base, linked)
    pidx = {k: i for i, k in enumerate(order)}
    nodes = sorted(linked, key=lambda k: pidx.get(k, 1 << 30))
    nidx = {k: i for i, k in enumerate(nodes)}
    m = metrics(papers, meta)
    return {
        "nodes": [[pidx[k], xy[k][0], xy[k][1], group_of(papers[k].get("topics") or [])] for k in nodes],
        "links": [[nidx[a], nidx[b]] for a, b in edges],
        "couple": {str(pidx[k]): [[pidx[o], n] for o, n in v] for k, v in sorted(rel["couple"].items()) if k in pidx},
        "cocite": {str(pidx[k]): [[pidx[o], n] for o, n in v] for k, v in sorted(rel["cocite"].items()) if k in pidx},
        "groups": [n for n, _ in TOPIC_GROUPS],
        "topics": [[t, TOPIC_LABEL.get(t, t)] for t in m["topics"]], "flows": m["flows"],
        "backbone": [pidx[b["key"]] for b in m["backbone"]],
        "box": head.get("box") or [1000.0, 760.0], "laid_out": head.get("made", ""), "unplaced": len(missing),
        "stats": {k: m[k] for k in ("links", "linked_papers", "isolated_papers", "giant_component", "components",
                                   "within_topic_share", "cross_group_share", "age_median", "with_reference_list")},
    }


# ------------------------------------------------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"                       # dataviz reference palette, slots 1-2
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]       # sequential blue 100 -> 700


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "svg.hashsalt": "citation_map", "path.simplify": False})
    return plt


def fig_network(papers, edges, xy, out: Path) -> Path:
    """Small multiples: the whole linked network in every panel, one topic group highlighted per panel (one accent hue
    -- seven hues on one scatter cannot all be told apart), sized by in-index citations; the last panel highlights the
    25-paper backbone."""
    plt = _plt()
    from matplotlib.collections import LineCollection
    linked = sorted(xy)
    indeg: Dict[str, int] = {}
    for _, b in edges:
        indeg[b] = indeg.get(b, 0) + 1
    grp = {k: group_of(papers[k].get("topics") or []) for k in linked}
    segs = [[xy[a], xy[b]] for a, b in edges]
    size = {k: 3 + 0.55 * indeg.get(k, 0) for k in linked}          # marker AREA grows with the citations
    backbone = sorted(indeg, key=lambda k: (-indeg[k], k))[:25]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    panels = [(n, {k for k in linked if grp[k] == i}) for i, (n, _) in enumerate(TOPIC_GROUPS)]
    panels.append(("Backbone: the 25 most-cited papers", set(backbone)))
    for ax in list(axes.flat)[len(panels):]:                         # a project with fewer than 7 groups
        ax.set_visible(False)
    for ax, (name, hot) in zip(axes.flat, panels):
        ax.add_collection(LineCollection(segs, colors=GRID, linewidths=0.35, zorder=1))
        cold = [k for k in linked if k not in hot]
        ax.scatter([xy[k][0] for k in cold], [xy[k][1] for k in cold], s=[size[k] for k in cold], c="#c9c8c2",
                   linewidths=0, zorder=2)
        hk = sorted(hot, key=lambda k: indeg.get(k, 0))
        ax.scatter([xy[k][0] for k in hk], [xy[k][1] for k in hk], s=[size[k] * 1.4 for k in hk], c=BLUE,
                   edgecolors=SURFACE, linewidths=0.6, zorder=3)
        placed: List[Tuple[float, float, float, float]] = []
        want = 3 if name != panels[-1][0] else 7
        for k in sorted(hot, key=lambda k: (-indeg.get(k, 0), k)):
            if len(placed) >= want:
                break
            w, h = 13.5 * len(k), 26.0                    # ~ data units of a 7-pt label in a ~4-inch panel
            x, y = xy[k]
            for bx in ((x + 8, y - h / 2), (x - 8 - w, y - h / 2), (x - w / 2, y - h - 6), (x - w / 2, y + 8)):
                box = (bx[0], bx[1], bx[0] + w, bx[1] + h)
                if 0 <= box[0] and box[2] <= 1000 and 0 <= box[1] and box[3] <= 760 and not any(
                        box[0] < o[2] and box[2] > o[0] and box[1] < o[3] and box[3] > o[1] for o in placed):
                    placed.append(box)
                    ax.annotate(k, (x, y), xytext=(box[0], box[1] + h * 0.75), textcoords="data", fontsize=7,
                                color=INK, zorder=4,
                                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5, shrinkA=0, shrinkB=2))
                    break
        n_hot = len(hot)
        ax.set_title(f"{name}\n{n_hot} linked papers" if name != panels[-1][0] else name, fontsize=9.5, loc="left")
        ax.set_xlim(0, 1000)
        ax.set_ylim(760, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"How the indexed papers cite each other: {len(edges):,} citations among {len(linked):,} papers "
                 "(OpenAlex reference lists; papers with no link in the index not drawn)", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    fig.text(0.01, 0.005, "Each panel shows the same fixed layout; blue = the panel's papers, size = citations from "
             "other papers in the index. Layout: ForceAtlas2-style (citation_map.py), positions in papers/_citemap.json.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_flows(m: dict, out: Path) -> Path:
    """Row-normalised topic-to-topic citation shares (sequential blue): row = the citing topic."""
    plt = _plt()
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    topics, flows = m["topics"], np.array(m["flows"])
    rows = flows.sum(axis=1, keepdims=True)
    share = np.divide(flows, rows, out=np.zeros_like(flows), where=rows > 0)
    cmap = LinearSegmentedColormap.from_list("seqblue", [SURFACE] + SEQ)
    fig, ax = plt.subplots(figsize=(11.5, 9.6))
    im = ax.imshow(share, cmap=cmap, vmin=0, vmax=max(0.5, float(share.max())))
    labels = [TOPIC_LABEL.get(t, t) for t in topics]
    ax.set_xticks(range(len(topics)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(topics)), [f"{l} ({int(round(r))})" for l, r in zip(labels, rows[:, 0])], fontsize=8)
    for i in range(len(topics)):
        for j in range(len(topics)):
            if share[i, j] >= 0.15:
                ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if share[i, j] >= 0.3 else INK)
    ax.set_xlabel("cited topic")
    ax.set_ylabel("citing topic (its links to papers in the index)")
    ax.set_title("Which topics build on which: share of each topic's in-index citations going to each topic",
                 loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("share of the citing topic's links", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_backbone(m: dict, out: Path) -> Path:
    """The 25 papers cited most often by other papers in the index (one series: no legend)."""
    plt = _plt()
    bb = list(reversed(m["backbone"]))
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ys = range(len(bb))
    ax.barh(ys, [b["cited_here"] for b in bb], height=0.62, color=BLUE)
    for y, b in zip(ys, bb):
        ax.text(b["cited_here"] + 0.6, y, str(b["cited_here"]), va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys), [f"{b['key']} ({b['year'] or 'n.d.'})  {b['title'][:52]}{'...' if len(b['title']) > 52 else ''}"
                             for b in bb], fontsize=7.6)
    ax.set_xlabel("citations from other papers in the index")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("The backbone: the 25 papers the index builds on most", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_age(papers, edges, out: Path) -> Path:
    """(a) how many years older the cited paper is than the citing one; (b) in-index citations received by the
    publication year of the cited paper -- two measures, two panels (never two y-axes)."""
    plt = _plt()
    ages, by_year = [], {}
    for a, b in edges:
        ya, yb = papers[a].get("year"), papers[b].get("year")
        if ya and yb:
            ages.append(ya - yb)
        if yb:
            by_year[yb] = by_year.get(yb, 0) + 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    lo, hi = max(-2, min(ages)), min(40, max(ages))
    bins = list(range(lo, hi + 2))
    a1.hist([min(max(x, lo), hi) for x in ages], bins=bins, color=BLUE, rwidth=0.8)
    med = sorted(ages)[len(ages) // 2]
    a1.axvline(med + 0.5, color=INK2, linewidth=1)
    a1.text(med + 1.0, a1.get_ylim()[1] * 0.92, f"median {med} years", fontsize=8, color=INK)
    a1.set_xlabel("citing year minus cited year (clipped to -2 .. 40)")
    a1.set_ylabel("citations")
    a1.set_title("How old is the work the index builds on?", loc="left", fontsize=10.5)
    ys = sorted(y for y in by_year if y >= 1990)
    a2.bar(ys, [by_year[y] for y in ys], width=0.75, color=BLUE)
    a2.set_xlabel("publication year of the cited paper (from 1990)")
    a2.set_ylabel("citations received from the index")
    a2.set_title("In-index citations received, by the cited paper's year", loc="left", fontsize=10.5)
    for ax in (a1, a2):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def render(root: Path = ROOT, base: Optional[Path] = None) -> Path:
    """Every citation figure + the metrics JSON from the committed inputs (papers/_meta.json, papers/_citemap.json);
    returns the network figure's path (compile_visuals.GENERATORS)."""
    base = Path(base) if base else Path(root) / "report" / "boat_world_model"
    papers, meta = inputs(base)
    edges = edges_of(meta, list(papers))
    linked = sorted({x for e in edges for x in e})
    xy, missing, _ = positions(base, linked)
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    m = metrics(papers, meta)
    m["unplaced"] = len(missing)
    (out / METRICS).write_text(json.dumps({k: v for k, v in m.items() if k not in ("flows",)}, indent=1,
                                          sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    net = fig_network(papers, edges, xy, out / "citation_network.png")
    fig_flows(m, out / "citation_topic_flows.png")
    fig_backbone(m, out / "citation_backbone.png")
    fig_age(papers, edges, out / "citation_age.png")
    return net


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["render", "layout", "stats"])
    ap.add_argument("--today", default="")
    ap.add_argument("--iters", type=int, default=700)
    a = ap.parse_args(argv)
    if a.cmd == "layout":
        write_citemap(HERE, a.today, a.iters)
        return 0
    if a.cmd == "stats":
        papers, meta = inputs(HERE)
        m = metrics(papers, meta)
        print(json.dumps({k: v for k, v in m.items() if k not in ("flows", "topics")}, indent=1))
        return 0
    print(f"wrote {render()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
