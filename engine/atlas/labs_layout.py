"""report/boat_world_model/labs_layout.py -- a FIXED, deterministic layout for the atlas's web of labs, computed when the
index is built, so the page draws still positions instead of running a physics simulation in the viewer's browser.

WHY (2026-10-12, owner: "the dot web section freaks out badly and is unusable"). The first atlas ran a live force
simulation in the page. REPLAYED on the real web (485 nodes, 260 links) it never settled: after its 600 frames 373 of
485 nodes were pinned against the canvas edges, 10,555 pairs of circles overlapped and single nodes still jumped up to
1,180 px a frame -- the unbounded 1/d repulsion between near-coincident nodes, no step cap, and a canvas sized once (a
tab opened at zero width stayed broken). A layout computed here is checked by tests before it ships (no overlaps, links
short, every number finite, the same bytes on every machine); the page only draws it, pans and zooms.

    layout(nodes, edges) -> {"xy": {id: [x, y]}, "r": {id: r}, "w": W, "h": H, "linked_h": y, "sections": [...]}

METHOD. Connected components are laid out separately by stress majorization (Gansner, Koren & North 2004: every
pair of groups pulled toward its graph distance x the link length, weighted 1/d^2 -- a monotone descent, no velocities)
from a seeded LCG start, then an overlap-removal pass pushes apart any two circles closer than their radii plus GAP.
Linked components are packed by a skyline packer into a band of ~2:1 aspect (the canvas's shape); the groups with no
recorded link are packed in rows beneath, under their own caption.
PORTABLE: only + - * / and sqrt on Python floats, summed in explicit loops (bit-identical on Windows and Linux; hypot /
sin / cos are NOT, and the built-in sum() of floats changed in Python 3.12), integer LCG seeds, sorted iteration -- so
`build_bwm.py check` compares bytes on every host.
"""
from __future__ import annotations

import math
import zlib
from typing import Dict, Iterable, List, Sequence, Tuple

EDGE = 46.0          # ideal link length between circle edges (layout units = CSS px at zoom 1)
GAP = 3.0            # minimum clear space between two circles
PAD = 26.0           # space between packed components
ITERS = 120          # stress-majorization sweeps per component
ASPECT = 2.0         # target width / height of the whole layout (the canvas is ~1080 x 520)


def radius(n_papers: int) -> float:
    return 3.0 + 2.2 * math.sqrt(max(0, int(n_papers)))


def components(ids: Sequence[str], edges: Iterable[Tuple[str, str]]) -> List[List[str]]:
    """Connected components, each sorted, ordered by size (desc) then first id."""
    adj: Dict[str, set] = {i: set() for i in ids}
    for a, b in edges:
        if a in adj and b in adj and a != b:
            adj[a].add(b)
            adj[b].add(a)
    seen, out = set(), []
    for i in sorted(ids):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        out.append(sorted(comp))
    out.sort(key=lambda c: (-len(c), c[0]))
    return out


def _lcg(seed: int):
    s = seed & 0x7FFFFFFF
    while True:
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        yield s / 2147483648.0


def _hops(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    out = []
    for s in range(n):
        d = [-1] * n
        d[s] = 0
        q = [s]
        for x in q:
            for y in adj[x]:
                if d[y] < 0:
                    d[y] = d[x] + 1
                    q.append(y)
        out.append(d)
    return out


def _stress(comp: List[str], edges: List[Tuple[int, int]], r: List[float]) -> List[List[float]]:
    """Stress majorization (Gansner, Koren & North 2004, the localized per-node update): every pair is pulled toward
    its graph distance x the link length, weighted 1/d^2 -- a bounded, monotone descent with no velocities to explode.
    Pure Python floats, fixed node order: the same bits on every machine."""
    n = len(comp)
    total = 0.0
    for x in r:
        total += x
    k = EDGE + 2.0 * (total / n)
    hops = _hops(n, edges)
    side = k * math.sqrt(n)
    rnd = _lcg(zlib.crc32(comp[0].encode("utf-8")))
    pos = [[next(rnd) * side, next(rnd) * side] for _ in range(n)]
    tgt = [[hops[i][j] * k for j in range(n)] for i in range(n)]
    wgt = [[(1.0 / (tgt[i][j] * tgt[i][j]) if j != i else 0.0) for j in range(n)] for i in range(n)]
    wsum = []
    for row in wgt:
        acc = 0.0
        for w in row:
            acc += w
        wsum.append(acc)
    for _ in range(ITERS):
        for i in range(n):
            xi, yi = pos[i]
            sx = sy = 0.0
            ti, wi = tgt[i], wgt[i]
            for j in range(n):
                if j == i:
                    continue
                xj, yj = pos[j]
                dx, dy = xi - xj, yi - yj
                d = math.sqrt(dx * dx + dy * dy)
                if d < 1e-9:                             # coincident: a fixed, index-derived direction, never a 1/0
                    dx, dy, d = (1.0 if i < j else -1.0), 0.0, 1.0
                w, s = wi[j], ti[j] / d
                sx += w * (xj + s * dx)
                sy += w * (yj + s * dy)
            pos[i][0] = sx / wsum[i]
            pos[i][1] = sy / wsum[i]
    return pos


def _separate(pos: List[List[float]], r: List[float], passes: int = 80) -> None:
    n = len(pos)
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                need = r[i] + r[j] + GAP + 0.5
                d2 = dx * dx + dy * dy
                if d2 >= need * need:
                    continue
                d = math.sqrt(d2)
                if d < 1e-9:
                    dx, dy, d = 1.0, 0.0, 1.0
                push = (need - d) / 2.0 / d
                pos[i][0] -= dx * push
                pos[i][1] -= dy * push
                pos[j][0] += dx * push
                pos[j][1] += dy * push
                moved = True
        if not moved:
            return


def _skyline(items: List[Tuple[float, float]], width: float) -> Tuple[List[Tuple[float, float]], float]:
    """Bottom-left skyline packing of (w, h) boxes, in the given order, into `width`. -> (top-left corners, height)."""
    sky = [[0.0, width, 0.0]]                            # segments [x, w, y]
    out = []
    for w, h in items:
        best = None
        for s in range(len(sky)):
            x = sky[s][0]
            if x + w > width + 1e-6 and x > 0.0:
                continue
            y, e, cover = 0.0, s, x
            while e < len(sky) and cover < x + w - 1e-6:
                y = max(y, sky[e][2])
                cover = sky[e][0] + sky[e][1]
                e += 1
            if best is None or (y, x) < (best[0], best[1]):
                best = (y, x)
        y, x = best
        out.append((x, y))
        top, x1 = y + h, x + w
        nxt = []
        for sx, sw, sy in sky:
            ex = sx + sw
            if ex <= x or sx >= x1:
                nxt.append([sx, sw, sy])
                continue
            if sx < x:
                nxt.append([sx, x - sx, sy])
            if ex > x1:
                nxt.append([x1, ex - x1, sy])
        nxt.append([x, max(w, 0.0), top])
        nxt.sort(key=lambda g: g[0])
        merged = []
        for g in nxt:
            if merged and abs(merged[-1][2] - g[2]) < 1e-9 and abs(merged[-1][0] + merged[-1][1] - g[0]) < 1e-6:
                merged[-1][1] += g[1]
            else:
                merged.append(g)
        sky = merged
    return out, max((y + h for (x, y), (w, h) in zip(out, items)), default=0.0)


def _rows(items: List[Tuple[float, float]], width: float) -> Tuple[List[Tuple[float, float]], float]:
    out, x, y, row_h = [], 0.0, 0.0, 0.0
    for w, h in items:
        if x > 0.0 and x + w > width:
            x, y, row_h = 0.0, y + row_h + GAP * 2, 0.0
        out.append((x, y))
        x += w + GAP * 2
        row_h = max(row_h, h)
    return out, (y + row_h if items else 0.0)


def layout(nodes: Sequence[dict], edges: Sequence[dict]) -> dict:
    """nodes: [{id, papers: [...]}], edges: [{a, b}] -> positions (circle centres), radii and the layout box."""
    ids = [n["id"] for n in nodes]
    rad = {n["id"]: radius(len(n.get("papers") or [])) for n in nodes}
    pairs = sorted({tuple(sorted((e["a"], e["b"]))) for e in edges if e["a"] in rad and e["b"] in rad and e["a"] != e["b"]})
    comps = components(ids, pairs)
    linked = [c for c in comps if len(c) > 1]
    single = sorted((c[0] for c in comps if len(c) == 1), key=lambda i: (-rad[i], i))
    boxes = []                                           # (w, h, {id: (x, y) inside the box})
    for comp in linked:
        ix = {c: i for i, c in enumerate(comp)}
        r = [rad[c] for c in comp]
        pos = _stress(comp, [(ix[a], ix[b]) for a, b in pairs if a in ix], r)
        _separate(pos, r)
        x0 = min(p[0] - r[i] for i, p in enumerate(pos))
        y0 = min(p[1] - r[i] for i, p in enumerate(pos))
        local = {c: (pos[i][0] - x0, pos[i][1] - y0) for i, c in enumerate(comp)}
        boxes.append((max(p[0] + r[i] for i, p in enumerate(pos)) - x0, max(p[1] + r[i] for i, p in enumerate(pos)) - y0,
                      local))
    order = sorted(range(len(boxes)), key=lambda b: (-boxes[b][1], -boxes[b][0], min(boxes[b][2])))
    sizes = [(boxes[b][0] + PAD, boxes[b][1] + PAD) for b in order]
    widest = max([w for w, _ in sizes] + [2 * max(rad.values(), default=1.0) + PAD])
    best = None
    for step in range(0, 13):                            # the band width whose whole layout is closest to ASPECT
        width = widest * (1.0 + step / 4.0)
        corners, lh = _skyline(sizes, width) if sizes else ([], 0.0)
        cap = 22.0 if single else 0.0
        _, sh = _rows([(2 * rad[i], 2 * rad[i]) for i in single], width)
        h = lh + cap + sh
        used = max([c[0] + s[0] for c, s in zip(corners, sizes)] + [0.0])
        score = abs((max(used, widest) / max(h, 1.0)) - ASPECT)
        if best is None or score < best[0] - 1e-9:
            best = (score, width, corners, lh)
    _, width, corners, lh = best
    xy: Dict[str, List[float]] = {}
    for b, (cx, cy) in zip(order, corners):
        for c, (x, y) in boxes[b][2].items():
            xy[c] = [cx + PAD / 2 + x, cy + PAD / 2 + y]
    sections = []
    top = lh + (22.0 if single else 0.0)
    if single:
        sections.append({"label": f"{len(single)} groups with no recorded link", "y": round(lh + 14.0, 1)})
        cs, _ = _rows([(2 * rad[i], 2 * rad[i]) for i in single], width)
        for i, (x, y) in zip(single, cs):
            xy[i] = [PAD / 2 + x + rad[i], top + y + rad[i]]
    xy = {k: [round(v[0], 1), round(v[1], 1)] for k, v in sorted(xy.items())}
    w = max([v[0] + rad[k] for k, v in xy.items()] + [0.0]) + PAD / 2
    h = max([v[1] + rad[k] for k, v in xy.items()] + [0.0]) + PAD / 2
    return {"xy": xy, "r": {k: round(rad[k], 2) for k in sorted(rad)}, "w": round(w, 1), "h": round(h, 1),
            "linked_h": round(lh, 1), "sections": sections}


def problems(lay: dict, edges: Sequence[dict] = ()) -> List[str]:
    """What is WRONG with a layout -- empty when it can ship. Every circle finite and inside the box, no two circles
    overlapping, links not stretched across the picture. The tests run this on the committed index AND on a replay of
    the old live simulation, which it must reject."""
    out = []
    xy, r = lay["xy"], lay["r"]
    ids = sorted(xy)
    for k in ids:
        x, y = xy[k]
        if not (math.isfinite(x) and math.isfinite(y)):
            out.append(f"{k}: position not finite")
        elif x - r[k] < -0.5 or y - r[k] < -0.5 or x + r[k] > lay["w"] + 0.5 or y + r[k] > lay["h"] + 0.5:
            out.append(f"{k}: outside the {lay['w']} x {lay['h']} box")
    over = []
    for a in range(len(ids)):
        xa, ya = xy[ids[a]]
        for b in range(a + 1, len(ids)):
            xb, yb = xy[ids[b]]
            need = r[ids[a]] + r[ids[b]] + 1.0
            if (xa - xb) * (xa - xb) + (ya - yb) * (ya - yb) < need * need:
                over.append((ids[a], ids[b]))
    if over:
        out.append(f"{len(over)} overlapping pairs, e.g. {over[0][0]} / {over[0][1]}")
    pinned = [k for k in ids if math.isfinite(xy[k][0]) and math.isfinite(xy[k][1]) and
              min(xy[k][0] - r[k], xy[k][1] - r[k], lay["w"] - xy[k][0] - r[k], lay["h"] - xy[k][1] - r[k]) < 0.5]
    if len(pinned) > max(2, len(ids) // 20):                 # the old simulation's signature: circles piled on the frame
        out.append(f"{len(pinned)} of {len(ids)} circles pressed against the frame")
    span = math.sqrt(lay["w"] * lay["w"] + lay["h"] * lay["h"])

    def length(e):
        dx, dy = xy[e["a"]][0] - xy[e["b"]][0], xy[e["a"]][1] - xy[e["b"]][1]
        return math.sqrt(dx * dx + dy * dy)

    long_ = [e for e in edges if e["a"] in xy and e["b"] in xy and
             length(e) > 0.5 * span]
    if long_:
        out.append(f"{len(long_)} links longer than half the picture")
    return out
