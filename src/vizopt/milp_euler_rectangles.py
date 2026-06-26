"""MILP-based Euler diagram layout with rectangular set boundaries."""

import numpy as np
import pulp


def _grid_positions(N, r, L):
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    spacing = 3 * r
    x0 = (L - (cols - 1) * spacing) / 2
    y0 = (L - (rows - 1) * spacing) / 2
    pts = [[x0 + (i % cols) * spacing, y0 + (i // cols) * spacing] for i in range(N)]
    return np.array(pts)


def solve_euler_rectangles(
    membership,
    r=0.15,
    offset=0.2,
    layout_size=5.0,
    initial_positions=None,
    position_penalty=0.0,
):
    """Find element positions and set rectangles forming an Euler diagram.

    Args:
        membership: (N, S) bool array — membership[i, j] = True if element i ∈ set j
        r: half-side of each element square
        offset: per-set margin (scalar or array of length S).
            Containment: element border ≥ offset[j] from each wall of rect j.
            Exclusion/non-overlap: borders ≥ offset[j] (or min(offset)) apart.
        layout_size: bounding box for all variables; big-M is derived from this
        initial_positions: (N, 2) array of preferred element centres. Defaults to a
            centred grid. Only used when position_penalty > 0.
        position_penalty: weight of the L1 penalty toward initial_positions. 0
            disables the penalty entirely (no extra variables or constraints added).

    Returns:
        dict with keys:
          status: PuLP status string
          element_positions: (N, 2) array of (x, y) centres
          rectangles: (S, 4) array of [x, y, w, h] (bottom-left + size)

    Notes:
        Set rectangles with no members in common are constrained to not overlap.
    """
    N, S = membership.shape
    offsets = np.broadcast_to(np.asarray(offset, dtype=float), (S,)).copy()
    elem_gap = float(offsets.min())
    L = layout_size
    M = 2 * L + 2 * r + float(offsets.max())

    prob = pulp.LpProblem("euler_rectangles", pulp.LpMinimize)

    ex = [pulp.LpVariable(f"ex_{i}", lowBound=0, upBound=L) for i in range(N)]
    ey = [pulp.LpVariable(f"ey_{i}", lowBound=0, upBound=L) for i in range(N)]

    rx = [pulp.LpVariable(f"rx_{j}", lowBound=0, upBound=L) for j in range(S)]
    ry = [pulp.LpVariable(f"ry_{j}", lowBound=0, upBound=L) for j in range(S)]
    rw = [
        pulp.LpVariable(f"rw_{j}", lowBound=2 * (r + offsets[j]), upBound=L)
        for j in range(S)
    ]
    rh = [
        pulp.LpVariable(f"rh_{j}", lowBound=2 * (r + offsets[j]), upBound=L)
        for j in range(S)
    ]

    objective = pulp.lpSum(2 * (rw[j] + rh[j]) for j in range(S))

    if position_penalty > 0:
        if initial_positions is None:
            initial_positions = _grid_positions(N, r, L)
        x0 = np.asarray(initial_positions, dtype=float)
        dx = [pulp.LpVariable(f"dx_{i}", lowBound=0) for i in range(N)]
        dy = [pulp.LpVariable(f"dy_{i}", lowBound=0) for i in range(N)]
        for i in range(N):
            prob += dx[i] >= ex[i] - x0[i, 0]
            prob += dx[i] >= x0[i, 0] - ex[i]
            prob += dy[i] >= ey[i] - x0[i, 1]
            prob += dy[i] >= x0[i, 1] - ey[i]
        objective += position_penalty * pulp.lpSum(dx[i] + dy[i] for i in range(N))

    prob += objective

    for i in range(N):
        for j in range(S):
            d = offsets[j]
            if membership[i, j]:
                prob += ex[i] - r >= rx[j] + d
                prob += ex[i] + r <= rx[j] + rw[j] - d
                prob += ey[i] - r >= ry[j] + d
                prob += ey[i] + r <= ry[j] + rh[j] - d
            else:
                bL = pulp.LpVariable(f"bL_{i}_{j}", cat="Binary")
                bR = pulp.LpVariable(f"bR_{i}_{j}", cat="Binary")
                bB = pulp.LpVariable(f"bB_{i}_{j}", cat="Binary")
                bT = pulp.LpVariable(f"bT_{i}_{j}", cat="Binary")
                prob += bL + bR + bB + bT >= 1
                prob += ex[i] + r <= rx[j] - d + M * (1 - bL)
                prob += ex[i] - r >= rx[j] + rw[j] + d - M * (1 - bR)
                prob += ey[i] + r <= ry[j] - d + M * (1 - bB)
                prob += ey[i] - r >= ry[j] + rh[j] + d - M * (1 - bT)

    for i in range(N):
        for k in range(i + 1, N):
            bL = pulp.LpVariable(f"ebL_{i}_{k}", cat="Binary")
            bR = pulp.LpVariable(f"ebR_{i}_{k}", cat="Binary")
            bB = pulp.LpVariable(f"ebB_{i}_{k}", cat="Binary")
            bT = pulp.LpVariable(f"ebT_{i}_{k}", cat="Binary")
            prob += bL + bR + bB + bT >= 1
            prob += ex[i] + 2 * r + elem_gap <= ex[k] + M * (1 - bL)
            prob += ex[k] + 2 * r + elem_gap <= ex[i] + M * (1 - bR)
            prob += ey[i] + 2 * r + elem_gap <= ey[k] + M * (1 - bB)
            prob += ey[k] + 2 * r + elem_gap <= ey[i] + M * (1 - bT)

    for j in range(S):
        for k in range(j + 1, S):
            if not np.any(membership[:, j] & membership[:, k]):
                cL = pulp.LpVariable(f"cL_{j}_{k}", cat="Binary")
                cR = pulp.LpVariable(f"cR_{j}_{k}", cat="Binary")
                cB = pulp.LpVariable(f"cB_{j}_{k}", cat="Binary")
                cT = pulp.LpVariable(f"cT_{j}_{k}", cat="Binary")
                prob += cL + cR + cB + cT >= 1
                prob += rx[j] + rw[j] <= rx[k] + M * (1 - cL)
                prob += rx[k] + rw[k] <= rx[j] + M * (1 - cR)
                prob += ry[j] + rh[j] <= ry[k] + M * (1 - cB)
                prob += ry[k] + rh[k] <= ry[j] + M * (1 - cT)

    solver = pulp.HiGHS(msg=True, mip_min_logging_interval=2.0)
    prob.solve(solver)

    elem_pos = np.array([[pulp.value(ex[i]), pulp.value(ey[i])] for i in range(N)])
    rects = np.array(
        [
            [pulp.value(rx[j]), pulp.value(ry[j]), pulp.value(rw[j]), pulp.value(rh[j])]
            for j in range(S)
        ]
    )

    return {
        "status": pulp.LpStatus[prob.status],
        "element_positions": elem_pos,
        "rectangles": rects,
    }
