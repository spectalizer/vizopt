"""Squarified treemap layout algorithm (Bruls et al. 2000)."""


def _worst_ratio_strip(row, row_sum, W, H, total):
    """Worst aspect ratio among tiles in a candidate strip."""
    if not row or row_sum == 0 or total == 0:
        return float("inf")
    worst = 0.0
    if W >= H:
        strip_w = row_sum / total * W
        for _, w in row:
            tile_h = w / row_sum * H
            if tile_h == 0:
                return float("inf")
            worst = max(worst, max(strip_w, tile_h) / min(strip_w, tile_h))
    else:
        strip_h = row_sum / total * H
        for _, w in row:
            tile_w = w / row_sum * W
            if tile_w == 0:
                return float("inf")
            worst = max(worst, max(tile_w, strip_h) / min(tile_w, strip_h))
    return worst


def _squarify_recursive(items, rect, out):
    """One squarified-treemap layout step, writing results into `out`."""
    if not items:
        return
    x0, y0, x1, y1 = rect
    W, H = x1 - x0, y1 - y0
    if len(items) == 1:
        out[items[0][0]] = rect
        return
    total = sum(w for _, w in items)
    row, row_sum = [], 0.0
    for name, w in items:
        trial_row = row + [(name, w)]
        trial_sum = row_sum + w
        new_worst = _worst_ratio_strip(trial_row, trial_sum, W, H, total)
        if row:
            curr_worst = _worst_ratio_strip(row, row_sum, W, H, total)
            if curr_worst < new_worst:
                break
        row, row_sum = trial_row, trial_sum
    fraction = row_sum / total
    if W >= H:
        strip_w = fraction * W
        y_cursor = y0
        for name, w in row:
            tile_h = (w / row_sum) * H
            out[name] = (x0, y_cursor, x0 + strip_w, y_cursor + tile_h)
            y_cursor += tile_h
        remaining_rect = (x0 + strip_w, y0, x1, y1)
    else:
        strip_h = fraction * H
        x_cursor = x0
        for name, w in row:
            tile_w = (w / row_sum) * W
            out[name] = (x_cursor, y0, x_cursor + tile_w, y0 + strip_h)
            x_cursor += tile_w
        remaining_rect = (x0, y0 + strip_h, x1, y1)
    _squarify_recursive(items[len(row) :], remaining_rect, out)


def squarify_layout(items, rect):
    """Squarified treemap layout (Bruls et al. 2000).

    Assigns each item a subrectangle of *rect* with area proportional to its
    weight, minimising the worst aspect ratio across all tiles.

    Args:
        items: Sequence of `(name, weight)` pairs with positive weights.
        rect: `(x0, y0, x1, y1)` bounding rectangle.

    Returns:
        Dict mapping each name to its `(x0, y0, x1, y1)` subrectangle.
    """
    if not items:
        return {}
    out: dict = {}
    _squarify_recursive(sorted(items, key=lambda kv: -kv[1]), rect, out)
    return out
