"""Generate the nested-rectangles treemap illustration for the blog post.

Run with `uv run python docs/blog/scripts/gen_rectangle_nesting.py`
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt

DEPTH = 3
ROOT_W = 8.0
ROOT_H = 8.0

fill = ["#f8f9fa", "#dee2e6", "#adb5bd", "#6c757d"]
edge = ["#9ca3af", "#6b7280", "#374151", "#111827"]


REDUCE = 0.9


def draw(ax, x, y, w, h, depth, direction="h"):
    level = DEPTH - depth
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            facecolor=fill[level],
            edgecolor=edge[level],
            linewidth=1.5,
            zorder=level,
        )
    )
    if depth > 0:
        nd = "v" if direction == "h" else "h"
        if direction == "h":
            cw, ch = w / 2 * REDUCE, h * REDUCE
            ox, oy = (w / 2 - cw) / 2, (h - ch) / 2
            draw(ax, x + ox, y + oy, cw, ch, depth - 1, nd)
            draw(ax, x + w / 2 + ox, y + oy, cw, ch, depth - 1, nd)
        else:
            cw, ch = w * REDUCE, h / 2 * REDUCE
            ox, oy = (w - cw) / 2, (h / 2 - ch) / 2
            draw(ax, x + ox, y + oy, cw, ch, depth - 1, nd)
            draw(ax, x + ox, y + h / 2 + oy, cw, ch, depth - 1, nd)


fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

draw(ax, -ROOT_W / 2, -ROOT_H / 2, ROOT_W, ROOT_H, DEPTH)

n_leaves = 2**DEPTH
eff = REDUCE**DEPTH
if False:
    ax.text(
        0,
        -ROOT_H / 2 - 1.3,
        f"Leaf rectangles cover {eff:.0%} of the enclosing area",
        ha="center",
        fontsize=10,
        color="#374151",
    )

pad = 1.5
ax.set_xlim(-ROOT_W / 2 - pad, ROOT_W / 2 + pad)
ax.set_ylim(-ROOT_H / 2 - pad - 1.5, ROOT_H / 2 + pad)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
out = "docs/blog/img/rectangle_nesting.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.show()
