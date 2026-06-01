"""Generate the nested-circles space-waste illustration for the blog post."""

import matplotlib.pyplot as plt
import numpy as np

DEPTH = 3
ROOT_R = 2 ** DEPTH  # = 8, leaf radius = 1

fill = ["#f8f9fa", "#dee2e6", "#adb5bd", "#6c757d"]
edge = ["#9ca3af", "#6b7280", "#374151", "#111827"]


def draw(ax, cx, cy, r, depth, direction="h"):
    level = DEPTH - depth
    ax.add_patch(
        plt.Circle(
            (cx, cy),
            r,
            facecolor=fill[level],
            edgecolor=edge[level],
            linewidth=1.5,
            zorder=level,
        )
    )
    if depth > 0:
        cr = r / 2
        nd = "v" if direction == "h" else "h"
        if direction == "h":
            draw(ax, cx - cr, cy, cr, depth - 1, nd)
            draw(ax, cx + cr, cy, cr, depth - 1, nd)
        else:
            draw(ax, cx, cy - cr, cr, depth - 1, nd)
            draw(ax, cx, cy + cr, cr, depth - 1, nd)


fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

draw(ax, 0, 0, ROOT_R, DEPTH)

n_leaves = 2**DEPTH
eff = n_leaves / ROOT_R**2  # leaf area / root area (r=1 at leaves)
ax.text(
    0,
    -ROOT_R - 1.3,
    f"Leaf circles cover {eff:.0%} of the enclosing area",
    ha="center",
    fontsize=10,
    color="#374151",
)

pad = 1.5
ax.set_xlim(-ROOT_R - pad, ROOT_R + pad)
ax.set_ylim(-ROOT_R - pad - 1.5, ROOT_R + pad)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
out = "docs/blog/img/circle_nesting.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.show()
