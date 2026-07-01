"""Generate the nested-circles space-waste illustration for the blog post.

Run with `uv run python docs/blog/scripts/gen_circle_nesting.py`
"""

import matplotlib.pyplot as plt

DEPTH = 3
ROOT_R = 2**DEPTH  # = 8, leaf radius = 1

fill = ["#f8f9fa", "#dee2e6", "#adb5bd", "#6c757d"]
edge = ["#9ca3af", "#6b7280", "#374151", "#111827"]


def draw(ax, cx, cy, r, depth, direction="h", reduce_factor=0.9):
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
        cr = r / 2 * reduce_factor
        nd = "v" if direction == "h" else "h"
        if direction == "h":
            draw(ax, cx - r / 2, cy, cr, depth - 1, nd)
            draw(ax, cx + r / 2, cy, cr, depth - 1, nd)
        else:
            draw(ax, cx, cy - r / 2, cr, depth - 1, nd)
            draw(ax, cx, cy + r / 2, cr, depth - 1, nd)


fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

reduce_factor = 0.9
draw(ax, 0, 0, ROOT_R, DEPTH, reduce_factor=reduce_factor)

n_leaves = 2**DEPTH
eff = n_leaves * (reduce_factor / ROOT_R) ** 2  # leaf area / root area (r=1 at leaves)
if False:
    ax.text(
        0,
        -ROOT_R - 1.3,
        f"Leaf circles cover {eff:.0%} of the enclosing area",
        ha="center",
        fontsize=10,
        color="#374151",
    )

pad = 1.0
ax.set_xlim(-ROOT_R - pad, ROOT_R + pad)
ax.set_ylim(-ROOT_R - pad, ROOT_R + pad)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
out = "docs/blog/img/circle_nesting.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.show()
