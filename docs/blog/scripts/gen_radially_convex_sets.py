"""Generate the radially convex sets illustration for the blog post."""

import matplotlib.pyplot as plt
import numpy as np

K = 300
angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
n_rays = 20
ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

fill_color = "#dee2e6"
edge_color = "#212529"
center_color = "#e63946"
ray_color = "#457b9d"


def draw_star_region(ax, r_fn, title):
    x = r_fn(angles) * np.cos(angles)
    y = r_fn(angles) * np.sin(angles)
    ax.fill(x, y, color=fill_color, zorder=1)
    ax.plot(np.append(x, x[0]), np.append(y, y[0]), color=edge_color, lw=1.5, zorder=2)
    r_rays = r_fn(ray_angles)
    for r, a in zip(r_rays, ray_angles):
        ax.plot(
            [0, r * np.cos(a)],
            [0, r * np.sin(a)],
            color=ray_color,
            lw=0.8,
            alpha=0.55,
            zorder=3,
        )
    ax.plot(0, 0, "o", color=center_color, markersize=5, zorder=4)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.85, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
fig.patch.set_facecolor("white")

# (a) Convex blob
draw_star_region(
    axes[0],
    lambda t: 1.0 + 0.28 * np.cos(2 * t) + 0.12 * np.sin(3 * t + 0.7),
    "(a) Convex",
)

# (b) Non-convex but star-shaped (5-arm star polygon via r = 1 + A·cos(5θ))
draw_star_region(
    axes[1],
    lambda t: 1.0 + 0.55 * np.cos(5 * t),
    "(b) Non-convex, star-shaped",
)

# (c) Non-radially-convex: C-shape (annulus sector)
ax = axes[2]
gap = 0.38
theta_out = np.linspace(gap, 2 * np.pi - gap, 200)
theta_in = theta_out[::-1]
r_out, r_in = 1.0, 0.45
x_c = np.concatenate([r_out * np.cos(theta_out), r_in * np.cos(theta_in)])
y_c = np.concatenate([r_out * np.sin(theta_out), r_in * np.sin(theta_in)])
ax.fill(x_c, y_c, color=fill_color, zorder=1)
ax.plot(np.append(x_c, x_c[0]), np.append(y_c, y_c[0]), color=edge_color, lw=1.5, zorder=2)
for t in [theta_out[0], theta_out[-1]]:
    ax.plot(
        [r_in * np.cos(t), r_out * np.cos(t)],
        [r_in * np.sin(t), r_out * np.sin(t)],
        color=edge_color,
        lw=1.5,
        zorder=2,
    )
ax.text(
    0,
    -1.35,
    "Try finding a center from which\nthe whole shape is visible.",
    ha="center",
    va="top",
    fontsize=8.5,
    color="#495057",
    style="italic",
)
ax.set_title("(c) Not star-shaped", fontsize=10, pad=8)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.95, 1.5)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
out = "docs/blog/img/radially_convex_sets.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.show()
