"""Generate the enclosure/exclusion ray-shadow geometry illustration."""

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.patch.set_facecolor("white")

circle_fill = "#dee2e6"
circle_edge = "#212529"
center_col = "#e63946"
ray_col = "#457b9d"
shadow_col = "#f4a261"
enc_col = "#2a9d8f"
exc_col = "#e76f51"
dim_col = "#868e96"
lbl_col = "#343a40"


def draw_panel(ax, title, theta_deg, tang, perp, r, boundary_r):
    theta = np.radians(theta_deg)
    ray_dir = np.array([np.cos(theta), np.sin(theta)])
    ray_perp = np.array([-np.sin(theta), np.cos(theta)])

    circle_center = tang * ray_dir + perp * ray_perp
    far_edge_r = tang + np.sqrt(r**2 - perp**2)
    near_edge_r = tang - np.sqrt(r**2 - perp**2)
    foot = tang * ray_dir

    # ---- Partial star boundary (0 to 90°) with faded color ----
    arc_angles = np.linspace(0, np.pi / 2, 200)
    # Slightly wavy radius anchored to boundary_r at theta
    wave = 0.12 * np.sin(5 * arc_angles + 0.8)
    wave_at_theta = 0.12 * np.sin(5 * theta + 0.8)
    arc_radii = boundary_r * (1 + wave - wave_at_theta)
    arc_x = arc_radii * np.cos(arc_angles)
    arc_y = arc_radii * np.sin(arc_angles)

    sector_x = np.concatenate([[0], arc_x, [0]])
    sector_y = np.concatenate([[0], arc_y, [0]])
    ax.fill(sector_x, sector_y, color=ray_col, alpha=0.08, zorder=1)
    ax.plot(arc_x, arc_y, "-", color=ray_col, lw=2.0, alpha=0.3, zorder=2)

    # ---- Theta angle indicator ----
    # Faint x-axis reference
    ax.plot(
        [0, 0.7 * boundary_r], [0, 0], "-", color=dim_col, lw=0.8, alpha=0.45, zorder=1
    )
    # Arc from 0 to theta
    t_arc = np.linspace(0, theta, 60)
    arc_r = 0.58
    ax.plot(
        arc_r * np.cos(t_arc),
        arc_r * np.sin(t_arc),
        "-",
        color=dim_col,
        lw=1.0,
        zorder=4,
    )
    # θ label
    t_mid = theta / 2
    t_lbl_r = 0.76
    ax.text(
        t_lbl_r * np.cos(t_mid),
        t_lbl_r * np.sin(t_mid),
        "θ",
        ha="center",
        va="center",
        fontsize=11,
        color=dim_col,
        style="italic",
    )

    # ---- Circle ----
    ax.add_patch(
        plt.Circle(
            circle_center,
            r,
            facecolor=circle_fill,
            edgecolor=circle_edge,
            linewidth=1.5,
            zorder=3,
        )
    )

    # ---- Ray ----
    ray_end = max(far_edge_r, boundary_r) + 0.5
    ax.annotate(
        "",
        xy=ray_end * ray_dir,
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "color": ray_col, "lw": 1.5},
        zorder=4,
    )

    # ---- Shadow segment on ray (near → far edge) ----
    near_pt = near_edge_r * ray_dir
    far_pt = far_edge_r * ray_dir
    ax.plot(
        [near_pt[0], far_pt[0]],
        [near_pt[1], far_pt[1]],
        "-",
        color=shadow_col,
        lw=5,
        alpha=0.7,
        zorder=4,
        solid_capstyle="round",
    )

    # ---- Dashed decomposition lines ----
    ax.plot([0, foot[0]], [0, foot[1]], "--", color=dim_col, lw=1.0, zorder=4)
    ax.plot(
        [foot[0], circle_center[0]],
        [foot[1], circle_center[1]],
        "--",
        color=dim_col,
        lw=1.0,
        zorder=4,
    )

    # ---- Right-angle mark at foot ----
    s = 0.1
    corner = foot - s * ray_dir + s * ray_perp
    ax.plot(
        [foot[0] - s * ray_dir[0], corner[0]],
        [foot[1] - s * ray_dir[1], corner[1]],
        "-",
        color=dim_col,
        lw=0.8,
    )
    ax.plot(
        [corner[0], foot[0] + s * ray_perp[0]],
        [corner[1], foot[1] + s * ray_perp[1]],
        "-",
        color=dim_col,
        lw=0.8,
    )

    # ---- Key points ----
    bp = boundary_r * ray_dir
    ax.plot(0, 0, "o", color=center_col, markersize=7, zorder=6)
    ax.plot(*foot, "s", color=dim_col, markersize=4, zorder=5)
    ax.plot(*circle_center, "o", color=lbl_col, markersize=4, zorder=5)
    ax.plot(*near_pt, "|", color=exc_col, markersize=12, markeredgewidth=2, zorder=5)
    ax.plot(*far_pt, "|", color=enc_col, markersize=12, markeredgewidth=2, zorder=5)
    ax.plot(*bp, "D", color=ray_col, markersize=7, zorder=6)

    # ---- Labels ----
    fs = 9.0
    ax.text(0, -0.32, "center", ha="center", fontsize=fs, color=center_col)

    # Double-headed arrow spanning origin → foot, offset below the ray
    tang_offset = -0.32 * ray_perp
    ax.annotate("", xy=foot + tang_offset, xytext=tang_offset,
                arrowprops={"arrowstyle": "<->", "color": lbl_col, "lw": 1.0})
    tang_mid = 0.5 * tang * ray_dir + tang_offset - 0.12 * ray_perp
    ax.text(*tang_mid, "tang", ha="center", va="top", fontsize=fs, color=lbl_col)

    perp_mid = 0.5 * (circle_center + foot) + 0.18 * ray_dir
    ax.text(*perp_mid, "perp", ha="left", fontsize=fs, color=lbl_col)

    ax.annotate(
        "far edge\n(enclosure threshold)",
        xy=far_pt,
        xytext=far_pt + np.array([0.5, -0.55]),
        fontsize=fs - 0.5,
        color=enc_col,
        ha="left",
        arrowprops={"arrowstyle": "->", "color": enc_col, "lw": 0.8},
    )

    ax.annotate(
        "near edge\n(exclusion threshold)",
        xy=near_pt,
        xytext=near_pt + np.array([-0.05, -0.75]),
        fontsize=fs - 0.5,
        color=exc_col,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": exc_col, "lw": 0.8},
    )

    ax.annotate(
        "boundary\nradius r_θ",
        xy=bp,
        xytext=bp + np.array([0.05, 0.65]),
        fontsize=fs - 0.5,
        color=ray_col,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": ray_col, "lw": 0.8},
    )

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlim(-0.5, ray_end + 0.6)
    ax.set_ylim(-1.1, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")


draw_panel(
    axes[0],
    "(a) Enclosure satisfied: r_θ ≥ far edge",
    theta_deg=16,
    tang=2.5,
    perp=0.45,
    r=0.8,
    boundary_r=4.45,
)

draw_panel(
    axes[1],
    "(b) Exclusion satisfied: r_θ ≤ near edge",
    theta_deg=28,
    tang=2.5,
    perp=0.45,
    r=0.8,
    boundary_r=1.42,
)

plt.tight_layout()
out = "docs/blog/img/enclosure_geometry.svg"
plt.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
plt.show()
