"""Render each GPS boundary over satellite imagery.

Downloads Esri World Imagery tiles (no API key required), stitches them into a
basemap, and overlays the boundary polygon, its centroid (the default rover
start/home), and the per-rover rest-offset start positions.

Usage:
    python spf/scripts/plot_gps_boundaries.py --output-dir docs/boundary_maps
    python spf/scripts/plot_gps_boundaries.py --boundary fort_baker_boundary
    python spf/scripts/plot_gps_boundaries.py --combined fort_baker   # overlay a family

Imagery: Esri, Maxar, Earthstar Geographics — attribution is drawn on each figure.
Tiles are cached under the output directory so repeat runs do not re-download.
"""

import argparse
import math
import os
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

from spf.gps.boundaries import boundaries
from spf.mavlink.mavlink_controller import drone_get_planner

TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ATTRIBUTION = "Imagery: Esri, Maxar, Earthstar Geographics"
TILE_PX = 256
USER_AGENT = "spf-boundary-plotter/1.0 (research; contact: repo maintainer)"

# Per-rover rest offsets (east_m, north_m) — keep in step with the
# rest-offset-m keys in data_collection/rover/rover_v3.1/capture_configs/.
ROVER_OFFSETS = {1: (1.0, 1.0), 2: (1.0, -1.0), 3: (-1.0, 1.0), 4: (-1.0, -1.0)}
ROVER_COLORS = {1: "#ff3b30", 2: "#ffcc00", 3: "#34c759", 4: "#5ac8fa"}


def deg2tile(lat, lon, zoom):
    """(lat, lon) -> fractional slippy-map tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile2deg(x, y, zoom):
    """Fractional tile coordinates -> (lat, lon)."""
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fetch_tile(zoom, x, y, cache_dir):
    path = os.path.join(cache_dir, f"{zoom}_{x}_{y}.jpg")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    url = TILE_URL.format(z=zoom, x=x, y=y)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))  # be polite to the tile server
    with open(path, "wb") as handle:
        handle.write(data)
    return Image.open(path).convert("RGB")


def build_basemap(lat_min, lat_max, lon_min, lon_max, zoom, cache_dir):
    """Stitch the tiles covering a bbox. Returns (image, extent_in_degrees)."""
    x0f, y0f = deg2tile(lat_max, lon_min, zoom)  # north-west
    x1f, y1f = deg2tile(lat_min, lon_max, zoom)  # south-east
    x0, y0, x1, y1 = (
        int(math.floor(x0f)),
        int(math.floor(y0f)),
        int(math.floor(x1f)),
        int(math.floor(y1f)),
    )
    canvas = Image.new("RGB", ((x1 - x0 + 1) * TILE_PX, (y1 - y0 + 1) * TILE_PX))
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            canvas.paste(
                fetch_tile(zoom, tx, ty, cache_dir),
                ((tx - x0) * TILE_PX, (ty - y0) * TILE_PX),
            )
    nw_lat, nw_lon = tile2deg(x0, y0, zoom)
    se_lat, se_lon = tile2deg(x1 + 1, y1 + 1, zoom)
    return canvas, (nw_lon, se_lon, se_lat, nw_lat)  # left, right, bottom, top


def pick_zoom(lat_span_m, target_px=1100):
    """Highest zoom whose ground resolution still fits the scene in target_px."""
    for zoom in range(21, 10, -1):
        m_per_px = 156543.03392 * math.cos(math.radians(37.8)) / (2**zoom)
        if lat_span_m / m_per_px <= target_px:
            return zoom
    return 17


def meters_between(a, b):
    """Local-ENU metres between two (lon, lat) points."""
    m_per_deg_lat = (math.pi / 180.0) * 6371008.8
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(a[1]))
    return math.hypot((a[0] - b[0]) * m_per_deg_lon, (a[1] - b[1]) * m_per_deg_lat)


def add_scale_bar(ax, extent, lat):
    """Draw a scale bar sized to a round number of metres."""
    m_per_deg_lon = (math.pi / 180.0) * 6371008.8 * math.cos(math.radians(lat))
    span_m = (extent[1] - extent[0]) * m_per_deg_lon
    for candidate in (100, 50, 25, 20, 10):
        if candidate < span_m * 0.35:
            bar_m = candidate
            break
    else:
        bar_m = 10
    bar_deg = bar_m / m_per_deg_lon
    x0 = extent[0] + (extent[1] - extent[0]) * 0.05
    y0 = extent[2] + (extent[3] - extent[2]) * 0.06
    ax.plot([x0, x0 + bar_deg], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    ax.text(
        x0 + bar_deg / 2,
        y0 + (extent[3] - extent[2]) * 0.012,
        f"{bar_m} m",
        color="white",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )


def draw_boundary(ax, points, color, label, fill_alpha=0.10):
    ring = np.vstack([points, points[0]])
    ax.add_patch(
        MplPolygon(points, closed=True, facecolor=color, alpha=fill_alpha, edgecolor="none")
    )
    ax.plot(ring[:, 0], ring[:, 1], color=color, lw=2.4, label=label, zorder=5)
    ax.scatter(
        points[:, 0], points[:, 1], s=34, facecolor="white", edgecolor=color, lw=1.6, zorder=6
    )


def render(names, out_path, title, show_rovers=True, cache_dir=".tilecache"):
    pts = np.vstack([boundaries[n] for n in names])
    lat_min, lat_max = pts[:, 1].min(), pts[:, 1].max()
    lon_min, lon_max = pts[:, 0].min(), pts[:, 0].max()
    pad_lat = max((lat_max - lat_min) * 0.22, 0.0004)
    pad_lon = max((lon_max - lon_min) * 0.22, 0.0004)
    lat_min, lat_max = lat_min - pad_lat, lat_max + pad_lat
    lon_min, lon_max = lon_min - pad_lon, lon_max + pad_lon

    span_m = (lat_max - lat_min) * (math.pi / 180.0) * 6371008.8
    zoom = pick_zoom(span_m)
    os.makedirs(cache_dir, exist_ok=True)
    basemap, extent = build_basemap(lat_min, lat_max, lon_min, lon_max, zoom, cache_dir)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(np.asarray(basemap), extent=extent, origin="upper", interpolation="bilinear")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    palette = ["#ff3b30", "#00d4ff", "#ffd60a", "#bf5af2"]
    for idx, name in enumerate(names):
        points = boundaries[name]
        color = palette[idx % len(palette)]
        centroid = points.mean(axis=0)
        draw_boundary(ax, points, color, f"{name}  ({len(points)} pts)")
        ax.scatter(
            *centroid, s=210, marker="*", color=color, edgecolor="black", lw=1.1, zorder=8
        )
        ax.annotate(
            f"{name}\ncentroid {centroid[1]:.6f}, {centroid[0]:.6f}",
            xy=centroid,
            xytext=(9, 9),
            textcoords="offset points",
            fontsize=8,
            color="white",
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.62, ec=color),
        )

    if show_rovers and len(names) == 1:
        boundary = boundaries[names[0]]
        centroid = boundary.mean(axis=0)
        homes = {
            rover: drone_get_planner("bounce", boundary, rest_offset_m=offset).get_home_point()
            for rover, offset in ROVER_OFFSETS.items()
        }
        # On a field-scale view the four offsets collapse into one dot — which is
        # the honest picture — so draw a magnified inset to make them legible.
        for rover, home in homes.items():
            ax.scatter(
                *home, s=52, marker="o", color=ROVER_COLORS[rover],
                edgecolor="black", lw=0.9, zorder=10,
            )
        offset_m = meters_between(centroid, homes[1])
        ax.annotate(
            f"4 rover start points\nwithin {offset_m:.2f} m — see inset",
            xy=centroid,
            xytext=(30, -46),
            textcoords="offset points",
            fontsize=8,
            color="white",
            zorder=11,
            arrowprops=dict(arrowstyle="->", color="white", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.65, ec="white", lw=0.6),
        )

        axins = ax.inset_axes([0.035, 0.60, 0.30, 0.36])
        # Main-map artists carry zorder 5-6; without this the fence line and its
        # vertex markers draw straight over the inset panel.
        axins.set_zorder(20)
        axins.patch.set_alpha(1.0)
        pad_m = 6.5
        m_per_deg_lat = (math.pi / 180.0) * 6371008.8
        m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(centroid[1]))
        axins.set_xlim(centroid[0] - pad_m / m_per_deg_lon, centroid[0] + pad_m / m_per_deg_lon)
        axins.set_ylim(centroid[1] - pad_m / m_per_deg_lat, centroid[1] + pad_m / m_per_deg_lat)
        axins.set_facecolor("#101418")
        # 5 m arrival tolerance, to scale — the reason these offsets are marginal
        # 5 m arrival tolerance drawn to scale. Ellipse, not Circle: a degree of
        # longitude is shorter than a degree of latitude, so equal metres are
        # unequal degrees on the two axes.
        tol = matplotlib.patches.Ellipse(
            (centroid[0], centroid[1]),
            width=2 * 5.0 / m_per_deg_lon,
            height=2 * 5.0 / m_per_deg_lat,
            fill=False, ec="#ff9f0a", ls="--", lw=1.4, zorder=7,
        )
        axins.add_patch(tol)
        axins.annotate(
            "5 m arrival tolerance", xy=(centroid[0], centroid[1] + 5.0 / m_per_deg_lat),
            xytext=(0, 4), textcoords="offset points", ha="center",
            fontsize=7, color="#ff9f0a", zorder=8,
        )
        axins.scatter(*centroid, s=150, marker="*", color="white", edgecolor="black", lw=0.8, zorder=8)
        for rover, home in homes.items():
            axins.scatter(
                *home, s=110, marker="o", color=ROVER_COLORS[rover],
                edgecolor="black", lw=1.0, zorder=9,
            )
            dx, dy = ROVER_OFFSETS[rover]
            axins.annotate(
                f"R{rover}",
                xy=home,
                xytext=(9 * (1 if dx > 0 else -1), 9 * (1 if dy > 0 else -1)),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="white",
                ha="center",
                zorder=10,
            )
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor("white")
        axins.text(
            0.5, 0.035,
            f"inset ±{pad_m:.1f} m about the centroid · offsets {offset_m:.2f} m",
            transform=axins.transAxes, ha="center", fontsize=7.2, color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.75, ec="none"),
            zorder=12,
        )

    add_scale_bar(ax, (lon_min, lon_max, lat_min, lat_max), (lat_min + lat_max) / 2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    # Absolute degrees, not matplotlib's "+3.776e1" offset notation — these are
    # coordinates an operator may need to read off and type in.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.5f"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
    ax.text(
        0.995,
        0.006,
        f"{ATTRIBUTION}  ·  zoom {zoom}",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, ec="none"),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
    plt.close(fig)
    print(f"wrote {out_path}  (zoom {zoom}, {len(names)} boundary/ies)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/boundary_maps")
    parser.add_argument("--boundary", help="render only this boundary")
    parser.add_argument("--combined", help="substring; overlay all matching boundaries")
    parser.add_argument("--cache-dir", default=None, help="tile cache (default <out>/.tiles)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, ".tiles")

    if args.combined:
        names = [n for n in boundaries if args.combined in n]
        if not names:
            raise SystemExit(f"no boundary matches {args.combined!r}")
        render(
            names,
            os.path.join(args.output_dir, f"{args.combined}_combined.png"),
            f"{args.combined}: {len(names)} overlapping fences",
            show_rovers=False,
            cache_dir=cache_dir,
        )
        return

    names = [args.boundary] if args.boundary else list(boundaries)
    for name in names:
        if name not in boundaries:
            raise SystemExit(f"unknown boundary {name!r}; have {list(boundaries)}")
        render(
            [name],
            os.path.join(args.output_dir, f"{name}.png"),
            f"{name} — fence, centroid, and per-rover start positions",
            cache_dir=cache_dir,
        )


if __name__ == "__main__":
    main()
