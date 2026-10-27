"""Map furniture for a `Basemap` axes: lon/lat ticks, scale bar, N/E arrows, azimuth arrows."""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pyproj import Geod

from .satellite import Basemap


def halo(linewidth: float = 3, color: str = "white") -> list:
    """Path effect that outlines a line or text, e.g. ``path_effects=halo()``."""
    return [pe.Stroke(linewidth=linewidth, foreground=color), pe.Normal()]


def add_lonlat_axes(
    ax,
    bm: Basemap,
    lon_step: float | None,
    lat_step: float | None,
    *,
    lon_decimals: int = 2,
    lat_decimals: int = 3,
    rotate_lat_labels: bool = True,
    labels: bool = True,
):
    """Replace the map-coordinate ticks with longitude/latitude ticks.

    Ticks are read along the lines through the map centre, which is what a
    reader expects from a small, nearly conformal UTM map. A step of None
    leaves that axis without ticks, e.g. the latitude of a right-hand panel.
    Returns the (longitude, latitude) secondary axes, None where skipped.
    """
    x_c, y_c = bm.center
    lon_c, lat_c = bm.lonlat(x_c, y_c)

    def x_to_lon(x):
        x = np.asarray(x, dtype=float)
        return bm.lonlat(x, np.full_like(x, y_c))[0]

    def lon_to_x(lon):
        lon = np.asarray(lon, dtype=float)
        return bm.xy(lon, np.full_like(lon, lat_c))[0]

    def y_to_lat(y):
        y = np.asarray(y, dtype=float)
        return bm.lonlat(np.full_like(y, x_c), y)[1]

    def lat_to_y(lat):
        lat = np.asarray(lat, dtype=float)
        return bm.xy(np.full_like(lat, lon_c), lat)[1]

    ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False,
                   right=False, labelbottom=False, labelleft=False)
    ax.grid(False)

    sx = sy = None
    if lon_step is not None:
        sx = ax.secondary_xaxis("bottom", functions=(x_to_lon, lon_to_x))
        sx.xaxis.set_major_locator(MultipleLocator(lon_step))
        sx.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.{lon_decimals}f}"))
        if labels:
            sx.set_xlabel("Longitude (deg)")
    if lat_step is not None:
        sy = ax.secondary_yaxis("left", functions=(y_to_lat, lat_to_y))
        sy.yaxis.set_major_locator(MultipleLocator(lat_step))
        sy.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.{lat_decimals}f}"))
        if labels:
            sy.set_ylabel("Latitude (deg)")
        if rotate_lat_labels:
            sy.tick_params(axis="y", pad=16)
            plt.setp(sy.get_yticklabels(), rotation=90, ha="center", va="top", rotation_mode="anchor")
    return sx, sy


def add_scalebar(
    ax,
    bm: Basemap,
    length_m: float,
    *,
    n_blocks: int = 2,
    corner: str = "lower left",
    pad: tuple[float, float] = (0.08, 0.05),
    bar_height: float | None = None,
    box_pad: float | None = None,
    fontsize: float = 14,
    zorder: float = 25,
):
    """Alternating black/white bar of true ground length `length_m`.

    The length is measured geodesically eastward from the bar's start, so it is
    right in any map CRS. `pad` is the gap from the chosen corner as a fraction
    of the map width and height; `bar_height` is in map units (default 1.5 % of
    the map height); `box_pad` is the white box's margin around the bar and
    label, in points (default 0.6 of the bar height). The box is widened to the
    label when the label is longer than the bar. Call it after `bm.show(ax)` and
    after the figure layout is set, since the box is sized to the label from the
    axes' current scale.
    """
    width, height = bm.size
    if bar_height is None:
        bar_height = 0.015 * height
    vert, horiz = corner.split()

    # Projected bar length from a geodesic `length_m` eastward at the map centre row.
    lon0, lat0 = bm.lonlat(*bm.center)
    lon1, lat1, _ = Geod(ellps="WGS84").fwd(float(lon0), float(lat0), 90.0, length_m)
    bar_len = float(bm.xy(lon1, lat1)[0] - bm.center[0])

    # Label size in map units, so the background box fits the text at any
    # map scale. Read from the axes as drawn now, i.e. after bm.show(ax).
    label = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    text = ax.text(0, 0, label, ha="center", va="bottom", fontsize=fontsize,
                   zorder=zorder + 2, path_effects=halo())
    text_px = text.get_window_extent(ax.figure.canvas.get_renderer())
    per_px = abs(np.diff(ax.get_ylim())[0]) / ax.get_window_extent().height
    text_w, text_h = text_px.width * per_px, text_px.height * per_px
    margin = 0.6 * bar_height if box_pad is None else box_pad * ax.figure.dpi / 72 * per_px
    gap = 0.3 * bar_height
    box_w = max(bar_len, text_w) + 2 * margin
    box_h = 2 * margin + bar_height + gap + text_h

    if horiz == "left":
        x_mid = bm.extent[0] + pad[0] * width - margin + box_w / 2
    else:
        x_mid = bm.extent[1] - pad[0] * width + margin - box_w / 2
    if vert == "lower":
        y0 = bm.extent[2] + pad[1] * height
    else:
        y0 = bm.extent[3] - pad[1] * height - (box_h - margin)
    x0 = x_mid - bar_len / 2

    block = bar_len / n_blocks
    for i in range(n_blocks):
        ax.add_patch(Rectangle((x0 + i * block, y0), block, bar_height,
                               facecolor="black" if i % 2 == 0 else "white",
                               edgecolor="black", linewidth=1.0, zorder=zorder + 1))
    ax.add_patch(Rectangle((x_mid - box_w / 2, y0 - margin), box_w, box_h,
                           facecolor="white", edgecolor="black", linewidth=1.0, alpha=0.8,
                           zorder=zorder))
    text.set_position((x_mid, y0 + bar_height + gap))


def add_ne_arrows(ax, x0: float = 0.10, y0: float = 0.10, length: float = 0.07, *,
                  north_color="red", east_color="black", linewidth=2.5,
                  mutation_scale=20, fontsize=16):
    """North and east arrows from (x0, y0), in axes-fraction coordinates."""
    for (dx, dy), color, text, ha, va, offset in [
        ((0, length), north_color, "N", "center", "bottom", (0, 0.015)),
        ((length, 0), east_color, "E", "left", "center", (0.015, 0)),
    ]:
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=linewidth,
                                    mutation_scale=mutation_scale))
        ax.text(x0 + dx + offset[0], y0 + dy + offset[1], text, transform=ax.transAxes,
                color=color, fontsize=fontsize, fontweight="bold", ha=ha, va=va)


def add_azimuth_arrow(ax, head, azimuth_deg: float, length: float, *, color="black",
                      linewidth=4, mutation_scale=24, zorder=30):
    """Arrow ending at `head` (map coordinates) travelling towards `azimuth_deg`.

    The azimuth is a map bearing (0 = north, 90 = east, clockwise), so a wind
    *from* 295 deg is drawn with ``azimuth_deg=295 + 180``. `length` is in map units.
    """
    theta = np.deg2rad(azimuth_deg)
    direction = np.array([np.sin(theta), np.cos(theta)])
    head = np.asarray(head, dtype=float)
    arrow = FancyArrowPatch(posA=head - direction * length, posB=head, arrowstyle="-|>",
                            mutation_scale=mutation_scale, linewidth=linewidth, color=color,
                            zorder=zorder, shrinkA=0, shrinkB=0,
                            path_effects=halo(linewidth + 2, "black"))
    ax.add_patch(arrow)
    return arrow
