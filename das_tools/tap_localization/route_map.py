"""Channel positions on the walked cable route, registered by the reviewed hammer taps.

The cable lies along the route walked and traced in the KML, so every channel
is placed *on* that route; the hammers only decide *where along it* each channel
sits. Each reviewed tap gives one knot (canonical channel, along-route distance
s of the route point nearest its GPS fix). Between knots, channels are spaced
evenly in arc length; beyond the first and last knots they continue at the
nominal channel spacing, along the route while it lasts and then straight along
its end direction.

The cross-track distance from each hammer to the route is reported, not fitted:
it is the GPS and tracing error the method chooses not to chase.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np

from das_tools.maps.geo import RouteCurve

from .epochs import Rules, canonical_channel_count, channel_shift_for_datetime
from .picking import load_hammer_table

# Onyx SpatialSamplingInterval for the whole deployment.
NOMINAL_CHANNEL_SPACING_M = 3.1904762684013206


@dataclass
class RouteMapConfig:
    channel_epoch_shift_rules: Rules = ()
    nominal_spacing_m: float = NOMINAL_CHANNEL_SPACING_M
    # Straight continuation past either end of the KML follows the route's
    # direction averaged over this length, not its last traced vertex.
    end_tangent_span_m: float = 20.0
    n_channels: int | None = None   # default: canonical count implied by the picks' number_of_loci


@dataclass
class RouteChannelGeometry:
    xy: np.ndarray              # (N, 2) local metres, frame of `route.projection`
    lonlat: np.ndarray          # (N, 2) lon, lat
    route_s_m: np.ndarray       # (N,) along-route position; < 0 or > route length off the KML
    source: np.ndarray          # (N,) 'anchor' | 'on_route' | 'off_route_start' | 'off_route_end'
    route: RouteCurve
    anchors: list               # the reviewed taps with their route diagnostics
    anchor_segments: list       # one row per pair of consecutive taps
    nominal_spacing_m: float
    route_reversed: bool

    @property
    def n_channels(self) -> int:
        return self.xy.shape[0]

    @property
    def spacing_m(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.xy, axis=0), axis=1)


# --------------------------------------------------------------------------
# reviewed picks
# --------------------------------------------------------------------------

def load_reviewed_anchors(picks_csv, rules: Rules) -> list[dict]:
    """Hammers with a reviewed `arrival_channel`, moved to canonical channel numbers.

    Rows with a blank arrival_channel (not reviewed, or skipped) are left out.
    Taps reviewed onto the same canonical channel are averaged into one anchor.
    """
    anchors = []
    for r in load_hammer_table(picks_csv):
        ch = r.get("arrival_channel", "")
        if ch == "" or not np.isfinite(float(ch)):
            continue
        shift = channel_shift_for_datetime(r["timestamp_dt"], rules)
        anchors.append({
            "timestamp": r["timestamp"],
            "lat": float(r["lat"]), "lon": float(r["lon"]),
            "channel_recorded": float(ch),
            "channel_epoch_shift": int(shift),
            "channel_canonical": float(ch) + shift,
            "number_of_loci": float(r.get("number_of_loci") or np.nan),
            "_dt": r["timestamp_dt"],
        })
    if len(anchors) < 2:
        raise ValueError(f"Need at least two reviewed picks in {picks_csv}")
    anchors.sort(key=lambda a: (a["channel_canonical"], a["timestamp"]))

    merged = []
    for _, group in groupby(anchors, key=lambda a: a["channel_canonical"]):
        group = list(group)
        m = dict(group[0])
        if len(group) > 1:
            m["lat"] = float(np.mean([g["lat"] for g in group]))
            m["lon"] = float(np.mean([g["lon"] for g in group]))
            m["timestamp"] = " | ".join(g["timestamp"] for g in group)
        merged.append(m)
    return merged


def canonical_channel_count_from_anchors(anchors, rules: Rules) -> int:
    """Largest canonical channel count of the recordings the picks came from."""
    counts = [canonical_channel_count(int(a["number_of_loci"]), a["_dt"], rules)
              for a in anchors if np.isfinite(a["number_of_loci"])]
    return max(counts) if counts else int(max(a["channel_canonical"] for a in anchors)) + 1


# --------------------------------------------------------------------------
# route, extended straight past its ends
# --------------------------------------------------------------------------

def reverse_route(route: RouteCurve) -> RouteCurve:
    return RouteCurve(lonlat=route.lonlat[::-1].copy(), xy=route.xy[::-1].copy(),
                      s=route.length_m - route.s[::-1], projection=route.projection)


class RouteEvaluator:
    """Position, tangent and normal at any arc length s, on or off the walked route."""

    def __init__(self, route: RouteCurve, end_tangent_span_m: float = 20.0):
        self.route = route
        self.length_m = route.length_m
        span = float(min(max(end_tangent_span_m, 1e-3), self.length_m))
        self._t_start = self._unit(*route.interpolate_xy([0.0, span]))
        self._t_end = self._unit(*route.interpolate_xy([self.length_m - span, self.length_m]))
        self._h = max(0.5, span / 4.0)   # half-step for the tangent finite difference

    @staticmethod
    def _unit(p0, p1) -> np.ndarray:
        v = p1 - p0
        return v / np.linalg.norm(v)

    def xy_at(self, s) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, dtype=float))
        pos = self.route.interpolate_xy(s)
        before, after = s < 0.0, s > self.length_m
        pos[before] += s[before, None] * self._t_start
        pos[after] += (s[after] - self.length_m)[:, None] * self._t_end
        return pos

    def tangent_at(self, s) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, dtype=float))
        v = self.xy_at(s + self._h) - self.xy_at(s - self._h)
        n = np.linalg.norm(v, axis=1)
        return v / np.where(n > 0, n, 1.0)[:, None]

    def normal_at(self, s) -> np.ndarray:
        """Unit normal to the left of the direction of increasing s."""
        t = self.tangent_at(s)
        return np.column_stack([-t[:, 1], t[:, 0]])


def _snap_to_route(anchors: list, ev: RouteEvaluator) -> None:
    """Along-route position and cross-track offset of each hammer GPS fix.

    The nearest point on the polyline is refined with a few Newton steps in the
    evaluator's smoothed tangent frame, so the offset is perpendicular to the
    route direction rather than to a single traced segment.
    """
    route = ev.route
    lon = np.asarray([a["lon"] for a in anchors])
    lat = np.asarray([a["lat"] for a in anchors])
    s0, dist, _ = route.project_lonlat(lon, lat)
    x, y = route.projection.ll_to_xy(lon, lat)
    p = np.column_stack([x, y])

    s = s0.copy()
    step = max(2.0 * ev._h, 1.0)
    for _ in range(4):
        ds = np.sum((p - ev.xy_at(s)) * ev.tangent_at(s), axis=1)
        s += np.clip(ds, -step, step)
        if np.max(np.abs(ds)) < 1e-9:
            break
    s = np.where(np.abs(s - s0) > 5.0 * step, s0, s)   # a sharp corner can derail it

    snapped = ev.xy_at(s)
    signed = np.sum((p - snapped) * ev.normal_at(s), axis=1)
    for i, a in enumerate(anchors):
        a.update(x_m=float(x[i]), y_m=float(y[i]), route_s_m=float(s[i]),
                 snapped_x_m=float(snapped[i, 0]), snapped_y_m=float(snapped[i, 1]),
                 cross_track_m=float(dist[i]), cross_track_signed_m=float(signed[i]))


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def build_route_channel_map(anchors: list, route: RouteCurve, cfg: RouteMapConfig) -> RouteChannelGeometry:
    """Place every canonical channel on the walked route."""
    anchors = [dict(a) for a in anchors]
    ev = RouteEvaluator(route, cfg.end_tangent_span_m)
    _snap_to_route(anchors, ev)

    # Run the route the same way as the channel numbering.
    c = np.asarray([a["channel_canonical"] for a in anchors])
    reversed_ = bool(np.polyfit(c, [a["route_s_m"] for a in anchors], 1)[0] < 0)
    if reversed_:
        route = reverse_route(route)
        ev = RouteEvaluator(route, cfg.end_tangent_span_m)
        _snap_to_route(anchors, ev)
    s_knot = np.asarray([a["route_s_m"] for a in anchors])

    folds = np.flatnonzero(np.diff(s_knot) < 0) + 1
    for i, a in enumerate(anchors):
        a["breaks_route_order"] = bool(i in folds)
    if folds.size:
        warnings.warn("Route order contradicts channel order at "
                      + ", ".join(f"channel {c[i]:.0f}" for i in folds)
                      + "; the channel map folds back there. Check those picks.", stacklevel=2)

    n = cfg.n_channels or canonical_channel_count_from_anchors(anchors, cfg.channel_epoch_shift_rules)
    n = max(n, int(math.floor(c[-1])) + 1)
    ch = np.arange(n, dtype=float)
    s = np.interp(ch, c, s_knot)
    head, tail = ch < c[0], ch > c[-1]
    s[head] = s_knot[0] + (ch[head] - c[0]) * cfg.nominal_spacing_m
    s[tail] = s_knot[-1] + (ch[tail] - c[-1]) * cfg.nominal_spacing_m

    xy = ev.xy_at(s)
    source = np.full(n, "on_route", dtype=object)
    source[s < 0.0] = "off_route_start"
    source[s > route.length_m] = "off_route_end"
    for a in anchors:
        k = a["channel_canonical"]
        if float(k).is_integer() and 0 <= k < n:
            source[int(k)] = "anchor"
            pos = xy[int(k)]
        else:
            pos = ev.xy_at([a["route_s_m"]])[0]
        a["assigned_x_m"], a["assigned_y_m"] = float(pos[0]), float(pos[1])
        a["assignment_error_m"] = float(math.hypot(pos[0] - a["x_m"], pos[1] - a["y_m"]))

    segments = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        dch = b["channel_canonical"] - a["channel_canonical"]
        ds = b["route_s_m"] - a["route_s_m"]
        segments.append({
            "channel_start": a["channel_canonical"], "channel_end": b["channel_canonical"],
            "timestamp_start": a["timestamp"], "timestamp_end": b["timestamp"],
            "n_channels": dch, "route_arc_length_m": ds,
            "implied_spacing_m": ds / dch,
            # How many channels off the pair is from the nominal spacing.
            "channel_gap_error": ds / cfg.nominal_spacing_m - dch,
        })

    lon, lat = route.projection.xy_to_ll(xy[:, 0], xy[:, 1])
    return RouteChannelGeometry(
        xy=xy, lonlat=np.column_stack([lon, lat]), route_s_m=s, source=source, route=route,
        anchors=anchors, anchor_segments=segments, nominal_spacing_m=cfg.nominal_spacing_m,
        route_reversed=reversed_,
    )


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def _write_rows(path: Path, rows) -> None:
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_channel_geometry(geom: RouteChannelGeometry, out_dir, *, name: str,
                          bend_channel: int, leg_half_width: int) -> dict:
    """Write the geometry in the forms the rest of the project reads.

    <name>.npz      ch_xy (2, N) local metres, ch_latlon (2, N) as (lat, lon),
                    origin_lat/origin_lon of the local frame
    <name>_L.npz    the channels [bend - half width, bend + half width) that
                    the hourly decimated files keep (pisces25.load_leg_geometry)
    channel_geometry.csv, tap_anchors.csv, anchor_segments.csv, summary.json
                    per-channel and per-tap tables for the figures
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proj = geom.route.projection
    origin = dict(origin_lat=np.float64(proj.origin_lat), origin_lon=np.float64(proj.origin_lon))
    ch_xy = geom.xy.T
    ch_latlon = geom.lonlat[:, ::-1].T
    np.savez(out / f"{name}.npz", ch_xy=ch_xy, ch_latlon=ch_latlon, **origin)

    leg = np.arange(bend_channel - leg_half_width, bend_channel + leg_half_width)
    np.savez(out / f"{name}_L.npz", ch_xy=ch_xy[:, leg], ch_latlon=ch_latlon[:, leg],
             channel=leg, channel_offset=leg - bend_channel, bend_ch=np.int64(bend_channel),
             source_file=f"{name}.npz", **origin)

    spacing = np.r_[geom.spacing_m, np.nan]
    _write_rows(out / "channel_geometry.csv", (
        {"channel": i, "lon": geom.lonlat[i, 0], "lat": geom.lonlat[i, 1],
         "x_m": geom.xy[i, 0], "y_m": geom.xy[i, 1], "route_s_m": geom.route_s_m[i],
         "source": geom.source[i], "spacing_to_next_m": spacing[i]}
        for i in range(geom.n_channels)))
    _write_rows(out / "tap_anchors.csv",
                ({k: v for k, v in a.items() if not k.startswith("_")} for a in geom.anchors))
    _write_rows(out / "anchor_segments.csv", geom.anchor_segments)

    cross = np.asarray([a["cross_track_m"] for a in geom.anchors])
    err = np.asarray([a["assignment_error_m"] for a in geom.anchors])
    summary = {
        "n_channels": geom.n_channels,
        "n_anchors": len(geom.anchors),
        "route_length_m": geom.route.length_m,
        "route_reversed": geom.route_reversed,
        "n_channels_on_route": int(np.sum((geom.route_s_m >= 0) & (geom.route_s_m <= geom.route.length_m))),
        "total_cable_length_m": float(np.sum(geom.spacing_m)),
        "nominal_cable_length_m": (geom.n_channels - 1) * geom.nominal_spacing_m,
        "nominal_spacing_m": geom.nominal_spacing_m,
        "hammer_cross_track_median_m": float(np.median(cross)),
        "hammer_cross_track_max_m": float(np.max(cross)),
        "hammer_assignment_error_median_m": float(np.median(err)),
        "hammer_assignment_error_max_m": float(np.max(err)),
        "origin_lat": proj.origin_lat,
        "origin_lon": proj.origin_lon,
        "bend_channel": bend_channel,
        "leg_channels": [int(leg[0]), int(leg[-1])],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
