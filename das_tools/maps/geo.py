"""Cable route and lon/lat <-> metres conversion.

Geometry in metres (channel positions, along-route distance, channel spacing)
is done in a local azimuthal-equidistant frame, `LocalProjection`. Drawing on a
satellite image uses the image's own CRS instead; see `das_tools.maps.satellite`.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS, Geod, Transformer

# Spherical-earth degree lengths used for the lon/lat boxes of the map figures.
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQUATOR = 111.320


def read_kml_lonlat(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) of the longest LineString in a KML file.

    The cable KML holds one route; if a file has several LineStrings the
    longest is used rather than stitching them across gaps.
    """
    root = ET.parse(path).getroot()
    lines = []
    for el in root.iter():
        if not el.tag.endswith("LineString"):
            continue
        for coords in el:
            if not coords.tag.endswith("coordinates"):
                continue
            pairs = [tok.split(",")[:2] for tok in (coords.text or "").split()]
            ll = np.asarray([(float(a), float(b)) for a, b in pairs if b], dtype=float)
            if len(ll) >= 2:
                lines.append(ll)
    if not lines:
        raise ValueError(f"No LineString coordinates found in {path}")
    geod = Geod(ellps="WGS84")
    ll = max(lines, key=lambda q: geod.line_length(q[:, 0], q[:, 1]))
    return ll[:, 0].copy(), ll[:, 1].copy()


def lonlat_box(lon0: float, lat0: float, half_width_km: float,
               half_height_km: float | None = None) -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) of a box centred on (lon0, lat0)."""
    if half_height_km is None:
        half_height_km = half_width_km
    dlat = half_height_km / KM_PER_DEG_LAT
    dlon = half_width_km / (KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(lat0)))
    return lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat


def points_box(lon, lat, pad_km: float = 0.1) -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) enclosing the points, padded by pad_km."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lat_mid = 0.5 * (np.nanmin(lat) + np.nanmax(lat))
    dlat = pad_km / KM_PER_DEG_LAT
    dlon = pad_km / (KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(lat_mid)))
    return np.nanmin(lon) - dlon, np.nanmax(lon) + dlon, np.nanmin(lat) - dlat, np.nanmax(lat) + dlat


@dataclass
class LocalProjection:
    """Azimuthal-equidistant frame in metres centred on (origin_lat, origin_lon)."""

    origin_lat: float
    origin_lon: float
    to_xy: Transformer
    to_ll: Transformer

    @classmethod
    def centered(cls, lat: float, lon: float) -> "LocalProjection":
        crs_ll = CRS.from_epsg(4326)
        crs_xy = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat:.12f} +lon_0={lon:.12f} +datum=WGS84 +units=m +no_defs"
        )
        return cls(
            origin_lat=float(lat),
            origin_lon=float(lon),
            to_xy=Transformer.from_crs(crs_ll, crs_xy, always_xy=True),
            to_ll=Transformer.from_crs(crs_xy, crs_ll, always_xy=True),
        )

    def ll_to_xy(self, lon, lat):
        x, y = self.to_xy.transform(lon, lat)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def xy_to_ll(self, x, y):
        lon, lat = self.to_ll.transform(x, y)
        return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


@dataclass
class RouteCurve:
    """A polyline route in a local frame, parameterized by arc length s."""

    lonlat: np.ndarray  # (N, 2): lon, lat
    xy: np.ndarray      # (N, 2): local metres
    s: np.ndarray       # (N,) cumulative arc length [m]
    projection: LocalProjection

    @property
    def length_m(self) -> float:
        return float(self.s[-1])

    def interpolate_xy(self, s_query) -> np.ndarray:
        sq = np.clip(np.asarray(s_query, dtype=float), 0.0, self.length_m)
        return np.column_stack([np.interp(sq, self.s, self.xy[:, 0]),
                                np.interp(sq, self.s, self.xy[:, 1])])

    def project_xy(self, points_xy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest point on the polyline: (along-route s, distance, projected xy)."""
        pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
        a = self.xy[:-1]
        v = self.xy[1:] - a
        vv = np.sum(v * v, axis=1)
        seg_len = np.sqrt(vv)
        safe_vv = np.where(vv > 0.0, vv, 1.0)
        out_s, out_d, out_q = [], [], []
        for p in pts:
            t = np.clip(np.sum((p - a) * v, axis=1) / safe_vv, 0.0, 1.0)
            q = a + t[:, None] * v
            d2 = np.sum((p - q) ** 2, axis=1)
            i = int(np.argmin(d2))
            out_s.append(float(self.s[i] + t[i] * seg_len[i]))
            out_d.append(math.sqrt(float(d2[i])))
            out_q.append(q[i])
        return np.asarray(out_s), np.asarray(out_d), np.asarray(out_q)

    def project_lonlat(self, lon, lat):
        x, y = self.projection.ll_to_xy(lon, lat)
        return self.project_xy(np.column_stack([x, y]))


def load_kml_route(path: str | Path) -> RouteCurve:
    """The KML route in a local frame centred on the mean of its vertices."""
    lon, lat = read_kml_lonlat(path)
    proj = LocalProjection.centered(float(np.mean(lat)), float(np.mean(lon)))
    x, y = proj.ll_to_xy(lon, lat)
    xy = np.column_stack([x, y])
    # Drop repeated vertices so every segment has a direction.
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]
    ll = np.column_stack([lon, lat])[keep]
    xy = xy[keep]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    return RouteCurve(lonlat=ll, xy=xy, s=s, projection=proj)
