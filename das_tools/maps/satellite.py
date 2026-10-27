"""Satellite basemaps: a GeoTIFF cropped to a lon/lat box, in the image's own CRS.

    box = geo.lonlat_box(lon0, lat0, half_width_km=0.6)
    bm = load_basemap(proj_path.data_dir("satellite", "..._visual_clip.tif"), *box)
    bm.show(ax)
    ax.plot(*bm.xy(cable_lon, cable_lat))

Everything drawn on the map goes through `bm.xy(lon, lat)`. data/satellite/ is
not tracked, so without the GeoTIFF (or without rasterio) the basemap falls back
to a plain background in `fallback_crs` over the same box, and the figure still
renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer

# CRS of the Planet scenes in data/satellite/ (UTM zone 4N).
PLANET_CRS = "EPSG:32604"


def stretch_percentile(img: np.ndarray, pmin: float = 0, pmax: float = 98) -> np.ndarray:
    """Scale each band to [0, 1] between its pmin and pmax percentiles."""
    img = np.asarray(img, dtype=float)
    bands = img[..., None] if img.ndim == 2 else img
    out = np.zeros_like(bands)
    for i in range(bands.shape[-1]):
        lo, hi = np.percentile(bands[..., i], (pmin, pmax))
        if hi > lo:
            out[..., i] = np.clip((bands[..., i] - lo) / (hi - lo), 0, 1)
    return out[..., 0] if img.ndim == 2 else out


@dataclass
class Basemap:
    image: np.ndarray | None   # stretched image in [0, 1]; None for a plain background
    extent: list[float]        # [left, right, bottom, top] in `crs`
    crs: object
    to_map: Transformer        # lon/lat -> map
    to_ll: Transformer         # map -> lon/lat

    def xy(self, lon, lat):
        """Map coordinates of lon/lat points."""
        x, y = self.to_map.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        return np.asarray(x), np.asarray(y)

    def lonlat(self, x, y):
        lon, lat = self.to_ll.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.asarray(lon), np.asarray(lat)

    @property
    def center(self) -> tuple[float, float]:
        return 0.5 * (self.extent[0] + self.extent[1]), 0.5 * (self.extent[2] + self.extent[3])

    @property
    def size(self) -> tuple[float, float]:
        """(width, height) in map units."""
        return self.extent[1] - self.extent[0], self.extent[3] - self.extent[2]

    def crop(self, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> "Basemap":
        """The part of this basemap inside a lon/lat box, for zoom panels and insets.

        The image keeps this basemap's contrast stretch, so a zoom matches its
        parent map instead of being re-stretched on a smaller, flatter crop.
        """
        x, y = self.xy([lon_min, lon_max], [lat_min, lat_max])
        left, right, bottom, top = self.extent
        if self.image is None:
            return Basemap(None, [x.min(), x.max(), y.min(), y.max()], self.crs, self.to_map, self.to_ll)
        h, w = self.image.shape[:2]
        px, py = (right - left) / w, (top - bottom) / h
        c0, c1 = max(int(np.floor((x.min() - left) / px)), 0), min(int(np.ceil((x.max() - left) / px)), w)
        r0, r1 = max(int(np.floor((top - y.max()) / py)), 0), min(int(np.ceil((top - y.min()) / py)), h)
        return Basemap(self.image[r0:r1, c0:c1],
                       [left + c0 * px, left + c1 * px, top - r1 * py, top - r0 * py],
                       self.crs, self.to_map, self.to_ll)

    def show(self, ax, facecolor="0.92", **imshow_kw):
        """Draw the image (or the plain background) and fix the axes to its extent."""
        if self.image is not None:
            ax.imshow(self.image, extent=self.extent, origin="upper", **imshow_kw)
        else:
            ax.set_facecolor(facecolor)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        ax.set_aspect("equal")
        return ax


def _transformers(crs):
    return (Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True))


def load_basemap(
    geotiff: str | Path | None,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    stretch: tuple[float, float] = (0, 98),
    downsample: int = 1,
    fallback_crs=PLANET_CRS,
) -> Basemap:
    """The GeoTIFF cropped to the lon/lat box and contrast-stretched.

    The box corners are transformed into the image CRS and the crop is clipped
    to the image bounds. `downsample` > 1 reads a coarser image for quick looks.
    """
    geotiff = Path(geotiff) if geotiff is not None else None
    try:
        if geotiff is None or not geotiff.exists():
            raise FileNotFoundError(geotiff)
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
    except (FileNotFoundError, ImportError) as exc:
        reason = "rasterio is not installed" if isinstance(exc, ImportError) else f"no GeoTIFF at {geotiff}"
        print(f"{reason}; drawing the map on a plain background in {fallback_crs}.")
        to_map, to_ll = _transformers(fallback_crs)
        x1, y1 = to_map.transform(lon_min, lat_min)
        x2, y2 = to_map.transform(lon_max, lat_max)
        extent = [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]
        return Basemap(None, extent, fallback_crs, to_map, to_ll)

    with rasterio.open(geotiff) as src:
        if src.crs is None:
            raise ValueError(f"{geotiff} has no CRS")
        to_map, to_ll = _transformers(src.crs)
        x1, y1 = to_map.transform(lon_min, lat_min)
        x2, y2 = to_map.transform(lon_max, lat_max)
        b = src.bounds
        xmin, xmax = max(min(x1, x2), b.left), min(max(x1, x2), b.right)
        ymin, ymax = max(min(y1, y2), b.bottom), min(max(y1, y2), b.top)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"The lon/lat box is outside {geotiff.name}")

        win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        out_h = max(int(win.height / downsample), 1)
        out_w = max(int(win.width / downsample), 1)
        if src.count >= 3:
            data = src.read([1, 2, 3], window=win, out_shape=(3, out_h, out_w),
                            resampling=Resampling.bilinear)
            img = np.moveaxis(data, 0, -1)
        else:
            img = src.read(1, window=win, out_shape=(out_h, out_w),
                           resampling=Resampling.bilinear)

        # Extent of the pixels actually read, which snap to the image grid.
        wt = src.window_transform(win)
        left, top = wt.c, wt.f
        right = left + wt.a * out_w * downsample
        bottom = top + wt.e * out_h * downsample
        crs = src.crs

    return Basemap(stretch_percentile(img, *stretch), [left, right, bottom, top], crs, to_map, to_ll)
