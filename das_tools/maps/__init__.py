"""Maps of the deployment: cable route, lon/lat <-> metres, satellite basemaps and map furniture."""
from .geo import (
    read_kml_lonlat, lonlat_box, points_box, LocalProjection, RouteCurve, load_kml_route,
)
from .satellite import PLANET_CRS, Basemap, load_basemap, stretch_percentile
from .decorations import halo, add_lonlat_axes, add_scalebar, add_ne_arrows, add_azimuth_arrow
