import math
import Rhino.Geometry as rg

# --------------------
# Utility Functions
# --------------------

def deg2rad(deg):
    return deg * (math.pi / 180)

def rad2deg(rad):
    return rad * (180 / math.pi)

def latlon_to_local_xy_feet(lon, lat, lon0, lat0):
    """
    Convert lon/lat to local X/Y coordinates in feet using equirectangular projection.
    """
    R = 6371000  # Earth radius in meters
    m_to_ft = 3.28084

    # Convert degrees to radians
    lon_r = deg2rad(lon)
    lat_r = deg2rad(lat) 
    lon0_r = deg2rad(lon0)
    lat0_r = deg2rad(lat0)

    x_m = (lon_r - lon0_r) * math.cos((lat_r + lat0_r) / 2.0) * R
    y_m = (lat_r - lat0_r) * R

    return x_m * m_to_ft, y_m * m_to_ft

def calculate_point_at_distance_and_bearing(center_lat, center_lon, distance_feet, bearing_degrees):
    """
    Calculates a new lat/lon given a center point, distance (in feet), and bearing.
    """
    R = 6378137  # Earth radius in meters
    distance_m = distance_feet * 0.3048
    bearing_rad = deg2rad(bearing_degrees)

    lat1 = deg2rad(center_lat)
    lon1 = deg2rad(center_lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance_m / R) +
                     math.cos(lat1) * math.sin(distance_m / R) * math.cos(bearing_rad))

    lon2 = lon1 + math.atan2(math.sin(bearing_rad) * math.sin(distance_m / R) * math.cos(lat1),
                             math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2))

    return [rad2deg(lon2), rad2deg(lat2)]

def calculate_square_corners(center_lon, center_lat, side_length):
    """
    Returns 4 corner coordinates (lon, lat) for a square centered at the given point.
    """
    half_side = side_length / 2
    north_mid = calculate_point_at_distance_and_bearing(center_lat, center_lon, half_side, 0)
    east_mid = calculate_point_at_distance_and_bearing(center_lat, center_lon, half_side, 90)
    south_mid = calculate_point_at_distance_and_bearing(center_lat, center_lon, half_side, 180)
    west_mid = calculate_point_at_distance_and_bearing(center_lat, center_lon, half_side, 270)

    return [
        [west_mid[0], north_mid[1]],   # Top-Left
        [east_mid[0], north_mid[1]],   # Top-Right
        [east_mid[0], south_mid[1]],   # Bottom-Right
        [west_mid[0], south_mid[1]],   # Bottom-Left
    ]

def convert_geo_to_local_points(lon_list, lat_list, origin_lon, origin_lat):
    """
    Converts lon/lat lists into Rhino Point3d list in feet relative to origin.
    """
    points = []
    x_coords = []
    y_coords = []
    for lon, lat in zip(lon_list, lat_list):
        x, y = latlon_to_local_xy_feet(lon, lat, origin_lon, origin_lat)
        pt = rg.Point3d(x, y, 0.0)
        points.append(pt)
        x_coords.append(x)
        y_coords.append(y)
    return points, x_coords, y_coords

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the geodesic distance in feet between two lat/lon points.
    """
    R = 6378137
    d_lat = deg2rad(lat2 - lat1)
    d_lon = deg2rad(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(d_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 3.28084  # Feet

def calc_mapbox_side_length(center_lat, zoom=16, image_px=1280):
    """
    Calculate the real-world side length (feet) of a Mapbox Static Image square
    given center latitude, zoom level, and image size in pixels.
    """
    # meters per pixel at this latitude
    resolution_m = (156543.03392 / (2 ** zoom)) * math.cos(deg2rad(center_lat))
    # total side length in meters
    side_m = resolution_m * image_px
    return side_m * 3.28084  # feet

# --------------------
# Grasshopper Inputs:
#   lonList (list of float or str) — parcel boundary longitudes
#   latList (list of float or str) — parcel boundary latitudes
#   zoom (int, optional) — Mapbox zoom level, default 16
#   imageSize (int, optional) — Mapbox static image size (width = height), default 1280
# --------------------

# Ensure inputs are floats
lon_list = [float(lon) for lon in lonList]
lat_list = [float(lat) for lat in latList]

if not lon_list or not lat_list or len(lon_list) != len(lat_list):
    raise ValueError("Longitude and latitude lists must be non-empty and equal length.")

# Use first point as origin
origin_lon = lon_list[0]
origin_lat = lat_list[0]
origin_lonlat = [origin_lon, origin_lat]

# Default params
if 'zoom' not in locals() or zoom is None:
    zoom = 16
if 'imageSize' not in locals() or imageSize is None:
    imageSize = 1280

# --------------------
# Auto-calculated Context Box Size
# --------------------
sideLength = calc_mapbox_side_length(origin_lat, zoom=zoom, image_px=imageSize)/2

# --------------------
# Parcel Boundary
# --------------------
if optional_origin_lonlat is not None:
    origin_lon = optional_origin_lonlat[0]
    origin_lat = optional_origin_lonlat[1]
    origin_lonlat = [origin_lon, origin_lat]

parcelPoints, parcelX, parcelY = convert_geo_to_local_points(lon_list, lat_list, origin_lon, origin_lat)

# --------------------
# Context Box (Square)
# --------------------
corner_coords = calculate_square_corners(origin_lon, origin_lat, sideLength)
context_lon = [pt[0] for pt in corner_coords]
context_lat = [pt[1] for pt in corner_coords]
contextPoints, contextX, contextY = convert_geo_to_local_points(context_lon, context_lat, origin_lon, origin_lat)

# Named corners (for convenience)
topLeft = contextPoints[0]
topRight = contextPoints[1]
bottomRight = contextPoints[2]
bottomLeft = contextPoints[3]

# --------------------
# Grasshopper Outputs:
#   parcelPoints   — [Point3d] Site boundary in feet
#   contextPoints  — [Point3d] Context square corners in feet
#   sideLength     — float, matched to Mapbox Static Image
#   topLeft, topRight, bottomRight, bottomLeft — individual corners
# --------------------