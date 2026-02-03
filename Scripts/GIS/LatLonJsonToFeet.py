import math
import json
import Rhino.Geometry as rg
import System
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path

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

def convert_geo_to_local_points(lon_tree, lat_tree, origin_lon_tree, origin_lat_tree):
    """
    Converts lon/lat DataTrees into Rhino Point3d DataTree in feet relative to origin per branch.
    """
    points_tree = DataTree[rg.Point3d]()
    x_coords_tree = DataTree[System.Double]()
    y_coords_tree = DataTree[System.Double]()
    
    if lon_tree is None or lat_tree is None:
        return points_tree, x_coords_tree, y_coords_tree
    
    try:
        lon_paths = list(lon_tree.Paths)
        lat_paths = list(lat_tree.Paths)
    except:
        lon_paths = []
        lat_paths = []
    
    all_paths = set()
    for p in lon_paths:
        all_paths.add(p)
    for p in lat_paths:
        all_paths.add(p)
    
    for path in all_paths:
        try:
            lon_branch = list(lon_tree.Branch(path)) if path in lon_tree.Paths else []
            lat_branch = list(lat_tree.Branch(path)) if path in lat_tree.Paths else []
            if len(lon_branch) == len(lat_branch) and len(lon_branch) > 0:
                origin_lon_val = None
                origin_lat_val = None
                if origin_lon_tree is not None and path in origin_lon_tree.Paths:
                    try:
                        origin_lon_branch = list(origin_lon_tree.Branch(path))
                        if len(origin_lon_branch) > 0:
                            origin_lon_val = float(origin_lon_branch[0])
                    except:
                        pass
                if origin_lat_tree is not None and path in origin_lat_tree.Paths:
                    try:
                        origin_lat_branch = list(origin_lat_tree.Branch(path))
                        if len(origin_lat_branch) > 0:
                            origin_lat_val = float(origin_lat_branch[0])
                    except:
                        pass
                
                if origin_lon_val is None or origin_lat_val is None:
                    if len(lon_branch) > 0:
                        try:
                            origin_lon_val = float(lon_branch[0])
                            origin_lat_val = float(lat_branch[0])
                        except:
                            origin_lon_val = None
                            origin_lat_val = None
                
                if origin_lon_val is not None and origin_lat_val is not None:
                    points_tree.EnsurePath(path)
                    x_coords_tree.EnsurePath(path)
                    y_coords_tree.EnsurePath(path)
                    for i in range(len(lon_branch)):
                        try:
                            lon_val = float(lon_branch[i])
                            lat_val = float(lat_branch[i])
                            x, y = latlon_to_local_xy_feet(lon_val, lat_val, origin_lon_val, origin_lat_val)
                            pt = rg.Point3d(x, y, 0.0)
                            points_tree.Add(pt, path)
                            x_coords_tree.Add(System.Double(x), path)
                            y_coords_tree.Add(System.Double(y), path)
                        except:
                            pass
        except:
            pass
    
    return points_tree, x_coords_tree, y_coords_tree

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

def parse_multipolygon_json(multipolygon_json_string):
    """
    Parse GeoJSON MultiPolygon or Polygon coordinate string and extract first ring lat/lon for each polygon.
    Supports both MultiPolygon format [[[ring1], [ring2]], [[ring3]]] and Polygon format [[ring1], [ring2]].
    
    Returns:
        lat_tree: DataTree[System.Double] - Latitudes, one branch per polygon
        lon_tree: DataTree[System.Double] - Longitudes, one branch per polygon
    """
    lat_tree = DataTree[System.Double]()
    lon_tree = DataTree[System.Double]()
    
    if not multipolygon_json_string:
        return lat_tree, lon_tree
    
    try:
        parsed = json.loads(multipolygon_json_string) if isinstance(multipolygon_json_string, str) else multipolygon_json_string
        
        if not isinstance(parsed, list) or len(parsed) == 0:
            return lat_tree, lon_tree
        
        is_multipolygon = False
        if len(parsed) > 0:
            first_elem = parsed[0]
            if isinstance(first_elem, list) and len(first_elem) > 0:
                first_ring = first_elem[0]
                if isinstance(first_ring, list) and len(first_ring) > 0:
                    first_coord = first_ring[0]
                    if isinstance(first_coord, list) and len(first_coord) >= 2:
                        is_multipolygon = True
        
        polygons_iter = parsed if is_multipolygon else [parsed]
        
        for poly_idx, polygon in enumerate(polygons_iter):
            if isinstance(polygon, list) and len(polygon) > 0:
                first_ring = polygon[0]
                if isinstance(first_ring, list) and len(first_ring) > 0:
                    path = GH_Path(poly_idx)
                    lat_tree.EnsurePath(path)
                    lon_tree.EnsurePath(path)
                    for coord in first_ring:
                        if isinstance(coord, list) and len(coord) >= 2:
                            try:
                                lon_val = float(coord[0])
                                lat_val = float(coord[1])
                                lon_tree.Add(System.Double(lon_val), path)
                                lat_tree.Add(System.Double(lat_val), path)
                            except:
                                pass
    except Exception as e:
        print("Failed to parse Polygon/MultiPolygon JSON: {}".format(e))
        pass
    
    return lat_tree, lon_tree

# --------------------
# Grasshopper Inputs:
#   multipolygon_json_string (str) — GeoJSON MultiPolygon string
#   optional_origin_lonlat (list of float, optional) — Origin [lon, lat] pair, default None
# --------------------

multipolygon_json_string = multipolygon_json_string if "multipolygon_json_string" in locals() else None

if multipolygon_json_string is not None:
    lat_tree, lon_tree = parse_multipolygon_json(multipolygon_json_string)
else:
    lat_tree = DataTree[System.Double]()
    lon_tree = DataTree[System.Double]()

origin_lon_tree = DataTree[System.Double]()
origin_lat_tree = DataTree[System.Double]()
origin_lonlat_tree = DataTree[System.Double]()

optional_origin_lonlat_tree = optional_origin_lonlat if "optional_origin_lonlat" in locals() else None

if optional_origin_lonlat_tree is not None:
    try:
        for path in optional_origin_lonlat_tree.Paths:
            branch = list(optional_origin_lonlat_tree.Branch(path))
            if len(branch) >= 2:
                try:
                    origin_lon_tree.EnsurePath(path)
                    origin_lat_tree.EnsurePath(path)
                    origin_lonlat_tree.EnsurePath(path)
                    origin_lon_tree.Add(System.Double(float(branch[0])), path)
                    origin_lat_tree.Add(System.Double(float(branch[1])), path)
                    origin_lonlat_tree.Add(System.Double(float(branch[0])), path)
                    origin_lonlat_tree.Add(System.Double(float(branch[1])), path)
                except:
                    pass
    except:
        pass
else:
    for path in lon_tree.Paths:
        try:
            lon_branch = list(lon_tree.Branch(path))
            lat_branch = list(lat_tree.Branch(path))
            if len(lon_branch) > 0 and len(lat_branch) > 0:
                try:
                    origin_lon_tree.EnsurePath(path)
                    origin_lat_tree.EnsurePath(path)
                    origin_lonlat_tree.EnsurePath(path)
                    origin_lon_val = float(lon_branch[0])
                    origin_lat_val = float(lat_branch[0])
                    origin_lon_tree.Add(System.Double(origin_lon_val), path)
                    origin_lat_tree.Add(System.Double(origin_lat_val), path)
                    origin_lonlat_tree.Add(System.Double(origin_lon_val), path)
                    origin_lonlat_tree.Add(System.Double(origin_lat_val), path)
                except:
                    pass
        except:
            pass

parcelPoints = convert_geo_to_local_points(lon_tree, lat_tree, origin_lon_tree, origin_lat_tree)

# --------------------
# Grasshopper Outputs:
#   origin_lonlat_tree — DataTree[System.Double] Origin [lon, lat] pairs per branch
#   parcelPoints       — DataTree[rg.Point3d] Site boundary points in feet, one branch per polygon
# --------------------