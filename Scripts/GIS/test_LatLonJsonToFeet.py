import pytest
import math
import json

def deg2rad(deg):
    return deg * (math.pi / 180)

def latlon_to_local_xy_feet(lon, lat, lon0, lat0):
    R = 6371000
    m_to_ft = 3.28084
    lon_r = deg2rad(lon)
    lat_r = deg2rad(lat)
    lon0_r = deg2rad(lon0)
    lat0_r = deg2rad(lat0)
    x_m = (lon_r - lon0_r) * math.cos((lat_r + lat0_r) / 2.0) * R
    y_m = (lat_r - lat0_r) * R
    return x_m * m_to_ft, y_m * m_to_ft

def parse_multipolygon_json_simple(multipolygon_json_string):
    if not multipolygon_json_string:
        return [], []
    
    try:
        parsed = json.loads(multipolygon_json_string) if isinstance(multipolygon_json_string, str) else multipolygon_json_string
        
        if not isinstance(parsed, list) or len(parsed) == 0:
            return [], []
        
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
        
        result_lat = []
        result_lon = []
        
        for poly_idx, polygon in enumerate(polygons_iter):
            if isinstance(polygon, list) and len(polygon) > 0:
                first_ring = polygon[0]
                if isinstance(first_ring, list) and len(first_ring) > 0:
                    poly_lat = []
                    poly_lon = []
                    for coord in first_ring:
                        if isinstance(coord, list) and len(coord) >= 2:
                            try:
                                lon_val = float(coord[0])
                                lat_val = float(coord[1])
                                poly_lon.append(lon_val)
                                poly_lat.append(lat_val)
                            except:
                                pass
                    if poly_lat:
                        result_lat.append(poly_lat)
                        result_lon.append(poly_lon)
    except Exception as e:
        print(f"Failed to parse: {e}")
        return [], []
    
    return result_lon, result_lat

class TestDeg2Rad:
    """Test deg2rad function."""
    
    @pytest.mark.parametrize("deg,expected", [
        (0, 0),
        (90, math.pi/2),
        (180, math.pi),
        (360, 2*math.pi),
        (-90, -math.pi/2),
        (720, 4*math.pi),
    ])
    def test_deg2rad(self, deg, expected):
        result = deg2rad(deg)
        assert abs(result - expected) < 1e-10, f"deg2rad({deg}) = {result}, expected {expected}"

class TestLatLonToLocalXYFeet:
    """Test latlon_to_local_xy_feet function."""
    
    def test_same_point_origin(self):
        x, y = latlon_to_local_xy_feet(0, 0, 0, 0)
        assert abs(x) < 1e-6 and abs(y) < 1e-6
    
    def test_east_offset(self):
        x, y = latlon_to_local_xy_feet(1, 0, 0, 0)
        assert x > 0 and abs(y) < 1e-6
    
    def test_north_offset(self):
        x, y = latlon_to_local_xy_feet(0, 1, 0, 0)
        assert abs(x) < 1e-6 and y > 0
    
    @pytest.mark.parametrize("lon,lat,lon0,lat0", [
        (-120, 40, -120, 40),
        (100, 30, 100, 30),
        (0, 0, 0, 0),
    ])
    def test_same_point_variations(self, lon, lat, lon0, lat0):
        x, y = latlon_to_local_xy_feet(lon, lat, lon0, lat0)
        assert abs(x) < 1e-6 and abs(y) < 1e-6
    
    @pytest.mark.parametrize("lon,lat,lon0,lat0", [
        (180, 90, 0, 0),
        (-180, -90, 0, 0),
    ])
    def test_extreme_coordinates(self, lon, lat, lon0, lat0):
        x, y = latlon_to_local_xy_feet(lon, lat, lon0, lat0)
        assert not (math.isnan(x) or math.isnan(y)), f"Result should be valid numbers, got ({x}, {y})"

class TestParseMultiPolygonJson:
    """Test parse_multipolygon_json function."""
    
    @pytest.mark.parametrize("input_val", [
        None,
        "",
        "[]",
        [],
    ])
    def test_empty_inputs(self, input_val):
        lon, lat = parse_multipolygon_json_simple(input_val)
        assert len(lon) == 0 and len(lat) == 0
    
    def test_simple_polygon(self):
        simple_polygon = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        lon, lat = parse_multipolygon_json_simple(json.dumps(simple_polygon))
        assert len(lon) == 1 and len(lat) == 1
        assert len(lon[0]) == 5 and len(lat[0]) == 5
    
    def test_multipolygon(self):
        multipolygon = [
            [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
        ]
        lon, lat = parse_multipolygon_json_simple(json.dumps(multipolygon))
        assert len(lon) == 2 and len(lat) == 2
    
    @pytest.mark.parametrize("input_val", [
        "{invalid json}",
        "not a list",
        [[]],
        [[[[]]]],
        [[[[0]]]],
    ])
    def test_invalid_inputs(self, input_val):
        lon, lat = parse_multipolygon_json_simple(input_val)
        assert len(lon) == 0 and len(lat) == 0
    
    def test_3d_coords(self):
        lon, lat = parse_multipolygon_json_simple([[[[0, 0, 100]]]])
        assert len(lon) == 1 and len(lat) == 1
        assert len(lon[0]) == 1
    
    def test_polygon_with_hole(self):
        polygon_with_hole = [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]
        ]
        lon, lat = parse_multipolygon_json_simple(json.dumps(polygon_with_hole))
        assert len(lon) == 1 and len(lon[0]) == 5
    
    def test_non_numeric_coords(self):
        malformed_coord = [[[["not", "numbers"], [1, 2]]]]
        lon, lat = parse_multipolygon_json_simple(json.dumps(malformed_coord))
        assert len(lon) == 1 and len(lon[0]) == 1
    
    def test_very_large_numbers(self):
        very_large = [[[1e10, 1e10], [-1e10, -1e10]]]
        lon, lat = parse_multipolygon_json_simple(json.dumps(very_large))
        assert len(lon) == 1 and len(lon[0]) == 2
