import pytest
import math
import json

class MockDouble:
    def __init__(self, val):
        self.Value = float(val)
    def __float__(self):
        return self.Value

class MockPath:
    def __init__(self, *indices):
        self.Indices = tuple(indices)
    def __hash__(self):
        return hash(self.Indices)
    def __eq__(self, other):
        return isinstance(other, MockPath) and self.Indices == other.Indices

class MockDataTree:
    def __init__(self):
        self._branches = {}
    def EnsurePath(self, path):
        if path not in self._branches:
            self._branches[path] = []
    def Add(self, item, path):
        self.EnsurePath(path)
        self._branches[path].append(item)
    @property
    def Paths(self):
        return list(self._branches.keys())
    def Branch(self, path):
        return self._branches.get(path, [])
    def __contains__(self, path):
        return path in self._branches

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
    
    def test_coordinate_order(self):
        polygon = [[[100, 50], [101, 50], [101, 51], [100, 51], [100, 50]]]
        lon, lat = parse_multipolygon_json_simple(json.dumps(polygon))
        assert len(lon) == 1 and len(lat) == 1
        assert lon[0][0] == 100
        assert lat[0][0] == 50
    
    def test_negative_coordinates(self):
        polygon = [[[-120, -40], [-119, -40], [-119, -39], [-120, -39], [-120, -40]]]
        lon, lat = parse_multipolygon_json_simple(json.dumps(polygon))
        assert len(lon) == 1 and len(lat) == 1
        assert lon[0][0] == -120
        assert lat[0][0] == -40
    
    def test_single_coordinate_polygon(self):
        polygon = [[[0, 0]]]
        lon, lat = parse_multipolygon_json_simple(json.dumps(polygon))
        assert len(lon) == 1 and len(lat) == 1
        assert len(lon[0]) == 1
    
    def test_mixed_valid_invalid_coords(self):
        polygon = [[[0, 0], ["invalid", "data"], [1, 1], [None, None], [2, 2]]]
        lon, lat = parse_multipolygon_json_simple(json.dumps(polygon))
        assert len(lon) == 1
        assert len(lon[0]) == 3

class TestParseMultiPolygonJsonWithDataTree:
    """Test parse_multipolygon_json returning DataTree structure."""
    
    def test_parse_returns_datatree_structure(self):
        simple_polygon = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        lon_tree = MockDataTree()
        lat_tree = MockDataTree()
        
        parsed = json.loads(json.dumps(simple_polygon))
        for poly_idx, polygon in enumerate([parsed]):
            if isinstance(polygon, list) and len(polygon) > 0:
                first_ring = polygon[0]
                if isinstance(first_ring, list) and len(first_ring) > 0:
                    path = MockPath(poly_idx)
                    lon_tree.EnsurePath(path)
                    lat_tree.EnsurePath(path)
                    for coord in first_ring:
                        if isinstance(coord, list) and len(coord) >= 2:
                            lon_val = float(coord[0])
                            lat_val = float(coord[1])
                            lon_tree.Add(MockDouble(lon_val), path)
                            lat_tree.Add(MockDouble(lat_val), path)
        
        assert len(list(lon_tree.Paths)) == 1
        assert len(list(lat_tree.Paths)) == 1
        assert len(list(lon_tree.Branch(MockPath(0)))) == 5
        assert len(list(lat_tree.Branch(MockPath(0)))) == 5
    
    def test_parse_multipolygon_datatree_structure(self):
        multipolygon = [
            [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
        ]
        lon_tree = MockDataTree()
        lat_tree = MockDataTree()
        
        parsed = json.dumps(multipolygon)
        parsed = json.loads(parsed)
        for poly_idx, polygon in enumerate(parsed):
            if isinstance(polygon, list) and len(polygon) > 0:
                first_ring = polygon[0]
                if isinstance(first_ring, list) and len(first_ring) > 0:
                    path = MockPath(poly_idx)
                    lon_tree.EnsurePath(path)
                    lat_tree.EnsurePath(path)
                    for coord in first_ring:
                        if isinstance(coord, list) and len(coord) >= 2:
                            lon_val = float(coord[0])
                            lat_val = float(coord[1])
                            lon_tree.Add(MockDouble(lon_val), path)
                            lat_tree.Add(MockDouble(lat_val), path)
        
        assert len(list(lon_tree.Paths)) == 2
        assert len(list(lat_tree.Paths)) == 2

class TestOriginBroadcasting:
    """Test optional_origin_lonlat broadcasting functionality."""
    
    def test_origin_broadcast_to_all_polygons(self):
        lon_tree = MockDataTree()
        lat_tree = MockDataTree()
        origin_lon_tree = MockDataTree()
        origin_lat_tree = MockDataTree()
        
        path0 = MockPath(0)
        path1 = MockPath(1)
        
        lon_tree.EnsurePath(path0)
        lat_tree.EnsurePath(path0)
        lon_tree.EnsurePath(path1)
        lat_tree.EnsurePath(path1)
        
        for lon_val, lat_val in zip([0, 1], [0, 1]):
            lon_tree.Add(MockDouble(lon_val), path0)
            lat_tree.Add(MockDouble(lat_val), path0)
        
        for lon_val, lat_val in zip([2, 3], [2, 3]):
            lon_tree.Add(MockDouble(lon_val), path1)
            lat_tree.Add(MockDouble(lat_val), path1)
        
        optional_origin_lonlat = MockDataTree()
        origin_path = MockPath(0)
        optional_origin_lonlat.EnsurePath(origin_path)
        optional_origin_lonlat.Add(MockDouble(-73.98867), origin_path)
        optional_origin_lonlat.Add(MockDouble(40.702559), origin_path)
        
        branch = list(optional_origin_lonlat.Branch(origin_path))
        if len(branch) >= 2:
            origin_lon_val = float(branch[0].Value)
            origin_lat_val = float(branch[1].Value)
            
            for path in lon_tree.Paths:
                origin_lon_tree.EnsurePath(path)
                origin_lat_tree.EnsurePath(path)
                origin_lon_tree.Add(MockDouble(origin_lon_val), path)
                origin_lat_tree.Add(MockDouble(origin_lat_val), path)
        
        assert len(list(origin_lon_tree.Paths)) == 2
        assert len(list(origin_lat_tree.Paths)) == 2
        
        for path in [path0, path1]:
            lon_branch = list(origin_lon_tree.Branch(path))
            lat_branch = list(origin_lat_tree.Branch(path))
            assert len(lon_branch) == 1
            assert len(lat_branch) == 1
            assert abs(float(lon_branch[0].Value) - (-73.98867)) < 1e-6
            assert abs(float(lat_branch[0].Value) - 40.702559) < 1e-6
