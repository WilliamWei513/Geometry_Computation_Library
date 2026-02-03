import sys
import os
import math
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def test_edge_cases():
    """Test edge cases for LatLonJsonToFeet functions."""
    passed = 0
    failed = 0
    tests = []
    
    def assert_test(name, condition, expected=None, got=None):
        nonlocal passed, failed
        if condition:
            passed += 1
            tests.append(f"PASS: {name}")
        else:
            failed += 1
            msg = f"FAIL: {name}"
            if expected is not None:
                msg += f" | Expected: {expected}"
            if got is not None:
                msg += f" | Got: {got}"
            tests.append(msg)
            print(msg)
    
    print("=" * 60)
    print("Edge Case Tests for LatLonJsonToFeet")
    print("=" * 60)
    print()
    
    print("Testing deg2rad...")
    assert_test("deg2rad(0)", abs(deg2rad(0)) < 1e-10)
    assert_test("deg2rad(90)", abs(deg2rad(90) - math.pi/2) < 1e-10)
    assert_test("deg2rad(180)", abs(deg2rad(180) - math.pi) < 1e-10)
    assert_test("deg2rad(360)", abs(deg2rad(360) - 2*math.pi) < 1e-10)
    assert_test("deg2rad(-90)", abs(deg2rad(-90) + math.pi/2) < 1e-10)
    assert_test("deg2rad(very large)", abs(deg2rad(720) - 4*math.pi) < 1e-10)
    print()
    
    print("Testing latlon_to_local_xy_feet...")
    x, y = latlon_to_local_xy_feet(0, 0, 0, 0)
    assert_test("Same point (0,0,0,0)", abs(x) < 1e-6 and abs(y) < 1e-6, 0, (x, y))
    
    x, y = latlon_to_local_xy_feet(1, 0, 0, 0)
    assert_test("1 deg east from origin", x > 0 and abs(y) < 1e-6, "x>0, y≈0", (x, y))
    
    x, y = latlon_to_local_xy_feet(0, 1, 0, 0)
    assert_test("1 deg north from origin", abs(x) < 1e-6 and y > 0, "x≈0, y>0", (x, y))
    
    x1, y1 = latlon_to_local_xy_feet(-120, 40, -120, 40)
    assert_test("Same point (-120,40)", abs(x1) < 1e-6 and abs(y1) < 1e-6, 0, (x1, y1))
    
    x, y = latlon_to_local_xy_feet(180, 90, 0, 0)
    assert_test("Extreme coordinates (180,90)", not (math.isnan(x) or math.isnan(y)), "valid numbers", (x, y))
    
    x, y = latlon_to_local_xy_feet(-180, -90, 0, 0)
    assert_test("Extreme negative (-180,-90)", not (math.isnan(x) or math.isnan(y)), "valid numbers", (x, y))
    print()
    
    print("Testing parse_multipolygon_json...")
    
    lon, lat = parse_multipolygon_json_simple(None)
    assert_test("None input", len(lon) == 0 and len(lat) == 0, ([], []), (lon, lat))
    
    lon, lat = parse_multipolygon_json_simple("")
    assert_test("Empty string", len(lon) == 0 and len(lat) == 0, ([], []), (lon, lat))
    
    lon, lat = parse_multipolygon_json_simple("[]")
    assert_test("Empty array", len(lon) == 0 and len(lat) == 0, ([], []), (lon, lat))
    
    simple_polygon = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
    lon, lat = parse_multipolygon_json_simple(json.dumps(simple_polygon))
    assert_test("Simple Polygon", len(lon) == 1 and len(lat) == 1 and len(lon[0]) == 5, 
              "1 polygon, 5 coords", (len(lon), len(lon[0]) if lon else None))
    
    multipolygon = [
        [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
    ]
    lon, lat = parse_multipolygon_json_simple(json.dumps(multipolygon))
    assert_test("MultiPolygon", len(lon) == 2 and len(lat) == 2, "2 polygons", len(lon))
    
    invalid_json = "{invalid json}"
    lon, lat = parse_multipolygon_json_simple(invalid_json)
    assert_test("Invalid JSON", len(lon) == 0 and len(lat) == 0, ([], []), (lon, lat))
    
    lon, lat = parse_multipolygon_json_simple("not a list")
    assert_test("Not a list", len(lon) == 0 and len(lat) == 0, ([], []), (lon, lat))
    
    lon, lat = parse_multipolygon_json_simple([[]])
    assert_test("Empty polygon", len(lon) == 0 and len(lat) == 0, ([], []), (lon, lat))
    
    lon, lat = parse_multipolygon_json_simple([[[[]]]])
    assert_test("Empty ring", len(lon) == 0 and len(lat) == 0, ([], []), (lon, lat))
    
    lon, lat = parse_multipolygon_json_simple([[[[0]]]])
    assert_test("Insufficient coords ([0])", len(lon) == 0 and len(lat) == 0, ([], []), (lon, lat))
    
    lon, lat = parse_multipolygon_json_simple([[[[0, 0, 100]]]])
    assert_test("3D coords", len(lon) == 1 and len(lat) == 1 and len(lon[0]) == 1, 
              "1 coord", len(lon[0]) if lon else None)
    
    polygon_with_hole = [
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]
    ]
    lon, lat = parse_multipolygon_json_simple(json.dumps(polygon_with_hole))
    assert_test("Polygon with hole (only first ring)", len(lon) == 1 and len(lon[0]) == 5, 
              "1 polygon, 5 coords", len(lon[0]) if lon else None)
    
    malformed_coord = [[[["not", "numbers"], [1, 2]]]]
    lon, lat = parse_multipolygon_json_simple(json.dumps(malformed_coord))
    assert_test("Non-numeric coords", len(lon) == 1 and len(lon[0]) == 1, 
              "1 valid coord", len(lon[0]) if lon else None)
    
    very_large = [[[1e10, 1e10], [-1e10, -1e10]]]
    lon, lat = parse_multipolygon_json_simple(json.dumps(very_large))
    assert_test("Very large numbers", len(lon) == 1 and len(lon[0]) == 2, 
              "2 coords", len(lon[0]) if lon else None)
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\nFailed tests:")
        for test in tests:
            if test.startswith("FAIL"):
                print(f"  {test}")
    return failed == 0

if __name__ == '__main__':
    success = test_edge_cases()
    sys.exit(0 if success else 1)
