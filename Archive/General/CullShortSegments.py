from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import Grasshopper as gh
import Rhino.Geometry as rg
import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System
import System.Collections.Generic as scg

def coerce_to_curve(obj):
    try:
        if isinstance(obj, rg.Curve):
            return obj
        if isinstance(obj, rg.Polyline):
            return rg.PolylineCurve(obj)
        if isinstance(obj, rg.Line):
            return rg.LineCurve(obj)
        if hasattr(obj, 'ToNurbsCurve'):
            c = obj.ToNurbsCurve()
            if c:
                return c
    except:
        pass
    return None

def data_tree_manager(processed_tree, initial_tree):
    new_tree = DataTree[object]()

    try:
        initial_paths = list(initial_tree.Paths)
    except:
        initial_paths = []

    for ip in initial_paths:
        try:
            new_tree.EnsurePath(ip)
        except:
            pass

    try:
        processed_paths = list(processed_tree.Paths) if (processed_tree is not None and processed_tree.DataCount > 0) else []
    except:
        processed_paths = []

    if not processed_paths:
        return new_tree

    try:
        def path_indices_tuple(p):
            try:
                return tuple(p.Indices)
            except:
                return tuple()

        initial_prefixes = [(ip, path_indices_tuple(ip)) for ip in initial_paths]

        for pp in processed_paths:
            try:
                pp_items = list(processed_tree.Branch(pp))
            except:
                pp_items = []
            if not pp_items:
                continue

            pp_idx = path_indices_tuple(pp)

            matched_path = None
            for ip, ip_idx in initial_prefixes:
                try:
                    if len(ip_idx) <= len(pp_idx) and tuple(pp_idx[:len(ip_idx)]) == ip_idx:
                        matched_path = ip
                        break
                except:
                    pass

            if matched_path is not None:
                for it in pp_items:
                    try:
                        new_tree.Add(it, matched_path)
                    except:
                        pass
    except Exception as e:
        print("data_tree_manager failed: {}".format(e))

    return new_tree

def offset_by_distances(segments_tree, distances_tree, side=1, tolerance=None, corner_style=rg.CurveOffsetCornerStyle.Sharp):

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.01

    result_tree = DataTree[rg.Curve]()

    if segments_tree is None:
        return result_tree

    # Detect global scalar distance (single branch with single value)
    global_distance = None
    try:
        if distances_tree is not None and len(list(distances_tree.Paths)) == 1:
            p0 = list(distances_tree.Paths)[0]
            vals = list(distances_tree.Branch(p0))
            if len(vals) == 1:
                try:
                    global_distance = float(vals[0])
                except:
                    global_distance = None
    except:
        pass

    for path in segments_tree.Paths:
        result_tree.EnsurePath(path)
        segs = list(segments_tree.Branch(path))
        if not segs:
            continue

        # Resolve distances for this path
        if global_distance is not None:
            dists = [global_distance] * len(segs)
        else:
            try:
                d_branch = list(distances_tree.Branch(path)) if (distances_tree is not None and path in distances_tree.Paths) else None
            except:
                d_branch = None
            if d_branch is None:
                # Try fallback to GH_Path(0) single value
                try:
                    d_branch = list(distances_tree.Branch(GH_Path(0))) if distances_tree is not None else None
                except:
                    d_branch = None
            if d_branch is None or len(d_branch) == 0:
                dists = [0.0] * len(segs)
            elif len(d_branch) == 1 and len(segs) > 1:
                try:
                    v = float(d_branch[0])
                except:
                    v = 0.0
                dists = [v] * len(segs)
            else:
                # One-to-one, clamp to min length
                dists = []
                for i in range(len(segs)):
                    try:
                        dists.append(float(d_branch[i]))
                    except:
                        dists.append(0.0)

        # Offset each segment
        for i, seg in enumerate(segs):
            try:
                crv = coerce_to_curve(seg)
                if crv is None or not hasattr(crv, 'IsValid') or not crv.IsValid:
                    continue
                # If offset distance is zero, return original curve
                try:
                    dist_val = float(dists[i])
                except:
                    dist_val = 0.0
                if abs(dist_val) <= 0.0:
                    try:
                        orig = crv.DuplicateCurve() if hasattr(crv, 'DuplicateCurve') else crv
                        if orig is not None and orig.IsValid:
                            result_tree.Add(orig, path)
                        continue
                    except:
                        pass
                d = dist_val * float(side)
                off = crv.Offset(rg.Plane.WorldXY, d, tolerance, corner_style)
                if off and len(off) > 0 and off[0] is not None and off[0].IsValid:
                    result_tree.Add(off[0], path)
            except:
                pass

    return result_tree

def shift_list_tree(tree, shift, wrap=True):

    new_tree = DataTree[object]()

    for path in tree.Paths:
        items = list(tree.Branch(path))
        n = len(items)
        if n == 0:
            continue

        if wrap:
            shift_mod = shift % n
            shifted = items[-shift_mod:] + items[:-shift_mod]
        else:
            shifted = [None] * n
            for i in range(n):
                j = i - shift
                if 0 <= j < n:
                    shifted[i] = items[j]

        new_tree.AddRange(shifted, path)

    return new_tree

def graft_tree(tree):

    new_tree = DataTree[object]()

    for path in tree.Paths:
        items = list(tree.Branch(path))
        for i, item in enumerate(items):
            new_path = GH_Path(path)   
            new_path = new_path.AppendElement(i)  
            new_tree.Add(item, new_path)

    return new_tree

def merge_trees(*trees):

    new_tree = DataTree[object]()
    if not trees:
        return new_tree

    all_paths = set()
    for t in trees:
        for p in t.Paths:
            all_paths.add(p)

    def path_key(p):
        return tuple(p.Indices)

    for path in sorted(all_paths, key=path_key):
        merged_items = []
        for t in trees:
            if path in t.Paths:
                merged_items.extend(list(t.Branch(path)))
        new_tree.AddRange(merged_items, path)

    return new_tree

def curve_curve_intersection_tree(treeA, treeB):

    new_tree = DataTree[object]()

    for path in treeA.Paths:
        curvesA = list(treeA.Branch(path))
        curvesB = list(treeB.Branch(path))

        n = min(len(curvesA), len(curvesB))

        results = []
        for i in range(n):
            cA = curvesA[i]
            cB = curvesB[i]
            
            events = rg.Intersect.Intersection.CurveCurve(cA, cB, 0.001, 0.001)
            results.append(events is not None and events.Count > 0)

        new_tree.AddRange(results, path)

    return new_tree

def dispatch_tree(treeData, treeMask):
   
    treeA = DataTree[object]()
    treeB = DataTree[object]()

    for path in treeData.Paths:
        data_items = list(treeData.Branch(path))
        mask_items = list(treeMask.Branch(path)) if path in treeMask.Paths else []

        mask_value = mask_items[0] if len(mask_items) > 0 else False

        if mask_value:
            treeA.AddRange(data_items, path)
            treeB.EnsurePath(path)  
        else:
            treeB.AddRange(data_items, path)
            treeA.EnsurePath(path) 

    return treeA, treeB

def extract_first_second_items(tree):

    first_line = DataTree[object]()
    second_line = DataTree[object]()
    
    for path in tree.Paths:
        items = list(tree.Branch(path))
        
        first_line.EnsurePath(path)
        second_line.EnsurePath(path)
        
        if len(items) >= 1:
            first_line.Add(items[0], path)
        
        if len(items) >= 2:
            second_line.Add(items[1], path)
    
    return first_line, second_line

def extract_start_end_points(tree):

    start_points = DataTree[rg.Point3d]()
    end_points = DataTree[rg.Point3d]()
    
    for path in tree.Paths:
        start_points.EnsurePath(path)
        end_points.EnsurePath(path)
        
        items = list(tree.Branch(path))
        
        for item in items:
            try:
                if hasattr(item, 'PointAtStart') and hasattr(item, 'PointAtEnd'):
                    start_pt = item.PointAtStart
                    end_pt = item.PointAtEnd
                    
                    start_points.Add(start_pt, path)
                    end_points.Add(end_pt, path)
                else:
                    pass
                    
            except:
                pass
    
    return start_points, end_points

def extend_lines(tree, extension_length=10000):

    extended_tree = DataTree[object]()
    
    for path in tree.Paths:

        extended_tree.EnsurePath(path)
        
        items = list(tree.Branch(path))
        
        for item in items:
            try:
                if hasattr(item, 'PointAtStart') and hasattr(item, 'PointAtEnd'):

                    start_pt = item.PointAtStart
                    end_pt = item.PointAtEnd
                    
                    direction = end_pt - start_pt
                    if direction.Length > 0:
                        direction.Unitize()
                        
                        new_start = start_pt - direction * extension_length
                        
                        new_end = end_pt + direction * extension_length
                        
                        extended_line = rg.Line(new_start, new_end)
                        extended_tree.Add(extended_line, path)
                    else:
                        extended_tree.Add(item, path)
                else:
                    extended_tree.Add(item, path)
                    
            except:
                extended_tree.Add(item, path)
    
    return extended_tree

def line_line_intersection_points(treeA, treeB, tolerance=0.001):

    intersection_points = DataTree[rg.Point3d]()
    
    for path in treeA.Paths:

        intersection_points.EnsurePath(path)
        
        linesA = list(treeA.Branch(path))
        linesB = list(treeB.Branch(path)) if path in treeB.Paths else []
        
        min_count = min(len(linesA), len(linesB))
        for i in range(min_count):
            try:
                lineA = linesA[i]
                lineB = linesB[i]
                
                if hasattr(lineA, 'PointAtStart') and hasattr(lineA, 'PointAtEnd'):
                    lineA_obj = rg.Line(lineA.PointAtStart, lineA.PointAtEnd)
                else:
                    lineA_obj = lineA
                    
                if hasattr(lineB, 'PointAtStart') and hasattr(lineB, 'PointAtEnd'):
                    lineB_obj = rg.Line(lineB.PointAtStart, lineB.PointAtEnd)
                else:
                    lineB_obj = lineB
                
                success, tA, tB = rg.Intersect.Intersection.LineLine(lineA_obj, lineB_obj)
                
                if success:

                    if 0 <= tA <= 1 and 0 <= tB <= 1:
                        intersection_pt = lineA_obj.PointAt(tA)
                        intersection_points.Add(intersection_pt, path)

                    else:
                        intersection_pt = lineA_obj.PointAt(tA)
                        intersection_points.Add(intersection_pt, path)
                        
            except Exception as e:

                print(f"Line intersection failed for path {path}, index {i}: {e}")
                pass
    
    return intersection_points

def two_pt_line(start_pts, end_pts):
    lines_tree = DataTree[rg.Line]()
    
    for path in start_pts.Paths:

        lines_tree.EnsurePath(path)
        
        start_points = list(start_pts.Branch(path))
        end_points = list(end_pts.Branch(path)) if path in end_pts.Paths else []
        
        min_count = min(len(start_points), len(end_points))
        for i in range(min_count):
            try:
                start_pt = start_points[i]
                end_pt = end_points[i]
                
                line = rg.Line(start_pt, end_pt)
                lines_tree.Add(line, path)
                
            except Exception as e:
                print(f"Line creation failed for path {path}, index {i}: {e}")
                pass
    
    return lines_tree

def points_to_closed_polyline(points_tree):

    polylines_tree = DataTree[rg.Polyline]()
    
    for path in points_tree.Paths:

        polylines_tree.EnsurePath(path)
        
        points = list(points_tree.Branch(path))
        
        if len(points) >= 3:  
            try:
                
                polyline = rg.Polyline(points)
                
                if not polyline.IsClosed:
                    polyline.Add(polyline[0])  
                
                polylines_tree.Add(polyline, path)
                
            except Exception as e:
                print(f"Polyline creation failed for path {path}: {e}")
                pass
        elif len(points) == 2:

            try:
                polyline = rg.Polyline([points[0], points[1], points[0]])
                polylines_tree.Add(polyline, path)
            except Exception as e:
                print(f"Line polyline creation failed for path {path}: {e}")
                pass
    
    return polylines_tree

def remove_small_fins(polylines_tree, tol=1e-6, min_ratio=0.15):

    def _remove_small_fins_one(crv, tol, min_ratio):
        events = rg.Intersect.Intersection.CurveSelf(crv, tol)
        if not events:
            return [crv]

        params = sorted(set([e.ParameterA for e in events] + [e.ParameterB for e in events]))
        segments = crv.Split(params)
        if not segments:
            return [crv]

        clean_segments = []
        crvlen = crv.GetLength()
        for seg in segments:
            seglen = seg.GetLength()
            ratio = seglen / crvlen if crvlen > 0 else 0
            if ratio > min_ratio:
                clean_segments.append(seg)

        joined = rg.Curve.JoinCurves(clean_segments, tol)
        if not joined or len(joined) == 0:
            return [crv]
        return list(joined)

    cleaned_tree = DataTree[rg.Curve]()

    for path in polylines_tree.Paths:
        cleaned_tree.EnsurePath(path)
        for polyline in polylines_tree.Branch(path):
            if not polyline.IsValid or polyline.Count < 2:
                continue

            if not polyline.IsClosed:
                polyline.Add(polyline[0])

            curve = rg.PolylineCurve(polyline)
            cleaned_list = _remove_small_fins_one(curve, tol=tol, min_ratio=min_ratio)

            for c in cleaned_list:
                if c is not None and c.IsValid:
                    cleaned_tree.Add(c, path)

    return cleaned_tree

def clean_tree(data_tree, remove_null=True, remove_invalid=True, remove_empty=True):
    cleaned_tree = DataTree[object]()

    for path in data_tree.Paths:
        items = list(data_tree.Branch(path))
        cleaned_items = []

        for item in items:
            if item is None and remove_null:
                continue

            if remove_invalid:
                if isinstance(item, rg.GeometryBase) and not item.IsValid:
                    continue
                if isinstance(item, rg.Curve) and not item.IsValid:
                    continue
                if isinstance(item, rg.Surface) and not item.IsValid:
                    continue
                if isinstance(item, rg.Brep) and not item.IsValid:
                    continue
                if isinstance(item, rg.Mesh) and not item.IsValid:
                    continue

            cleaned_items.append(item)

        if not (remove_empty and len(cleaned_items) == 0):
            for c in cleaned_items:
                cleaned_tree.Add(c, path)

    return cleaned_tree

def explode_curves(curve_tree, recursive=True, tol=1e-6):
    segments_tree = DataTree[object]()
    vertices_tree = DataTree[object]()

    for path in curve_tree.Paths:
        # Ensure paths exist in output trees
        segments_tree.EnsurePath(path)
        vertices_tree.EnsurePath(path)
        
        curves = list(curve_tree.Branch(path))
        all_vertices = []  # Collect all vertices first, then deduplicate
        
        for crv in curves:
            # Check if it's a valid curve or polyline
            if not crv or not hasattr(crv, 'IsValid') or not crv.IsValid:
                print(f"Skipping invalid curve in path {path}")
                continue
            
            # Convert polyline to curve if needed
            if isinstance(crv, rg.Polyline):
                try:
                    crv = rg.PolylineCurve(crv)
                except:
                    print(f"Failed to convert polyline to curve in path {path}")
                    continue

            pieces = crv.DuplicateSegments() if recursive else [crv]

            for piece in pieces:
                if not piece.IsValid:
                    continue

                segments_tree.Add(piece, path)
            
                try:
                    # Try different methods to get vertices
                    vertices_added = False
                    
                    # Method 1: Try ToPolyline with minimal parameters
                    try:
                        poly = piece.ToPolyline()
                        for i, pt in enumerate(poly):
                            # Skip last point if curve is closed and it's the same as first point
                            if i == len(poly) - 1 and piece.IsClosed and pt.EpsilonEquals(poly[0], tol):
                                continue
                            all_vertices.append(pt)
                        vertices_added = True
                    except:
                        pass
                    
                    # Method 2: If ToPolyline fails, try getting start and end points
                    if not vertices_added:
                        try:
                            start_pt = piece.PointAtStart
                            end_pt = piece.PointAtEnd
                            all_vertices.extend([start_pt, end_pt])
                            vertices_added = True
                        except:
                            pass
                    
                    # Method 3: If still no vertices, try sampling the curve
                    if not vertices_added:
                        try:
                            # Sample curve at regular intervals
                            domain = piece.Domain
                            num_samples = max(2, int(piece.GetLength() / (tol * 10)))
                            for i in range(num_samples + 1):
                                t = domain.T0 + (domain.T1 - domain.T0) * i / num_samples
                                pt = piece.PointAt(t)
                                all_vertices.append(pt)
                        except:
                            pass
                        
                except Exception as e:
                    print(f"Error processing curve in path {path}: {e}")
                    pass

        # Deduplicate vertices
        unique_vertices = []
        for pt in all_vertices:
            is_duplicate = False
            for existing_pt in unique_vertices:
                if pt.EpsilonEquals(existing_pt, tol):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_vertices.append(pt)
        
        # Add unique vertices to output tree
        for vertex in unique_vertices:
            vertices_tree.Add(vertex, path)

    return segments_tree, vertices_tree

def remove_sharp_corner(polylines_tree, extend_len=500, area_ratio=0.05):

    def to_curve(obj):
        if isinstance(obj, Rhino.Geometry.Curve):
            return obj
        elif isinstance(obj, System.Guid):
            return rs.coercecurve(obj)
        return None

    def remove_fins_by_extension(curve, extend_len, area_ratio):
        crv = to_curve(curve)
        if not crv:
            return None
        
        if isinstance(crv, Rhino.Geometry.PolylineCurve):
            success, polyline = crv.TryGetPolyline()
            if not success:
                return None
        else:
            polyline = crv.ToPolyline(0,0,0,0,0,0,True)  
        
        if not polyline.IsClosed:
            polyline.Add(polyline[0])
        
        base_brep_list = Rhino.Geometry.Brep.CreatePlanarBreps(
            [Rhino.Geometry.PolylineCurve(polyline)], 
            sc.doc.ModelAbsoluteTolerance
        )
        if not base_brep_list:
            return None
        base_brep = base_brep_list[0]
        
        lines = []
        for i in range(len(polyline)-1):
            seg = Rhino.Geometry.Line(polyline[i], polyline[i+1])
            seg.Extend(extend_len, extend_len)
            lines.append(seg.ToNurbsCurve())
        
        split_breps = [base_brep]
        for ln in lines:
            temp = []
            cutter = scg.List[Rhino.Geometry.Curve]([ln]) 
            for b in split_breps:
                pieces = b.Split(cutter, sc.doc.ModelAbsoluteTolerance)
                if pieces:
                    temp.extend(pieces)
                else:
                    temp.append(b)
            split_breps = temp
        
        total_area = sum(b.GetArea() for b in split_breps)
        big_faces = [b for b in split_breps if b.GetArea() >= total_area * area_ratio]
        
        result_crvs = []
        for b in big_faces:
            edges = b.DuplicateNakedEdgeCurves(True, False)
            joined = Rhino.Geometry.Curve.JoinCurves(edges, sc.doc.ModelAbsoluteTolerance)
            if joined:
                result_crvs.extend(joined)
        
        return result_crvs

    cleaned_tree = DataTree[object]()
    
    for path in polylines_tree.Paths:

        cleaned_tree.EnsurePath(path)
        
        polylines = list(polylines_tree.Branch(path))
        
        for polyline in polylines:
            try:
                if isinstance(polyline, rg.Polyline):
                    curve = rg.PolylineCurve(polyline)
                else:
                    curve = polyline
                
                cleaned_curves = remove_fins_by_extension(curve, extend_len, area_ratio)
                
                if cleaned_curves:
                    for cleaned_curve in cleaned_curves:
                        cleaned_tree.Add(cleaned_curve, path)
                        
            except Exception as e:
                print(f"Fin removal failed for path {path}: {e}")
                pass
    
    return cleaned_tree

def boundary_surface(curves_tree, tolerance=None):

    if tolerance is None:
        tolerance = sc.doc.ModelAbsoluteTolerance
    
    surfaces_tree = DataTree[rg.Brep]()
    
    for path in curves_tree.Paths:

        surfaces_tree.EnsurePath(path)
        curves = list(curves_tree.Branch(path))
        
        if not curves:
            continue
            
        try:

            valid_curves = []
            for curve in curves:
                if curve is not None and hasattr(curve, 'IsValid') and curve.IsValid:
                    valid_curves.append(curve)
            
            if not valid_curves:
                continue
            
            breps = rg.Brep.CreatePlanarBreps(valid_curves, tolerance)
            
            if breps and len(breps) > 0:
                for brep in breps:
                    if brep is not None and brep.IsValid:
                        surfaces_tree.Add(brep, path)
                        
        except Exception as e:
            print(f"Boundary surface creation failed for path {path}: {e}")
            pass
    
    return surfaces_tree

def join_breps(surfaces_tree, tolerance=None):

    if tolerance is None:
        tolerance = sc.doc.ModelAbsoluteTolerance

    joined_tree = DataTree[rg.Brep]()

    for path in surfaces_tree.Paths:
        breps = list(surfaces_tree.Branch(path))
        if not breps:
            continue
        try:

            valid_breps = [b for b in breps if b is not None and b.IsValid]
            if not valid_breps:
                continue

            joined_breps = rg.Brep.JoinBreps(valid_breps, tolerance)
            if joined_breps:
                for jb in joined_breps:
                    if jb is not None and jb.IsValid:
                        joined_tree.Add(jb, path)
        except Exception as e:
            print(f"Brep join failed for path {path}: {e}")
            pass

    return joined_tree

def merge_coplanar_faces(surfaces_tree, tolerance=None):

    if tolerance is None:
        tolerance = sc.doc.ModelAbsoluteTolerance

    merged_tree = DataTree[rg.Brep]()

    for path in surfaces_tree.Paths:
        breps = list(surfaces_tree.Branch(path))
        
        merged_breps = []
        for brep in breps:
            if brep is not None and brep.IsValid:
                try:
                    merged_brep = brep.DuplicateBrep()
                    merged_brep.MergeCoplanarFaces(tolerance)
                    if merged_brep is not None and merged_brep.IsValid:
                        merged_breps.append(merged_brep)
                    else:
                        merged_breps.append(brep)
                except Exception as e:
                    print(f"MergeCoplanarFaces failed for path {path}: {e}")
                    merged_breps.append(brep)
            else:
                merged_breps.append(brep)
        for merged in merged_breps:
            merged_tree.Add(merged, path)

    return merged_tree

def deconstruct_brep(brep_tree):

    faces_tree = DataTree[rg.BrepFace]()
    edges_tree = DataTree[rg.BrepEdge]()
    vertices_tree = DataTree[rg.BrepVertex]()

    for path in brep_tree.Paths:
        breps = list(brep_tree.Branch(path))
        for brep in breps:
            if brep is not None and brep.IsValid:
                
                for face in brep.Faces:
                    faces_tree.Add(face, path)
                
                for edge in brep.Edges:
                    edges_tree.Add(edge, path)
                
                for vertex in brep.Vertices:
                    vertices_tree.Add(vertex, path)
            else:
                pass

    return faces_tree, edges_tree, vertices_tree

def join_curves(curves_tree, tolerance=1e-6):

    joined_tree = DataTree[rg.Curve]()
    for path in curves_tree.Paths:
        raw_items = list(curves_tree.Branch(path))
        curves = []
        for item in raw_items:
            try:
                c = coerce_to_curve(item)
                if c is not None and hasattr(c, 'IsValid') and c.IsValid:
                    curves.append(c)
            except:
                pass
        if curves:
            try:
                joined_curves = rg.Curve.JoinCurves(curves, tolerance)
                if joined_curves:
                    for jc in joined_curves:
                        joined_tree.Add(jc, path)
                else:
                    # If nothing was joined, add coerced curves back
                    for c in curves:
                        if c is not None and c.IsValid:
                            joined_tree.Add(c, path)
            except Exception as e:
                print(f"JoinCurves failed for path {path}: {e}")
                for c in curves:
                    if c is not None and hasattr(c, 'IsValid') and c.IsValid:
                        joined_tree.Add(c, path)
        else:
            # If no valid curves, add None to preserve structure
            joined_tree.Add(None, path)
    return joined_tree

def is_point_inside_region(point, region_curve, plane):

    try:
        projected_point = plane.ClosestPoint(point)
        
        ray_direction = rg.Vector3d(1, 0, 0)
        ray = rg.Ray3d(projected_point, ray_direction)
        
        intersection_events = rg.Intersect.Intersection.CurveRay(region_curve, ray)
        
        if intersection_events and intersection_events.Count > 0:
            return intersection_events.Count % 2 == 1
        
        return False
        
    except:
        return False

def trim_with_region(curves_tree, regions_tree):

    plane = rg.Plane.WorldXY
    
    inside_tree = DataTree[object]()
    outside_tree = DataTree[object]()
    
    for path in curves_tree.Paths:
        
        inside_tree.EnsurePath(path)
        outside_tree.EnsurePath(path)
        
        curves = list(curves_tree.Branch(path))
        regions = list(regions_tree.Branch(path)) if path in regions_tree.Paths else []
        
        if not regions:
            
            for curve in curves:
                if curve is not None and hasattr(curve, 'IsValid') and curve.IsValid:
                    outside_tree.Add(curve, path)
            continue
        
        for curve in curves:
            curve_obj = coerce_to_curve(curve)
            if curve_obj is None or not hasattr(curve_obj, 'IsValid') or not curve_obj.IsValid:
                continue
                
            try:
                projected_curve = curve_obj
                try:
                    if not curve_obj.IsInPlane(plane, sc.doc.ModelAbsoluteTolerance):
                        projected_curve = rg.Curve.ProjectToPlane(curve_obj, plane)
                        if not projected_curve or not projected_curve.IsValid:
                            projected_curve = curve_obj
                except:
                    projected_curve = curve_obj
                
                valid_regions = []
                for region in regions:
                    if region is not None and hasattr(region, 'IsValid') and region.IsValid:
                        region_curve = coerce_to_curve(region)
                        if region_curve and region_curve.IsValid:
                            valid_regions.append(region_curve)
                
                if not valid_regions:
                    outside_tree.Add(curve_obj, path)
                    continue
                
                inside_curves = []
                outside_curves = []
                
                try:
                    for region_curve in valid_regions:
                        if not region_curve.IsClosed:
                            region_curve = region_curve.ToNurbsCurve()
                            if region_curve:
                                region_curve.MakeClosed(sc.doc.ModelAbsoluteTolerance)
                        
                        intersection_events = rg.Intersect.Intersection.CurveCurve(
                            projected_curve, region_curve, sc.doc.ModelAbsoluteTolerance, sc.doc.ModelAbsoluteTolerance
                        )
                        
                        if intersection_events and intersection_events.Count > 0:
                            params = []
                            for i in range(intersection_events.Count):
                                params.append(intersection_events[i].ParameterA)
                            
                            params = sorted(set(params))
                            
                            split_curves = projected_curve.Split(params)
                            
                            if split_curves:
                                for split_curve in split_curves:
                                    if split_curve and split_curve.IsValid:
                                        mid_param = split_curve.Domain.Mid
                                        mid_point = split_curve.PointAt(mid_param)
                                        
                                        if is_point_inside_region(mid_point, region_curve, plane):
                                            inside_curves.append(split_curve)
                                        else:
                                            outside_curves.append(split_curve)
                        else:
                            mid_param = projected_curve.Domain.Mid
                            mid_point = projected_curve.PointAt(mid_param)
                            
                            if is_point_inside_region(mid_point, region_curve, plane):
                                inside_curves.append(projected_curve)
                            else:
                                outside_curves.append(projected_curve)
                
                except Exception as e:
                    print(f"Region trimming failed for path {path}: {e}")
                    outside_curves = [curve_obj]
                
                for inside_curve in inside_curves:
                    if inside_curve and inside_curve.IsValid:
                        inside_tree.Add(inside_curve, path)
                
                for outside_curve in outside_curves:
                    if outside_curve and outside_curve.IsValid:
                        outside_tree.Add(outside_curve, path)
                    
            except Exception as e:
                print(f"Trim operation failed for path {path}: {e}")
                if curve_obj and curve_obj.IsValid:
                    outside_tree.Add(curve_obj, path)
    
    return inside_tree, outside_tree

def curve_middle(curves_tree, by_length=True):

    points_tree = DataTree[rg.Point3d]()
    
    for path in curves_tree.Paths:
        points_tree.EnsurePath(path)
        items = list(curves_tree.Branch(path))
        if not items:
            continue
        try:
            for item in items:
                try:
                    curve = coerce_to_curve(item)
                    if not curve or not curve.IsValid:
                        continue
                    if by_length:
                        t = None
                        success, t = curve.NormalizedLengthParameter(0.5)
                        if success:
                            pt = curve.PointAt(t)
                        else:
                            mid_t = curve.Domain.Mid
                            pt = curve.PointAt(mid_t)
                    else:
                        mid_t = curve.Domain.Mid
                        pt = curve.PointAt(mid_t)
                    points_tree.Add(pt, path)
                except:
                    pass
        except Exception as e:
            print("Curve middle failed for path {}: {}".format(path, e))
            pass
    return points_tree

def plane_surface(planes_tree, x_size, y_size, centered=True, tolerance=None):

    if tolerance is None:
        tolerance = sc.doc.ModelAbsoluteTolerance
    surfaces_tree = DataTree[rg.Surface]()
    
    for path in planes_tree.Paths:
        surfaces_tree.EnsurePath(path)
        planes = list(planes_tree.Branch(path))
        if not planes:
            continue
        try:
            for pl in planes:
                try:
                    if not isinstance(pl, rg.Plane):
                        continue
                    if centered:
                        x_dom = rg.Interval(-abs(x_size) * 0.5, abs(x_size) * 0.5)
                        y_dom = rg.Interval(-abs(y_size) * 0.5, abs(y_size) * 0.5)
                    else:
                        x_dom = rg.Interval(0.0, abs(x_size))
                        y_dom = rg.Interval(0.0, abs(y_size))
                    ps = rg.PlaneSurface(pl, x_dom, y_dom)
                    if ps and ps.IsValid:
                        surfaces_tree.Add(ps, path)
                except Exception as ie:
                    print("Plane surface failed for path {}: {}".format(path, ie))
                    pass
        except Exception as e:
            print("Plane surface generation failed for path {}: {}".format(path, e))
            pass
    return surfaces_tree

def flip_surface_normal(surfaces_tree, guide_tree=None):

    flipped_tree = DataTree[rg.Brep]()
    
    global_guide_plane = None
    if guide_tree is not None:
        try:
            for gp in guide_tree.Paths:
                for g in list(guide_tree.Branch(gp)):
                    if isinstance(g, rg.Plane):
                        global_guide_plane = g
                        break
                if global_guide_plane is not None:
                    break
        except:
            pass
    
    for path in surfaces_tree.Paths:
        flipped_tree.EnsurePath(path)
        items = list(surfaces_tree.Branch(path))
        if not items:
            continue
        try:
            for item in items:
                try:
                    brep = None
                    if isinstance(item, rg.Brep):
                        brep = item.DuplicateBrep()
                    elif isinstance(item, rg.BrepFace):
                        brep = item.DuplicateFace(True)
                    elif isinstance(item, rg.Surface):
                        brep = item.ToBrep()

                    if not brep or not brep.IsValid:
                        continue

                    do_flip = True
                    guide_plane = None
                    if guide_tree is not None:
                        if path in guide_tree.Paths:
                            guides = list(guide_tree.Branch(path))
                            for g in guides:
                                if isinstance(g, rg.Plane):
                                    guide_plane = g
                                    break
                        if guide_plane is None:
                            guide_plane = global_guide_plane

                    if guide_plane is not None:
                        ref_z = guide_plane.ZAxis
                        face = brep.Faces[0] if brep.Faces.Count > 0 else None
                        if face is not None:
                            u = face.Domain(0).Mid
                            v = face.Domain(1).Mid
                            ok, frm = face.FrameAt(u, v)
                            if ok:
                                cur_z = frm.ZAxis
                                dot = rg.Vector3d.Multiply(cur_z, ref_z)
                                do_flip = (dot < 0)

                    if do_flip:
                        brep.Flip()
                    if brep.IsValid:
                        flipped_tree.Add(brep, path)
                except:
                    pass
        except:
            pass
    
    return flipped_tree

def curve_closest_point_length(points_tree, curves_tree, tolerance=None):

    if tolerance is None:
        tolerance = sc.doc.ModelAbsoluteTolerance
    length_params = DataTree[System.Double]()
    
    for path in points_tree.Paths:
        length_params.EnsurePath(path)
        pts = list(points_tree.Branch(path))
        crvs = list(curves_tree.Branch(path)) if path in curves_tree.Paths else []
        if not crvs:
            continue
        curve = coerce_to_curve(crvs[0])
        if not curve or not curve.IsValid:
            continue
        try:
            dom = curve.Domain
            for pt in pts:
                try:
                    if not isinstance(pt, rg.Point3d):
                        # Try to coerce basic tuple-like into Point3d
                        if hasattr(pt, 'X') and hasattr(pt, 'Y') and hasattr(pt, 'Z'):
                            pt = rg.Point3d(pt.X, pt.Y, pt.Z)
                        elif isinstance(pt, (tuple, list)) and len(pt) >= 3:
                            pt = rg.Point3d(pt[0], pt[1], pt[2])
                        else:
                            continue
                    ok, t = curve.ClosestPoint(pt)
                    if not ok:
                        # Fallback: sample mid parameter
                        t = dom.Mid
                    # Clamp to domain
                    if t < dom.T0:
                        t = dom.T0
                    elif t > dom.T1:
                        t = dom.T1
                    arc_len = curve.GetLength(rg.Interval(dom.T0, t))
                    length_params.Add(System.Double(arc_len), path)
                except Exception as ie:
                    print("Closest length failed for path {}: {}".format(path, ie))
                    pass
        except Exception as e:
            print("Curve closest length failed for path {}: {}".format(path, e))
            pass
    return length_params

def evaluate_curve(curves_tree, params_tree, use_length=True, ref_plane=None, tolerance=None):

    if tolerance is None:
        tolerance = sc.doc.ModelAbsoluteTolerance
    if ref_plane is None:
        ref_plane = rg.Plane.WorldXY

    points_tree = DataTree[rg.Point3d]()
    tangents_tree = DataTree[rg.Vector3d]()
    angles_tree = DataTree[System.Double]()

    # Detect if params_tree is a single scalar applied globally
    is_global_param = False
    global_param_value = None
    try:
        if not hasattr(params_tree, 'Paths'):
            # Likely a scalar
            global_param_value = float(params_tree)
            is_global_param = True
    except:
        pass

    def eval_one(curve, param_value):
        try:
            if not curve or not curve.IsValid:
                return None, None, None
            if use_length:
                try:
                    length_value = float(param_value)
                except:
                    length_value = 0.0
                total_len = curve.GetLength()
                if length_value < 0:
                    length_value = 0.0
                if total_len is not None and length_value > total_len:
                    length_value = total_len
                ok, t = curve.LengthParameter(length_value)
                if not ok:
                    t = curve.Domain.Mid
            else:
                # If a single global float is provided and use_length is False,
                # interpret it as a ratio in [0,1] along the parameter domain
                if is_global_param:
                    try:
                        r = float(param_value)
                    except:
                        r = 0.5
                    if r < 0.0:
                        r = 0.0
                    if r > 1.0:
                        r = 1.0
                    dom = curve.Domain
                    t = dom.T0 + r * (dom.T1 - dom.T0)
                else:
                    try:
                        t = float(param_value)
                    except:
                        t = curve.Domain.Mid

            pt = curve.PointAt(t)
            tan = curve.TangentAt(t)
            if not tan.IsZero:
                tan.Unitize()

            tan_proj = tan
            try:
                n = ref_plane.Normal
                comp = rg.Vector3d.Multiply(tan, n)
                tan_proj = tan - n * comp
                if not tan_proj.IsZero:
                    tan_proj.Unitize()
            except:
                pass
            ang = rg.Vector3d.VectorAngle(tan_proj, ref_plane.XAxis)
            return pt, tan, System.Double(ang)
        except Exception as e:
            return None, None, None

    for path in curves_tree.Paths:
        points_tree.EnsurePath(path)
        tangents_tree.EnsurePath(path)
        angles_tree.EnsurePath(path)

        crvs = list(curves_tree.Branch(path))
        if is_global_param:
            pars = [global_param_value]
        else:
            pars = list(params_tree.Branch(path)) if path in params_tree.Paths else []
        if not crvs or not pars:
            continue

        num_crvs = len(crvs)
        num_pars = len(pars)

        try:
            if num_crvs == 1 and num_pars >= 1:
                curve = coerce_to_curve(crvs[0])
                for i in range(num_pars):
                    pt, tan, ang = eval_one(curve, pars[i])
                    if pt is not None:
                        points_tree.Add(pt, path)
                    if tan is not None:
                        tangents_tree.Add(tan, path)
                    if ang is not None:
                        angles_tree.Add(ang, path)
            elif num_pars == 1 and num_crvs >= 1:
                param_value = pars[0]
                for i in range(num_crvs):
                    curve = coerce_to_curve(crvs[i])
                    pt, tan, ang = eval_one(curve, param_value)
                    if pt is not None:
                        points_tree.Add(pt, path)
                    if tan is not None:
                        tangents_tree.Add(tan, path)
                    if ang is not None:
                        angles_tree.Add(ang, path)
            else:
                count = min(num_crvs, num_pars)
                for i in range(count):
                    curve = coerce_to_curve(crvs[i])
                    pt, tan, ang = eval_one(curve, pars[i])
                    if pt is not None:
                        points_tree.Add(pt, path)
                    if tan is not None:
                        tangents_tree.Add(tan, path)
                    if ang is not None:
                        angles_tree.Add(ang, path)
        except Exception as e:
            print("Evaluate curve failed for path {}: {}".format(path, e))
            pass

    return points_tree, tangents_tree, angles_tree

def unitize_vectors(vectors_tree):

    unit_tree = DataTree[rg.Vector3d]()
    
    for path in vectors_tree.Paths:
        unit_tree.EnsurePath(path)
        items = list(vectors_tree.Branch(path))
        if not items:
            continue
        try:
            for v in items:
                try:
                    vec = v
                    if not isinstance(vec, rg.Vector3d):
                        if hasattr(vec, 'X') and hasattr(vec, 'Y') and hasattr(vec, 'Z'):
                            vec = rg.Vector3d(vec.X, vec.Y, vec.Z)
                        elif isinstance(vec, (tuple, list)) and len(vec) >= 3:
                            vec = rg.Vector3d(vec[0], vec[1], vec[2])
                    if isinstance(vec, rg.Vector3d):
                        if vec.IsZero:
                            unit_tree.Add(vec, path)
                        else:
                            u = rg.Vector3d(vec)
                            u.Unitize()
                            unit_tree.Add(u, path)
                except:
                    pass
        except:
            pass
    
    return unit_tree

def multiplication(treeA, treeB):

    def is_number(x):
        try:
            if isinstance(x, (int, float, System.Double)):
                return True
        except:
            pass
        return False

    def to_vector(obj):
        if isinstance(obj, rg.Vector3d):
            return obj
        if hasattr(obj, 'X') and hasattr(obj, 'Y') and hasattr(obj, 'Z'):
            try:
                return rg.Vector3d(obj.X, obj.Y, obj.Z)
            except:
                pass
        if isinstance(obj, (tuple, list)) and len(obj) >= 3:
            try:
                return rg.Vector3d(obj[0], obj[1], obj[2])
            except:
                pass
        return None

    result_tree = DataTree[object]()

    for path in treeA.Paths:
        result_tree.EnsurePath(path)
        a_items = list(treeA.Branch(path))
        b_items = list(treeB.Branch(path)) if path in treeB.Paths else []
        n = min(len(a_items), len(b_items))
        if n == 0:
            continue
        try:
            for i in range(n):
                try:
                    a = a_items[i]
                    b = b_items[i]

                    if is_number(a) and is_number(b):
                        result_tree.Add(a * b, path)
                        continue

                    va = to_vector(a)
                    vb = to_vector(b)

                    if va is not None and is_number(b):
                        s = float(b)
                        vec = rg.Vector3d(va)
                        vec.X *= s; vec.Y *= s; vec.Z *= s
                        result_tree.Add(vec, path)
                    elif vb is not None and is_number(a):
                        s = float(a)
                        vec = rg.Vector3d(vb)
                        vec.X *= s; vec.Y *= s; vec.Z *= s
                        result_tree.Add(vec, path)
                    elif va is not None and vb is not None:
                        # Dot product when both operands are vectors
                        dot = rg.Vector3d.Multiply(va, vb)
                        result_tree.Add(dot, path)
                    else:
                        # Unsupported combination; skip
                        pass
                except Exception as ie:
                    print("Multiplication failed for path {} index {}: {}".format(path, i, ie))
                    pass
        except Exception as e:
            print("Multiplication failed for path {}: {}".format(path, e))
            pass

    return result_tree

def equality(treeA, treeB):

    def is_number(x):
        try:
            if isinstance(x, (int, float, System.Double)):
                return True
        except:
            pass
        return False

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    result_tree = DataTree[bool]()

    is_tree_a = is_tree(treeA)
    is_tree_b = is_tree(treeB)

    # Determine paths to iterate
    all_paths = set()
    if is_tree_a:
        for p in treeA.Paths:
            all_paths.add(p)
    if is_tree_b:
        for p in treeB.Paths:
            all_paths.add(p)

    # If both are scalars, create a default path
    if not all_paths and (not is_tree_a) and (not is_tree_b):
        try:
            val = bool(abs(float(treeA) - float(treeB)) < 1e-10) if is_number(treeA) and is_number(treeB) else False
            path0 = GH_Path(0)
            result_tree.EnsurePath(path0)
            result_tree.Add(val, path0)
            return result_tree
        except:
            return result_tree

    def path_key(p):
        return tuple(p.Indices)

    # Precompute global scalars if provided
    global_a = None
    global_b = None
    if not is_tree_a and is_number(treeA):
        global_a = float(treeA)
    if not is_tree_b and is_number(treeB):
        global_b = float(treeB)

    for path in sorted(all_paths, key=path_key):
        result_tree.EnsurePath(path)

        a_items = []
        b_items = []

        if is_tree_a and path in treeA.Paths:
            a_items = list(treeA.Branch(path))
        elif global_a is not None:
            a_items = [global_a]

        if is_tree_b and path in treeB.Paths:
            b_items = list(treeB.Branch(path))
        elif global_b is not None:
            b_items = [global_b]

        # If one side has no items and the other does, broadcast scalar if available; otherwise skip
        if len(a_items) == 0 and global_a is not None and len(b_items) > 0:
            a_items = [global_a] * len(b_items)
        if len(b_items) == 0 and global_b is not None and len(a_items) > 0:
            b_items = [global_b] * len(a_items)

        if len(a_items) == 0 or len(b_items) == 0:
            continue

        # Per-path broadcasting when one branch has a single value
        if len(a_items) == 1 and len(b_items) > 1 and is_number(a_items[0]):
            a_items = [float(a_items[0])] * len(b_items)
        if len(b_items) == 1 and len(a_items) > 1 and is_number(b_items[0]):
            b_items = [float(b_items[0])] * len(a_items)

        n = min(len(a_items), len(b_items))
        try:
            for i in range(n):
                try:
                    a = a_items[i]
                    b = b_items[i]
                    if is_number(a) and is_number(b):
                        result_tree.Add(bool(abs(float(a) - float(b)) < 1e-10), path)
                except:
                    pass
        except:
            pass
    return result_tree

def larger_than(treeA, treeB):

    def is_number(x):
        try:
            if isinstance(x, (int, float, System.Double)):
                return True
        except:
            pass
        return False

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    result_tree = DataTree[bool]()

    is_tree_a = is_tree(treeA)
    is_tree_b = is_tree(treeB)

    # Determine paths to iterate
    all_paths = set()
    if is_tree_a:
        for p in treeA.Paths:
            all_paths.add(p)
    if is_tree_b:
        for p in treeB.Paths:
            all_paths.add(p)

    # If both are scalars, create a default path
    if not all_paths and (not is_tree_a) and (not is_tree_b):
        try:
            val = bool(float(treeA) > float(treeB)) if is_number(treeA) and is_number(treeB) else False
            path0 = GH_Path(0)
            result_tree.EnsurePath(path0)
            result_tree.Add(val, path0)
            return result_tree
        except:
            return result_tree

    def path_key(p):
        return tuple(p.Indices)

    # Precompute global scalars if provided
    global_a = None
    global_b = None
    if not is_tree_a and is_number(treeA):
        global_a = float(treeA)
    if not is_tree_b and is_number(treeB):
        global_b = float(treeB)

    for path in sorted(all_paths, key=path_key):
        result_tree.EnsurePath(path)

        a_items = []
        b_items = []

        if is_tree_a and path in treeA.Paths:
            a_items = list(treeA.Branch(path))
        elif global_a is not None:
            a_items = [global_a]

        if is_tree_b and path in treeB.Paths:
            b_items = list(treeB.Branch(path))
        elif global_b is not None:
            b_items = [global_b]

        # If one side has no items and the other does, broadcast scalar if available; otherwise skip
        if len(a_items) == 0 and global_a is not None and len(b_items) > 0:
            a_items = [global_a] * len(b_items)
        if len(b_items) == 0 and global_b is not None and len(a_items) > 0:
            b_items = [global_b] * len(a_items)

        if len(a_items) == 0 or len(b_items) == 0:
            continue

        # Per-path broadcasting when one branch has a single value
        if len(a_items) == 1 and len(b_items) > 1 and is_number(a_items[0]):
            a_items = [float(a_items[0])] * len(b_items)
        if len(b_items) == 1 and len(a_items) > 1 and is_number(b_items[0]):
            b_items = [float(b_items[0])] * len(a_items)

        n = min(len(a_items), len(b_items))
        try:
            for i in range(n):
                try:
                    a = a_items[i]
                    b = b_items[i]
                    if is_number(a) and is_number(b):
                        result_tree.Add(bool(float(a) > float(b)), path)
                except:
                    pass
        except:
            pass
    return result_tree

def pick_n_choose(tree0, tree1, pattern_tree):

    result_tree = DataTree[object]()

    all_paths = set()
    for t in (tree0, tree1, pattern_tree):
        for p in t.Paths:
            all_paths.add(p)

    def path_key(p):
        return tuple(p.Indices)

    for path in sorted(all_paths, key=path_key):
        result_tree.EnsurePath(path)
        items0 = list(tree0.Branch(path)) if path in tree0.Paths else []
        items1 = list(tree1.Branch(path)) if path in tree1.Paths else []
        pats = list(pattern_tree.Branch(path)) if path in pattern_tree.Paths else []

        max_len = max(len(items0), len(items1))
        if max_len == 0:
            continue

        broadcast = None
        if len(pats) == 0:
            broadcast = False
        elif len(pats) == 1:
            try:
                broadcast = bool(pats[0])
            except:
                broadcast = False

        try:
            for i in range(max_len):
                try:
                    if broadcast is None:
                        if i < len(pats):
                            sel = bool(pats[i])
                        else:
                            sel = bool(pats[-1]) if len(pats) > 0 else False
                    else:
                        sel = broadcast

                    choice = None
                    if sel:
                        if i < len(items1):
                            choice = items1[i]
                        elif i < len(items0):
                            choice = items0[i]
                    else:
                        if i < len(items0):
                            choice = items0[i]
                        elif i < len(items1):
                            choice = items1[i]

                    if choice is not None:
                        result_tree.Add(choice, path)
                except:
                    pass
        except:
            pass

    return result_tree

def flip_curve(curves_tree):

    flipped_tree = DataTree[rg.Curve]()
    
    for path in curves_tree.Paths:
        flipped_tree.EnsurePath(path)
        items = list(curves_tree.Branch(path))
        
        for item in items:
            try:
                curve = coerce_to_curve(item)
                if curve and curve.IsValid:
                    flipped = curve.DuplicateCurve()
                    flipped.Reverse()
                    flipped_tree.Add(flipped, path)
            except:
                pass
    
    return flipped_tree

def flatten_tree(tree):

    flat_tree = DataTree[object]()
    path0 = GH_Path(0)
    
    for path in tree.Paths:
        items = list(tree.Branch(path))
        for item in items:
            flat_tree.Add(item, path0)
    
    return flat_tree

def unflatten_tree(input_tree, guide_tree):

    """
    Rebuild a tree to match the guide tree's structure by distributing items sequentially.

    Parameters:
        input_tree: DataTree or sequence; items are consumed in order (flattened if tree).
        guide_tree: DataTree; its branch paths and per-branch item counts define the layout.

    Returns:
        DataTree[object]: Same paths as guide; items filled in order. Empty branches preserved.

    Behavior:
        - Flattens input (if DataTree) into a single ordered list.
        - For each guide branch, adds up to N items where N = len(guide branch).
        - If items run out, remaining branches stay present but empty.
        - Extra input items beyond guide capacity are ignored.
    """

    result_tree = DataTree[object]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    flat_items = []
    try:
        if is_tree(input_tree):
            for p in input_tree.Paths:
                try:
                    for it in list(input_tree.Branch(p)):
                        flat_items.append(it)
                except:
                    pass
        else:
            try:
                for it in list(input_tree):
                    flat_items.append(it)
            except:
                if input_tree is not None:
                    flat_items.append(input_tree)
    except:
        pass

    idx = 0

    def path_key(p):
        try:
            return tuple(p.Indices)
        except:
            return (0,)

    for path in sorted(guide_tree.Paths, key=path_key):
        result_tree.EnsurePath(path)
        try:
            required = len(list(guide_tree.Branch(path)))
        except:
            required = 0
        if required <= 0:
            continue
        try:
            for _ in range(required):
                if idx >= len(flat_items):
                    break
                result_tree.Add(flat_items[idx], path)
                idx += 1
        except:
            pass

    return result_tree

def curve_length(curves_tree):

    length_tree = DataTree[System.Double]()
    
    for path in curves_tree.Paths:
        length_tree.EnsurePath(path)
        items = list(curves_tree.Branch(path))
        
        for item in items:
            try:
                curve = coerce_to_curve(item)
                if curve and curve.IsValid:
                    length = curve.GetLength()
                    length_tree.Add(System.Double(length), path)
            except:
                pass
    
    return length_tree

def sort_list(values_tree, reverse=False):

    sorted_tree = DataTree[System.Double]()
    
    for path in values_tree.Paths:
        sorted_tree.EnsurePath(path)
        items = list(values_tree.Branch(path))
        
        try:
            # Convert items to float for sorting
            numbers = [float(x) for x in items if x is not None]
            # Sort the numbers
            sorted_numbers = sorted(numbers, reverse=reverse)
            # Add sorted numbers back to tree
            for num in sorted_numbers:
                sorted_tree.Add(System.Double(num), path)
        except:
            # If conversion fails, add original items unsorted
            for item in items:
                try:
                    sorted_tree.Add(System.Double(item), path)
                except:
                    pass
    
    return sorted_tree

def list_item(list_tree, index_input, wrap=True):

    result_tree = DataTree[object]()
    
    # Detect if index_input is a single scalar
    is_global_index = False
    global_index_value = None
    try:
        if not hasattr(index_input, 'Paths'):
            # Likely a scalar
            global_index_value = int(index_input)
            is_global_index = True
    except:
        pass
    
    def get_item(items, idx):
        try:
            i = int(idx)
            n = len(items)
            if n == 0:
                return None
                
            if wrap:
                # Wrap around if index is out of bounds
                i = i % n if n > 0 else 0
            else:
                # Clamp to valid range if not wrapping
                i = max(0, min(i, n - 1))
                
            return items[i]
        except:
            return None
    
    for path in list_tree.Paths:
        result_tree.EnsurePath(path)
        items = list(list_tree.Branch(path))
        
        if not items:
            continue
            
        if is_global_index:
            # Single index value applied to all items
            result = get_item(items, global_index_value)
            if result is not None:
                result_tree.Add(result, path)
        else:
            # DataTree of indices
            indices = list(index_input.Branch(path)) if path in index_input.Paths else []
            
            if len(indices) == 1:
                # Single index in this branch - apply to all items
                idx = indices[0]
                result = get_item(items, idx)
                if result is not None:
                    result_tree.Add(result, path)
            else:
                # Multiple indices - process each
                for idx in indices:
                    result = get_item(items, idx)
                    if result is not None:
                        result_tree.Add(result, path)
    
    return result_tree

def bool_to_integer(data_tree):
    result_tree = DataTree[object]()
    for path in data_tree.Paths:
            data_tree.EnsurePath(path)
            items = list(data_tree.Branch(path))
            
            if not items:
                continue
            else:
                for i in items:
                    i = int(i)
                    result_tree.Add(i, path)
    return result_tree

def true_indices(bool_tree):

    indices_tree = DataTree[System.Int32]()
    for path in bool_tree.Paths:
        indices_tree.EnsurePath(path)
        items = list(bool_tree.Branch(path))
        for idx, value in enumerate(items):
            try:
                if bool(value):
                    indices_tree.Add(System.Int32(idx), path)
            except:
                pass
    return indices_tree

def list_length(list_tree):

    lengths_tree = DataTree[System.Int32]()
    if list_tree is None:
        return lengths_tree
    for path in list_tree.Paths:
        lengths_tree.EnsurePath(path)
        try:
            items = list(list_tree.Branch(path))
            lengths_tree.Add(System.Int32(len(items)), path)
        except Exception as e:
            print("List length failed for path {}: {}".format(path, e))
            lengths_tree.Add(System.Int32(0), path)
    return lengths_tree

def series(start_tree, step_tree, count_input):
    result_tree = DataTree[System.Double]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    all_paths = set()
    # Collect paths from start tree
    if start_tree is not None and is_tree(start_tree):
        for p in start_tree.Paths:
            all_paths.add(p)

    # Support scalar or tree for step
    is_step_tree = is_tree(step_tree)
    global_step = None
    if not is_step_tree:
        try:
            global_step = float(step_tree)
        except:
            global_step = None
    else:
        for p in step_tree.Paths:
            all_paths.add(p)

    is_count_tree = is_tree(count_input)
    global_count = None
    if not is_count_tree:
        try:
            global_count = int(max(0, int(float(count_input))))
        except:
            global_count = 0
    else:
        for p in count_input.Paths:
            all_paths.add(p)

    def path_key(p):
        return tuple(p.Indices)

    for path in sorted(all_paths, key=path_key):
        result_tree.EnsurePath(path)
        try:
            s_items = list(start_tree.Branch(path)) if (start_tree is not None and is_tree(start_tree) and path in start_tree.Paths) else []
            t_items = list(step_tree.Branch(path)) if (is_step_tree and path in step_tree.Paths) else []

            if len(s_items) == 0 or len(t_items) == 0:
                # allow scalar step broadcast if available
                if len(s_items) == 0 or (len(t_items) == 0 and global_step is None):
                    continue

            try:
                s0 = float(s_items[0])
            except:
                continue
            # Determine step per path
            if len(t_items) > 0:
                try:
                    d0 = float(t_items[0])
                except:
                    d0 = 0.0
            else:
                d0 = float(global_step) if global_step is not None else 0.0

            local_count = global_count
            if is_count_tree:
                c_items = list(count_input.Branch(path)) if path in count_input.Paths else []
                if len(c_items) > 0:
                    try:
                        local_count = int(max(0, int(float(c_items[0]))))
                    except:
                        local_count = 0
                else:
                    if local_count is None:
                        local_count = 0
            else:
                if local_count is None:
                    local_count = 0

            for i in range(local_count):
                try:
                    val = s0 + d0 * i
                    result_tree.Add(System.Double(val), path)
                except:
                    pass
        except Exception as e:
            print("Series failed for path {}: {}".format(path, e))
            pass

    return result_tree

def reset_curve_domain_start(curves_tree):

    result_tree = DataTree[rg.Curve]()
    for path in curves_tree.Paths:
        result_tree.EnsurePath(path)
        items = list(curves_tree.Branch(path))
        for item in items:
            try:
                curve = coerce_to_curve(item)
                if curve and curve.IsValid:
                    dup = curve.DuplicateCurve()
                    d = dup.Domain
                    span = d.T1 - d.T0
                    try:
                        dup.Domain = rg.Interval(0.0, span)
                    except:
                        pass
                    if dup and dup.IsValid:
                        result_tree.Add(dup, path)
            except:
                pass
    return result_tree

def bin_pack_segments(segment_length_tree, lot_length, lot_width, lot_count, lot_type=None):

    result_positions = DataTree[System.Double]()
    status_tree = DataTree[object]()
    used_counts_tree = DataTree[System.Int32]()
    sorted_len_tree = DataTree[System.Double]()
    sorted_wid_tree = DataTree[System.Double]()
    sorted_type_tree = DataTree[object]()

    packed_widths_tree = DataTree[System.Double]()
    packed_lengths_tree = DataTree[System.Double]()
    packed_types_tree = DataTree[object]()

    for branch in segment_length_tree.Paths:
        try:
            segs_raw = list(segment_length_tree.Branch(branch))
            segment_lengths = []
            for s in segs_raw:
                try:
                    segment_lengths.append(float(s))
                except:
                    segment_lengths.append(0.0)

            if lot_type is None:
                lot_data = list(zip(lot_length, lot_width, lot_count))
                lot_data.sort(key=lambda x: x[0], reverse=True)
                sorted_len = [float(x[0]) for x in lot_data]
                sorted_wid = [float(x[1]) for x in lot_data]
                sorted_cnt = [int(x[2]) for x in lot_data]
                sorted_typ = [None for _ in lot_data]
            else:
                lot_data = list(zip(lot_length, lot_width, lot_count, lot_type))
                lot_data.sort(key=lambda x: x[0], reverse=True)
                sorted_len = [float(x[0]) for x in lot_data]
                sorted_wid = [float(x[1]) for x in lot_data]
                sorted_cnt = [int(x[2]) for x in lot_data]
                sorted_typ = [x[3] for x in lot_data]

            per_segment_positions = [[] for _ in segment_lengths]
            per_segment_widths = [[] for _ in segment_lengths]
            per_segment_lengths = [[] for _ in segment_lengths]
            per_segment_types = [[] for _ in segment_lengths]
            used_counts_local = [0] * len(sorted_wid)
            finished = [False] * len(sorted_wid)

            seg_idx = 0
            current_sum = 0.0

            for i in range(len(sorted_wid)):
                w = sorted_wid[i]
                c = sorted_cnt[i]
                for _ in range(c):
                    if seg_idx >= len(segment_lengths):
                        break
                    if current_sum + w <= segment_lengths[seg_idx] + 1e-9:
                        current_sum += w
                        per_segment_positions[seg_idx].append(current_sum)
                        per_segment_widths[seg_idx].append(w)
                        per_segment_lengths[seg_idx].append(sorted_len[i])
                        per_segment_types[seg_idx].append(sorted_typ[i])
                        used_counts_local[i] += 1
                    else:
                        seg_idx += 1
                        if seg_idx >= len(segment_lengths):
                            break
                        current_sum = 0.0
                        if current_sum + w <= segment_lengths[seg_idx] + 1e-9:
                            current_sum += w
                            per_segment_positions[seg_idx].append(current_sum)
                            per_segment_widths[seg_idx].append(w)
                            per_segment_lengths[seg_idx].append(sorted_len[i])
                            per_segment_types[seg_idx].append(sorted_typ[i])
                            used_counts_local[i] += 1
                finished[i] = (used_counts_local[i] == sorted_cnt[i])

            status_values = [w if f else "not packed" for w, f in zip(sorted_wid, finished)]

            for i, lst in enumerate(per_segment_positions):
                new_path = GH_Path(branch)
                new_path = new_path.AppendElement(i)
                result_positions.EnsurePath(new_path)
                for val in lst:
                    try:
                        result_positions.Add(System.Double(float(val)), new_path)
                    except:
                        pass

                packed_widths_tree.EnsurePath(new_path)
                for val in per_segment_widths[i]:
                    try:
                        packed_widths_tree.Add(System.Double(float(val)), new_path)
                    except:
                        pass

                packed_lengths_tree.EnsurePath(new_path)
                for val in per_segment_lengths[i]:
                    try:
                        packed_lengths_tree.Add(System.Double(float(val)), new_path)
                    except:
                        pass

                packed_types_tree.EnsurePath(new_path)
                for val in per_segment_types[i]:
                    try:
                        packed_types_tree.Add(val, new_path)
                    except:
                        pass

            status_tree.EnsurePath(branch)
            used_counts_tree.EnsurePath(branch)
            sorted_len_tree.EnsurePath(branch)
            sorted_wid_tree.EnsurePath(branch)
            sorted_type_tree.EnsurePath(branch)

            for v in status_values:
                status_tree.Add(v, branch)
            for v in used_counts_local:
                used_counts_tree.Add(System.Int32(int(v)), branch)
            for v in sorted_len:
                sorted_len_tree.Add(System.Double(float(v)), branch)
            for v in sorted_wid:
                sorted_wid_tree.Add(System.Double(float(v)), branch)
            for v in (sorted_typ if lot_type is not None else []):
                try:
                    sorted_type_tree.Add(v, branch)
                except:
                    pass

        except Exception as e:
            print("Bin packing failed for path {}: {}".format(branch, e))
            pass

    return result_positions, status_tree, used_counts_tree, sorted_len_tree, sorted_wid_tree, packed_widths_tree, packed_lengths_tree, packed_types_tree, sorted_type_tree

def first_change_or_constant(values_tree):

    result_tree = DataTree[System.Double]()

    def to_float(x):
        try:
            return float(x)
        except:
            return None

    def path_key(p):
        try:
            return tuple(p.Indices)
        except:
            return (0,)

    paths_sorted = sorted(values_tree.Paths, key=path_key)

    per_path = {}
    for path in paths_sorted:
        result_tree.EnsurePath(path)
        items = list(values_tree.Branch(path))
        if not items:
            per_path[path] = (None, None)
            continue
        base = to_float(items[0])
        change_val = None
        if base is not None:
            for i in range(1, len(items)):
                v = to_float(items[i])
                if v is None:
                    continue
                if abs(v - base) > 1e-9:
                    change_val = v
                    break
        per_path[path] = (base, change_val)

    carry = None
    for path in paths_sorted:
        base, change_val = per_path.get(path, (None, None))
        try:
            value_to_add = carry if carry is not None else base
            if value_to_add is not None:
                result_tree.Add(System.Double(float(value_to_add)), path)
        except:
            pass
        carry = change_val

    return result_tree

def cull_short_curves(curves_tree, min_length, include_equal=False):

    output = DataTree[rg.Curve]()

    try:
        input_paths = list(curves_tree.Paths)
    except Exception as e:
        return output

    for path in input_paths:
        output.EnsurePath(path)
        try:
            items = list(curves_tree.Branch(path))
            if not items:
                continue

            for item in items:
                crv = coerce_to_curve(item)
                if crv is None:
                    continue
                if not crv.IsValid:
                    continue
                try:
                    length = crv.GetLength()
                except Exception:
                    continue

                if include_equal:
                    if length >= float(min_length):
                        output.Add(crv, path)
                else:
                    if length > float(min_length):
                        output.Add(crv, path)

        except Exception as e:
            try:
                print("Processing failed for path {}: {}".format(path, e))
            except Exception:
                pass
            pass

    return output

exploded = explode_curves(curves)[0]
exploded = data_tree_manager(exploded, curves)
long_segments = cull_short_curves(exploded, min_segment_length)
shifted_list = shift_list_tree(long_segments, -1, True)

grafted2 = graft_tree(shifted_list)
grafted1 = graft_tree(long_segments)
merged = merge_trees(grafted1, grafted2)
intersection_bool = curve_curve_intersection_tree(grafted1, grafted2)

tree_A, tree_B = dispatch_tree(merged, intersection_bool)
first_line, second_line = extract_first_second_items(tree_B)
line1_start, line1_end = extract_start_end_points(first_line)
line2_start, line2_end = extract_start_end_points(second_line)

extended_first_line = extend_lines(first_line, 10000)
extended_second_line = extend_lines(second_line, 10000)
intersection_pts = line_line_intersection_points(extended_first_line, extended_second_line)

joined_line1 = two_pt_line(line1_start, intersection_pts)
joined_line2 = two_pt_line(intersection_pts, line2_end)
joined_offset = merge_trees(joined_line1, joined_line2, tree_A)

first_line_all, second_line_all = extract_first_second_items(joined_offset)
intersection_pts_all = line_line_intersection_points(first_line_all, second_line_all)

comform_isct_pts_all = data_tree_manager(intersection_pts_all, curves)
closed_polylines = points_to_closed_polyline(comform_isct_pts_all)

min_ratio = 0.15
fin_removed1 = remove_small_fins(closed_polylines, tol=sc.doc.ModelAbsoluteTolerance, min_ratio=min_ratio)
fin_removed1 = clean_tree(fin_removed1, True, True, True)
exploded_vertices = explode_curves(fin_removed1)[1]
major_curves = points_to_closed_polyline(exploded_vertices)