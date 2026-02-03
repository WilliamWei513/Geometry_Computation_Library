from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import Grasshopper as gh
import Rhino.Geometry as rg
import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System
import System.Collections.Generic as scg

def is_tree(x):
    return hasattr(x, 'Paths') and hasattr(x, 'Branch')

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

    # Special rule: when initial tree has a single branch (e.g., {0}) treat its depth as 0
    # → flatten all processed items into that single initial path, preserving output depth.
    try:
        if len(initial_paths) == 1:
            single_path = initial_paths[0]
            if processed_paths:
                try:
                    for pp in processed_paths:
                        try:
                            for it in list(processed_tree.Branch(pp)):
                                try:
                                    new_tree.Add(it, single_path)
                                except:
                                    pass
                        except:
                            pass
                    return new_tree
                except:
                    # Even if flatten fails, keep existing paths and fall through
                    pass
            # No processed paths → return ensured single path
            return new_tree
    except:
        pass

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

    # Detect global scalar distance (plain number, or tree with a single value)
    global_distance = None
    # Case 1: direct scalar (no Paths attribute)
    try:
        if distances_tree is not None and not hasattr(distances_tree, 'Paths'):
            global_distance = float(distances_tree)
    except:
        global_distance = None
    # Case 2: DataTree with exactly one value (possibly under any branch)
    if global_distance is None:
        try:
            total_count = 0
            the_value = None
            if distances_tree is not None and hasattr(distances_tree, 'Paths'):
                for p in distances_tree.Paths:
                    try:
                        vals = list(distances_tree.Branch(p))
                        total_count += len(vals)
                        if len(vals) > 0 and the_value is None:
                            the_value = vals[0]
                        if total_count > 1:
                            break
                    except:
                        pass
                if total_count == 1 and the_value is not None:
                    try:
                        global_distance = float(the_value)
                    except:
                        global_distance = None
        except:
            pass
    # Case 3: original single-branch single-value heuristic
    if global_distance is None:
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
        if len(items) == 0:
            # Preserve empty branch by grafting to an empty child path
            try:
                new_path = GH_Path(path)
                new_path = new_path.AppendElement(0)
                new_tree.EnsurePath(new_path)
            except:
                pass
            continue
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

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    # Helper: resolve mask list for a given path with broadcasting rules
    def resolve_mask_for_path(path, count):
        masks = []
        try:
            if is_tree(treeMask):
                if path in treeMask.Paths:
                    masks = list(treeMask.Branch(path))
                elif len(list(treeMask.Paths)) == 1:
                    # Single-branch mask → broadcast
                    p0 = list(treeMask.Paths)[0]
                    masks = list(treeMask.Branch(p0))
            else:
                # Scalar or list-like
                try:
                    # Try list-like
                    masks = list(treeMask)
                except:
                    masks = [treeMask]
        except:
            masks = []

        # Normalize to booleans
        masks_bool = []
        for m in masks:
            try:
                masks_bool.append(bool(m))
            except:
                masks_bool.append(False)

        if count <= 0:
            return []

        if len(masks_bool) == 0:
            return [False] * count
        if len(masks_bool) == 1:
            return [masks_bool[0]] * count
        # If more than 1 but not enough, extend using last value
        if len(masks_bool) < count:
            last = masks_bool[-1]
            masks_bool = masks_bool + [last] * (count - len(masks_bool))
        # Trim to count if longer
        return masks_bool[:count]

    # Support scalar/list data broadcast across mask paths
    is_data_tree = is_tree(treeData)
    data_paths = list(treeData.Paths) if is_data_tree else []
    paths_to_iterate = data_paths

    if not is_data_tree:
        # No data tree → broadcast scalar/list to all mask paths; if mask is scalar, default path {0}
        try:
            if is_tree(treeMask):
                paths_to_iterate = list(treeMask.Paths)
            else:
                paths_to_iterate = [GH_Path(0)]
        except:
            paths_to_iterate = [GH_Path(0)]

    for path in paths_to_iterate:
        treeA.EnsurePath(path)
        treeB.EnsurePath(path)

        if is_data_tree:
            data_items = list(treeData.Branch(path))
        else:
            # Normalize scalar/list data
            try:
                data_items = list(treeData)
            except:
                data_items = [treeData]

        n = len(data_items)
        if n == 0:
            continue

        masks = resolve_mask_for_path(path, n)
        # If mask is shorter, it will be extended by last value; if longer, trimmed inside resolver
        for i in range(n):
            try:
                if masks[i]:
                    treeA.Add(data_items[i], path)
                else:
                    treeB.Add(data_items[i], path)
            except:
                pass

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

def extend_lines(tree, extension_length = None):
    if extension_length is None:
        extension_length = 10000

    extended_tree = DataTree[object]()
    
    for path in tree.Paths:

        extended_tree.EnsurePath(path)
        
        items = list(tree.Branch(path))
        
        for item in items:
            try:
                start_pt = None
                end_pt = None

                if isinstance(item, rg.Line):
                    try:
                        start_pt = item.From
                        end_pt = item.To
                    except:
                        start_pt = None; end_pt = None
                else:
                    # Try to coerce to curve and use start/end points
                    crv = None
                    try:
                        crv = coerce_to_curve(item)
                    except:
                        crv = None
                    if crv is not None and hasattr(crv, 'PointAtStart') and hasattr(crv, 'PointAtEnd'):
                        start_pt = crv.PointAtStart
                        end_pt = crv.PointAtEnd

                if start_pt is not None and end_pt is not None:
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
                try:
                    c = coerce_to_curve(curve)
                except:
                    c = None
                if c is not None and hasattr(c, 'IsValid') and c.IsValid:
                    # Ensure closed for planar brep creation when appropriate
                    if hasattr(c, 'IsClosed') and not c.IsClosed:
                        try:
                            nc = c.ToNurbsCurve()
                            if nc is not None:
                                nc.MakeClosed(tolerance)
                                if nc.IsValid:
                                    c = nc
                        except:
                            pass
                    valid_curves.append(c)
            
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

def solid_union(breps_tree, tolerance=None):

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.001

    result_tree = DataTree[rg.Brep]()

    def to_brep(obj):
        try:
            if isinstance(obj, rg.Brep):
                return obj
            if isinstance(obj, rg.Surface):
                return obj.ToBrep()
            if isinstance(obj, rg.BrepFace):
                return obj.DuplicateFace(True)
        except:
            pass
        return None

    for path in breps_tree.Paths:
        result_tree.EnsurePath(path)
        items = list(breps_tree.Branch(path))
        if not items:
            continue
        try:
            breps = []
            for it in items:
                b = to_brep(it)
                if b is None or not b.IsValid:
                    continue
                # Try to ensure solid if possible
                try:
                    if not b.IsSolid:
                        dup = b.DuplicateBrep()
                        dup.CapPlanarHoles(tolerance)
                        if dup is not None and dup.IsValid:
                            b = dup
                except:
                    pass
                breps.append(b)

            if not breps:
                continue

            # First try a single union of all breps
            unioned = None
            try:
                unioned = rg.Brep.CreateBooleanUnion(breps, tolerance)
            except:
                unioned = None

            if unioned and len(unioned) > 0:
                for ub in unioned:
                    if ub is not None and ub.IsValid:
                        result_tree.Add(ub, path)
                continue

            # Fallback: pairwise union accumulation
            try:
                acc = breps[0]
                leftovers = []
                for i in range(1, len(breps)):
                    step = rg.Brep.CreateBooleanUnion([acc, breps[i]], tolerance)
                    if step and len(step) > 0 and step[0] is not None and step[0].IsValid:
                        acc = step[0]
                    else:
                        leftovers.append(breps[i])
                if acc is not None and acc.IsValid:
                    result_tree.Add(acc, path)
                for lf in leftovers:
                    result_tree.Add(lf, path)
            except Exception as e:
                for b in breps:
                    if b is not None and b.IsValid:
                        result_tree.Add(b, path)
        except Exception as e:
            print("Solid union failed for path {}: {}".format(path, e))
            pass

    return result_tree

def merge_brep_coplanar_faces(breps_tree, angle_or_tolerance=None):

    if angle_or_tolerance is None:
        try:
            angle_or_tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            angle_or_tolerance = 0.001

    result_tree = DataTree[rg.Brep]()

    for path in breps_tree.Paths:
        result_tree.EnsurePath(path)
        items = list(breps_tree.Branch(path))
        if not items:
            continue
        try:
            for it in items:
                try:
                    b = None
                    if isinstance(it, rg.Brep):
                        b = it.DuplicateBrep()
                    elif isinstance(it, rg.BrepFace):
                        b = it.DuplicateFace(True)
                    elif isinstance(it, rg.Surface):
                        b = it.ToBrep()
                    if b is None or not b.IsValid:
                        continue
                    try:
                        b.MergeCoplanarFaces(angle_or_tolerance)
                    except:
                        pass
                    if b.IsValid:
                        result_tree.Add(b, path)
                except:
                    pass
        except Exception as e:
            print("Merge coplanar faces failed for path {}: {}".format(path, e))
            pass

    return result_tree

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

def join_curves(curves_tree, tolerance=1e-6, plane=None):

    def _resolve_plane_for_path(plane_param, path):
        try:
            if isinstance(plane_param, rg.Plane):
                return plane_param
        except:
            pass
        # treat as DataTree[rg.Plane]
        try:
            if plane_param is not None and hasattr(plane_param, 'Paths'):
                # prefer same path
                if path in plane_param.Paths:
                    branch = list(plane_param.Branch(path))
                    for g in branch:
                        if isinstance(g, rg.Plane):
                            return g
                # fallback: first plane anywhere
                for gp in plane_param.Paths:
                    branch = list(plane_param.Branch(gp))
                    for g in branch:
                        if isinstance(g, rg.Plane):
                            return g
        except:
            pass
        return None

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
                    plane_for_path = _resolve_plane_for_path(plane, path)
                    if plane_for_path is not None:
                        tol = None
                        try:
                            tol = sc.doc.ModelAbsoluteTolerance
                        except:
                            tol = 1e-3
                        def _signed_area_2d(pts2d):
                            area = 0.0
                            n = len(pts2d)
                            if n < 3:
                                return 0.0
                            for i in range(n):
                                x1, y1 = pts2d[i]
                                x2, y2 = pts2d[(i + 1) % n]
                                area += (x1 * y2 - x2 * y1)
                            return area * 0.5
                        def _curve_area_sign_on_plane(curve, ref_plane):
                            try:
                                prj = rg.Curve.ProjectToPlane(curve, ref_plane)
                                c2 = prj if (prj and prj.IsValid) else curve
                                plc = rg.Polyline()
                                ok = False
                                try:
                                    ok, plc = c2.TryGetPolyline()
                                except:
                                    ok = False
                                pts = []
                                if ok and plc is not None and plc.Count > 2:
                                    for p in plc:
                                        ok_uv, u, v = ref_plane.ClosestParameter(p)
                                        if ok_uv:
                                            pts.append((u, v))
                                else:
                                    # sample points when not a polyline
                                    try:
                                        div = max(16, int(c2.GetLength() / (tol * 2.0)))
                                    except:
                                        div = 32
                                    if div < 8:
                                        div = 8
                                    params = c2.DivideByCount(div, True)
                                    if params:
                                        for t in params:
                                            p = c2.PointAt(t)
                                            ok_uv, u, v = ref_plane.ClosestParameter(p)
                                            if ok_uv:
                                                pts.append((u, v))
                                if len(pts) >= 3:
                                    return _signed_area_2d(pts)
                            except:
                                pass
                            return 0.0
                        fixed = []
                        for jc in joined_curves:
                            if jc is not None and jc.IsValid and jc.IsClosed:
                                reversed_needed = False
                                try:
                                    # Prefer Rhino's built-in orientation test when available
                                    try:
                                        ori = jc.ClosedCurveOrientation(plane_for_path)
                                    except:
                                        ori = rg.Curve.ClosedCurveOrientation(jc, plane_for_path)
                                    # Enforce clockwise
                                    if ori == rg.CurveOrientation.CounterClockwise:
                                        reversed_needed = True
                                except:
                                    # Fallback to signed area method
                                    s = _curve_area_sign_on_plane(jc, plane_for_path)
                                    if s > 0:
                                        reversed_needed = True
                                if reversed_needed:
                                    try:
                                        jc.Reverse()
                                    except:
                                        pass
                            fixed.append(jc)
                        for fc in fixed:
                            joined_tree.Add(fc, path)
                    else:
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
        # Project test point to the given plane and use robust containment
        test_pt = plane.ClosestPoint(point)
        tol = sc.doc.ModelAbsoluteTolerance if sc and sc.doc else 1e-3
        # Prefer Curve.Contains for planar closed regions
        res = region_curve.Contains(test_pt, plane, tol)
        if res == rg.PointContainment.Inside or res == rg.PointContainment.Coincident:
            return True
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
        # Fallback: if regions provided as a single flattened branch, reuse it for all paths
        if not regions:
            try:
                if hasattr(regions_tree, 'Paths') and len(list(regions_tree.Paths)) == 1:
                    only_path = list(regions_tree.Paths)[0]
                    regions = list(regions_tree.Branch(only_path))
            except:
                pass
        
        if not regions:
            
            for curve in curves:
                if curve is not None and hasattr(curve, 'IsValid') and curve.IsValid:
                    outside_tree.Add(curve, path)
            continue
        
        # Build projected/closed regions list for this branch (preserve order)
        projected_regions = []
        for r in regions:
            rc = coerce_to_curve(r)
            if rc is None or not rc.IsValid:
                continue
            try:
                prj = rg.Curve.ProjectToPlane(rc, plane)
                if prj and prj.IsValid:
                    rc2 = prj
                else:
                    rc2 = rc
            except:
                rc2 = rc
            try:
                if not rc2.IsClosed:
                    nr = rc2.ToNurbsCurve()
                    if nr:
                        nr.MakeClosed(sc.doc.ModelAbsoluteTolerance)
                        if nr.IsValid:
                            rc2 = nr
            except:
                pass
            projected_regions.append(rc2)

        if len(projected_regions) == 0:
            for curve in curves:
                cobj = coerce_to_curve(curve)
                if cobj is not None and hasattr(cobj, 'IsValid') and cobj.IsValid:
                    outside_tree.Add(cobj, path)
            continue

        for c_idx, curve in enumerate(curves):
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
                
                inside_curves = []
                outside_curves = []
                
                try:
                    # Union over regions: collect all intersection params from all regions,
                    # split once, and classify a fragment as inside if it's inside ANY region.
                    all_params = []
                    for region_curve in projected_regions:
                        intersection_events = rg.Intersect.Intersection.CurveCurve(
                            projected_curve, region_curve, sc.doc.ModelAbsoluteTolerance, sc.doc.ModelAbsoluteTolerance
                        )
                        if intersection_events and intersection_events.Count > 0:
                            for i in range(intersection_events.Count):
                                all_params.append(intersection_events[i].ParameterA)

                    dom = projected_curve.Domain
                    t0 = dom.T0; t1 = dom.T1
                    eps = max(1e-9 * (t1 - t0), 1e-12)
                    filtered = []
                    for p in sorted(set(all_params)):
                        if p > t0 + eps and p < t1 - eps:
                            filtered.append(p)

                    if len(filtered) == 0:
                        mid_param = projected_curve.Domain.Mid
                        mid_point = projected_curve.PointAt(mid_param)
                        inside_any = False
                        for region_curve in projected_regions:
                            if is_point_inside_region(mid_point, region_curve, plane):
                                inside_any = True; break
                        if inside_any:
                            inside_curves.append(projected_curve)
                        else:
                            outside_curves.append(projected_curve)
                    else:
                        split_curves = projected_curve.Split(filtered)
                        if split_curves:
                            for split_curve in split_curves:
                                if split_curve and split_curve.IsValid and split_curve.GetLength() > sc.doc.ModelAbsoluteTolerance:
                                    mid_param = split_curve.Domain.Mid
                                    mid_point = split_curve.PointAt(mid_param)
                                    inside_any = False
                                    for region_curve in projected_regions:
                                        if is_point_inside_region(mid_point, region_curve, plane):
                                            inside_any = True; break
                                    if inside_any:
                                        inside_curves.append(split_curve)
                                    else:
                                        outside_curves.append(split_curve)
                        else:
                            # Split returned nothing, fallback to whole-curve classification
                            mid_param = projected_curve.Domain.Mid
                            mid_point = projected_curve.PointAt(mid_param)
                            inside_any = False
                            for region_curve in projected_regions:
                                if is_point_inside_region(mid_point, region_curve, plane):
                                    inside_any = True; break
                            if inside_any:
                                inside_curves.append(projected_curve)
                            else:
                                outside_curves.append(projected_curve)
                
                except Exception as e:
                    print(f"Region trimming failed for path {path}: {e}")
                    outside_curves = [curve_obj]
                
                # Final robust classification by midpoint to counter precision issues
                def classify_and_add(crv):
                    try:
                        if crv is None or not crv.IsValid:
                            return
                        mid_t = crv.Domain.Mid
                        mid_pt = crv.PointAt(mid_t)
                        inside_any = False
                        for region_curve in projected_regions:
                            if is_point_inside_region(mid_pt, region_curve, plane):
                                inside_any = True; break
                        if inside_any:
                            inside_tree.Add(crv, path)
                        else:
                            outside_tree.Add(crv, path)
                    except:
                        outside_tree.Add(crv, path)

                for inside_curve in inside_curves:
                    classify_and_add(inside_curve)
                for outside_curve in outside_curves:
                    classify_and_add(outside_curve)
                    
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

                    do_flip = False
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
                        cur_z = None
                        centroid = None
                        try:
                            amp = rg.AreaMassProperties.Compute(brep)
                            if amp is not None:
                                centroid = amp.Centroid
                        except:
                            centroid = None
                        if centroid is None:
                            try:
                                centroid = brep.GetBoundingBox(True).Center
                            except:
                                centroid = None
                        if face is not None:
                            got_uv = False
                            if centroid is not None:
                                try:
                                    ok, u, v = face.ClosestPoint(centroid)
                                    if ok:
                                        cur_z = face.NormalAt(u, v)
                                        got_uv = True
                                except:
                                    pass
                            if not got_uv:
                                try:
                                    u = face.Domain(0).Mid
                                    v = face.Domain(1).Mid
                                    cur_z = face.NormalAt(u, v)
                                except:
                                    cur_z = None
                        if cur_z is not None and ref_z is not None:
                            try:
                                cur_z.Unitize()
                                ref_z.Unitize()
                                dot = rg.Vector3d.Multiply(cur_z, ref_z)
                                do_flip = (dot < 0)
                            except:
                                do_flip = False

                    if do_flip:
                        brep.Flip()
                    if brep.IsValid:
                        flipped_tree.Add(brep, path)
                except:
                    pass
        except:
            pass
    
    return flipped_tree

def curve_closest_point(points_tree, curves_tree, tolerance=None):

    if tolerance is None:
        tolerance = sc.doc.ModelAbsoluteTolerance

    points_out = DataTree[rg.Point3d]()
    length_params = DataTree[System.Double]()
    distances_out = DataTree[System.Double]()

    # Detect hierarchical broadcasting: curves are one level deeper under each points path
    try:
        points_paths = list(points_tree.Paths)
        curves_paths = list(curves_tree.Paths)
        points_path_tuples = [tuple(p.Indices) for p in points_paths]
        # Map parent tuple -> child curve paths at exactly +1 depth
        parent_to_children = {}
        for q in curves_paths:
            try:
                inds = tuple(q.Indices)
                if len(inds) == 0:
                    continue
                parent = inds[:-1]
                if parent in [tuple(p.Indices) for p in points_paths]:
                    parent_to_children.setdefault(parent, []).append(q)
            except:
                pass
        has_hier = any(len(v) >= 1 for v in parent_to_children.values())
    except:
        has_hier = False

    if has_hier:
        for p in points_paths:
            pts = list(points_tree.Branch(p))
            parent_key = tuple(p.Indices)
            child_paths = parent_to_children.get(parent_key, [])
            # If no child paths, skip; preserve empties implicitly by not adding
            for cpath in child_paths:
                try:
                    points_out.EnsurePath(cpath)
                    length_params.EnsurePath(cpath)
                    distances_out.EnsurePath(cpath)
                    crv_list = list(curves_tree.Branch(cpath))
                    if not crv_list:
                        # preserve empty branch structure
                        continue
                    curve = coerce_to_curve(crv_list[0])
                    if not curve or not curve.IsValid:
                        continue
                    dom = curve.Domain
                    for pt in pts:
                        try:
                            if not isinstance(pt, rg.Point3d):
                                if hasattr(pt, 'X') and hasattr(pt, 'Y') and hasattr(pt, 'Z'):
                                    pt = rg.Point3d(pt.X, pt.Y, pt.Z)
                                elif isinstance(pt, (tuple, list)) and len(pt) >= 3:
                                    pt = rg.Point3d(pt[0], pt[1], pt[2])
                                else:
                                    continue
                            ok, t = curve.ClosestPoint(pt)
                            if not ok:
                                t = dom.Mid
                            if t < dom.T0:
                                t = dom.T0
                            elif t > dom.T1:
                                t = dom.T1
                            cp = curve.PointAt(t)
                            arc_len = curve.GetLength(rg.Interval(dom.T0, t))
                            dist = pt.DistanceTo(cp)
                            points_out.Add(cp, cpath)
                            length_params.Add(System.Double(arc_len), cpath)
                            distances_out.Add(System.Double(dist), cpath)
                        except Exception as ie:
                            print("Closest length failed for path {}: {}".format(cpath, ie))
                            pass
                except Exception as e:
                    print("Curve closest length failed for path {}: {}".format(cpath, e))
                    pass
        return points_out, length_params, distances_out

    # Detect broadcasting case: points in a single branch, curves with one item per branch
    try:
        is_single_points_branch = (len(points_paths) == 1)
        has_curves = (len(curves_paths) > 0)
        curves_one_each = has_curves and all(len(list(curves_tree.Branch(p))) == 1 for p in curves_paths)
    except Exception as e:
        is_single_points_branch = False
        curves_one_each = False

    if is_single_points_branch and curves_one_each:
        base_path = points_paths[0]
        pts = list(points_tree.Branch(base_path))
        for cpath in curves_paths:
            try:
                points_out.EnsurePath(cpath)
                length_params.EnsurePath(cpath)
                distances_out.EnsurePath(cpath)
                crv_list = list(curves_tree.Branch(cpath))
                if not crv_list:
                    continue
                curve = coerce_to_curve(crv_list[0])
                if not curve or not curve.IsValid:
                    continue
                dom = curve.Domain
                for pt in pts:
                    try:
                        if not isinstance(pt, rg.Point3d):
                            if hasattr(pt, 'X') and hasattr(pt, 'Y') and hasattr(pt, 'Z'):
                                pt = rg.Point3d(pt.X, pt.Y, pt.Z)
                            elif isinstance(pt, (tuple, list)) and len(pt) >= 3:
                                pt = rg.Point3d(pt[0], pt[1], pt[2])
                            else:
                                continue
                        ok, t = curve.ClosestPoint(pt)
                        if not ok:
                            t = dom.Mid
                        if t < dom.T0:
                            t = dom.T0
                        elif t > dom.T1:
                            t = dom.T1
                        cp = curve.PointAt(t)
                        arc_len = curve.GetLength(rg.Interval(dom.T0, t))
                        dist = pt.DistanceTo(cp)
                        points_out.Add(cp, cpath)
                        length_params.Add(System.Double(arc_len), cpath)
                        distances_out.Add(System.Double(dist), cpath)
                    except Exception as ie:
                        print("Closest length failed for path {}: {}".format(cpath, ie))
                        pass
            except Exception as e:
                print("Curve closest length failed for path {}: {}".format(cpath, e))
                pass
    else:
        for path in points_tree.Paths:
            points_out.EnsurePath(path)
            length_params.EnsurePath(path)
            distances_out.EnsurePath(path)
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
                            if hasattr(pt, 'X') and hasattr(pt, 'Y') and hasattr(pt, 'Z'):
                                pt = rg.Point3d(pt.X, pt.Y, pt.Z)
                            elif isinstance(pt, (tuple, list)) and len(pt) >= 3:
                                pt = rg.Point3d(pt[0], pt[1], pt[2])
                            else:
                                continue
                        ok, t = curve.ClosestPoint(pt)
                        if not ok:
                            t = dom.Mid
                        if t < dom.T0:
                            t = dom.T0
                        elif t > dom.T1:
                            t = dom.T1
                        cp = curve.PointAt(t)
                        arc_len = curve.GetLength(rg.Interval(dom.T0, t))
                        dist = pt.DistanceTo(cp)
                        points_out.Add(cp, path)
                        length_params.Add(System.Double(arc_len), path)
                        distances_out.Add(System.Double(dist), path)
                    except Exception as ie:
                        print("Closest length failed for path {}: {}".format(path, ie))
                        pass
            except Exception as e:
                print("Curve closest length failed for path {}: {}".format(path, e))
                pass
    return points_out, length_params, distances_out

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

            # Prepare plane axes (unitized)
            try:
                xaxis = rg.Vector3d(ref_plane.XAxis)
            except:
                xaxis = rg.Vector3d(1, 0, 0)
            try:
                if not xaxis.IsZero:
                    xaxis.Unitize()
            except:
                pass

            # Project tangent onto plane using a unit normal; fallback if projection is tiny
            ang_rad = 0.0
            try:
                n = rg.Vector3d(ref_plane.Normal)
                if not n.IsZero:
                    n.Unitize()
                comp = rg.Vector3d.Multiply(tan, n)
                tan_proj = tan - (n * comp)
                tiny = False
                try:
                    tiny = tan_proj.Length < (tolerance * 10.0)
                except:
                    tiny = False
                if not tiny and not tan_proj.IsZero:
                    tan_proj.Unitize()
                    ang_rad = rg.Vector3d.VectorAngle(tan_proj, xaxis)
                else:
                    # Fallback: use unprojected tangent when projection degenerates
                    ang_rad = rg.Vector3d.VectorAngle(tan, xaxis)
            except:
                ang_rad = rg.Vector3d.VectorAngle(tan, xaxis)

            ang_deg = ang_rad * 180.0 / System.Math.PI
            return pt, tan, System.Double(ang_deg)
        except Exception as e:
            return None, None, None

    # Build union of paths, prioritizing curve paths but including parameter-only paths as empty outputs
    all_paths = set()
    try:
        for p in curves_tree.Paths:
            all_paths.add(p)
    except:
        all_paths = set()
    try:
        if not is_global_param:
            for p in params_tree.Paths:
                all_paths.add(p)
    except:
        pass

    def path_key(p):
        try:
            return tuple(p.Indices)
        except:
            return (0,)

    for path in sorted(all_paths, key=path_key):
        points_tree.EnsurePath(path)
        tangents_tree.EnsurePath(path)
        angles_tree.EnsurePath(path)

        crvs = list(curves_tree.Branch(path)) if path in curves_tree.Paths else []
        if is_global_param:
            pars = [global_param_value]
        else:
            pars = list(params_tree.Branch(path)) if (path in getattr(params_tree, 'Paths', [])) else []

        # Priority: empty curve branch → keep outputs empty; else empty params branch (when not global) → keep empty
        if not crvs:
            continue
        if (not is_global_param) and (len(pars) == 0):
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

def perp_frames(curves_tree, length_params):

    frames_tree = DataTree[rg.Plane]()

    # Detect if length_params is a single scalar applied globally
    is_global_param = False
    global_param_value = None
    try:
        if not hasattr(length_params, 'Paths'):
            global_param_value = float(length_params)
            is_global_param = True
    except:
        pass

    def frame_at_length(curve, length_value):
        try:
            if not curve or not curve.IsValid:
                return None
            # Clamp and convert length to parameter
            try:
                target_len = float(length_value)
            except:
                target_len = 0.0
            total_len = curve.GetLength()
            if target_len < 0.0:
                target_len = 0.0
            if total_len is not None and target_len > total_len:
                target_len = total_len
            ok, t = curve.LengthParameter(target_len)
            if not ok:
                t = curve.Domain.Mid
            # Try perpendicular frame first, fallback to generic frame
            try:
                okp, pl = curve.PerpendicularFrameAt(t)
                if okp:
                    return pl
            except:
                pass
            try:
                okf, pl2 = curve.FrameAt(t)
                if okf:
                    return pl2
            except:
                pass
            # Final fallback: build plane from point and tangent
            pt = curve.PointAt(t)
            tan = curve.TangentAt(t)
            if not tan.IsZero:
                tan.Unitize()
            # Choose an arbitrary normal that is not parallel to tangent
            arbitrary = rg.Vector3d(0, 0, 1)
            if abs(rg.Vector3d.Multiply(tan, arbitrary)) > 0.999:
                arbitrary = rg.Vector3d(1, 0, 0)
            xaxis = tan
            yaxis = rg.Vector3d.CrossProduct(arbitrary, xaxis)
            if not yaxis.IsZero:
                yaxis.Unitize()
            zaxis = rg.Vector3d.CrossProduct(xaxis, yaxis)
            if not zaxis.IsZero:
                zaxis.Unitize()
            return rg.Plane(pt, xaxis, yaxis)
        except:
            return None

    for path in curves_tree.Paths:
        frames_tree.EnsurePath(path)
        crvs = list(curves_tree.Branch(path))
        if not crvs:
            continue
        if is_global_param:
            pars = [global_param_value]
        else:
            pars = list(length_params.Branch(path)) if path in length_params.Paths else []
        if not pars:
            continue

        num_crvs = len(crvs)
        num_pars = len(pars)

        try:
            if num_crvs == 1 and num_pars >= 1:
                curve = coerce_to_curve(crvs[0])
                for i in range(num_pars):
                    pl = frame_at_length(curve, pars[i])
                    if pl is not None:
                        frames_tree.Add(pl, path)
            elif num_pars == 1 and num_crvs >= 1:
                param_value = pars[0]
                for i in range(num_crvs):
                    curve = coerce_to_curve(crvs[i])
                    pl = frame_at_length(curve, param_value)
                    if pl is not None:
                        frames_tree.Add(pl, path)
            else:
                count = min(num_crvs, num_pars)
                for i in range(count):
                    curve = coerce_to_curve(crvs[i])
                    pl = frame_at_length(curve, pars[i])
                    if pl is not None:
                        frames_tree.Add(pl, path)
        except Exception as e:
            print("Perp frames failed for path {}: {}".format(path, e))
            pass

    return frames_tree

def deconstruct_plane(planes_tree):

    origins_tree = DataTree[rg.Point3d]()
    xaxis_tree = DataTree[rg.Vector3d]()
    yaxis_tree = DataTree[rg.Vector3d]()
    zaxis_tree = DataTree[rg.Vector3d]()

    for path in planes_tree.Paths:
        origins_tree.EnsurePath(path)
        xaxis_tree.EnsurePath(path)
        yaxis_tree.EnsurePath(path)
        zaxis_tree.EnsurePath(path)

        items = list(planes_tree.Branch(path))
        if not items:
            continue

        for pl in items:
            try:
                plane_obj = None
                if isinstance(pl, rg.Plane):
                    plane_obj = pl
                elif hasattr(pl, 'Origin') and hasattr(pl, 'XAxis') and hasattr(pl, 'YAxis') and hasattr(pl, 'ZAxis'):
                    plane_obj = rg.Plane(pl.Origin, pl.XAxis, pl.YAxis)
                if plane_obj is None:
                    continue

                origins_tree.Add(plane_obj.Origin, path)
                xaxis_tree.Add(plane_obj.XAxis, path)
                yaxis_tree.Add(plane_obj.YAxis, path)
                zaxis_tree.Add(plane_obj.ZAxis, path)
            except:
                pass

    return origins_tree, xaxis_tree, yaxis_tree, zaxis_tree

def area(geometry_tree, tolerance=None):

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.001

    area_tree = DataTree[System.Double]()
    centroid_tree = DataTree[rg.Point3d]()

    def to_curve(obj):
        try:
            if isinstance(obj, rg.Curve):
                return obj
            if isinstance(obj, rg.Polyline):
                return rg.PolylineCurve(obj)
            if isinstance(obj, rg.Line):
                return rg.LineCurve(obj)
        except:
            pass
        return None

    for path in geometry_tree.Paths:
        area_tree.EnsurePath(path)
        centroid_tree.EnsurePath(path)
        items = list(geometry_tree.Branch(path))
        if not items:
            continue
        for it in items:
            try:
                amp = None
                # Try direct types first
                if isinstance(it, rg.Brep):
                    amp = rg.AreaMassProperties.Compute(it)
                elif isinstance(it, rg.BrepFace):
                    amp = rg.AreaMassProperties.Compute(it)
                elif isinstance(it, rg.Surface):
                    amp = rg.AreaMassProperties.Compute(it)
                elif isinstance(it, rg.Mesh):
                    amp = rg.AreaMassProperties.Compute(it)
                else:
                    crv = to_curve(it)
                    if crv is not None and crv.IsValid:
                        # For curves, AMP works for closed planar curves. If fails, try planar brep from curve.
                        amp = rg.AreaMassProperties.Compute(crv)
                        if amp is None:
                            try:
                                pb = rg.Brep.CreatePlanarBreps(crv, tolerance)
                                if pb and len(pb) > 0:
                                    amp = rg.AreaMassProperties.Compute(pb[0])
                            except:
                                amp = None

                if amp is not None:
                    try:
                        area_tree.Add(System.Double(float(amp.Area)), path)
                        centroid_tree.Add(amp.Centroid, path)
                        continue
                    except:
                        pass

                # Fallback: area 0, centroid by bounding box center if possible
                area_tree.Add(System.Double(0.0), path)
                try:
                    bbox = None
                    if hasattr(it, 'GetBoundingBox'):
                        bbox = it.GetBoundingBox(True)
                    elif isinstance(it, rg.Curve):
                        bbox = it.GetBoundingBox(True)
                    if bbox is not None and bbox.IsValid:
                        centroid_tree.Add(bbox.Center, path)
                    else:
                        centroid_tree.Add(rg.Point3d(0,0,0), path)
                except:
                    centroid_tree.Add(rg.Point3d(0,0,0), path)
            except Exception as e:
                try:
                    area_tree.Add(System.Double(0.0), path)
                    centroid_tree.Add(rg.Point3d(0,0,0), path)
                except:
                    pass

    return area_tree, centroid_tree

def deconstruct_point(points_tree):

    x_tree = DataTree[System.Double]()
    y_tree = DataTree[System.Double]()
    z_tree = DataTree[System.Double]()

    for path in points_tree.Paths:
        x_tree.EnsurePath(path)
        y_tree.EnsurePath(path)
        z_tree.EnsurePath(path)
        items = list(points_tree.Branch(path))
        if not items:
            continue
        for p in items:
            try:
                pt = None
                if isinstance(p, rg.Point3d):
                    pt = p
                elif hasattr(p, 'X') and hasattr(p, 'Y') and hasattr(p, 'Z'):
                    pt = rg.Point3d(p.X, p.Y, p.Z)
                elif isinstance(p, (tuple, list)) and len(p) >= 3:
                    pt = rg.Point3d(float(p[0]), float(p[1]), float(p[2]))
                if pt is None:
                    continue
                x_tree.Add(System.Double(float(pt.X)), path)
                y_tree.Add(System.Double(float(pt.Y)), path)
                z_tree.Add(System.Double(float(pt.Z)), path)
            except:
                pass

    return x_tree, y_tree, z_tree

def project_point(points_tree, direction_input, geometry_input, tolerance=None):
    """
    Project points along a direction onto geometry, preserving DataTree structure.

    Inputs:
        points_tree: DataTree[rg.Point3d]
        direction_input: rg.Vector3d or DataTree[rg.Vector3d]
        geometry_input: Brep/Mesh/Curve/Surface or DataTree containing those
        tolerance: optional, defaults to document tolerance

    Returns:
        (projected_points_tree, hit_index_tree)
    """

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.001

    projected_tree = DataTree[rg.Point3d]()
    index_tree = DataTree[System.Int32]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    # Resolve direction: allow scalar or tree (per-path first item)
    is_dir_tree = is_tree(direction_input)
    global_dir = None
    if not is_dir_tree:
        try:
            if isinstance(direction_input, rg.Vector3d):
                global_dir = rg.Vector3d(direction_input)
            elif hasattr(direction_input, 'X') and hasattr(direction_input, 'Y') and hasattr(direction_input, 'Z'):
                global_dir = rg.Vector3d(direction_input.X, direction_input.Y, direction_input.Z)
        except:
            global_dir = None

    def _sanitize_vector_inplace(v, eps=1e-12):
        try:
            if abs(v.X) < eps:
                v.X = 0.0
            if abs(v.Y) < eps:
                v.Y = 0.0
            if abs(v.Z) < eps:
                v.Z = 0.0
        except:
            pass

    def resolve_directions(path, count):
        # Return a list of vectors, length == count
        dirs = []
        if is_dir_tree and path in direction_input.Paths:
            try:
                items = list(direction_input.Branch(path))
                for i in range(min(len(items), count)):
                    v = items[i]
                    if isinstance(v, rg.Vector3d):
                        vv = rg.Vector3d(v)
                        _sanitize_vector_inplace(vv)
                        dirs.append(vv)
                    elif hasattr(v, 'X') and hasattr(v, 'Y') and hasattr(v, 'Z'):
                        vv = rg.Vector3d(v.X, v.Y, v.Z)
                        _sanitize_vector_inplace(vv)
                        dirs.append(vv)
                # If not enough, pad with last or global/default
                while len(dirs) < count:
                    if len(items) > 0:
                        v = items[-1]
                        if isinstance(v, rg.Vector3d):
                            vv = rg.Vector3d(v)
                            _sanitize_vector_inplace(vv)
                            dirs.append(vv)
                        elif hasattr(v, 'X') and hasattr(v, 'Y') and hasattr(v, 'Z'):
                            vv = rg.Vector3d(v.X, v.Y, v.Z)
                            _sanitize_vector_inplace(vv)
                            dirs.append(vv)
                    else:
                        break
            except:
                pass
        # If still not enough, fill with global or default
        while len(dirs) < count:
            if global_dir is not None:
                vv = rg.Vector3d(global_dir)
                _sanitize_vector_inplace(vv)
                dirs.append(vv)
            else:
                dirs.append(rg.Vector3d(0, 0, -1))
        return dirs

    # Resolve geometry: allow scalar (single or iterable) or per-path tree
    is_geom_tree = is_tree(geometry_input)
    global_geoms = None
    if not is_geom_tree:
        try:
            global_geoms = []
            try:
                for g in list(geometry_input):
                    global_geoms.append(g)
            except:
                global_geoms.append(geometry_input)
        except:
            global_geoms = None

    def resolve_geometries(path):
        if is_geom_tree and path in geometry_input.Paths:
            return list(geometry_input.Branch(path))
        return list(global_geoms) if global_geoms is not None else []

    def try_project_one(pt, direction_vec, geom):
        if direction_vec.IsZero or geom is None:
            return None
        d = rg.Vector3d(direction_vec)
        _sanitize_vector_inplace(d)
        d.Unitize()
        _sanitize_vector_inplace(d)
        ray = rg.Line(pt, pt + d * 1e9)

        candidates = []
        try:
            if isinstance(geom, rg.Brep):
                events = rg.Intersect.Intersection.BrepLine(geom, ray, tolerance)
                if events:
                    for ev in events:
                        p = None
                        try:
                            p = ev.Point
                        except:
                            try:
                                p = ev.PointA
                            except:
                                p = None
                        if p is not None:
                            candidates.append(p)
            elif isinstance(geom, rg.Mesh):
                pts = rg.Intersect.Intersection.LineMesh(geom, ray)
                if pts:
                    for p in pts:
                        candidates.append(p)
            elif isinstance(geom, rg.Curve):
                events = rg.Intersect.Intersection.CurveLine(geom, ray, tolerance, tolerance)
                if events:
                    for ev in events:
                        p = None
                        try:
                            p = ev.Point
                        except:
                            try:
                                p = ev.PointA
                            except:
                                p = None
                        if p is not None:
                            candidates.append(p)
            elif isinstance(geom, rg.Surface):
                b = geom.ToBrep()
                if b is not None:
                    events = rg.Intersect.Intersection.BrepLine(b, ray, tolerance)
                    if events:
                        for ev in events:
                            p = None
                            try:
                                p = ev.Point
                            except:
                                try:
                                    p = ev.PointA
                                except:
                                    p = None
                            if p is not None:
                                candidates.append(p)
        except:
            pass

        def shoot(origin_pt):
            best = None
            best_dist = float('inf')
            line = rg.Line(origin_pt, origin_pt + d * 1e9)
            loc_candidates = []
            try:
                if isinstance(geom, rg.Brep):
                    events = rg.Intersect.Intersection.BrepLine(geom, line, tolerance)
                    if events:
                        for ev in events:
                            p = None
                            try:
                                p = ev.Point
                            except:
                                try:
                                    p = ev.PointA
                                except:
                                    p = None
                            if p is not None:
                                loc_candidates.append(p)
                elif isinstance(geom, rg.Mesh):
                    pts = rg.Intersect.Intersection.LineMesh(geom, line)
                    if pts:
                        for p in pts:
                            loc_candidates.append(p)
                elif isinstance(geom, rg.Curve):
                    events = rg.Intersect.Intersection.CurveLine(geom, line, tolerance, tolerance)
                    if events:
                        for ev in events:
                            p = None
                            try:
                                p = ev.Point
                            except:
                                try:
                                    p = ev.PointA
                                except:
                                    p = None
                            if p is not None:
                                loc_candidates.append(p)
                elif isinstance(geom, rg.Surface):
                    b = geom.ToBrep()
                    if b is not None:
                        events = rg.Intersect.Intersection.BrepLine(b, line, tolerance)
                        if events:
                            for ev in events:
                                p = None
                                try:
                                    p = ev.Point
                                except:
                                    try:
                                        p = ev.PointA
                                    except:
                                        p = None
                                if p is not None:
                                    loc_candidates.append(p)
            except:
                pass

            for cp in loc_candidates:
                vec = cp - origin_pt
                if rg.Vector3d.Multiply(vec, d) >= 0.0:
                    dist = vec.Length
                    if dist < best_dist:
                        best_dist = dist
                        best = cp
            return best, best_dist

        # First attempt
        first_pt, first_dist = shoot(pt)
        # If no move (on geometry) or no hit, nudge and retry
        step = max(100.0 * tolerance, tolerance)
        if first_pt is None or first_dist <= step * 0.5:
            nudged = pt + d * step
            second_pt, _ = shoot(nudged)
            if second_pt is not None:
                return second_pt
        return first_pt

    for path in points_tree.Paths:
        projected_tree.EnsurePath(path)
        index_tree.EnsurePath(path)
        pts = list(points_tree.Branch(path))
        if not pts:
            continue

        dirs = resolve_directions(path, len(pts))
        geoms = resolve_geometries(path)
        if len(geoms) == 0:
            continue

        for i, p in enumerate(pts):
            try:
                if not isinstance(p, rg.Point3d):
                    if hasattr(p, 'X') and hasattr(p, 'Y') and hasattr(p, 'Z'):
                        p = rg.Point3d(p.X, p.Y, p.Z)
                    elif isinstance(p, (tuple, list)) and len(p) >= 3:
                        p = rg.Point3d(p[0], p[1], p[2])
                dvec = dirs[i] if i < len(dirs) else (rg.Vector3d(global_dir) if global_dir is not None else rg.Vector3d(0,0,-1))
                _sanitize_vector_inplace(dvec)
                for gi, g in enumerate(geoms):
                    res_pt = try_project_one(p, dvec, g)
                    if res_pt is not None:
                        projected_tree.Add(res_pt, path)
                        index_tree.Add(System.Int32(int(gi)), path)
            except:
                pass

    return projected_tree, index_tree


def find_adjacent_point(points_tree, grafted_curves_tree, threshold_input=None, tolerance=None):

    """
    For each branch in a grafted curves tree (one curve per branch), find the nearest point
    from the parent path's points and output exactly one point per curve branch.

    Inputs:
        points_tree: DataTree[rg.Point3d] – candidate points under parent paths
        grafted_curves_tree: DataTree[rg.Curve] – expected one valid curve per branch (grafted)
        tolerance: optional, defaults to document tolerance

    Returns:
        DataTree[rg.Point3d] – structure mirrors grafted_curves_tree; one selected point per branch when available.
        If threshold_input is provided, only outputs when nearest distance < threshold; otherwise leaves branch empty.
    """

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.001

    result = DataTree[rg.Point3d]()

    def to_point(obj):
        try:
            if isinstance(obj, rg.Point3d):
                return rg.Point3d(obj)
            if hasattr(obj, 'X') and hasattr(obj, 'Y') and hasattr(obj, 'Z'):
                return rg.Point3d(float(obj.X), float(obj.Y), float(obj.Z))
            if isinstance(obj, (tuple, list)) and len(obj) >= 3:
                return rg.Point3d(float(obj[0]), float(obj[1]), float(obj[2]))
        except:
            pass
        return None

    def path_tuple(p):
        try:
            return tuple(p.Indices)
        except:
            return tuple()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    def to_float(x):
        try:
            return float(x)
        except:
            return None

    # Threshold resolvers with flexible broadcasting
    thresholds_by_path = {}
    single_threshold = None
    global_threshold = None
    if threshold_input is not None:
        if is_tree(threshold_input):
            try:
                th_paths = list(threshold_input.Paths)
                for tp in th_paths:
                    vals = []
                    try:
                        vals = list(threshold_input.Branch(tp))
                    except:
                        vals = []
                    tv = None
                    for v in vals:
                        tv = to_float(v)
                        if tv is not None:
                            break
                    thresholds_by_path[path_tuple(tp)] = tv
                if len(th_paths) == 1:
                    try:
                        br = list(threshold_input.Branch(th_paths[0]))
                    except:
                        br = []
                    for v in br:
                        single_threshold = to_float(v)
                        if single_threshold is not None:
                            break
                    if single_threshold is None:
                        single_threshold = 0.0
            except:
                thresholds_by_path = {}
        else:
            # Scalar or list-like (take first)
            try:
                if isinstance(threshold_input, (list, tuple)) and len(threshold_input) > 0:
                    global_threshold = to_float(threshold_input[0])
                else:
                    global_threshold = to_float(threshold_input)
            except:
                global_threshold = to_float(threshold_input)

    def resolve_threshold_for_path(cpath):
        if threshold_input is None:
            return None
        ptup = path_tuple(cpath)
        parent_tup = ptup[:-1] if len(ptup) > 0 else ptup
        # Exact cpath
        tv = thresholds_by_path.get(ptup, None)
        if tv is not None:
            return tv
        # Parent path
        tv = thresholds_by_path.get(parent_tup, None)
        if tv is not None:
            return tv
        # Prefix matches
        try:
            for k, v in thresholds_by_path.items():
                try:
                    if v is None:
                        continue
                    if len(k) >= len(parent_tup) and tuple(k[:len(parent_tup)]) == parent_tup:
                        return v
                except:
                    pass
        except:
            pass
        # Single-branch tree broadcast
        if single_threshold is not None:
            return single_threshold
        # Global scalar
        return global_threshold

    # Pre-index points by path tuple for quick lookup
    points_by_path = {}
    all_points = []
    single_branch_points = None
    try:
        pts_paths = list(points_tree.Paths)
        for pp in pts_paths:
            try:
                br = list(points_tree.Branch(pp))
            except:
                br = []
            points_by_path[path_tuple(pp)] = br
            for it in br:
                all_points.append(it)
        if len(pts_paths) == 1:
            try:
                single_branch_points = list(points_tree.Branch(pts_paths[0]))
            except:
                single_branch_points = []
    except:
        points_by_path = {}
        all_points = []
        single_branch_points = None

    for cpath in grafted_curves_tree.Paths:
        result.EnsurePath(cpath)

        # Get curve (first valid in branch)
        curve = None
        try:
            items = list(grafted_curves_tree.Branch(cpath))
        except:
            items = []
        for it in items:
            try:
                c = coerce_to_curve(it)
                if c is not None and hasattr(c, 'IsValid') and c.IsValid:
                    curve = c
                    break
            except:
                pass
        if curve is None:
            continue

        # Resolve parent path to fetch candidate points, with robust fallbacks
        ptup = path_tuple(cpath)
        parent_tup = ptup[:-1] if len(ptup) > 0 else ptup

        candidates = []

        # 1) Exact parent path
        try:
            candidates = list(points_by_path.get(parent_tup, []))
        except:
            candidates = []

        # 2) If empty, try exact curve path tuple (some inputs may align on same depth)
        if not candidates:
            try:
                candidates = list(points_by_path.get(ptup, []))
            except:
                candidates = []

        # 3) If still empty, try any points whose path starts with parent prefix
        if not candidates and len(parent_tup) > 0:
            try:
                prefix_hits = []
                for k, vals in points_by_path.items():
                    try:
                        if len(k) >= len(parent_tup) and tuple(k[:len(parent_tup)]) == parent_tup:
                            if vals:
                                prefix_hits.extend(vals)
                    except:
                        pass
                candidates = prefix_hits
            except:
                candidates = []

        # 4) Single-branch broadcast
        if not candidates and single_branch_points is not None:
            candidates = list(single_branch_points)

        # 5) Global flatten as last resort
        if not candidates and all_points:
            candidates = list(all_points)

        if not candidates:
            # No available points – leave branch empty to preserve structure
            continue

        # Find nearest candidate by point-to-curve distance
        best_pt = None
        best_dist = float('inf')
        try:
            dom = curve.Domain
        except:
            dom = None

        for cand in candidates:
            try:
                p = to_point(cand)
                if p is None:
                    continue
                ok, t = curve.ClosestPoint(p)
                if not ok:
                    # Fallback: use domain midpoint when projection fails
                    t = curve.Domain.Mid if dom is not None else 0.5
                try:
                    if dom is not None:
                        if t < dom.T0:
                            t = dom.T0
                        elif t > dom.T1:
                            t = dom.T1
                except:
                    pass
                cp = curve.PointAt(t)
                d = p.DistanceTo(cp)
                if d < best_dist:
                    best_dist = d
                    best_pt = p
            except:
                pass

        if best_pt is not None:
            # Apply threshold filter when provided
            th = resolve_threshold_for_path(cpath)
            if th is not None:
                try:
                    if not (best_dist < th):
                        # keep branch empty (filtered out)
                        continue
                except:
                    pass
            try:
                result.Add(best_pt, cpath)
            except:
                pass

    return result

def extrude_solid(base_tree, direction_input, cap_ends=True):

    solids_tree = DataTree[rg.Brep]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    # Resolve direction: allow tuple/list, Vector3d, or per-branch
    is_dir_tree = is_tree(direction_input)
    global_dir = None
    try:
        if not is_dir_tree and direction_input is not None:
            if isinstance(direction_input, rg.Vector3d):
                global_dir = rg.Vector3d(direction_input)
            elif hasattr(direction_input, 'X') and hasattr(direction_input, 'Y') and hasattr(direction_input, 'Z'):
                global_dir = rg.Vector3d(direction_input.X, direction_input.Y, direction_input.Z)
            elif isinstance(direction_input, (tuple, list)) and len(direction_input) >= 3:
                global_dir = rg.Vector3d(float(direction_input[0]), float(direction_input[1]), float(direction_input[2]))
    except:
        global_dir = None

    def _sanitize_vector_inplace(v, eps=1e-12):
        try:
            if abs(v.X) < eps: v.X = 0.0
            if abs(v.Y) < eps: v.Y = 0.0
            if abs(v.Z) < eps: v.Z = 0.0
        except:
            pass

    def resolve_direction(path):
        if is_dir_tree and path in direction_input.Paths:
            try:
                items = list(direction_input.Branch(path))
                if len(items) > 0:
                    v = items[0]
                    if isinstance(v, rg.Vector3d):
                        vv = rg.Vector3d(v)
                        _sanitize_vector_inplace(vv)
                        return vv
                    if hasattr(v, 'X') and hasattr(v, 'Y') and hasattr(v, 'Z'):
                        vv = rg.Vector3d(v.X, v.Y, v.Z)
                        _sanitize_vector_inplace(vv)
                        return vv
                    if isinstance(v, (tuple, list)) and len(v) >= 3:
                        vv = rg.Vector3d(float(v[0]), float(v[1]), float(v[2]))
                        _sanitize_vector_inplace(vv)
                        return vv
            except:
                pass
        if global_dir is not None:
            vv = rg.Vector3d(global_dir)
            _sanitize_vector_inplace(vv)
            return vv
        return rg.Vector3d(0,0,1)

    def extrude_closed_curve_to_brep(crv, dir_vec):
        try:
            if crv is None or not crv.IsValid:
                return []
            # Ensure closed for solid
            c = crv
            try:
                if hasattr(c, 'IsClosed') and not c.IsClosed:
                    cc = c.ToNurbsCurve()
                    if cc:
                        cc.MakeClosed(sc.doc.ModelAbsoluteTolerance)
                        if cc.IsValid:
                            c = cc
            except:
                pass
            # Create extrusion using Surface.CreateExtrusion
            b = None
            try:
                srf = rg.Surface.CreateExtrusion(c, dir_vec)
                if srf is not None:
                    b = srf.ToBrep()
                    if b is not None and cap_ends:
                        b.CapPlanarHoles(sc.doc.ModelAbsoluteTolerance)
            except:
                pass
            
            if b is None:
                # Try Extrusion primitive then convert to Brep
                try:
                    ex = rg.Extrusion.Create(c, dir_vec.Length, cap_ends)
                    if ex is not None:
                        b = ex.ToBrep()
                except:
                    b = None
            if b is None:
                return []
            # Cap planar holes if any to ensure solid
            try:
                b.CapPlanarHoles(sc.doc.ModelAbsoluteTolerance)
            except:
                pass
            return [b] if b is not None and b.IsValid else []
        except Exception as e:
            print("Extrude curve to brep failed: {}".format(e))
            return []

    for path in base_tree.Paths:
        solids_tree.EnsurePath(path)
        dir_vec = resolve_direction(path)
        if dir_vec.IsZero:
            continue

        items = list(base_tree.Branch(path))
        for it in items:
            try:
                breps_out = []
                # If it's a curve directly
                if isinstance(it, rg.Curve):
                    breps_out = extrude_closed_curve_to_brep(it, dir_vec)
                else:
                    # Try to coerce to curve first
                    ctry = None
                    try:
                        ctry = coerce_to_curve(it)
                    except:
                        ctry = None
                    if ctry is not None and ctry.IsValid:
                        breps_out = extrude_closed_curve_to_brep(ctry, dir_vec)
                    else:
                        # If it's a Brep or Surface, collect closed outer loops by joining naked edges
                        b = None
                        s = None
                        if isinstance(it, rg.Brep):
                            b = it
                        elif isinstance(it, rg.Surface):
                            try:
                                b = it.ToBrep()
                            except:
                                b = None
                        if b is not None and b.IsValid:
                            try:
                                edges = b.DuplicateNakedEdgeCurves(True, False)
                                if edges and len(edges) > 0:
                                    joined = rg.Curve.JoinCurves(edges, sc.doc.ModelAbsoluteTolerance)
                                    if joined:
                                        for jc in joined:
                                            if jc is not None and jc.IsValid:
                                                breps_out.extend(extrude_closed_curve_to_brep(jc, dir_vec))
                                else:
                                    # If no naked edges (closed brep), try to use outer loop from first face
                                    try:
                                        if b.Faces.Count > 0:
                                            face = b.Faces[0]
                                            outer_loop = face.OuterLoop
                                            if outer_loop is not None:
                                                loop_curve = outer_loop.To3dCurve()
                                                if loop_curve is not None and loop_curve.IsValid:
                                                    breps_out.extend(extrude_closed_curve_to_brep(loop_curve, dir_vec))
                                    except Exception as e:
                                        print("Failed to extract outer loop: {}".format(e))
                                        pass
                            except Exception as e:
                                print("Failed to process Brep edges: {}".format(e))
                                pass

                for bo in breps_out:
                    if bo is not None and bo.IsValid:
                        solids_tree.Add(bo, path)
            except Exception as e:
                print("Extrude item failed for path {}: {}".format(path, e))
                pass

    return solids_tree

def offset_surface_solid(brep_tree, distance_input, tolerance=None):

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.01

    result_tree = DataTree[rg.Brep]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    # Resolve distance: allow scalar or per-branch first item
    is_dist_tree = is_tree(distance_input)
    global_distance = None
    if not is_dist_tree:
        try:
            global_distance = float(distance_input)
        except:
            global_distance = None

    def resolve_distance_for_path(path):
        if is_dist_tree and path in distance_input.Paths:
            try:
                items = list(distance_input.Branch(path))
                if len(items) > 0:
                    return float(items[0])
            except:
                pass
        if global_distance is not None:
            return float(global_distance)
        # Fallback: GH_Path(0) single value
        try:
            if is_dist_tree and len(list(distance_input.Paths)) == 1:
                p0 = list(distance_input.Paths)[0]
                vals = list(distance_input.Branch(p0))
                if len(vals) > 0:
                    return float(vals[0])
        except:
            pass
        return None

    def to_brep(obj):
        try:
            if isinstance(obj, rg.Brep):
                return obj
            if isinstance(obj, rg.Surface):
                return obj.ToBrep()
            if isinstance(obj, rg.BrepFace):
                return obj.DuplicateFace(True)
        except:
            pass
        return None

    for path in brep_tree.Paths:
        result_tree.EnsurePath(path)
        dist_val = resolve_distance_for_path(path)
        if dist_val is None:
            continue
        items = list(brep_tree.Branch(path))
        if not items:
            continue
        for it in items:
            try:
                b = to_brep(it)
                if b is None or not b.IsValid:
                    continue

                added_any = False

                # Prefer offsetting a face to make a capped solid (like GH Offset Surface Solid = True)
                try:
                    if b.Faces.Count > 0:
                        face = b.Faces[0]
                        created = rg.Brep.CreateFromOffsetFace(face, dist_val, tolerance, False, True)
                        # Handle return as single Brep or sequence defensively
                        if created is not None:
                            if isinstance(created, rg.Brep):
                                if created.IsValid:
                                    try:
                                        created.CapPlanarHoles(tolerance)
                                    except:
                                        pass
                                    result_tree.Add(created, path)
                                    added_any = True
                            else:
                                try:
                                    for nb in created:
                                        if nb is not None and nb.IsValid:
                                            try:
                                                nb.CapPlanarHoles(tolerance)
                                            except:
                                                pass
                                            result_tree.Add(nb, path)
                                            added_any = True
                                except:
                                    pass
                except:
                    pass

                # Fallback: offset whole brep
                if not added_any:
                    try:
                        offs = rg.Brep.CreateOffsetBrep(b, dist_val, True, True, tolerance)
                        if offs:
                            for nb in offs:
                                if nb is not None and nb.IsValid:
                                    try:
                                        nb.CapPlanarHoles(tolerance)
                                    except:
                                        pass
                                    result_tree.Add(nb, path)
                                    added_any = True
                    except:
                        pass

                # Last fallback: thicken by extruding boundary loops and capping
                if not added_any:
                    try:
                        # Build solids by extruding all face loops (outer + inner), then boolean-difference holes
                        loops_curves = []
                        try:
                            if b.Faces.Count > 0:
                                face = b.Faces[0]
                                for lp in face.Loops:
                                    try:
                                        c = lp.To3dCurve()
                                        if c is not None and c.IsValid:
                                            loops_curves.append(c)
                                    except:
                                        pass
                        except:
                            pass

                        if not loops_curves:
                            # Fallback: try naked edges join
                            edges = b.DuplicateNakedEdgeCurves(True, False)
                            loops_curves = list(rg.Curve.JoinCurves(edges, tolerance)) if edges else []

                        solids = []
                        for c in loops_curves:
                            srf = rg.Surface.CreateExtrusion(c, rg.Vector3d(0,0,dist_val))
                            if srf is not None:
                                nb = srf.ToBrep()
                                if nb is not None:
                                    try:
                                        nb.CapPlanarHoles(tolerance)
                                    except:
                                        pass
                                    if nb.IsValid:
                                        solids.append(nb)

                        if solids:
                            # Identify outer by max area of base curve
                            try:
                                areas = []
                                for c in loops_curves:
                                    am = rg.AreaMassProperties.Compute(c)
                                    areas.append(abs(am.Area) if am else 0.0)
                                outer_idx = max(range(len(areas)), key=lambda i: areas[i]) if areas else 0
                            except:
                                outer_idx = 0

                            outer = solids[outer_idx]
                            inners = [solids[i] for i in range(len(solids)) if i != outer_idx]
                            if inners:
                                try:
                                    diff = rg.Brep.CreateBooleanDifference(outer, inners, tolerance)
                                    if diff:
                                        for nb in diff:
                                            if nb is not None and nb.IsValid:
                                                result_tree.Add(nb, path)
                                                added_any = True
                                except:
                                    pass
                            if not added_any:
                                # As-is union of solids if boolean failed
                                for nb in solids:
                                    result_tree.Add(nb, path)
                                    added_any = True
                    except:
                        pass
            except:
                pass

    return result_tree
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


def amplitude(vectors_input=None, lengths=None, tolerance=None):
	"""
	Scale vectors to a target length, similar to Grasshopper's Amplitude component.

	Parameters:
		vectors_input: DataTree[rg.Vector3d] or vector-like or iterable of vector-like
		lengths: DataTree[float] or float or iterable of floats (broadcast rules apply)
		tolerance: optional, defaults to document tolerance

	Returns:
		DataTree[rg.Vector3d] with the same primary structure as the vectors input when it is a tree;
		otherwise falls back to the lengths tree; otherwise returns a single-path {0} tree.
	"""

	if tolerance is None:
		try:
			tolerance = sc.doc.ModelAbsoluteTolerance
		except:
			tolerance = 0.001

	result = DataTree[rg.Vector3d]()

	def is_tree(x):
		return hasattr(x, 'Paths') and hasattr(x, 'Branch')

	def to_vector(obj):
		try:
			if isinstance(obj, rg.Vector3d):
				return rg.Vector3d(obj)
			if hasattr(obj, 'X') and hasattr(obj, 'Y') and hasattr(obj, 'Z'):
				return rg.Vector3d(float(obj.X), float(obj.Y), float(obj.Z))
			if isinstance(obj, (tuple, list)) and len(obj) >= 3:
				return rg.Vector3d(float(obj[0]), float(obj[1]), float(obj[2]))
		except:
			pass
		return None

	def to_float(x):
		try:
			return float(x)
		except:
			return None

	def get_vectors_for_path(inp, path, expected_len):
		if is_tree(inp):
			items = list(inp.Branch(path)) if path in inp.Paths else []
			if len(items) == 0:
				return []
			if len(items) == expected_len:
				return [to_vector(v) for v in items]
			if len(items) == 1:
				v0 = to_vector(items[0])
				return [v0] * expected_len
			out = [to_vector(v) for v in items]
			if len(out) < expected_len and len(out) > 0:
				out = out + [out[-1]] * (expected_len - len(out))
			return out[:expected_len]
		# Non-tree
		if isinstance(inp, (list, tuple)):
			if len(inp) == expected_len:
				return [to_vector(v) for v in inp]
			if len(inp) >= 1:
				v0 = to_vector(inp[0])
				return [v0] * expected_len
		v0 = to_vector(inp)
		return [v0] * expected_len

	def get_lengths_for_path(inp, path, expected_len):
		if is_tree(inp):
			vals = list(inp.Branch(path)) if path in inp.Paths else []
			if len(vals) == 0:
				return []
			if len(vals) == expected_len:
				return [to_float(v) for v in vals]
			if len(vals) == 1:
				f0 = to_float(vals[0])
				return [f0] * expected_len
			out = [to_float(v) for v in vals]
			if len(out) < expected_len and len(out) > 0:
				out = out + [out[-1]] * (expected_len - len(out))
			return out[:expected_len]
		# Non-tree
		if isinstance(inp, (list, tuple)):
			if len(inp) == expected_len:
				return [to_float(v) for v in inp]
			if len(inp) >= 1:
				f0 = to_float(inp[0])
				return [f0] * expected_len
		f0 = to_float(inp)
		return [f0] * expected_len

	primary = vectors_input if is_tree(vectors_input) else (lengths if is_tree(lengths) else None)

	if primary is None:
		path0 = GH_Path(0)
		result.EnsurePath(path0)
		vecs = get_vectors_for_path(vectors_input, path0, 1)
		lens = get_lengths_for_path(lengths, path0, 1)
		if len(vecs) and len(lens):
			v = vecs[0]
			L = lens[0]
			if v is not None and L is not None:
				try:
					if v.IsZero:
						result.Add(rg.Vector3d(0.0, 0.0, 0.0), path0)
					else:
						u = rg.Vector3d(v)
						u.Unitize()
						u.X *= L; u.Y *= L; u.Z *= L
						result.Add(u, path0)
				except:
					pass
		return result

	for path in primary.Paths:
		result.EnsurePath(path)
		branch_len = len(list(primary.Branch(path)))
		vecs = get_vectors_for_path(vectors_input, path, branch_len)
		lens = get_lengths_for_path(lengths, path, branch_len)
		n = min(len(vecs), len(lens)) if branch_len == 0 else max(len(vecs), len(lens))
		if len(vecs) != n:
			vecs = (vecs + [vecs[-1]] * (n - len(vecs))) if vecs else [None] * n
		if len(lens) != n:
			lens = (lens + [lens[-1]] * (n - len(lens))) if lens else [None] * n
		if n == 0:
			continue
		try:
			for i in range(n):
				v = vecs[i]
				L = lens[i]
				if v is None or L is None:
					continue
				try:
					if v.IsZero:
						result.Add(rg.Vector3d(0.0, 0.0, 0.0), path)
					else:
						u = rg.Vector3d(v)
						u.Unitize()
						u.X *= L; u.Y *= L; u.Z *= L
						result.Add(u, path)
				except:
					pass
		except Exception as e:
			print("Amplitude failed for path {}: {}".format(path, e))
			pass

	return result


def move_points(points_tree, vectors_input, distances=None, tolerance=None):
	"""
	Move points by vectors, similar to Grasshopper's Move component (points-only).

	Parameters:
		points_tree: DataTree[rg.Point3d]
		vectors_input: DataTree[rg.Vector3d] or vector-like or iterable (direction or full displacement)
		distances: optional DataTree[float] or float; when provided, vectors are unitized and scaled by distance
		tolerance: optional, defaults to document tolerance

	Returns:
		DataTree[rg.Point3d] preserving the structure of points_tree.
	"""

	if tolerance is None:
		try:
			tolerance = sc.doc.ModelAbsoluteTolerance
		except:
			tolerance = 0.001

	result = DataTree[rg.Point3d]()

	def is_tree(x):
		return hasattr(x, 'Paths') and hasattr(x, 'Branch')

	def to_point(obj):
		try:
			if isinstance(obj, rg.Point3d):
				return rg.Point3d(obj)
			if hasattr(obj, 'X') and hasattr(obj, 'Y') and hasattr(obj, 'Z'):
				return rg.Point3d(float(obj.X), float(obj.Y), float(obj.Z))
			if isinstance(obj, (tuple, list)) and len(obj) >= 3:
				return rg.Point3d(float(obj[0]), float(obj[1]), float(obj[2]))
		except:
			pass
		return None

	def to_vector(obj):
		try:
			if isinstance(obj, rg.Vector3d):
				return rg.Vector3d(obj)
			if hasattr(obj, 'X') and hasattr(obj, 'Y') and hasattr(obj, 'Z'):
				return rg.Vector3d(float(obj.X), float(obj.Y), float(obj.Z))
			if isinstance(obj, (tuple, list)) and len(obj) >= 3:
				return rg.Vector3d(float(obj[0]), float(obj[1]), float(obj[2]))
		except:
			pass
		return None

	def to_float(x):
		try:
			return float(x)
		except:
			return None

	def resolve_vectors_for_path(path, count):
		dirs = []
		if is_tree(vectors_input) and path in vectors_input.Paths:
			items = list(vectors_input.Branch(path))
			for i in range(min(len(items), count)):
				dirs.append(to_vector(items[i]))
			while len(dirs) < count and len(items) > 0:
				dirs.append(to_vector(items[-1]))
		else:
			# Non-tree broadcast
			v0 = None
			try:
				if isinstance(vectors_input, (list, tuple)) and len(vectors_input) >= 1 and not hasattr(vectors_input, 'X'):
					# If a list/tuple was provided, prefer first item as direction
					v0 = to_vector(vectors_input[0])
				else:
					v0 = to_vector(vectors_input)
			except:
				v0 = to_vector(vectors_input)
			dirs = [v0] * count
		return dirs

	def resolve_distances_for_path(path, count):
		if distances is None:
			return [None] * count
		vals = []
		if is_tree(distances) and path in distances.Paths:
			items = list(distances.Branch(path))
			for i in range(min(len(items), count)):
				vals.append(to_float(items[i]))
			while len(vals) < count and len(items) > 0:
				vals.append(to_float(items[-1]))
		else:
			# Non-tree broadcast
			try:
				if isinstance(distances, (list, tuple)) and len(distances) >= 1:
					f0 = to_float(distances[0])
					vals = [f0] * count
				else:
					f0 = to_float(distances)
					vals = [f0] * count
			except:
				f0 = to_float(distances)
				vals = [f0] * count
		return vals

	for path in points_tree.Paths:
		result.EnsurePath(path)
		pts = list(points_tree.Branch(path))
		if not pts:
			continue
		count = len(pts)
		dirs = resolve_vectors_for_path(path, count)
		dists = resolve_distances_for_path(path, count)
		try:
			for i in range(count):
				pt = to_point(pts[i])
				v = dirs[i]
				if pt is None or v is None:
					continue
				disp = None
				if dists[i] is None:
					# Use input vector magnitude as displacement
					disp = rg.Vector3d(v)
				else:
					L = dists[i]
					if v.IsZero or L is None:
						continue
					u = rg.Vector3d(v)
					u.Unitize()
					u.X *= L; u.Y *= L; u.Z *= L
					disp = u
				try:
					new_pt = pt + disp
					result.Add(new_pt, path)
				except:
					pass
		except Exception as e:
			print("Move points failed for path {}: {}".format(path, e))
			pass

	return result


def vector_2pt(point_a_input, point_b_input, unitize=False, tolerance=None):
	"""
	Compute vector from Point A to Point B, similar to Grasshopper's Vector 2Pt.

	Parameters:
		point_a_input: DataTree[rg.Point3d] or point-like or iterable
		point_b_input: DataTree[rg.Point3d] or point-like or iterable
		unitize: optional bool or DataTree[bool]; when True, output vectors are unitized
		tolerance: optional, defaults to document tolerance

	Returns:
		(vector_tree: DataTree[rg.Vector3d], length_tree: DataTree[System.Double])
		Primary structure is preserved from point_b_input if it is a tree; otherwise from point_a_input; otherwise {0}.
	"""

	if tolerance is None:
		try:
			tolerance = sc.doc.ModelAbsoluteTolerance
		except:
			tolerance = 0.001

	vector_tree = DataTree[rg.Vector3d]()
	length_tree = DataTree[System.Double]()

	def is_tree(x):
		return hasattr(x, 'Paths') and hasattr(x, 'Branch')

	def to_point(obj):
		try:
			if isinstance(obj, rg.Point3d):
				return rg.Point3d(obj)
			if hasattr(obj, 'X') and hasattr(obj, 'Y') and hasattr(obj, 'Z'):
				return rg.Point3d(float(obj.X), float(obj.Y), float(obj.Z))
			if isinstance(obj, (tuple, list)) and len(obj) >= 3:
				return rg.Point3d(float(obj[0]), float(obj[1]), float(obj[2]))
		except:
			pass
		return None

	def to_bool(x):
		try:
			if isinstance(x, bool):
				return x
			return bool(int(x))
		except:
			return False

	def resolve_points_for_path(inp, path, count):
		pts = []
		if is_tree(inp) and path in inp.Paths:
			items = list(inp.Branch(path))
			for i in range(min(len(items), count)):
				pts.append(to_point(items[i]))
			while len(pts) < count and len(items) > 0:
				pts.append(to_point(items[-1]))
		else:
			p0 = None
			try:
				if isinstance(inp, (list, tuple)) and len(inp) >= 1 and not hasattr(inp, 'X'):
					p0 = to_point(inp[0])
				else:
					p0 = to_point(inp)
			except:
				p0 = to_point(inp)
			pts = [p0] * count
		return pts

	def resolve_unitize_for_path(inp, path, count):
		if is_tree(inp) and path in inp.Paths:
			items = list(inp.Branch(path))
			if len(items) == 0:
				return [False] * count
			if len(items) == 1:
				return [to_bool(items[0])] * count
			vals = [to_bool(x) for x in items]
			if len(vals) < count:
				vals = vals + [vals[-1]] * (count - len(vals))
			return vals[:count]
		# Non-tree
		return [to_bool(inp)] * count

	primary = point_b_input if is_tree(point_b_input) else (point_a_input if is_tree(point_a_input) else None)

	if primary is None:
		path0 = GH_Path(0)
		vector_tree.EnsurePath(path0)
		length_tree.EnsurePath(path0)
		pa = resolve_points_for_path(point_a_input, path0, 1)[0]
		pb = resolve_points_for_path(point_b_input, path0, 1)[0]
		flag = resolve_unitize_for_path(unitize, path0, 1)[0]
		if pa is not None and pb is not None:
			v = pb - pa
			try:
				L = v.Length
				x = rg.Vector3d(v)
				if flag and not x.IsZero:
					x.Unitize()
				vector_tree.Add(x, path0)
				length_tree.Add(System.Double(float(L)), path0)
			except:
				pass
		return vector_tree, length_tree

	for path in primary.Paths:
		vector_tree.EnsurePath(path)
		length_tree.EnsurePath(path)
		count = len(list(primary.Branch(path)))
		As = resolve_points_for_path(point_a_input, path, count)
		Bs = resolve_points_for_path(point_b_input, path, count)
		flags = resolve_unitize_for_path(unitize, path, count)
		try:
			for i in range(count):
				pa = As[i]; pb = Bs[i]
				if pa is None or pb is None:
					continue
				v = pb - pa
				L = v.Length
				x = rg.Vector3d(v)
				if flags[i] and not x.IsZero:
					x.Unitize()
				vector_tree.Add(x, path)
				length_tree.Add(System.Double(float(L)), path)
		except Exception as e:
			print("Vector 2Pt failed for path {}: {}".format(path, e))
			pass

	return vector_tree, length_tree


def move_geometry(geometry_tree, motion_input):
	"""
	Move geometry by a vector or transform, similar to Grasshopper's Move.

	Parameters:
		geometry_tree: DataTree[object]
		motion_input: DataTree[rg.Vector3d] or rg.Vector3d or rg.Transform or iterables (broadcast rules apply)

	Returns:
		(moved_tree: DataTree[object], xform_tree: DataTree[rg.Transform])
		Primary structure is preserved from geometry_tree.
	"""

	moved_tree = DataTree[object]()
	xform_tree = DataTree[rg.Transform]()

	def is_tree(x):
		return hasattr(x, 'Paths') and hasattr(x, 'Branch')

	def to_vector(obj):
		try:
			if isinstance(obj, rg.Vector3d):
				return rg.Vector3d(obj)
			if hasattr(obj, 'X') and hasattr(obj, 'Y') and hasattr(obj, 'Z'):
				return rg.Vector3d(float(obj.X), float(obj.Y), float(obj.Z))
			if isinstance(obj, (tuple, list)) and len(obj) >= 3:
				return rg.Vector3d(float(obj[0]), float(obj[1]), float(obj[2]))
		except:
			pass
		return None

	def to_transform(obj):
		if isinstance(obj, rg.Transform):
			return rg.Transform(obj)
		v = to_vector(obj)
		if v is not None:
			try:
				xf = rg.Transform.Translation(v)
				return xf
			except:
				return None
		return None

	def duplicate_geometry(g):
		try:
			if isinstance(g, rg.Point3d):
				return rg.Point3d(g)
			if hasattr(g, 'Duplicate'):
				return g.Duplicate()
			if isinstance(g, rg.Curve):
				return g.DuplicateCurve()
			if isinstance(g, rg.Brep):
				return g.DuplicateBrep()
			if isinstance(g, rg.Mesh):
				return g.DuplicateMesh()
		except:
			pass
		return None

	def apply_xform(g, xf):
		if isinstance(g, rg.Point3d):
			try:
				pt = rg.Point3d(g)
				pt.Transform(xf)
				return pt
			except:
				return None
		try:
			dup = duplicate_geometry(g)
			geom = dup if dup is not None else g
			ok = geom.Transform(xf)
			return geom if ok is None or ok else geom
		except:
			try:
				# Fallback: translate when possible
				v = rg.Vector3d(xf.M03, xf.M13, xf.M23)
				if hasattr(g, 'Translate'):
					geom = duplicate_geometry(g) or g
					geom.Translate(v)
					return geom
			except:
				return None

	def resolve_motions_for_path(path, count):
		xforms = []
		if is_tree(motion_input) and path in motion_input.Paths:
			items = list(motion_input.Branch(path))
			for i in range(min(len(items), count)):
				xforms.append(to_transform(items[i]))
			while len(xforms) < count and len(items) > 0:
				xforms.append(to_transform(items[-1]))
		else:
			# Non-tree broadcast
			try:
				if isinstance(motion_input, (list, tuple)) and len(motion_input) >= 1 and not isinstance(motion_input, rg.Transform) and not hasattr(motion_input, 'M00'):
					x0 = to_transform(motion_input[0])
				else:
					x0 = to_transform(motion_input)
			except:
				x0 = to_transform(motion_input)
			xforms = [x0] * count
		return xforms

	for path in geometry_tree.Paths:
		moved_tree.EnsurePath(path)
		xform_tree.EnsurePath(path)
		geoms = list(geometry_tree.Branch(path))
		if not geoms:
			continue
		count = len(geoms)
		xforms = resolve_motions_for_path(path, count)
		try:
			for i in range(count):
				g = geoms[i]
				xf = xforms[i]
				if g is None or xf is None:
					continue
				mg = apply_xform(g, xf)
				if mg is not None:
					moved_tree.Add(mg, path)
					xform_tree.Add(xf, path)
		except Exception as e:
			print("Move geometry failed for path {}: {}".format(path, e))
			pass

	return moved_tree, xform_tree


def to_boolean(values_input):
	"""
	Convert numbers to booleans per item, following DataTree rules.

	- 0 -> False, non-zero -> True
	- Preserves input DataTree structure; keeps empty branches
	- Accepts scalar, list/tuple, or DataTree
	"""

	result = DataTree[bool]()

	def is_tree(x):
		return hasattr(x, 'Paths') and hasattr(x, 'Branch')

	def num_to_bool(x):
		try:
			if isinstance(x, bool):
				return x
			v = float(x)
			return (v != 0.0)
		except:
			return False

	# Non-tree: produce {0}
	if not is_tree(values_input):
		path0 = GH_Path(0)
		result.EnsurePath(path0)
		try:
			if isinstance(values_input, (list, tuple)):
				for it in values_input:
					result.Add(bool(num_to_bool(it)), path0)
			else:
				result.Add(bool(num_to_bool(values_input)), path0)
		except:
			pass
		return result

	# Tree input: preserve structure
	for path in values_input.Paths:
		result.EnsurePath(path)
		items = []
		try:
			items = list(values_input.Branch(path))
		except:
			items = []
		if not items:
			continue
		try:
			for it in items:
				b = num_to_bool(it)
				result.Add(bool(b), path)
		except Exception as e:
			print("to_boolean failed for path {}: {}".format(path, e))
			pass

	return result

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
    
    # Support scalar broadcasting when treeB is a number or vector-like
    is_treeB_scalar = not hasattr(treeB, 'Paths')
    scalar_b = treeB if is_treeB_scalar else None

    for path in treeA.Paths:
        result_tree.EnsurePath(path)
        a_items = list(treeA.Branch(path))
        b_items = ([scalar_b] * len(a_items)) if is_treeB_scalar else (list(treeB.Branch(path)) if path in treeB.Paths else [])
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

def division(treeA, treeB):

    def is_number(x):
        try:
            if isinstance(x, (int, float, System.Double)):
                return True
        except:
            pass
        return False

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

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

    is_tree_a = is_tree(treeA)
    is_tree_b = is_tree(treeB)

    # Both scalars → default path {0}
    if not is_tree_a and not is_tree_b:
        path0 = GH_Path(0)
        result_tree.EnsurePath(path0)
        try:
            a = treeA; b = treeB
            if is_number(a) and is_number(b):
                denom = float(b)
                if denom != 0.0:
                    result_tree.Add(System.Double(float(a)/denom), path0)
            else:
                va = to_vector(a)
                if va is not None and is_number(b):
                    denom = float(b)
                    if denom != 0.0:
                        s = 1.0/denom
                        vec = rg.Vector3d(va); vec.X *= s; vec.Y *= s; vec.Z *= s
                        result_tree.Add(vec, path0)
        except:
            pass
        return result_tree

    # Collect all paths
    all_paths = set()
    if is_tree_a:
        for p in treeA.Paths:
            all_paths.add(p)
    if is_tree_b:
        for p in treeB.Paths:
            all_paths.add(p)

    # Global broadcast values (when one side is scalar)
    global_a = None
    global_b = None
    if not is_tree_a:
        global_a = treeA
    if not is_tree_b:
        global_b = treeB

    def path_key(p):
        try:
            return tuple(p.Indices)
        except:
            return (0,)

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

        if len(a_items) == 0 and global_a is not None and len(b_items) > 0:
            a_items = [global_a] * len(b_items)
        if len(b_items) == 0 and global_b is not None and len(a_items) > 0:
            b_items = [global_b] * len(a_items)

        if len(a_items) == 0 or len(b_items) == 0:
            continue

        # Per-path broadcasting when one branch has a single value
        if len(a_items) == 1 and len(b_items) > 1:
            a_items = [a_items[0]] * len(b_items)
        if len(b_items) == 1 and len(a_items) > 1:
            b_items = [b_items[0]] * len(a_items)

        n = min(len(a_items), len(b_items))
        try:
            for i in range(n):
                a = a_items[i]; b = b_items[i]
                # number / number
                if is_number(a) and is_number(b):
                    try:
                        denom = float(b)
                        if denom == 0.0:
                            continue
                        result_tree.Add(System.Double(float(a)/denom), path)
                    except:
                        pass
                    continue
                # vector / number
                va = to_vector(a)
                if va is not None and is_number(b):
                    try:
                        denom = float(b)
                        if denom == 0.0:
                            continue
                        s = 1.0/denom
                        vec = rg.Vector3d(va); vec.X *= s; vec.Y *= s; vec.Z *= s
                        result_tree.Add(vec, path)
                    except:
                        pass
                    continue
                # Unsupported combos: skip
        except Exception as e:
            print("Division failed for path {}: {}".format(path, e))
            pass

    return result_tree

def equality(treeA, treeB):

    def is_number(x):
        try:
            float(x)
            return True
        except:
            return False

    result_tree = DataTree[bool]()

    is_tree_a = is_tree(treeA)
    is_tree_b = is_tree(treeB)

    # Special case: treeB is a single scalar number, treeA is a tree
    # Output structure matches treeA (emit one bool per item, non-numbers -> False)
    if is_tree_a and not is_tree_b and is_number(treeB):
        try:
            b_scalar = float(treeB)
            for path in treeA.Paths:
                result_tree.EnsurePath(path)
                a_items = list(treeA.Branch(path))
                for a_item in a_items:
                    try:
                        if is_number(a_item):
                            result_tree.Add(bool(abs(float(a_item) - b_scalar) < 1e-10), path)
                        else:
                            result_tree.Add(False, path)
                    except:
                        result_tree.Add(False, path)
            return result_tree
        except:
            return result_tree
    
    # Special case: treeA is a single scalar number, treeB is a tree
    # Output structure matches treeB (emit one bool per item, non-numbers -> False)
    if is_tree_b and not is_tree_a and is_number(treeA):
        try:
            a_scalar = float(treeA)
            for path in treeB.Paths:
                result_tree.EnsurePath(path)
                b_items = list(treeB.Branch(path))
                for b_item in b_items:
                    try:
                        if is_number(b_item):
                            result_tree.Add(bool(abs(a_scalar - float(b_item)) < 1e-10), path)
                        else:
                            result_tree.Add(False, path)
                    except:
                        result_tree.Add(False, path)
            return result_tree
        except:
            return result_tree

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

    # Also support broadcasting when input is a single-branch single-item numeric DataTree
    singleton_a = None
    singleton_b = None
    if is_tree_a:
        try:
            paths_a = list(treeA.Paths)
            if len(paths_a) == 1:
                br = list(treeA.Branch(paths_a[0]))
                if len(br) == 1 and is_number(br[0]):
                    singleton_a = float(br[0])
        except:
            singleton_a = None
    if is_tree_b:
        try:
            paths_b = list(treeB.Paths)
            if len(paths_b) == 1:
                br = list(treeB.Branch(paths_b[0]))
                if len(br) == 1 and is_number(br[0]):
                    singleton_b = float(br[0])
        except:
            singleton_b = None

    for path in sorted(all_paths, key=path_key):
        result_tree.EnsurePath(path)

        a_items = []
        b_items = []

        if is_tree_a and path in treeA.Paths:
            a_items = list(treeA.Branch(path))
        elif singleton_a is not None:
            a_items = [singleton_a]
        elif global_a is not None:
            a_items = [global_a]

        if is_tree_b and path in treeB.Paths:
            b_items = list(treeB.Branch(path))
        elif singleton_b is not None:
            b_items = [singleton_b]
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

def subtraction(treeA, treeB):

    def is_number(x):
        try:
            float(x)
            return True
        except:
            return False

    result_tree = DataTree[System.Double]()

    is_tree_a = is_tree(treeA)
    is_tree_b = is_tree(treeB)

    # Scalar - scalar
    if (not is_tree_a) and (not is_tree_b):
        try:
            path0 = GH_Path(0)
            result_tree.EnsurePath(path0)
            if is_number(treeA) and is_number(treeB):
                result_tree.Add(System.Double(float(treeA) - float(treeB)), path0)
            return result_tree
        except:
            return result_tree

    # Tree vs scalar → preserve tree structure (A - b)
    if is_tree_a and (not is_tree_b) and is_number(treeB):
        try:
            b_scalar = float(treeB)
            for path in treeA.Paths:
                result_tree.EnsurePath(path)
                a_items = list(treeA.Branch(path))
                for a_item in a_items:
                    try:
                        if is_number(a_item):
                            result_tree.Add(System.Double(float(a_item) - b_scalar), path)
                    except:
                        pass
            return result_tree
        except:
            return result_tree

    # Scalar vs tree → preserve tree structure (a - B)
    if is_tree_b and (not is_tree_a) and is_number(treeA):
        try:
            a_scalar = float(treeA)
            for path in treeB.Paths:
                result_tree.EnsurePath(path)
                b_items = list(treeB.Branch(path))
                for b_item in b_items:
                    try:
                        if is_number(b_item):
                            result_tree.Add(System.Double(a_scalar - float(b_item)), path)
                    except:
                        pass
            return result_tree
        except:
            return result_tree

    # Tree vs tree (path-wise)
    all_paths = set()
    if is_tree_a:
        for p in treeA.Paths:
            all_paths.add(p)
    if is_tree_b:
        for p in treeB.Paths:
            all_paths.add(p)

    def path_key(p):
        try:
            return tuple(p.Indices)
        except:
            return (0,)

    for path in sorted(all_paths, key=path_key):
        result_tree.EnsurePath(path)

        a_items = []
        b_items = []

        if is_tree_a and path in treeA.Paths:
            a_items = list(treeA.Branch(path))
        if is_tree_b and path in treeB.Paths:
            b_items = list(treeB.Branch(path))

        # Per-path broadcasting: if one side single, broadcast to the other
        if len(a_items) == 1 and len(b_items) > 1 and is_number(a_items[0]):
            a_items = [float(a_items[0])] * len(b_items)
        if len(b_items) == 1 and len(a_items) > 1 and is_number(b_items[0]):
            b_items = [float(b_items[0])] * len(a_items)

        n = min(len(a_items), len(b_items))
        for i in range(n):
            try:
                a = a_items[i]
                b = b_items[i]
                if is_number(a) and is_number(b):
                    result_tree.Add(System.Double(float(a) - float(b)), path)
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

        # If one branch is empty but the other has items, fill the empty side with None to match counts
        if len(items0) == 0 and len(items1) > 0:
            items0 = [None] * len(items1)
        if len(items1) == 0 and len(items0) > 0:
            items1 = [None] * len(items0)

        # Selection pattern broadcasting
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

                    a = items0[i] if i < len(items0) else None
                    b = items1[i] if i < len(items1) else None
                    choice = b if sel else a

                    # Only add non-null; if both null, branch remains empty
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

    # If guide_tree is not a DataTree, place all items under a default path {0}
    try:
        if not is_tree(guide_tree):
            path0 = GH_Path(0)
            result_tree.EnsurePath(path0)
            for it in flat_items:
                result_tree.Add(it, path0)
            return result_tree
    except:
        # Fallback: still try to return flat items under {0}
        path0 = GH_Path(0)
        result_tree.EnsurePath(path0)
        for it in flat_items:
            result_tree.Add(it, path0)
        return result_tree

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

def simplify_tree(tree):
    """
    Remove the longest common prefix AND suffix from all branch paths,
    preserving order and empty branches.

    Replaces Grasshopper's Simplify Tree.

    Parameters:
        tree: DataTree[Any]

    Returns:
        DataTree[object]: New tree with simplified paths.
    """

    result = DataTree[object]()
    try:
        paths = list(tree.Paths)
    except:
        return result

    if not paths:
        return result

    idx_lists = []
    try:
        for p in paths:
            idx_lists.append(list(p.Indices))
    except:
        idx_lists = [list(getattr(p, 'Indices', [])) for p in paths]

    min_len = min((len(l) for l in idx_lists)) if idx_lists else 0
    common_prefix_len = 0

    for i in range(min_len):
        try:
            val = idx_lists[0][i]
            if all((len(l) > i and l[i] == val) for l in idx_lists[1:]):
                common_prefix_len += 1
            else:
                break
        except:
            break

    common_suffix_len = 0
    while common_suffix_len < max(0, min_len - common_prefix_len):
        try:
            val = idx_lists[0][-1 - common_suffix_len]
            # Ensure we don't overlap prefix and suffix
            if all((len(l) > common_prefix_len + common_suffix_len and l[-1 - common_suffix_len] == val) for l in idx_lists[1:]):
                common_suffix_len += 1
            else:
                break
        except:
            break

    for i, p in enumerate(paths):
        try:
            lst = idx_lists[i]
            end = len(lst) - common_suffix_len if common_suffix_len > 0 else len(lst)
            idxs = lst[common_prefix_len:end]
            new_path = GH_Path(*idxs) if idxs else GH_Path(0)
            result.EnsurePath(new_path)
            for item in list(tree.Branch(p)):
                result.Add(item, new_path)
        except:
            pass

    return result

def curve_length(curves_input):

    # Handle single curve object
    if not is_tree(curves_input):
        try:
            curve = coerce_to_curve(curves_input)
            if curve and curve.IsValid:
                return System.Double(curve.GetLength())
            return System.Double(0.0)
        except:
            return System.Double(0.0)
    
    # Handle DataTree
    length_tree = DataTree[System.Double]()
    
    for path in curves_input.Paths:
        length_tree.EnsurePath(path)
        items = list(curves_input.Branch(path))
        
        for item in items:
            try:
                curve = coerce_to_curve(item)
                if curve and curve.IsValid:
                    length = curve.GetLength()
                    length_tree.Add(System.Double(length), path)
            except:
                pass
    
    return length_tree

def sort_list(keys_tree, values_tree=None, reverse=False):

    sorted_keys = DataTree[object]()
    sorted_values = DataTree[object]() if values_tree is not None else None

    def sort_key(v):
        try:
            return (0, float(v))
        except:
            try:
                return (1, str(v))
            except:
                return (2, 0)

    def add_item(tree, item, path):
        try:
            if isinstance(item, (int, float, System.Double)):
                tree.Add(System.Double(float(item)), path)
            else:
                tree.Add(item, path)
        except:
            try:
                tree.Add(item, path)
            except:
                pass

    # Detect values single-branch broadcasting case
    use_values_broadcast = False
    values_single_path = None
    values_single_items = []
    if values_tree is not None:
        try:
            vpaths = list(values_tree.Paths)
            if len(vpaths) == 1:
                values_single_path = vpaths[0]
                values_single_items = list(values_tree.Branch(values_single_path))
                # Verify every non-empty keys branch count matches
                counts = []
                for p in keys_tree.Paths:
                    try:
                        k = list(keys_tree.Branch(p))
                        if k:
                            counts.append(len(k))
                    except:
                        pass
                if counts and all(c == len(values_single_items) for c in counts):
                    use_values_broadcast = True
        except:
            use_values_broadcast = False

    # Precompute: path tuple helper
    def path_tuple(p):
        try:
            return tuple(getattr(p, 'Indices', []))
        except:
            return tuple()

    # Precompute: keys counts per branch and grouping by parent
    key_counts = {}
    parent_to_children = {}
    try:
        for kp in keys_tree.Paths:
            kt = path_tuple(kp)
            try:
                k_items = list(keys_tree.Branch(kp))
                key_counts[kt] = len(k_items)
            except:
                key_counts[kt] = 0
            parent = kt[:-1] if len(kt) > 0 else None
            parent_to_children.setdefault(parent, []).append(kt)
        # Sort children by their last index to keep deterministic slice order
        for par, childs in parent_to_children.items():
            try:
                childs.sort(key=lambda t: t[-1] if t else -1)
            except:
                pass
    except:
        key_counts = {}
        parent_to_children = {}

    # Precompute: values by exact path tuple
    values_by_path = {}
    if values_tree is not None:
        try:
            for vp in values_tree.Paths:
                values_by_path[path_tuple(vp)] = list(values_tree.Branch(vp))
        except:
            values_by_path = {}

    # Allocate values from parent branches to children by slicing using keys counts
    allocated_values = {}
    if values_tree is not None and parent_to_children:
        for parent, childs in parent_to_children.items():
            if parent is None:
                continue
            try:
                parent_vals = values_by_path.get(parent, [])
                if not parent_vals:
                    continue
                total_needed = 0
                try:
                    for ch in childs:
                        total_needed += int(key_counts.get(ch, 0))
                except:
                    total_needed = sum(key_counts.get(ch, 0) for ch in childs)
                if total_needed <= 0:
                    continue
                if len(parent_vals) != total_needed:
                    # Not an exact match; skip slicing for this parent
                    continue
                offset = 0
                for ch in childs:
                    cnt = int(key_counts.get(ch, 0))
                    if cnt <= 0:
                        allocated_values[ch] = []
                        continue
                    allocated_values[ch] = parent_vals[offset:offset + cnt]
                    offset += cnt
            except:
                pass

    for path in keys_tree.Paths:
        sorted_keys.EnsurePath(path)
        if sorted_values is not None:
            sorted_values.EnsurePath(path)

        keys = list(keys_tree.Branch(path))
        if not keys:
            continue

        idx = list(range(len(keys)))
        try:
            idx.sort(key=lambda i: sort_key(keys[i]), reverse=bool(reverse))
        except:
            # Fallback: leave order unchanged
            pass

        # Rebuild keys
        for i in idx:
            add_item(sorted_keys, keys[i], path)

        # Rebuild values if provided and counts match
        if values_tree is not None:
            try:
                vals = []
                key_t = path_tuple(path)

                # Priority 1: exact same-path values
                cand = values_by_path.get(key_t, [])
                if len(cand) == len(keys):
                    vals = cand

                # Priority 2: allocated slice from parent values
                if not vals:
                    cand2 = allocated_values.get(key_t, [])
                    if len(cand2) == len(keys):
                        vals = cand2

                # Priority 3: global single-branch broadcast when counts match across branches
                if (not vals) and use_values_broadcast and len(values_single_items) == len(keys):
                    vals = values_single_items
            except:
                vals = []
            if len(vals) == len(keys):
                for i in idx:
                    add_item(sorted_values, vals[i], path)
            else:
                # Ensure path exists; leave empty when mismatch
                pass

    if values_tree is None:
        return sorted_keys
    return sorted_keys, sorted_values


def geometry_linear_sequencing_mapper(base_curves_tree, outlines_tree, tolerance=None):

    """
    Reorder inner row outlines per branch by the linear coordinate on a base line,
    with distance as a tie-breaker. Structure is preserved; empty branches kept.

    Inputs per branch:
        - base_curves_tree: one valid curve in each branch (used as linear base)
        - outlines_tree: multiple closed curves/breps per branch to be ordered

    Returns:
        (sorted_outlines_tree, sorted_linear_tree, sorted_distance_tree)
    """

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.001

    # 1) Centroids of outlines
    try:
        centroid_tree = area(outlines_tree)[1]
    except:
        centroid_tree = DataTree[rg.Point3d]()

    # 2) Closest along-length and distances to base curve
    try:
        closest_pts, length_params, distances = curve_closest_point(centroid_tree, base_curves_tree, tolerance)
    except:
        # Fallback: empty trees matching outlines structure
        length_params = DataTree[System.Double]()
        distances = DataTree[System.Double]()

    # 3) Per-branch composite-key sort (length_param primary, distance secondary)
    sorted_outlines = DataTree[object]()
    sorted_linear_tree = DataTree[System.Double]()
    sorted_distance_tree = DataTree[System.Double]()

    def path_key(p):
        try:
            return tuple(p.Indices)
        except:
            return (0,)

    for path in sorted(outlines_tree.Paths, key=path_key):
        sorted_outlines.EnsurePath(path)
        sorted_linear_tree.EnsurePath(path)
        sorted_distance_tree.EnsurePath(path)

        items = list(outlines_tree.Branch(path))
        if not items:
            continue

        lens = list(length_params.Branch(path)) if path in length_params.Paths else []
        dists = list(distances.Branch(path)) if path in distances.Paths else []

        # Align counts; if mismatch, fallback to unsorted
        if len(lens) != len(items) or len(dists) != len(items):
            for it in items:
                sorted_outlines.Add(it, path)
            continue

        triples = []
        for i in range(len(items)):
            try:
                lval = float(lens[i])
            except:
                try:
                    lval = float(getattr(lens[i], 'ToString', lambda: '0')())
                except:
                    lval = 0.0
            try:
                dval = float(dists[i])
            except:
                try:
                    dval = float(getattr(dists[i], 'ToString', lambda: '0')())
                except:
                    dval = 0.0
            triples.append((lval, dval, items[i]))

        triples.sort(key=lambda t: (t[0], t[1]))

        for lval, dval, it in triples:
            try:
                sorted_outlines.Add(it, path)
            except:
                pass
            try:
                sorted_linear_tree.Add(System.Double(float(lval)), path)
            except:
                pass
            try:
                sorted_distance_tree.Add(System.Double(float(dval)), path)
            except:
                pass

    return sorted_outlines, sorted_linear_tree, sorted_distance_tree

def list_item(list_tree, index_input, wrap=True):

    result_tree = DataTree[object]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    # Helper to fetch one item with wrap/clamp
    def get_item(items, idx):
        try:
            i = int(idx)
            n = len(items)
            if n == 0:
                return None
            if wrap:
                i = i % n if n > 0 else 0
            else:
                i = max(0, min(i, n - 1))
            return items[i]
        except:
            return None

    # Helper: if branch只有一个元素且该元素本身是可迭代序列（list/tuple/.NET List），将其视为本地列表参与取项
    def maybe_unwrap_single_list(items):
        try:
            if len(items) == 1:
                it0 = items[0]
                # Python list/tuple
                if isinstance(it0, (list, tuple)):
                    return list(it0)
                # .NET IEnumerable (exclude strings)
                try:
                    from System.Collections import IEnumerable
                    from System import String as NetString
                    if isinstance(it0, IEnumerable) and not isinstance(it0, NetString):
                        return [x for x in it0]
                except:
                    pass
        except:
            pass
        return items

    # Case A: list_tree is a DataTree (original behavior, extended to accept list/tuple indices too)
    if is_tree(list_tree):
        # Detect if index_input is a single scalar
        is_global_index = False
        global_index_value = None
        try:
            if not is_tree(index_input):
                # Scalar or list-like
                if isinstance(index_input, (list, tuple)):
                    is_global_index = False
                else:
                    global_index_value = int(index_input)
                    is_global_index = True
        except:
            pass

        for path in list_tree.Paths:
            result_tree.EnsurePath(path)
            items = list(list_tree.Branch(path))
            # unwrap "single item that is itself a list" → behave like GH List Item on list
            items = maybe_unwrap_single_list(items)
            if not items:
                continue

            if is_global_index:
                val = get_item(items, global_index_value)
                if val is not None:
                    result_tree.Add(val, path)
            else:
                # indices could be a DataTree, list/tuple, or scalar fallback
                indices = []
                if is_tree(index_input):
                    indices = list(index_input.Branch(path)) if path in index_input.Paths else []
                    if len(indices) == 0 and len(list(index_input.Paths)) == 1:
                        p0 = list(index_input.Paths)[0]
                        indices = list(index_input.Branch(p0))
                else:
                    if isinstance(index_input, (list, tuple)):
                        indices = list(index_input)
                    else:
                        indices = [index_input]

                if len(indices) == 1:
                    val = get_item(items, indices[0])
                    if val is not None:
                        result_tree.Add(val, path)
                else:
                    for idx in indices:
                        val = get_item(items, idx)
                        if val is not None:
                            result_tree.Add(val, path)

        return result_tree

    # Case B: list_tree is a plain list/tuple/scalar (Grasshopper 'list' input)
    items = []
    try:
        if isinstance(list_tree, (list, tuple)):
            items = list(list_tree)
        else:
            items = [list_tree]
    except:
        items = [list_tree]
    # Unwrap list payload if the only element is a nested list
    items = maybe_unwrap_single_list(items)

    path0 = GH_Path(0)
    result_tree.EnsurePath(path0)

    # Resolve indices for list input
    indices = []
    if is_tree(index_input):
        # Prefer first (or only) branch
        try:
            p_used = None
            if len(list(index_input.Paths)) > 0:
                p_used = list(index_input.Paths)[0]
            if p_used is not None:
                indices = list(index_input.Branch(p_used))
        except:
            indices = []
    elif isinstance(index_input, (list, tuple)):
        indices = list(index_input)
    else:
        indices = [index_input]

    if len(indices) == 1:
        val = get_item(items, indices[0])
        if val is not None:
            result_tree.Add(val, path0)
    else:
        for idx in indices:
            val = get_item(items, idx)
            if val is not None:
                result_tree.Add(val, path0)

    return result_tree

def split_list(list_tree, index_tree, clamp=True):
    """
    Split each branch of a list DataTree at a per-branch index. Replaces Grasshopper's Split List.

    Parameters:
        list_tree: DataTree[object] – Input items to split per branch
        index_tree: DataTree[int|float] – Per-branch split index; must have a matching branch. If a branch is missing or index_tree is not a tree, the split is not executed for that branch
        clamp: bool – Clamp index to [0, len] when true; otherwise skip when out of range

    Returns:
        tuple(DataTree[object], DataTree[object]): (before_tree, after_tree), paths mirror list_tree, empty branches preserved
    """
    before_tree = DataTree[object]()
    after_tree = DataTree[object]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    if not is_tree(index_tree):
        for path in list_tree.Paths:
            before_tree.EnsurePath(path)
            after_tree.EnsurePath(path)
        return before_tree, after_tree

    for path in list_tree.Paths:
        before_tree.EnsurePath(path)
        after_tree.EnsurePath(path)

        items = list(list_tree.Branch(path))
        if not items:
            continue

        if path not in index_tree.Paths:
            continue

        indices = list(index_tree.Branch(path))
        if len(indices) == 0:
            continue

        try:
            idx = int(float(indices[0]))
        except:
            continue

        n = len(items)
        if clamp:
            if idx < 0:
                idx = 0
            if idx > n:
                idx = n
        else:
            if idx < 0 or idx > n:
                continue

        for i in range(0, idx):
            before_tree.Add(items[i], path)
        for i in range(idx, n):
            after_tree.Add(items[i], path)

    return before_tree, after_tree

def brep_edges(breps_tree):

    naked_tree = DataTree[rg.Curve]()
    interior_tree = DataTree[rg.Curve]()
    nonmanifold_tree = DataTree[rg.Curve]()

    def classify_valence(valence):
        try:
            name = str(valence)
        except:
            name = ""
        name_lower = name.lower()
        if ("naked" in name_lower) or ("boundary" in name_lower):
            return "naked"
        if "interior" in name_lower:
            return "interior"
        if ("nonmanifold" in name_lower) or ("non-manifold" in name_lower):
            return "nonmanifold"
        return "interior"

    for path in breps_tree.Paths:
        naked_tree.EnsurePath(path)
        interior_tree.EnsurePath(path)
        nonmanifold_tree.EnsurePath(path)
        items = list(breps_tree.Branch(path))
        if not items:
            continue
        for it in items:
            try:
                brep = None
                if isinstance(it, rg.Brep):
                    brep = it
                elif isinstance(it, rg.BrepFace):
                    try:
                        brep = it.DuplicateFace(True)
                    except:
                        brep = None
                elif isinstance(it, rg.Surface):
                    try:
                        brep = it.ToBrep()
                    except:
                        brep = None
                if brep is None or not brep.IsValid:
                    continue
                for e in brep.Edges:
                    try:
                        c = e.DuplicateCurve()
                        if c is None or not c.IsValid:
                            continue
                        cls = classify_valence(e.Valence)
                        if cls == "naked":
                            naked_tree.Add(c, path)
                        elif cls == "nonmanifold":
                            nonmanifold_tree.Add(c, path)
                        else:
                            interior_tree.Add(c, path)
                    except:
                        pass
            except:
                pass

    return naked_tree, interior_tree, nonmanifold_tree


def snap_by_collision(base_tree, to_snap_tree, tolerance=None):

    """
    Assign flattened geometries into grafted branches of base geometries by first-hit collision.

    Replaces GH's Collision One|Many style distribution with DataTree fidelity:
    - Graft base geometries (each base item becomes its own child branch under original path).
    - Flatten to-snap geometries (scan in order across all branches).
    - For each to-snap item, find the first base item that collides and append it to that base's grafted branch.
    - Preserve empty grafted branches when no items collide.

    Parameters:
        base_tree: DataTree[object]  # Curves/Breps/Surfaces/Meshes supported
        to_snap_tree: DataTree[object]
        tolerance: float | None  # defaults to document absolute tolerance

    Returns:
        DataTree[object]  # Structure mirrors graft(base_tree): {p;i} per base item
    """

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.001

    result = DataTree[object]()

    # Build list of base items with their grafted target paths and precomputed bboxes
    base_items = []  # list of (target_path, geom, bbox)
    try:
        for path in base_tree.Paths:
            items = list(base_tree.Branch(path))
            for i, it in enumerate(items):
                try:
                    tgt = GH_Path(path)
                    tgt = tgt.AppendElement(i)
                    result.EnsurePath(tgt)
                    bbox = None
                    try:
                        if hasattr(it, 'GetBoundingBox'):
                            bbox = it.GetBoundingBox(True)
                    except:
                        bbox = None
                    base_items.append((tgt, it, bbox))
                except:
                    pass
    except:
        pass

    # Flatten to-snap items preserving order across branches
    to_snap_items = []
    try:
        for p in to_snap_tree.Paths:
            for it in list(to_snap_tree.Branch(p)):
                to_snap_items.append(it)
    except:
        pass

    def bbox_intersects(a, b):
        try:
            if a is None or b is None:
                return True  # no bbox → do not cull
            return a.Intersects(b)
        except:
            return True

    def to_brep(obj):
        try:
            if isinstance(obj, rg.Brep):
                return obj
            if isinstance(obj, rg.Surface):
                return obj.ToBrep()
            if isinstance(obj, rg.BrepFace):
                return obj.DuplicateFace(True)
        except:
            pass
        return None

    def to_mesh(obj):
        try:
            if isinstance(obj, rg.Mesh):
                return obj
        except:
            pass
        return None

    def try_intersect(g1, g2):
        # Broad-phase bbox check
        bb1 = None; bb2 = None
        try:
            if hasattr(g1, 'GetBoundingBox'):
                bb1 = g1.GetBoundingBox(True)
        except:
            bb1 = None
        try:
            if hasattr(g2, 'GetBoundingBox'):
                bb2 = g2.GetBoundingBox(True)
        except:
            bb2 = None
        if not bbox_intersects(bb1, bb2):
            return False

        # Normalize common types
        c1 = None; c2 = None
        try:
            c1 = coerce_to_curve(g1)
        except:
            c1 = None
        try:
            c2 = coerce_to_curve(g2)
        except:
            c2 = None

        b1 = to_brep(g1); b2 = to_brep(g2)
        m1 = to_mesh(g1); m2 = to_mesh(g2)

        # Curve-Curve
        if c1 is not None and c1.IsValid and c2 is not None and c2.IsValid:
            try:
                ev = rg.Intersect.Intersection.CurveCurve(c1, c2, tolerance, tolerance)
                if ev and ev.Count > 0:
                    return True
            except:
                pass
            return False

        # Curve-Brep / Brep-Curve
        if (c1 is not None and b2 is not None) or (b1 is not None and c2 is not None):
            try:
                if c1 is not None and b2 is not None:
                    # Typical RhinoCommon python binding: returns (curves, points)
                    try:
                        crvs, pts = rg.Intersect.Intersection.CurveBrep(c1, b2, tolerance)
                        if (crvs and len(crvs) > 0) or (pts and len(pts) > 0):
                            return True
                    except:
                        pass
                    # Fallback: try CurveSurface against first face
                    try:
                        face = b2.Faces[0] if b2.Faces.Count > 0 else None
                        if face is not None:
                            ev = rg.Intersect.Intersection.CurveSurface(c1, face, tolerance, tolerance)
                            if ev and ev.Count > 0:
                                return True
                    except:
                        pass
                    try:
                        crvs, pts = rg.Intersect.Intersection.BrepCurve(b2, c1, tolerance)
                        if (crvs and len(crvs) > 0) or (pts and len(pts) > 0):
                            return True
                    except:
                        pass
                else:
                    try:
                        crvs, pts = rg.Intersect.Intersection.CurveBrep(c2, b1, tolerance)
                        if (crvs and len(crvs) > 0) or (pts and len(pts) > 0):
                            return True
                    except:
                        pass
                    # Fallback: CurveSurface
                    try:
                        face = b1.Faces[0] if b1.Faces.Count > 0 else None
                        if face is not None:
                            ev = rg.Intersect.Intersection.CurveSurface(c2, face, tolerance, tolerance)
                            if ev and ev.Count > 0:
                                return True
                    except:
                        pass
                    try:
                        crvs, pts = rg.Intersect.Intersection.BrepCurve(b1, c2, tolerance)
                        if (crvs and len(crvs) > 0) or (pts and len(pts) > 0):
                            return True
                    except:
                        pass
            except:
                pass
            return False

        # Brep-Brep
        if b1 is not None and b2 is not None:
            try:
                # Python binding usually returns (curves, points)
                crvs, pts = rg.Intersect.Intersection.BrepBrep(b1, b2, tolerance)
                if (crvs and len(crvs) > 0) or (pts and len(pts) > 0):
                    return True
            except:
                pass
            return False

        # Mesh-Mesh
        if m1 is not None and m2 is not None:
            try:
                pls = rg.Intersect.Intersection.MeshMeshFast(m1, m2)
                if pls and len(pls) > 0:
                    return True
            except:
                try:
                    pls = rg.Intersect.Intersection.MeshMeshAccurate(m1, m2)
                    if pls and len(pls) > 0:
                        return True
                except:
                    pass
            return False

        # Mesh-Brep
        if (m1 is not None and b2 is not None) or (b1 is not None and m2 is not None):
            try:
                if m1 is not None and b2 is not None:
                    polylines, pts = rg.Intersect.Intersection.MeshBrep(m1, b2, tolerance)
                    if (polylines and len(polylines) > 0) or (pts and len(pts) > 0):
                        return True
                else:
                    polylines, pts = rg.Intersect.Intersection.MeshBrep(m2, b1, tolerance)
                    if (polylines and len(polylines) > 0) or (pts and len(pts) > 0):
                        return True
            except:
                pass
            return False

        # Surface-Surface (fallback)
        try:
            if isinstance(g1, rg.Surface) and isinstance(g2, rg.Surface):
                crvs, pts = rg.Intersect.Intersection.SurfaceSurface(g1, g2, tolerance)
                if (crvs and len(crvs) > 0) or (pts and len(pts) > 0):
                    return True
        except:
            pass

        # Last resort: proximity/cointainment-style checks
        try:
            # If a curve is entirely on/inside a brep without clear intersection curves,
            # check distance from several points on the curve to the brep.
            cur = c1 or c2
            brp = b1 or b2
            if cur is not None and brp is not None:
                dom = cur.Domain
                cnt = 8
                step = (dom.T1 - dom.T0) / float(cnt)
                for k in range(cnt + 1):
                    t = dom.T0 + step * k
                    p = cur.PointAt(t)
                    ok, cp = brp.ClosestPoint(p)
                    if ok and p.DistanceTo(cp) <= max(tolerance, 1e-6):
                        return True
        except:
            pass

        # Finally: treat bbox overlap as collision when precise checks fail
        return True if bbox_intersects(bb1, bb2) else False

    # Assign each to-snap item to first colliding base item (first-come first-served)
    for idx, g in enumerate(to_snap_items):
        try:
            gb = None
            try:
                if hasattr(g, 'GetBoundingBox'):
                    gb = g.GetBoundingBox(True)
            except:
                gb = None
            for tgt_path, base_g, base_bb in base_items:
                try:
                    if not bbox_intersects(gb, base_bb):
                        continue
                    if try_intersect(g, base_g):
                        result.Add(g, tgt_path)
                        break
                except:
                    pass
        except Exception as e:
            try:
                print("Snapping failed for item {}: {}".format(idx, e))
            except:
                pass

    return result

def cull_index(list_tree, indices_input, wrap=True):

    result_tree = DataTree[object]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    # Resolve per-path indices with broadcasting
    def resolve_indices_for_path(path, count):
        idxs = []
        try:
            if is_tree(indices_input):
                if path in indices_input.Paths:
                    idxs = list(indices_input.Branch(path))
                elif len(list(indices_input.Paths)) == 1:
                    p0 = list(indices_input.Paths)[0]
                    idxs = list(indices_input.Branch(p0))
            else:
                try:
                    idxs = list(indices_input)
                except:
                    idxs = [indices_input]
        except:
            idxs = []
        # Convert to integers
        norm = []
        for v in idxs:
            try:
                norm.append(int(float(v)))
            except:
                pass
        if count <= 0:
            return set()
        if not wrap:
            # Keep only in-range
            norm = [i for i in norm if 0 <= i < count]
        else:
            # Map via modulo
            norm = [(i % count) for i in norm]
        return set(norm)

    for path in list_tree.Paths:
        result_tree.EnsurePath(path)
        items = list(list_tree.Branch(path))
        n = len(items)
        if n == 0:
            continue
        remove_set = resolve_indices_for_path(path, n)
        for i, it in enumerate(items):
            if i in remove_set:
                continue
            try:
                result_tree.Add(it, path)
            except:
                pass

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

def series(start_input, step_input, count_input):
    
    # Handle all scalar inputs - return list instead of DataTree
    if not is_tree(start_input) and not is_tree(step_input) and not is_tree(count_input):
        try:
            start_val = float(start_input)
            step_val = float(step_input)
            count_val = int(max(0, int(float(count_input))))
            
            result = []
            for i in range(count_val):
                result.append(start_val + step_val * i)
            return result
        except Exception as e:
            print("Series scalar calculation failed: {}".format(e))
            return []
    
    # Handle DataTree inputs
    result_tree = DataTree[System.Double]()

    all_paths = set()
    # Collect paths from start tree
    is_start_tree = is_tree(start_input)
    if start_input is not None and is_start_tree:
        for p in start_input.Paths:
            all_paths.add(p)

    # Support scalar or tree for step
    is_step_tree = is_tree(step_input)
    global_step = None
    if not is_step_tree:
        try:
            global_step = float(step_input)
        except:
            global_step = None
    else:
        for p in step_input.Paths:
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

    # Support scalar or tree for start
    global_start = None
    if not is_start_tree:
        try:
            global_start = float(start_input)
        except:
            global_start = None

    def path_key(p):
        return tuple(p.Indices)

    for path in sorted(all_paths, key=path_key):
        result_tree.EnsurePath(path)
        try:
            s_items = list(start_input.Branch(path)) if (start_input is not None and is_start_tree and path in start_input.Paths) else []
            t_items = list(step_input.Branch(path)) if (is_step_tree and path in step_input.Paths) else []
            
            has_start = (len(s_items) > 0) or (global_start is not None)
            has_step = (len(t_items) > 0) or (global_step is not None)
            if not has_start or not has_step:
                continue
            
            if len(s_items) > 0:
                try:
                    s0 = float(s_items[0])
                except:
                    s0 = float(global_start) if global_start is not None else 0.0
            else:
                s0 = float(global_start) if global_start is not None else 0.0
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

def bin_pack_segments(segment_length_tree, lot_length, lot_width, lot_count, lot_type=None, chain_across_branches=False, include_start_zero=False):

    result_positions = DataTree[System.Double]()
    remaining_count_tree = DataTree[System.Int32]()
    used_counts_tree = DataTree[System.Int32]()
    sorted_len_tree = DataTree[System.Double]()
    sorted_wid_tree = DataTree[System.Double]()
    sorted_type_tree = DataTree[object]()

    packed_widths_tree = DataTree[System.Double]()
    packed_lengths_tree = DataTree[System.Double]()
    packed_types_tree = DataTree[object]()

    # Compute global sorted lot specs once (used for counts and sorted outputs)
    if lot_type is None:
        _lot_data = list(zip(lot_length, lot_width, lot_count))
        _lot_data.sort(key=lambda x: x[0], reverse=True)
        sorted_len_global = [float(x[0]) for x in _lot_data]
        sorted_wid_global = [float(x[1]) for x in _lot_data]
        sorted_cnt_global = [int(x[2]) for x in _lot_data]
        sorted_typ_global = [None for _ in _lot_data]
    else:
        _lot_data = list(zip(lot_length, lot_width, lot_count, lot_type))
        _lot_data.sort(key=lambda x: x[0], reverse=True)
        sorted_len_global = [float(x[0]) for x in _lot_data]
        sorted_wid_global = [float(x[1]) for x in _lot_data]
        sorted_cnt_global = [int(x[2]) for x in _lot_data]
        sorted_typ_global = [x[3] for x in _lot_data]

    used_counts_global = [0] * len(sorted_wid_global)

    if chain_across_branches:
        try:
            def compute_sorted_lots():
                if lot_type is None:
                    data = list(zip(lot_length, lot_width, lot_count))
                    data.sort(key=lambda x: x[0], reverse=True)
                    return ([float(x[0]) for x in data],
                            [float(x[1]) for x in data],
                            [int(x[2]) for x in data],
                            [None for _ in data])
                else:
                    data = list(zip(lot_length, lot_width, lot_count, lot_type))
                    data.sort(key=lambda x: x[0], reverse=True)
                    return ([float(x[0]) for x in data],
                            [float(x[1]) for x in data],
                            [int(x[2]) for x in data],
                            [x[3] for x in data])

            def path_key(p):
                try:
                    return tuple(p.Indices)
                except:
                    return (0,)

            paths_sorted = sorted(list(segment_length_tree.Paths), key=path_key)

            segments_info = []
            for branch in paths_sorted:
                segs_raw = list(segment_length_tree.Branch(branch))
                segment_lengths = []
                for s in segs_raw:
                    try:
                        segment_lengths.append(float(s))
                    except:
                        segment_lengths.append(0.0)

                if len(segment_lengths) == 0:
                    try:
                        empty_path = GH_Path(branch)
                        empty_path = empty_path.AppendElement(0)
                        result_positions.EnsurePath(empty_path)
                        packed_widths_tree.EnsurePath(empty_path)
                        packed_lengths_tree.EnsurePath(empty_path)
                        packed_types_tree.EnsurePath(empty_path)
                    except:
                        pass
                else:
                    for i, sl in enumerate(segment_lengths):
                        segments_info.append((branch, i, sl))

            sorted_len, sorted_wid, sorted_cnt, sorted_typ = compute_sorted_lots()

            used_counts_by_branch = {}
            for branch in paths_sorted:
                used_counts_by_branch[branch] = [0] * len(sorted_wid)

            per_seg_positions = {}
            per_seg_widths = {}
            per_seg_lengths = {}
            per_seg_types = {}

            seg_ptr = 0
            current_sum = 0.0
            for i in range(len(sorted_wid)):
                w = sorted_wid[i]
                c = sorted_cnt[i]
                for _ in range(c):
                    while seg_ptr < len(segments_info):
                        bpath, sidx, slen = segments_info[seg_ptr]
                        if current_sum + w <= slen + 1e-9:
                            current_sum += w
                            key = (bpath, sidx)
                            per_seg_positions.setdefault(key, []).append(current_sum)
                            per_seg_widths.setdefault(key, []).append(w)
                            per_seg_lengths.setdefault(key, []).append(sorted_len[i])
                            per_seg_types.setdefault(key, []).append(sorted_typ[i])
                            used_counts_by_branch[bpath][i] += 1
                            break
                        else:
                            seg_ptr += 1
                            current_sum = 0.0
                    if seg_ptr >= len(segments_info):
                        break

            for (bpath, sidx, slen) in segments_info:
                new_path = GH_Path(bpath)
                new_path = new_path.AppendElement(sidx)
                result_positions.EnsurePath(new_path)
                vals = per_seg_positions.get((bpath, sidx), [])
                if include_start_zero and len(vals) > 0:
                    try:
                        result_positions.Add(System.Double(0.0), new_path)
                    except:
                        pass
                for val in vals:
                    try:
                        result_positions.Add(System.Double(float(val)), new_path)
                    except:
                        pass
                packed_widths_tree.EnsurePath(new_path)
                for val in per_seg_widths.get((bpath, sidx), []):
                    try:
                        packed_widths_tree.Add(System.Double(float(val)), new_path)
                    except:
                        pass
                packed_lengths_tree.EnsurePath(new_path)
                for val in per_seg_lengths.get((bpath, sidx), []):
                    try:
                        packed_lengths_tree.Add(System.Double(float(val)), new_path)
                    except:
                        pass
                packed_types_tree.EnsurePath(new_path)
                for val in per_seg_types.get((bpath, sidx), []):
                    try:
                        packed_types_tree.Add(val, new_path)
                    except:
                        pass

            # Aggregate used counts globally
            for branch in paths_sorted:
                for j in range(len(sorted_cnt)):
                    try:
                        used_counts_global[j] += int(used_counts_by_branch[branch][j])
                    except:
                        pass

            # Emit global single-branch outputs for counts and sorted specs
            path0 = GH_Path(0)
            remaining_count_tree.EnsurePath(path0)
            used_counts_tree.EnsurePath(path0)
            sorted_len_tree.EnsurePath(path0)
            sorted_wid_tree.EnsurePath(path0)
            sorted_type_tree.EnsurePath(path0)

            for j in range(len(sorted_cnt_global)):
                try:
                    rem = int(sorted_cnt_global[j] - used_counts_global[j])
                except:
                    rem = 0
                remaining_count_tree.Add(System.Int32(rem), path0)
                used_counts_tree.Add(System.Int32(int(used_counts_global[j])), path0)
            for v in sorted_len_global:
                sorted_len_tree.Add(System.Double(float(v)), path0)
            for v in sorted_wid_global:
                sorted_wid_tree.Add(System.Double(float(v)), path0)
            for v in (sorted_typ_global if lot_type is not None else []):
                try:
                    sorted_type_tree.Add(v, path0)
                except:
                    pass

        except Exception as e:
            print("Bin packing (chain_across_branches=True) failed: {}".format(e))
            pass
        return result_positions, remaining_count_tree, used_counts_tree, sorted_len_tree, sorted_wid_tree, packed_widths_tree, packed_lengths_tree, packed_types_tree, sorted_type_tree

    for branch in segment_length_tree.Paths:
        try:
            segs_raw = list(segment_length_tree.Branch(branch))
            segment_lengths = []
            for s in segs_raw:
                try:
                    segment_lengths.append(float(s))
                except:
                    segment_lengths.append(0.0)

            # Preserve empty branch in relative positioning output
            if len(segment_lengths) == 0:
                try:
                    empty_path = GH_Path(branch)
                    empty_path = empty_path.AppendElement(0)
                    result_positions.EnsurePath(empty_path)
                    packed_widths_tree.EnsurePath(empty_path)
                    packed_lengths_tree.EnsurePath(empty_path)
                    packed_types_tree.EnsurePath(empty_path)
                except:
                    pass

            # Use global sorted specs
            sorted_len = sorted_len_global
            sorted_wid = sorted_wid_global
            sorted_cnt = sorted_cnt_global
            sorted_typ = sorted_typ_global

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

            remaining_count_values = [sorted_cnt[i] - used_counts_local[i] for i in range(len(sorted_cnt))]

            for i, lst in enumerate(per_segment_positions):
                new_path = GH_Path(branch)
                new_path = new_path.AppendElement(i)
                result_positions.EnsurePath(new_path)
                if include_start_zero and len(lst) > 0:
                    try:
                        result_positions.Add(System.Double(0.0), new_path)
                    except:
                        pass
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

            # Accumulate used counts globally (for single-branch summary)
            for j in range(len(used_counts_local)):
                try:
                    used_counts_global[j] += int(used_counts_local[j])
                except:
                    pass

        except Exception as e:
            print("Bin packing failed for path {}: {}".format(branch, e))
            pass

    # Emit global single-branch outputs after per-branch processing
    path0 = GH_Path(0)
    remaining_count_tree.EnsurePath(path0)
    used_counts_tree.EnsurePath(path0)
    sorted_len_tree.EnsurePath(path0)
    sorted_wid_tree.EnsurePath(path0)
    sorted_type_tree.EnsurePath(path0)

    for j in range(len(sorted_cnt_global)):
        try:
            rem = int(sorted_cnt_global[j] - used_counts_global[j])
        except:
            rem = 0
        remaining_count_tree.Add(System.Int32(rem), path0)
        used_counts_tree.Add(System.Int32(int(used_counts_global[j])), path0)
    for v in sorted_len_global:
        sorted_len_tree.Add(System.Double(float(v)), path0)
    for v in sorted_wid_global:
        sorted_wid_tree.Add(System.Double(float(v)), path0)
    for v in (sorted_typ_global if lot_type is not None else []):
        try:
            sorted_type_tree.Add(v, path0)
        except:
            pass

    return result_positions, remaining_count_tree, used_counts_tree, sorted_len_tree, sorted_wid_tree, packed_widths_tree, packed_lengths_tree, packed_types_tree, sorted_type_tree

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


def duplicate_data(data_tree, number, order=False):

    """
    Duplicates items similar to Grasshopper's Duplicate Data component.

    Parameters:
        data_tree: DataTree[object] - Primary input; structure is preserved per path.
        number: int | float | list | DataTree - Duplication count per item or global.
        order: bool - If True, cycle items by copy index (a,b,a,b,...). If False, group by item (a,a,...,b,b,...).

    Returns:
        DataTree[object] - Items duplicated with the same path structure. Empty branches preserved.
    """

    result = DataTree[object]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    def to_int_safe(x):
        try:
            return max(0, int(float(x)))
        except:
            return 0

    def get_counts_for_path(path, item_count):
        try:
            # Per-branch counts from DataTree
            if hasattr(number, 'Paths'):
                nums = list(number.Branch(path)) if path in number.Paths else []
                if len(nums) == item_count:
                    return [to_int_safe(v) for v in nums]
                if len(nums) == 1:
                    return [to_int_safe(nums[0])] * item_count
                if len(nums) > 1:
                    # Truncate or pad with last value
                    vals = [to_int_safe(v) for v in nums]
                    if len(vals) < item_count:
                        vals = vals + [vals[-1]] * (item_count - len(vals))
                    return vals[:item_count]
        except:
            pass
        # List/tuple scalar
        if isinstance(number, (list, tuple)):
            if len(number) == item_count:
                return [to_int_safe(v) for v in number]
            if len(number) >= 1:
                return [to_int_safe(number[0])] * item_count
        # Scalar
        return [to_int_safe(number)] * item_count

    if not is_tree(data_tree):
        # Normalize items (scalar or list)
        if isinstance(data_tree, (list, tuple)):
            items = list(data_tree)
        else:
            items = [data_tree]

        # If number is a DataTree, drive structure from its branches
        if hasattr(number, 'Paths') and hasattr(number, 'Branch'):
            try:
                for p in list(number.Paths):
                    try:
                        result.EnsurePath(p)
                    except:
                        pass
                    try:
                        nums_branch = list(number.Branch(p))
                    except:
                        nums_branch = []
                    if not nums_branch:
                        continue
                    counts_vals = [to_int_safe(v) for v in nums_branch]
                    if len(items) == 1:
                        item = items[0]
                        for cnt in counts_vals:
                            for _ in range(cnt):
                                try:
                                    result.Add(item, p)
                                except:
                                    pass
                    else:
                        # Multiple items: align counts to items
                        if len(counts_vals) == len(items):
                            pairs = zip(items, counts_vals)
                        elif len(counts_vals) == 1:
                            pairs = [(it, counts_vals[0]) for it in items]
                        else:
                            # Pad/truncate counts to items length
                            vals = counts_vals + [counts_vals[-1]] * (len(items) - len(counts_vals)) if len(counts_vals) < len(items) else counts_vals[:len(items)]
                            pairs = zip(items, vals)
                        for it, cnt in pairs:
                            for _ in range(max(0, int(cnt))):
                                try:
                                    result.Add(it, p)
                                except:
                                    pass
            except:
                pass
            return result

        # Non-tree number: single output path {0}
        path0 = GH_Path(0)
        result.EnsurePath(path0)
        # Determine counts for non-tree number
        if isinstance(number, (list, tuple)):
            counts = [to_int_safe(v) for v in number]
            if len(items) == 1:
                for cnt in counts:
                    for _ in range(cnt):
                        try:
                            result.Add(items[0], path0)
                        except:
                            pass
            else:
                # Align counts to items
                if len(counts) == len(items):
                    pairs = zip(items, counts)
                elif len(counts) == 1:
                    pairs = [(it, counts[0]) for it in items]
                else:
                    vals = counts + [counts[-1]] * (len(items) - len(counts)) if len(counts) < len(items) else counts[:len(items)]
                    pairs = zip(items, vals)
                for it, cnt in pairs:
                    for _ in range(max(0, int(cnt))):
                        try:
                            result.Add(it, path0)
                        except:
                            pass
        else:
            cnt = to_int_safe(number)
            for it in items:
                for _ in range(cnt):
                    try:
                        result.Add(it, path0)
                    except:
                        pass
        return result

    # Tree case
    try:
        input_paths = list(data_tree.Paths)
    except:
        input_paths = []

    # Special case A: data_tree contains exactly one NON-NULL item overall → drive structure from 'number' tree
    try:
        total_count = 0
        non_null_count = 0
        single_item = None
        for p in input_paths:
            br = list(data_tree.Branch(p))
            total_count += len(br)
            for it in br:
                if it is not None:
                    if single_item is None:
                        single_item = it
                    non_null_count += 1
        if non_null_count == 1 and hasattr(number, 'Paths') and hasattr(number, 'Branch'):
            try:
                for p in list(number.Paths):
                    result.EnsurePath(p)
                    try:
                        nums_branch = list(number.Branch(p))
                    except:
                        nums_branch = []
                    if len(nums_branch) == 0:
                        continue
                    counts_vals = [to_int_safe(v) for v in nums_branch]
                    # If more than one count in a branch, repeat pattern per count value
                    for cnt in counts_vals:
                        for _ in range(max(0, int(cnt))):
                            try:
                                result.Add(single_item, p)
                            except:
                                pass
            except:
                pass
            return result
    except:
        pass

    # Special case B: data_tree items across all branches are the SAME scalar value → broadcast to number tree
    try:
        if hasattr(number, 'Paths') and hasattr(number, 'Branch'):
            candidate = None
            same = True
            for p in input_paths:
                for it in list(data_tree.Branch(p)):
                    if it is None:
                        continue
                    if candidate is None:
                        candidate = it
                    else:
                        eq = False
                        try:
                            # numeric compare when possible
                            eq = float(it) == float(candidate)
                        except:
                            try:
                                eq = (str(it) == str(candidate))
                            except:
                                eq = False
                        if not eq:
                            same = False
                            break
                if not same:
                    break
            if candidate is not None and same:
                for p in list(number.Paths):
                    result.EnsurePath(p)
                    try:
                        nums_branch = list(number.Branch(p))
                    except:
                        nums_branch = []
                    if len(nums_branch) == 0:
                        continue
                    counts_vals = [to_int_safe(v) for v in nums_branch]
                    for cnt in counts_vals:
                        for _ in range(max(0, int(cnt))):
                            try:
                                result.Add(candidate, p)
                            except:
                                pass
                return result
    except:
        pass

    for path in input_paths:
        result.EnsurePath(path)
        items = list(data_tree.Branch(path))
        if not items:
            continue
        counts = get_counts_for_path(path, len(items))
        if not order:
            # Item-major: a,a,a,b,b,b
            for item, cnt in zip(items, counts):
                for _ in range(cnt):
                    try:
                        result.Add(item, path)
                    except:
                        pass
        else:
            # Copy-index-major: a,b,a,b,...
            max_cnt = max(counts) if counts else 0
            for k in range(max_cnt):
                for idx, item in enumerate(items):
                    if counts[idx] > k:
                        try:
                            result.Add(item, path)
                        except:
                            pass
    return result


def addition(a_input, b_input):

    """
    Adds values like Grasshopper's Addition component with broadcasting.

    Parameters:
        a_input: DataTree or scalar/list - Primary if tree; structure preserved.
        b_input: DataTree or scalar/list - Broadcast across primary when needed.

    Returns:
        DataTree[System.Double] - Sum per item; empty branches preserved.
    """

    result = DataTree[System.Double]()

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    def to_float(x):
        try:
            return float(x)
        except:
            return None

    def get_values_for_path(inp, path, expected_len):
        if is_tree(inp):
            vals = list(inp.Branch(path)) if path in inp.Paths else []
            if len(vals) == 0:
                return []
            if len(vals) == expected_len:
                return [to_float(v) for v in vals]
            if len(vals) == 1:
                return [to_float(vals[0])] * expected_len
            # Truncate or pad with last
            out = [to_float(v) for v in vals]
            if len(out) < expected_len and len(out) > 0:
                out = out + [out[-1]] * (expected_len - len(out))
            return out[:expected_len]
        # Non-tree
        if isinstance(inp, (list, tuple)):
            if len(inp) == expected_len:
                return [to_float(v) for v in inp]
            if len(inp) >= 1:
                return [to_float(inp[0])] * expected_len
        return [to_float(inp)] * expected_len

    primary = a_input if is_tree(a_input) else (b_input if is_tree(b_input) else None)

    if primary is None:
        # Both scalars/lists; produce single path {0}
        path0 = GH_Path(0)
        result.EnsurePath(path0)
        a_vals = [to_float(a_input)]
        b_vals = [to_float(b_input)]
        try:
            s = (a_vals[0] if a_vals[0] is not None else 0.0) + (b_vals[0] if b_vals[0] is not None else 0.0)
            result.Add(System.Double(float(s)), path0)
        except:
            pass
        return result

    for path in primary.Paths:
        result.EnsurePath(path)
        a_vals = get_values_for_path(a_input, path, len(list(primary.Branch(path)))) if is_tree(a_input) else get_values_for_path(a_input, path, len(list(primary.Branch(path))))
        b_vals = get_values_for_path(b_input, path, len(list(primary.Branch(path))))
        if len(a_vals) == 0 and len(b_vals) == 0:
            continue
        n = max(len(a_vals), len(b_vals))
        if len(a_vals) != n:
            a_vals = (a_vals + [a_vals[-1]] * (n - len(a_vals))) if a_vals else [0.0] * n
        if len(b_vals) != n:
            b_vals = (b_vals + [b_vals[-1]] * (n - len(b_vals))) if b_vals else [0.0] * n
        for i in range(n):
            a = a_vals[i] if a_vals[i] is not None else 0.0
            b = b_vals[i] if b_vals[i] is not None else 0.0
            try:
                result.Add(System.Double(float(a + b)), path)
            except:
                pass
    return result


def mass_addition(input_tree):

    """
    Mass Addition equivalent.

    Parameters:
        input_tree: DataTree of numeric-like values.

    Returns:
        tuple(DataTree[System.Double], DataTree[System.Double]) - (Result, Partial Results)
            Result: one total per path (if items exist), else empty branch preserved.
            Partial Results: cumulative sums per path, same item count as input branch.
    """

    result = DataTree[System.Double]()
    partial = DataTree[System.Double]()

    try:
        input_paths = list(input_tree.Paths)
    except:
        input_paths = []

    def to_float(x):
        try:
            return float(x)
        except:
            return None

    for path in input_paths:
        result.EnsurePath(path)
        partial.EnsurePath(path)
        items = list(input_tree.Branch(path))
        if not items:
            continue
        running = 0.0
        added_any = False
        for it in items:
            v = to_float(it)
            if v is None:
                continue
            running += v
            added_any = True
            try:
                partial.Add(System.Double(float(running)), path)
            except:
                pass
        if added_any:
            try:
                result.Add(System.Double(float(running)), path)
            except:
                pass
    return result, partial


def fill_empty_branches_by_neighbors(values_tree):
	"""
	Fill empty branches using neighboring branches' values.

	Rules:
	- Each branch ideally contains two numeric values.
	- For an empty branch, inspect the nearest previous non-empty branch (prev)
	  and the nearest next non-empty branch (next) in path order.
	  * If both exist and prev.last == next.last: fill with [value, value].
	  * If prev missing or has no value: use next.first duplicated.
	  * If next missing or has no value: use prev.last duplicated.
	  * If both exist but next.first != prev.last: use prev.last duplicated.
	- For consecutive empty branches, extend the search to both sides and
	  fill the whole run using the same rule based on the outer neighbors.

	Parameters:
		values_tree: DataTree of numeric-like values.

	Returns:
		DataTree[System.Double] with empty branches filled per the rules, preserving paths.
	"""

	result = DataTree[System.Double]()

	def to_float(x):
		try:
			return float(x)
		except:
			return None

	try:
		paths = list(values_tree.Paths)
	except:
		paths = []

	# Pre-read all branch values as float lists
	branch_vals = []
	for p in paths:
		items = []
		try:
			raw = list(values_tree.Branch(p))
		except:
			raw = []
		for it in raw:
			v = to_float(it)
			if v is not None:
				items.append(v)
		branch_vals.append(items)

	idx = 0
	n = len(paths)
	while idx < n:
		p = paths[idx]
		result.EnsurePath(p)
		vals = branch_vals[idx]
		if len(vals) > 0:
			# Copy existing values as-is
			for v in vals:
				try:
					result.Add(System.Double(float(v)), p)
				except:
					pass
			idx += 1
			continue

		# Empty run detection
		run_start = idx
		run_end = idx
		while run_end < n and len(branch_vals[run_end]) == 0:
			run_end += 1

		# Find previous non-empty
		prev_idx = run_start - 1
		while prev_idx >= 0 and len(branch_vals[prev_idx]) == 0:
			prev_idx -= 1

		# Find next non-empty
		next_idx = run_end if run_end < n else -1
		if next_idx >= 0:
			while next_idx < n and len(branch_vals[next_idx]) == 0:
				next_idx += 1
			if next_idx >= n:
				next_idx = -1

		# Determine fill value based on rules
		fill_value = None
		prev_last = None
		next_first = None
		next_last = None

		if prev_idx >= 0 and len(branch_vals[prev_idx]) > 0:
			prev_list = branch_vals[prev_idx]
			prev_last = prev_list[-1]
		if next_idx >= 0 and len(branch_vals[next_idx]) > 0:
			next_list = branch_vals[next_idx]
			next_first = next_list[0]
			next_last = next_list[-1]

		if (prev_last is None) and (next_first is not None):
			fill_value = next_first
		elif (next_last is None) and (prev_last is not None):
			fill_value = prev_last
		elif (prev_last is not None) and (next_last is not None):
			if prev_last == next_last:
				fill_value = prev_last
			else:
				fill_value = prev_last
		else:
			# Both sides missing values; leave branches empty
			fill_value = None

		# Fill all branches in the empty run
		for k in range(run_start, run_end):
			pk = paths[k]
			result.EnsurePath(pk)
			if fill_value is not None:
				try:
					result.Add(System.Double(float(fill_value)), pk)
					result.Add(System.Double(float(fill_value)), pk)
				except:
					pass

		idx = run_end

	return result

def offset_by_distances_master(input_segments, offset_distances, offset_side=1, min_ratio=0.15, tolerance=None):
    """
    Creates offset polylines with intersection handling and fin removal.
    
    Replaces a complex Grasshopper workflow that offsets segments, handles intersections,
    and creates clean closed polylines by removing small fins.
    
    Parameters:
        input_segments: DataTree[rg.Curve] - Input curve segments to offset
        offset_distances: DataTree[float] or float - Offset distances for each segment
        offset_side: int - Side to offset (1 or -1, default 1)
        min_ratio: float - Minimum ratio for fin removal (default 0.15)
        tolerance: float - Tolerance for operations (default: document tolerance)
        
    Returns:
        DataTree[rg.Polyline] - Clean closed polylines with intersections handled
    """
    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.01
    
    output = DataTree[rg.Polyline]()
    
    for path in input_segments.Paths:
        output.EnsurePath(path)
        segments = list(input_segments.Branch(path))
        if not segments:
            continue
            
        try:
            # Step 1: Offset segments by distances
            offset_segments = offset_by_distances(input_segments, offset_distances, offset_side)
            
            # Step 2: Create shifted list and graft both trees
            shifted_list = shift_list_tree(offset_segments, -1, True)
            grafted2 = graft_tree(shifted_list)
            grafted1 = graft_tree(offset_segments)
            
            # Step 3: Merge trees and find intersections
            merged = merge_trees(grafted1, grafted2)
            intersection_bool = curve_curve_intersection_tree(grafted1, grafted2)
            
            # Step 4: Dispatch based on intersection results
            tree_A, tree_B = dispatch_tree(merged, intersection_bool)
            first_line, second_line = extract_first_second_items(tree_B)
            line1_start, line1_end = extract_start_end_points(first_line)
            line2_start, line2_end = extract_start_end_points(second_line)
            
            # Step 5: Extend lines and find intersection points
            extended_first_line = extend_lines(first_line, 10000)
            extended_second_line = extend_lines(second_line, 10000)
            intersection_pts = line_line_intersection_points(extended_first_line, extended_second_line)
            
            # Step 6: Create joined lines
            joined_line1 = two_pt_line(line1_start, intersection_pts)
            joined_line2 = two_pt_line(intersection_pts, line2_end)
            joined_offset = merge_trees(joined_line1, joined_line2, tree_A)
            
            # Step 7: Process all intersection points
            first_line_all, second_line_all = extract_first_second_items(joined_offset)
            intersection_pts_all = line_line_intersection_points(first_line_all, second_line_all)
            
            # Step 8: Conform intersection points to input tree structure
            conform_isct_pts_all = data_tree_manager(intersection_pts_all, input_segments)
            closed_polylines = points_to_closed_polyline(conform_isct_pts_all)
            
            # Step 9: Remove small fins and clean up
            fin_removed1 = remove_small_fins(closed_polylines, tol=tolerance, min_ratio=min_ratio)
            fin_removed1 = clean_tree(fin_removed1, True, True, True)
            
            # Step 10: Explode curves and rebuild polylines
            exploded_vertices = explode_curves(fin_removed1)[1]
            final_offset = points_to_closed_polyline(exploded_vertices)
            
            # Add results to output tree
            final_polylines = list(final_offset.Branch(path)) if path in final_offset.Paths else []
            for polyline in final_polylines:
                if polyline is not None and polyline.IsValid:
                    output.Add(polyline, path)
                    
        except Exception as e:
            print("Offset with intersection handling failed for path {}: {}".format(path, e))
            pass
    
    return output

def cull_curve_short_segments(curves, min_segment_length, min_ratio=0.15):
    """
    Master function for culling short curves and processing geometry workflow.
    Replaces the manual workflow of exploding curves, culling short segments,
    finding intersections, joining lines, and creating closed polylines.
    
    Parameters:
        curves (DataTree[rg.Curve]): Input curves to process.
        min_segment_length (float): Minimum length threshold for culling.
        min_ratio (float): Minimum ratio for fin removal (default: 0.15).
    
    Returns:
        DataTree[rg.Curve]: Processed closed polylines with short segments removed.
    """
    
    # Step 1: Explode curves and conform to original tree structure
    exploded = explode_curves(curves)[0]
    exploded = data_tree_manager(exploded, curves)
    
    # Step 2: Cull short segments
    long_segments = cull_short_curves(exploded, min_segment_length)
    
    # Step 3: Prepare for intersection analysis
    shifted_list = shift_list_tree(long_segments, -1, True)
    grafted2 = graft_tree(shifted_list)
    grafted1 = graft_tree(long_segments)
    merged = merge_trees(grafted1, grafted2)
    intersection_bool = curve_curve_intersection_tree(grafted1, grafted2)
    
    # Step 4: Process intersections
    tree_A, tree_B = dispatch_tree(merged, intersection_bool)
    first_line, second_line = extract_first_second_items(tree_B)
    line1_start, line1_end = extract_start_end_points(first_line)
    line2_start, line2_end = extract_start_end_points(second_line)
    
    # Step 5: Extend lines and find intersection points
    extended_first_line = extend_lines(first_line, 10000)
    extended_second_line = extend_lines(second_line, 10000)
    intersection_pts = line_line_intersection_points(extended_first_line, extended_second_line)
    
    # Step 6: Join lines at intersection points
    joined_line1 = two_pt_line(line1_start, intersection_pts)
    joined_line2 = two_pt_line(intersection_pts, line2_end)
    joined_offset = merge_trees(joined_line1, joined_line2, tree_A)
    
    # Step 7: Process all intersections and create closed polylines
    first_line_all, second_line_all = extract_first_second_items(joined_offset)
    intersection_pts_all = line_line_intersection_points(first_line_all, second_line_all)
    
    # Step 8: Conform intersection points and create closed polylines
    comform_isct_pts_all = data_tree_manager(intersection_pts_all, curves)
    closed_polylines = points_to_closed_polyline(comform_isct_pts_all)
    
    # Step 9: Remove small fins and clean up
    fin_removed = remove_small_fins(closed_polylines, tol=sc.doc.ModelAbsoluteTolerance, min_ratio=min_ratio)
    fin_removed = clean_tree(fin_removed, True, True, True)
    
    # Step 10: Final processing - explode vertices and create final polylines
    exploded_vertices = explode_curves(fin_removed)[1]
    major_curves = points_to_closed_polyline(exploded_vertices)
    
    return major_curves

def offset_curve_universal(curves_tree, distance_input, plane_input=None, both_sides=False, kinks=True, cap_type=None, tolerance=None):
    """
    Offset curves by one universal distance (Pufferfish 'Offset Curve (Universal)' replacement).

    Parameters:
        curves_tree: DataTree[rg.Curve] – primary input; output preserves its paths (empty branches kept)
        distance_input: number or DataTree single numeric value (broadcast to all)
        plane_input: None | rg.Plane | DataTree[rg.Plane]; per-branch plane when provided
        both_sides: bool – when True, output +d and -d for each curve
        kinks: bool – True = Sharp corners; False = Round corners
        cap_type: None | 'Flat' – when both_sides=True, Flat will cap ends to form closed offset
        tolerance: float – default document tolerance

    Returns:
        DataTree[rg.Curve]
    """

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.01

    result = DataTree[rg.Curve]()
    if curves_tree is None:
        return result

    # Resolve distance (scalar or first numeric in a tree)
    def to_float(x):
        try:
            return float(x)
        except:
            return None

    uni_distance = None
    try:
        if distance_input is not None and not hasattr(distance_input, 'Paths'):
            uni_distance = to_float(distance_input)
    except:
        uni_distance = None
    if uni_distance is None and hasattr(distance_input, 'Paths'):
        try:
            for p in distance_input.Paths:
                vals = list(distance_input.Branch(p))
                for v in vals:
                    uni_distance = to_float(v)
                    if uni_distance is not None:
                        break
                if uni_distance is not None:
                    break
        except:
            pass
    if uni_distance is None:
        uni_distance = 0.0

    corner_style = rg.CurveOffsetCornerStyle.Sharp if kinks else rg.CurveOffsetCornerStyle.Round

    # Normalize cap flag: True or 'Flat' enables flat caps (only when two offsets exist)
    cap_flat = False
    try:
        if isinstance(cap_type, bool):
            cap_flat = bool(cap_type)
        elif isinstance(cap_type, (str, System.String)):
            cap_flat = str(cap_type).strip().lower() in ('flat', 'true', '1', 'yes', 'y')
    except:
        cap_flat = False

    def to_plane(obj):
        if isinstance(obj, rg.Plane):
            return obj
        if hasattr(obj, 'Origin') and hasattr(obj, 'XAxis') and hasattr(obj, 'YAxis'):
            try:
                return rg.Plane(obj.Origin, obj.XAxis, obj.YAxis)
            except:
                return None
        return None

    plane_is_tree = hasattr(plane_input, 'Paths') if plane_input is not None else False
    global_plane = None if plane_is_tree else (to_plane(plane_input) if plane_input is not None else None)

    for path in curves_tree.Paths:
        result.EnsurePath(path)
        crvs = list(curves_tree.Branch(path))
        if not crvs:
            continue

        # Resolve per-path plane when provided as tree
        path_plane = None
        if plane_is_tree:
            try:
                if path in plane_input.Paths:
                    items = list(plane_input.Branch(path))
                    if items:
                        path_plane = to_plane(items[0])
            except:
                path_plane = None

        for it in crvs:
            try:
                c = coerce_to_curve(it)
                if c is None or not hasattr(c, 'IsValid') or not c.IsValid:
                    continue

                # Choose plane: branch > global > curve plane > WorldXY
                pl = path_plane or global_plane
                if pl is None:
                    try:
                        ok, pl2 = c.TryGetPlane()
                        pl = pl2 if ok else None
                    except:
                        pl = None
                if pl is None:
                    pl = rg.Plane.WorldXY

                if abs(uni_distance) <= 0.0:
                    try:
                        dup = c.DuplicateCurve() if hasattr(c, 'DuplicateCurve') else c
                        if dup is not None and dup.IsValid:
                            result.Add(dup, path)
                        continue
                    except:
                        pass

                if not both_sides:
                    off = c.Offset(pl, uni_distance, tolerance, corner_style)
                    try:
                        if off and len(off) > 0:
                            oc = None
                            for _c in off:
                                if _c is not None and _c.IsValid:
                                    oc = _c
                                    break
                            if oc is not None:
                                if cap_flat:
                                    try:
                                        c_s = c.PointAtStart; c_e = c.PointAtEnd
                                        p_s = oc.PointAtStart; p_e = oc.PointAtEnd
                                        d1 = c_s.DistanceTo(p_s) + c_e.DistanceTo(p_e)
                                        d2 = c_s.DistanceTo(p_e) + c_e.DistanceTo(p_s)
                                        if d2 < d1:
                                            cap_a = rg.Line(c_s, p_e).ToNurbsCurve()
                                            cap_b = rg.Line(c_e, p_s).ToNurbsCurve()
                                        else:
                                            cap_a = rg.Line(c_s, p_s).ToNurbsCurve()
                                            cap_b = rg.Line(c_e, p_e).ToNurbsCurve()
                                        pieces = [c, cap_a, oc, cap_b]
                                        joined = rg.Curve.JoinCurves(pieces, tolerance)
                                        if joined and len(joined) > 0 and joined[0] is not None and joined[0].IsValid:
                                            result.Add(joined[0], path)
                                        else:
                                            for pc in pieces:
                                                if pc is not None and pc.IsValid:
                                                    result.Add(pc, path)
                                    except:
                                        result.Add(oc, path)
                                else:
                                    result.Add(oc, path)
                    except:
                        pass
                else:
                    pos_off = None
                    neg_off = None
                    try:
                        tmp = c.Offset(pl, abs(uni_distance), tolerance, corner_style)
                        if tmp and len(tmp) > 0 and tmp[0] is not None and tmp[0].IsValid:
                            pos_off = tmp[0]
                    except:
                        pos_off = None
                    try:
                        tmp = c.Offset(pl, -abs(uni_distance), tolerance, corner_style)
                        if tmp and len(tmp) > 0 and tmp[0] is not None and tmp[0].IsValid:
                            neg_off = tmp[0]
                    except:
                        neg_off = None

                    if pos_off is None and neg_off is None:
                        pass
                    elif cap_flat and (pos_off is not None) and (neg_off is not None):
                        try:
                            p_s = pos_off.PointAtStart; p_e = pos_off.PointAtEnd
                            n_s = neg_off.PointAtStart; n_e = neg_off.PointAtEnd
                            d1 = p_s.DistanceTo(n_s) + p_e.DistanceTo(n_e)
                            d2 = p_s.DistanceTo(n_e) + p_e.DistanceTo(n_s)
                            if d2 < d1:
                                cap_a = rg.Line(p_s, n_e).ToNurbsCurve()
                                cap_b = rg.Line(p_e, n_s).ToNurbsCurve()
                                pieces = [pos_off, cap_a, neg_off, cap_b]
                            else:
                                cap_a = rg.Line(p_s, n_s).ToNurbsCurve()
                                cap_b = rg.Line(p_e, n_e).ToNurbsCurve()
                                pieces = [pos_off, cap_a, neg_off, cap_b]
                            joined = rg.Curve.JoinCurves(pieces, tolerance)
                            if joined and len(joined) > 0 and joined[0] is not None and joined[0].IsValid:
                                result.Add(joined[0], path)
                            else:
                                for pc in pieces:
                                    if pc is not None and pc.IsValid:
                                        result.Add(pc, path)
                        except:
                            # Fallback: add offsets without capping
                            if pos_off is not None and pos_off.IsValid:
                                result.Add(pos_off, path)
                            if neg_off is not None and neg_off.IsValid:
                                result.Add(neg_off, path)
                    else:
                        if pos_off is not None and pos_off.IsValid:
                            result.Add(pos_off, path)
                        if neg_off is not None and neg_off.IsValid:
                            result.Add(neg_off, path)
            except:
                pass

    return result

def calculate_division_results_general(A, B, C):

    # Helper
    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    is_tree_a = is_tree(A)
    is_tree_b = is_tree(B)
    is_tree_c = is_tree(C)

    # All scalars → return tuple of scalars
    if not is_tree_a and not is_tree_b and not is_tree_c:
        try:
            a = float(A)
            b = float(B)
            c = float(C)
            spacing_count = a // b if b != 0 else 0
            residue_space = a % b if b != 0 else 0.0
            single_lot_row_count = residue_space // c if c != 0 else 0
            return (spacing_count, single_lot_row_count, residue_space)
        except:
            return (0, 0, 0.0)

    # At least one is a DataTree → return DataTrees preserving paths
    result_spacing = DataTree[System.Int32]()
    result_single_lot = DataTree[System.Int32]()
    result_residue = DataTree[System.Double]()

    all_paths = set()
    if is_tree_a:
        for p in A.Paths:
            all_paths.add(p)
    if is_tree_b:
        for p in B.Paths:
            all_paths.add(p)
    if is_tree_c:
        for p in C.Paths:
            all_paths.add(p)

    def path_key(p):
        try:
            return tuple(p.Indices)
        except:
            return (0,)

    for path in sorted(all_paths, key=path_key):
        result_spacing.EnsurePath(path)
        result_single_lot.EnsurePath(path)
        result_residue.EnsurePath(path)

        try:
            # Resolve per-path values with scalar broadcast
            a_val = None
            b_val = None
            c_val = None

            if is_tree_a and path in A.Paths:
                a_items = list(A.Branch(path))
                a_val = float(a_items[0]) if len(a_items) > 0 else 0.0
            elif not is_tree_a:
                a_val = float(A)

            if is_tree_b and path in B.Paths:
                b_items = list(B.Branch(path))
                b_val = float(b_items[0]) if len(b_items) > 0 else 0.0
            elif not is_tree_b:
                b_val = float(B)

            if is_tree_c and path in C.Paths:
                c_items = list(C.Branch(path))
                c_val = float(c_items[0]) if len(c_items) > 0 else 0.0
            elif not is_tree_c:
                c_val = float(C)

            if a_val is None or b_val is None or c_val is None:
                continue

            if b_val == 0.0:
                spacing_count = 0
                residue_space = 0.0
            else:
                spacing_count = a_val // b_val
                residue_space = a_val % b_val

            if c_val == 0.0:
                single_lot_row_count = 0
            else:
                single_lot_row_count = residue_space // c_val

            result_spacing.Add(System.Int32(int(spacing_count)), path)
            result_single_lot.Add(System.Int32(int(single_lot_row_count)), path)
            result_residue.Add(System.Double(float(residue_space)), path)
        except:
            pass

    return (result_spacing, result_single_lot, result_residue)


## calculate initial linear coordinates of branch rd along the longest side of inner row outline
longest_lot_length = lot_length[0]
shortest_lot_length = list_item(lot_length, -1)
initial_first_branch_rd_cline_coordinate = (longest_lot_length * 2) + (right_of_way_width / 2) + back_of_lot_gap
length_longest_side = curve_length(longest_side_inner_loop)
initial_branch_rd_spacing = (lot_length[0] * 2) + right_of_way_width + back_of_lot_gap

initial_spacing_count, single_lot_row_count, residue_space = calculate_division_results_general(length_longest_side, initial_branch_rd_spacing, longest_lot_length)

branch_rd_cline_initial_coordinates = series(initial_first_branch_rd_cline_coordinate, initial_branch_rd_spacing, initial_spacing_count)
is_single_lot_row_count = unflatten_tree(equality(single_lot_row_count, 0), longest_side_inner_loop)
coordinates_count = subtraction(list_length(branch_rd_cline_initial_coordinates), bool_to_integer(is_single_lot_row_count))
branch_rd_cline_initial_coordinates = split_list(branch_rd_cline_initial_coordinates, coordinates_count)[0]

## draw branch rd center lines & include extra branch rd
coordinate_points = evaluate_curve(longest_side_inner_loop, branch_rd_cline_initial_coordinates)[0]
frames = perp_frames(longest_side_inner_loop, branch_rd_cline_initial_coordinates)
x_axes = deconstruct_plane(frames)[1]

amp1_x_axes = amplitude(x_axes, 10000)
amp2_x_axes = amplitude(x_axes, -10000)
moved_pts1 = move_points(coordinate_points, amp1_x_axes)
moved_pts2 = move_points(coordinate_points, amp2_x_axes)
cline = two_pt_line(moved_pts2, moved_pts1)

all_branch_rd_cline = trim_with_region(cline, right_of_way_cline)[0]

## get initial branch rd outlines
offset1 = offset_by_distances(all_branch_rd_cline, right_of_way_width / 2, 1)
offset2 = offset_by_distances(all_branch_rd_cline, right_of_way_width / 2, -1)
cap1 = two_pt_line(extract_start_end_points(offset1)[0], extract_start_end_points(offset2)[0])
cap2 = two_pt_line(extract_start_end_points(offset1)[1], extract_start_end_points(offset2)[1])
merge = merge_trees(graft_tree(offset1), graft_tree(offset2), graft_tree(cap1), graft_tree(cap2))
joined_offset = join_curves(merge) 

## region union branch rd outlines with main rd outlines
brep1 = boundary_surface(joined_offset)
brep2 = boundary_surface(flatten_tree(right_of_way_outline))
extrusion1 = offset_surface_solid(brep1, 10)
extrusion2 = offset_surface_solid(brep2, 10)
union = merge_brep_coplanar_faces(solid_union(flatten_tree(merge_trees(extrusion1, extrusion2))))

## get inner row outlines & force counter-clockwise
faces = deconstruct_brep(union)[0]
centroid = area(faces)[1]
z = deconstruct_point(centroid)[2]
z0 = equality(z, 0)
z0_faces = dispatch_tree(faces, z0)[0]

snapped_z0_faces = snap_by_collision(longest_side_inner_loop, z0_faces)
edges = brep_edges(snapped_z0_faces)[0]

planes = DataTree[rg.Plane]()
planes.Add(rg.Plane.WorldXY, GH_Path(0))
joined_edges = join_curves(edges, tolerance=1e-6, plane=planes)

lengths = curve_length(joined_edges)
sorted_joined_edges = sort_list(lengths, joined_edges)[1]
inner_row_outlines = graft_tree(simplify_tree(cull_index(sorted_joined_edges, -1))) # remove outer outlines
inner_row_outlines_counter_clockwise = graft_tree(geometry_linear_sequencing_mapper(longest_side_inner_loop, data_tree_manager(flip_curve(inner_row_outlines), longest_side_inner_loop))[0])

## find the longest side of each ring
exploded2 = explode_curves(inner_row_outlines_counter_clockwise)[0]
crv_length2 = curve_length(exploded2)
sorted_length2 = sort_list(crv_length2, exploded2, True)[0]
longest_segment = list_item(sort_list(crv_length2, exploded2, True)[1], 0)

longest_length2 = list_item(sorted_length2, 0)
list_length2 = list_length(crv_length2)
series2 = series(0, 1, list_length2)
equality2 = equality(crv_length2, longest_length2)
longest_crv_index = dispatch_tree(series2, equality2)[0]
other_segments = cull_index(exploded2, longest_crv_index)

frames2 = perp_frames(longest_segment, division(longest_length2, 2))
index_projected_objects = project_point(deconstruct_plane(frames2)[0], deconstruct_plane(frames2)[1], other_segments)[1]
raw_second_side = list_item(other_segments, index_projected_objects)

## getting trimmed first side to create lots
offset0 = offset_by_distances(longest_segment, shortest_lot_length, -1) # prevent missing packing inside narrow regions
offset3 = offset_by_distances(longest_segment, longest_lot_length * -1)
trimmed_offset0 = trim_with_region(offset0, inner_row_outlines_counter_clockwise)[0]
trimmed_offset3 = trim_with_region(offset3, inner_row_outlines_counter_clockwise)[0]

rect_trimmed_offset3 = offset_curve_universal(trimmed_offset3, longest_lot_length, rg.Plane.WorldXY, False, True, True)
first_side = simplify_tree(offset_curve_universal(trimmed_offset0, shortest_lot_length, rg.Plane.WorldXY, False, True, False))

## getting second side by trimming (de-toleranced) with the bounding rectangle of lot rows of the first side
tol_rect_trimmed_offset3 = offset_curve_universal(rect_trimmed_offset3, 0.3)
offset_second_side = offset_by_distances(raw_second_side, longest_lot_length * -1)
trimmed_offset_second_side_inside = trim_with_region(offset_second_side, tol_rect_trimmed_offset3)[0]
trimmed_offset_second_side_outside = trim_with_region(offset_second_side, tol_rect_trimmed_offset3)[1]

midpt_trimmed_offset3 = curve_middle(trimmed_offset3)
distance_check = curve_closest_point(midpt_trimmed_offset3, trimmed_offset_second_side_inside)[2]
checked = larger_than(distance_check, 1)
checked_trimmed_offset_second_side_inside = dispatch_tree(trimmed_offset_second_side_inside, checked)[1]
checked_second_side = merge_trees(trimmed_offset_second_side_outside, checked_trimmed_offset_second_side_inside)
second_side = simplify_tree(offset_curve_universal(trim_with_region(checked_second_side, inner_row_outlines_counter_clockwise)[0], longest_lot_length, rg.Plane.WorldXY, False, True, False))

## bin-packing lots
both_sides = merge_trees(first_side, second_side)
both_sides_length = curve_length(both_sides)
bin_pack_output_tup = bin_pack_segments(both_sides_length, lot_length, lot_width, lot_count, lot_type, True, True)

relative_positioning = bin_pack_output_tup[0]
remaining_count = bin_pack_output_tup[1]
used_counts = bin_pack_output_tup[2]
sorted_lot_length = bin_pack_output_tup[3]
sorted_lot_width = bin_pack_output_tup[4]
packed_widths = bin_pack_output_tup[5]
packed_lengths = bin_pack_output_tup[6]
packed_types = bin_pack_output_tup[7]
sorted_lot_type = bin_pack_output_tup[8]

############# new iteration starts #############

## modify branch road cline coordinates for next iteration
point_of_excess = list_item(packed_lengths, 0)
both_point_of_excess = data_tree_manager(point_of_excess, both_sides)
list_length3 = list_length(both_point_of_excess)
equality3 = equality(list_length3, 1)
single_values = dispatch_tree(both_point_of_excess, equality3)[0]
double_values = dispatch_tree(both_point_of_excess, equality3)[1]
doubled_single_values = merge_trees(single_values, single_values)
double_both_point_of_excess = merge_trees(doubled_single_values, double_values)
filled_double_both_point_of_excess = fill_empty_branches_by_neighbors(double_both_point_of_excess)

raw_modified_branch_rd_spacing = data_tree_manager(clean_tree(addition(mass_addition(filled_double_both_point_of_excess)[0], right_of_way_width)), simplify_tree(longest_side_inner_loop))
first_item_deduction = list_item(duplicate_data(right_of_way_width / 2, list_length(raw_modified_branch_rd_spacing)), 0)
zero_placeholder = duplicate_data(0, subtraction(list_length(raw_modified_branch_rd_spacing), 1))
first_item_dedudction_list = merge_trees(first_item_deduction, zero_placeholder)

raw_modified_branch_rd_spacing = subtraction(raw_modified_branch_rd_spacing, first_item_dedudction_list)
tail_modified_branch_rd_spacing = list_item(raw_modified_branch_rd_spacing, -1) # trying to add one more road at the end
modified_branch_rd_spacing = addition(merge_trees(raw_modified_branch_rd_spacing, tail_modified_branch_rd_spacing), back_of_lot_gap)
raw_modified_branch_rd_cline_coordinates = mass_addition(modified_branch_rd_spacing)[1]

last_modified_branch_rd_cline_coordinates = list_item(raw_modified_branch_rd_cline_coordinates, -1)
minimal_module_width = addition(longest_lot_length, right_of_way_width)
is_larger = larger_than(subtraction(length_longest_side, last_modified_branch_rd_cline_coordinates), minimal_module_width)
indices_to_cull = dispatch_tree(-1, is_larger)[1]
modified_branch_rd_cline_coordinates = cull_index(raw_modified_branch_rd_cline_coordinates, indices_to_cull) # remove redundant road at the end

last_modified_branch_rd_cline_coordinates = list_item(modified_branch_rd_cline_coordinates, -1)
is_larger = larger_than(subtraction(length_longest_side, last_modified_branch_rd_cline_coordinates), minimal_module_width)
indices_to_cull = dispatch_tree(-1, is_larger)[1]
modified_branch_rd_cline_coordinates = cull_index(modified_branch_rd_cline_coordinates, indices_to_cull) # once more remove redundant road at the end

# last_modified_branch_rd_cline_coordinates = list_item(modified_branch_rd_cline_coordinates, -1)
# is_larger = larger_than(subtraction(length_longest_side, last_modified_branch_rd_cline_coordinates), minimal_module_width)
# indices_to_cull = dispatch_tree(-1, is_larger)[1]
# modified_branch_rd_cline_coordinates = cull_index(modified_branch_rd_cline_coordinates, indices_to_cull) # once more remove redundant road at the end

### inputs from initial iteration: modified_branch_rd_cline_coordinates, both_point_of_excess

## draw branch rd center lines & include extra branch rd
coordinate_points = evaluate_curve(longest_side_inner_loop, modified_branch_rd_cline_coordinates)[0]
frames = perp_frames(longest_side_inner_loop, modified_branch_rd_cline_coordinates)
x_axes = deconstruct_plane(frames)[1]

amp1_x_axes = amplitude(x_axes, 10000)
amp2_x_axes = amplitude(x_axes, -10000)
moved_pts1 = move_points(coordinate_points, amp1_x_axes)
moved_pts2 = move_points(coordinate_points, amp2_x_axes)
cline = two_pt_line(moved_pts2, moved_pts1)

all_branch_rd_cline = trim_with_region(cline, right_of_way_cline)[0]

# ## snap branch rd cline to closest main rd vertices
# vertices = explode_curves(simplify_tree(right_of_way_cline))[1]
# grafted_all_branch_rd_cline = simplify_tree(graft_tree(all_branch_rd_cline))
# vts_to_snap = find_adjacent_point(vertices, grafted_all_branch_rd_cline, longest_lot_length) 
# vts_to_snap_proj = curve_closest_point(vts_to_snap, grafted_all_branch_rd_cline)[0]
# snap_vector = vector_2pt(vts_to_snap_proj, vts_to_snap)[0]
# snapped_branch_rd_cline = move_geometry(grafted_all_branch_rd_cline, snap_vector)[0]
# boolean3 = to_boolean(list_length(snapped_branch_rd_cline))
# snapped_all_branch_rd_cline = pick_n_choose(grafted_all_branch_rd_cline, snapped_branch_rd_cline, boolean3)

## get initial branch rd outlines
offset1 = offset_by_distances(all_branch_rd_cline, right_of_way_width / 2, 1)
offset2 = offset_by_distances(all_branch_rd_cline, right_of_way_width / 2, -1)
cap1 = two_pt_line(extract_start_end_points(offset1)[0], extract_start_end_points(offset2)[0])
cap2 = two_pt_line(extract_start_end_points(offset1)[1], extract_start_end_points(offset2)[1])
merge = merge_trees(graft_tree(offset1), graft_tree(offset2), graft_tree(cap1), graft_tree(cap2))
joined_offset = join_curves(merge) 

## region union branch rd outlines with main rd outlines
brep1 = boundary_surface(joined_offset)
brep2 = boundary_surface(flatten_tree(right_of_way_outline))
extrusion1 = offset_surface_solid(brep1, 10)
extrusion2 = offset_surface_solid(brep2, 10)
union = merge_brep_coplanar_faces(solid_union(flatten_tree(merge_trees(extrusion1, extrusion2))))

## get inner row outlines & force counter-clockwise
faces = deconstruct_brep(union)[0]
centroid = area(faces)[1]
z = deconstruct_point(centroid)[2]
z0 = equality(z, 0)
z0_faces = dispatch_tree(faces, z0)[0]

snapped_z0_faces = snap_by_collision(longest_side_inner_loop, z0_faces) # sequence stabilizer
edges = brep_edges(snapped_z0_faces)[0]

planes = DataTree[rg.Plane]()
planes.Add(rg.Plane.WorldXY, GH_Path(0))
joined_edges = join_curves(edges, tolerance=1e-6, plane=planes)

lengths = curve_length(joined_edges)
sorted_joined_edges = sort_list(lengths, joined_edges)[1]
inner_row_outlines = graft_tree(simplify_tree(cull_index(sorted_joined_edges, -1))) # remove outer outlines
inner_row_outlines_counter_clockwise = graft_tree(geometry_linear_sequencing_mapper(longest_side_inner_loop, data_tree_manager(flip_curve(inner_row_outlines), longest_side_inner_loop))[0])

## find the longest side of each ring
exploded2 = explode_curves(inner_row_outlines_counter_clockwise)[0]
crv_length2 = curve_length(exploded2)
sorted_length2 = sort_list(crv_length2, exploded2, True)[0]
longest_segment = list_item(sort_list(crv_length2, exploded2, True)[1], 0)

longest_length2 = list_item(sorted_length2, 0)
list_length2 = list_length(crv_length2)
series2 = series(0, 1, list_length2)
equality2 = equality(crv_length2, longest_length2)
longest_crv_index = dispatch_tree(series2, equality2)[0]
other_segments = cull_index(exploded2, longest_crv_index)

frames2 = perp_frames(longest_segment, division(longest_length2, 2))
index_projected_objects = project_point(deconstruct_plane(frames2)[0], deconstruct_plane(frames2)[1], other_segments)[1]
raw_second_side = list_item(other_segments, index_projected_objects)

## getting trimmed first side to create lots
list_length_longest_segment = list_length(data_tree_manager(longest_segment, longest_side_inner_loop)) 
list_length_modified_offset_distances_first_side = list_length(data_tree_manager(list_item(both_point_of_excess, 0), longest_side_inner_loop)) 
diff1 = subtraction(list_length_longest_segment, list_length_modified_offset_distances_first_side)
dup1_shortest_lot_length = duplicate_data(list_item(lot_length, -1), diff1)
modified_offset_distances_first_side = graft_tree(merge_trees(data_tree_manager(list_item(both_point_of_excess, 0), longest_side_inner_loop), dup1_shortest_lot_length))

list_length_second_side = list_length(data_tree_manager(raw_second_side, longest_side_inner_loop)) 
list_length_modified_offset_distances_second_side = list_length(data_tree_manager(list_item(both_point_of_excess, 1), longest_side_inner_loop)) 
diff2 = subtraction(list_length_second_side, list_length_modified_offset_distances_second_side)
dup2_shortest_lot_length = duplicate_data(list_item(lot_length, -1), diff2)
modified_offset_distances_second_side = graft_tree(merge_trees(data_tree_manager(list_item(both_point_of_excess, 1), longest_side_inner_loop), dup2_shortest_lot_length))

offset3 = offset_by_distances(longest_segment, modified_offset_distances_first_side, -1)
trimmed_offset3 = trim_with_region(offset3, inner_row_outlines_counter_clockwise)[0]
trimmed_longest_segment = offset_by_distances(trimmed_offset3, modified_offset_distances_first_side, 1)

startpts_trimmed_longest_segment = extract_start_end_points(trimmed_longest_segment)[0]
endpts_trimmed_longest_segment = extract_start_end_points(trimmed_longest_segment)[1]
startpts_trimmed_offset3 = extract_start_end_points(trimmed_offset3)[0]
endpts_trimmed_offset3 = extract_start_end_points(trimmed_offset3)[1]
cap1 = two_pt_line(startpts_trimmed_longest_segment, startpts_trimmed_offset3)
cap2 = two_pt_line(endpts_trimmed_longest_segment, endpts_trimmed_offset3)
rect_trimmed_offset3 = join_curves(merge_trees(trimmed_longest_segment, cap1, trimmed_offset3, cap2), tolerance=1e-6, plane=planes) 

first_side = simplify_tree(offset_by_distances(trimmed_offset3, modified_offset_distances_first_side, 1))

## getting second side by trimming (de-toleranced) with the bounding rectangle of lot rows of the first side
tol_rect_trimmed_offset3 = offset_curve_universal(rect_trimmed_offset3, 0.3)
offset_second_side = offset_by_distances(raw_second_side, modified_offset_distances_second_side, -1)
trimmed_offset_second_side_inside = trim_with_region(offset_second_side, tol_rect_trimmed_offset3)[0]
trimmed_offset_second_side_outside = trim_with_region(offset_second_side, tol_rect_trimmed_offset3)[1]

midpt_trimmed_offset3 = curve_middle(trimmed_offset3)
distance_check = curve_closest_point(midpt_trimmed_offset3, trimmed_offset_second_side_inside)[2]
checked = larger_than(distance_check, 1)
checked_trimmed_offset_second_side_inside = dispatch_tree(trimmed_offset_second_side_inside, checked)[1]
checked_second_side = merge_trees(trimmed_offset_second_side_outside, checked_trimmed_offset_second_side_inside)
second_side = simplify_tree(offset_by_distances(trim_with_region(checked_second_side, inner_row_outlines_counter_clockwise)[0], modified_offset_distances_second_side))

## bin-packing lots
both_sides = merge_trees(first_side, second_side)
both_sides_length = curve_length(both_sides)
bin_pack_output_tup = bin_pack_segments(both_sides_length, lot_length, lot_width, lot_count, lot_type, True, True)

relative_positioning = bin_pack_output_tup[0]
remaining_count = bin_pack_output_tup[1]
used_counts = bin_pack_output_tup[2]
sorted_lot_length = bin_pack_output_tup[3]
sorted_lot_width = bin_pack_output_tup[4]
packed_widths = bin_pack_output_tup[5]
packed_lengths = bin_pack_output_tup[6]
packed_types = bin_pack_output_tup[7]
sorted_lot_type = bin_pack_output_tup[8]

############# iteration ends #############

division_pts = evaluate_curve(graft_tree(both_sides), relative_positioning, True, rg.Plane.WorldXY)[0]