from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import Rhino.Geometry as rg
import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System
import System.Collections.Generic as scg

def get_distances_for_branch(offset_distances, sp, branch_length):

    distances = offset_distances.Branch(sp)
    
    if distances is None:
        
        default_branch = offset_distances.Branch(GH_Path(0))
        if default_branch is not None and len(default_branch) == 1:
            distances = [default_branch[0]] * branch_length
        else:
            distances = [0] * branch_length
    elif len(distances) == 1 and branch_length > 1:
        
        distances = [distances[0]] * branch_length

    return distances


def offset_segments(input_segments, offset_distances, side=1):
    
    if input_segments is None or input_segments.DataCount == 0:
        return DataTree[rg.Curve]()

    new_tree = DataTree[rg.Curve]()

    for sp in input_segments.Paths:
        branch = input_segments.Branch(sp)
        if not branch:
            continue
        distances = get_distances_for_branch(offset_distances, sp, len(branch))
        for i, crv in enumerate(branch):
            d = distances[i] * side
            try:
                offset_crvs = crv.Offset(rg.Plane.WorldXY, d, 0.01, rg.CurveOffsetCornerStyle.Sharp)
                if offset_crvs:
                    new_tree.Add(offset_crvs[0], sp)
            except:
                pass

    return new_tree


def data_tree_manager(processed_tree, initial_tree):
    
    if processed_tree is None or processed_tree.DataCount == 0:
        return DataTree[object]()

    new_tree = DataTree[object]()
    
    initial_paths = initial_tree.Paths
    flatten_all = (len(initial_paths) == 1)

    if flatten_all:
        top_path = GH_Path(0)
        for sp in processed_tree.Paths:
            branch = processed_tree.Branch(sp)
            for item in branch:
                new_tree.Add(item, top_path)
    else:
        path_map = {}
        for sp in initial_paths:
            top_index = sp.Indices[0]
            path_map[top_index] = GH_Path(top_index)

        for sp in processed_tree.Paths:
            branch = processed_tree.Branch(sp)
            if not branch:
                continue
            top_index = sp.Indices[0]
            top_path = path_map.get(top_index, GH_Path(top_index))
            for item in branch:
                new_tree.Add(item, top_path)

    return new_tree

if input_segments is not None and offset_distances is not None:
    raw_offsets = offset_segments(input_segments, offset_distances, side)
    offset_segments = data_tree_manager(raw_offsets, input_segments)
else:
    offset_segments = None

### non-joined offset generated at this point

### start joining raw offset lines

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

shifted_list = shift_list_tree(offset_segments, -1, True)

def graft_tree(tree):

    new_tree = DataTree[object]()

    for path in tree.Paths:
        items = list(tree.Branch(path))
        for i, item in enumerate(items):
            new_path = GH_Path(path)   
            new_path = new_path.AppendElement(i)  
            new_tree.Add(item, new_path)

    return new_tree

grafted2 = graft_tree(shifted_list)
grafted1 = graft_tree(offset_segments)

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

merged = merge_trees(grafted1, grafted2)

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

intersection_bool = curve_curve_intersection_tree(grafted1, grafted2)

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

tree_A, tree_B = dispatch_tree(merged, intersection_bool)

def extract_first_second_items(tree):

    first_line = DataTree[object]()
    second_line = DataTree[object]()
    
    for path in tree.Paths:
        items = list(tree.Branch(path))
        
        # Ensure path exists in both output trees, even if empty
        first_line.EnsurePath(path)
        second_line.EnsurePath(path)
        
        if len(items) >= 1:
            first_line.Add(items[0], path)
        
        if len(items) >= 2:
            second_line.Add(items[1], path)
    
    return first_line, second_line

first_line, second_line = extract_first_second_items(tree_B)


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

line1_start, line1_end = extract_start_end_points(first_line)
line2_start, line2_end = extract_start_end_points(second_line)


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

extended_first_line = extend_lines(first_line, 10000)
extended_second_line = extend_lines(second_line, 10000)



def line_line_intersection_points(treeA, treeB, tolerance=0.001):

    intersection_points = DataTree[rg.Point3d]()
    
    for path in treeA.Paths:
        # Ensure path exists in output tree
        intersection_points.EnsurePath(path)
        
        linesA = list(treeA.Branch(path))
        linesB = list(treeB.Branch(path)) if path in treeB.Paths else []
        
        min_count = min(len(linesA), len(linesB))
        for i in range(min_count):
            try:
                lineA = linesA[i]
                lineB = linesB[i]
                
                # Convert to Line objects if they are curves
                if hasattr(lineA, 'PointAtStart') and hasattr(lineA, 'PointAtEnd'):
                    lineA_obj = rg.Line(lineA.PointAtStart, lineA.PointAtEnd)
                else:
                    lineA_obj = lineA
                    
                if hasattr(lineB, 'PointAtStart') and hasattr(lineB, 'PointAtEnd'):
                    lineB_obj = rg.Line(lineB.PointAtStart, lineB.PointAtEnd)
                else:
                    lineB_obj = lineB
                
                # Use Line-Line intersection
                success, tA, tB = rg.Intersect.Intersection.LineLine(lineA_obj, lineB_obj)
                
                if success:
                    # Check if intersection is within both line segments
                    if 0 <= tA <= 1 and 0 <= tB <= 1:
                        intersection_pt = lineA_obj.PointAt(tA)
                        intersection_points.Add(intersection_pt, path)
                    # Also check for extended intersection (since we extended the lines)
                    else:
                        intersection_pt = lineA_obj.PointAt(tA)
                        intersection_points.Add(intersection_pt, path)
                        
            except Exception as e:
                # Debug: print error for troubleshooting
                print(f"Line intersection failed for path {path}, index {i}: {e}")
                pass
    
    return intersection_points

intersection_pts = line_line_intersection_points(extended_first_line, extended_second_line)

def two_pt_line(start_pts, end_pts):
    lines_tree = DataTree[rg.Line]()
    
    for path in start_pts.Paths:
        # Ensure path exists in output tree
        lines_tree.EnsurePath(path)
        
        # Get start and end points for this path
        start_points = list(start_pts.Branch(path))
        end_points = list(end_pts.Branch(path)) if path in end_pts.Paths else []
        
        # Create lines from corresponding points
        min_count = min(len(start_points), len(end_points))
        for i in range(min_count):
            try:
                start_pt = start_points[i]
                end_pt = end_points[i]
                
                # Create line from start to end point
                line = rg.Line(start_pt, end_pt)
                lines_tree.Add(line, path)
                
            except Exception as e:
                print(f"Line creation failed for path {path}, index {i}: {e}")
                pass
    
    return lines_tree

joined_line1 = two_pt_line(line1_start, intersection_pts)
joined_line2 = two_pt_line(intersection_pts, line2_end)

joined_offset = merge_trees(joined_line1, joined_line2, tree_A)

first_line_all, second_line_all = extract_first_second_items(joined_offset)
intersection_pts_all = line_line_intersection_points(first_line_all, second_line_all)

comform_isct_pts_all = data_tree_manager(intersection_pts_all, input_segments)

def points_to_closed_polyline(points_tree):

    polylines_tree = DataTree[rg.Polyline]()
    
    for path in points_tree.Paths:
        # Ensure path exists in output tree
        polylines_tree.EnsurePath(path)
        
        points = list(points_tree.Branch(path))
        
        if len(points) >= 3:  # Need at least 3 points for a closed polyline
            try:
                # Create polyline from points
                polyline = rg.Polyline(points)
                
                # Ensure it's closed
                if not polyline.IsClosed:
                    polyline.Add(polyline[0])  # Add first point at the end to close
                
                polylines_tree.Add(polyline, path)
                
            except Exception as e:
                print(f"Polyline creation failed for path {path}: {e}")
                pass
        elif len(points) == 2:
            # If only 2 points, create a line and close it
            try:
                polyline = rg.Polyline([points[0], points[1], points[0]])
                polylines_tree.Add(polyline, path)
            except Exception as e:
                print(f"Line polyline creation failed for path {path}: {e}")
                pass
        # Skip if less than 2 points
    
    return polylines_tree

closed_polylines = points_to_closed_polyline(comform_isct_pts_all)

### offset join completed 

### start cleaning & simplifying geometry

def remove_small_fins(polylines_tree, tol=1e-6, min_ratio=0.15):

    def remove_small_fins(crv, tol, min_ratio):
        events = rg.Intersect.Intersection.CurveSelf(crv, tol)
        if not events:
            return crv  

        params = sorted(set([e.ParameterA for e in events] + [e.ParameterB for e in events]))
        segments = crv.Split(params)
        if not segments:
            return crv

        clean_segments = []
        crvlen = crv.GetLength()
        for seg in segments:
            seglen = seg.GetLength()
            ratio = seglen / crvlen if crvlen > 0 else 0
            if ratio > min_ratio:
                clean_segments.append(seg)

        joined = rg.Curve.JoinCurves(clean_segments, tol)
        if not joined or len(joined) == 0:
            return crv  
        return joined[0] if len(joined) == 1 else joined

    cleaned_tree = DataTree[rg.Curve]()

    for path in polylines_tree.Paths:
        for polyline in polylines_tree.Branch(path):
            if not polyline.IsValid or polyline.Count < 2:
                continue

            if not polyline.IsClosed:
                polyline.Add(polyline[0])

            curve = rg.PolylineCurve(polyline)
            cleaned = remove_small_fins(curve, tol=tol, min_ratio=min_ratio)

            if isinstance(cleaned, list):
                for c in cleaned:
                    cleaned_tree.Add(c, path)
            elif cleaned:
                cleaned_tree.Add(cleaned, path)

    return cleaned_tree

min_ratio = 0.15 
fin_removed1 = remove_small_fins(closed_polylines, tol=sc.doc.ModelAbsoluteTolerance, min_ratio=min_ratio)

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

fin_removed1 = clean_tree(fin_removed1, True, True, True)

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

exploded_vertices = explode_curves(fin_removed1)[1]
reconstructed_polyline = points_to_closed_polyline(exploded_vertices)


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

major_curves = remove_sharp_corner(reconstructed_polyline, extend_len=500, area_ratio=0.06)

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

boundary_surfaces = boundary_surface(major_curves)

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

joined_surfaces = join_breps(boundary_surfaces)

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

merged_surfaces = merge_coplanar_faces(boundary_surfaces)

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

brep_edges = deconstruct_brep(merged_surfaces)[1]

def join_curves(curves_tree, tolerance=1e-6):

    joined_tree = DataTree[rg.Curve]()
    for path in curves_tree.Paths:
        curves = [c for c in curves_tree.Branch(path) if c is not None and c.IsValid]
        if curves:
            try:
                joined_curves = rg.Curve.JoinCurves(curves, tolerance)
                if joined_curves:
                    for jc in joined_curves:
                        joined_tree.Add(jc, path)
                else:
                    # If nothing was joined, add original curves
                    for c in curves:
                        joined_tree.Add(c, path)
            except Exception as e:
                print(f"JoinCurves failed for path {path}: {e}")
                for c in curves:
                    joined_tree.Add(c, path)
        else:
            # If no valid curves, add None to preserve structure
            joined_tree.Add(None, path)
    return joined_tree

joined_edges = join_curves(brep_edges)
brep_vertices = explode_curves(joined_edges)[1]
brep_polyline = points_to_closed_polyline(brep_vertices)

explode_brep_polyline = explode_curves(brep_polyline)[0]