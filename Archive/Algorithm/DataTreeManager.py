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

conformed_tree = data_tree_manager(processed_tree, initial_tree)