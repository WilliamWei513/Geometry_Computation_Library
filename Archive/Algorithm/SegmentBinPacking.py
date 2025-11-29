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

def bin_pack_segments(segment_length_tree, lot_length, lot_width, lot_count, lot_type=None, chain_across_branches=False, include_start_zero=False, reverse_sort=True):

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
        _lot_data.sort(key=lambda x: x[0], reverse=reverse_sort)
        sorted_len_global = [float(x[0]) for x in _lot_data]
        sorted_wid_global = [float(x[1]) for x in _lot_data]
        sorted_cnt_global = [int(x[2]) for x in _lot_data]
        sorted_typ_global = [None for _ in _lot_data]
    else:
        _lot_data = list(zip(lot_length, lot_width, lot_count, lot_type))
        _lot_data.sort(key=lambda x: x[0], reverse=reverse_sort)
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
                    data.sort(key=lambda x: x[0], reverse=reverse_sort)
                    return ([float(x[0]) for x in data],
                            [float(x[1]) for x in data],
                            [int(x[2]) for x in data],
                            [None for _ in data])
                else:
                    data = list(zip(lot_length, lot_width, lot_count, lot_type))
                    data.sort(key=lambda x: x[0], reverse=reverse_sort)
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

segments_length = curve_length(segments)
bin_pack_output_tup = bin_pack_segments(segments_length, lot_length, lot_width, lot_count, lot_type, True, True, True)

relative_positioning = bin_pack_output_tup[0]
remaining_count = bin_pack_output_tup[1]
used_count = bin_pack_output_tup[2]
sorted_lot_length = bin_pack_output_tup[3]
sorted_lot_width = bin_pack_output_tup[4]
packed_widths = bin_pack_output_tup[5]
packed_lengths = bin_pack_output_tup[6]
packed_types = bin_pack_output_tup[7]
sorted_lot_type = bin_pack_output_tup[8]

############# iteration ends #############

division_pts = evaluate_curve(graft_tree(segments), relative_positioning, True, rg.Plane.WorldXY)[0]