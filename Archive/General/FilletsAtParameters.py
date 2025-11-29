from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import Grasshopper as gh
import Rhino.Geometry as rg
import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System
import System.Collections.Generic as scg

def fillet_at_parameters(curves_tree, params_tree, radii_tree, use_length=True, tolerance=None):
    """
    Create multiple fillets on a single curve per branch at specified parameters with matching radii.

    Replaces Grasshopper's Fillet-at-Parameter workflow (multi-parameter, per-branch).

    Inputs:
        curves_tree: DataTree[rg.Curve] – primary input, one curve per branch (preserved)
        params_tree: DataTree[float] – per-branch list of parameters/lengths (paired by index)
        radii_tree: DataTree[float] – per-branch list of radii (paired by index)
        use_length: bool – when True, interpret parameters as lengths along the curve; otherwise treat
                          values as curve parameters, with [0,1] interpreted as normalized domain ratio
        tolerance: float – optional, defaults to document tolerance

    Returns:
        DataTree[rg.Curve] – filleted curve per branch (structure preserved; empty branches kept). If a branch
        cannot be processed, the original curve is returned for that branch when available.
    """

    if tolerance is None:
        try:
            tolerance = sc.doc.ModelAbsoluteTolerance
        except:
            tolerance = 0.001

    result_tree = DataTree[rg.Curve]()

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

    def is_tree(x):
        return hasattr(x, 'Paths') and hasattr(x, 'Branch')

    def to_float(x):
        try:
            return float(x)
        except:
            return None

    def get_branch_values(tree_like, path):
        vals = []
        try:
            if is_tree(tree_like):
                if path in tree_like.Paths:
                    vals = list(tree_like.Branch(path))
                elif len(list(tree_like.Paths)) == 1:
                    p0 = list(tree_like.Paths)[0]
                    vals = list(tree_like.Branch(p0))
            else:
                try:
                    vals = list(tree_like)
                except:
                    vals = [tree_like]
        except:
            vals = []
        return vals

    def param_to_t(curve, val):
        if curve is None or not curve.IsValid:
            return None
        try:
            if use_length:
                L = curve.GetLength()
                if L is None:
                    L = 0.0
                try:
                    s = float(val)
                except:
                    s = 0.0
                if s < 0.0:
                    s = 0.0
                if s > L:
                    s = L
                ok, t = curve.LengthParameter(s)
                if ok:
                    return t
                return curve.Domain.Mid

            dom = curve.Domain
            try:
                r = float(val)
            except:
                r = dom.Mid
            if 0.0 <= r <= 1.0:
                return dom.T0 + r * (dom.T1 - dom.T0)
            return r
        except:
            try:
                return curve.Domain.Mid
            except:
                return None

    def try_get_polyline(curve):
        if curve is None or not curve.IsValid:
            return None

        try:
            if isinstance(curve, rg.PolylineCurve):
                ok, pl = curve.TryGetPolyline()
                if ok:
                    return pl
        except:
            pass
        try:
            if isinstance(curve, rg.Polyline):
                return curve
        except:
            pass

        try:
            plc = curve.ToPolyline()
            if plc is not None:
                ok, pl = plc.TryGetPolyline()
                if ok:
                    return pl
        except:
            pass
        return None

    def point_distance_sq(a, b):
        try:
            v = a - b
            return v.X * v.X + v.Y * v.Y + v.Z * v.Z
        except:
            return float('inf')

    def compute_corner_data(prev_pt, cur_pt, next_pt, radius):
        try:
            if radius is None:
                return None
            r = float(radius)
            if r <= 0.0:
                return None
        except:
            return None

        # Clamp radius so tangency distances fit within both incident segments
        try:
            v_in = cur_pt - prev_pt
            v_out = next_pt - cur_pt
            if not v_in.IsZero and not v_out.IsZero:
                v_in.Unitize(); v_out.Unitize()
                theta = rg.Vector3d.VectorAngle(v_in, v_out)
                # avoid degenerate
                if theta > 1e-9 and abs(System.Math.PI - theta) > 1e-9:
                    denom = System.Math.Tan(theta * 0.5)
                    if abs(denom) > 1e-12:
                        max_d = min(prev_pt.DistanceTo(cur_pt), cur_pt.DistanceTo(next_pt)) - tolerance
                        if max_d > 0:
                            max_r = max_d / denom
                            if r > max_r:
                                r = max_r
        except:
            pass

        try:
            ln1 = rg.Line(prev_pt, cur_pt)
            ln1.Extend(1000000.0, 1000000.0)
            c1 = ln1.ToNurbsCurve()
            ln2 = rg.Line(cur_pt, next_pt)
            ln2.Extend(1000000.0, 1000000.0)
            c2 = ln2.ToNurbsCurve()
        except:
            return None
        if c1 is None or c2 is None or (not c1.IsValid) or (not c2.IsValid):
            return None

        try:
            ok0, t0 = c1.ClosestPoint(cur_pt)
            ok1, t1 = c2.ClosestPoint(cur_pt)
            if not ok0 or not ok1:
                return None
        except:
            return None

        fillet = None
        try:
            fillet = rg.Curve.CreateFilletCurve(c1, t0, c2, t1, r, False, False, tolerance)
        except:
            fillet = None
        if fillet is None or not fillet.IsValid:
            try:
                fillet = rg.Curve.CreateFilletCurve(c1, t0, c2, t1, r, True, False, tolerance)
            except:
                fillet = None
        if fillet is None or not fillet.IsValid:
            return None

        try:
            ps = fillet.PointAtStart
            pe = fillet.PointAtEnd
            okas, tas = c1.ClosestPoint(ps)
            okee, tae = c1.ClosestPoint(pe)
            d_s_c1 = (ps - c1.PointAt(tas)).Length if okas else ps.DistanceTo(prev_pt)
            d_e_c1 = (pe - c1.PointAt(tae)).Length if okee else pe.DistanceTo(prev_pt)
            if d_s_c1 <= d_e_c1:
                A = ps; B = pe
            else:
                A = pe; B = ps
        except:
            A = fillet.PointAtStart
            B = fillet.PointAtEnd

        return (A, B, fillet)

    for path in curves_tree.Paths:
        result_tree.EnsurePath(path)
        try:
            items = list(curves_tree.Branch(path))
        except:
            items = []
        if not items:
            continue

        curve = None
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

        param_vals = [to_float(v) for v in get_branch_values(params_tree, path)]
        rad_vals = [to_float(v) for v in get_branch_values(radii_tree, path)]

        n_pair = min(len(param_vals), len(rad_vals))
        if n_pair <= 0:
            try:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            except:
                pass
            continue

        t_list = []
        for i in range(n_pair):
            t = param_to_t(curve, param_vals[i])
            if t is not None:
                t_list.append((t, rad_vals[i]))

        if not t_list:
            try:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            except:
                pass
            continue

        poly = try_get_polyline(curve)
        if poly is None or not poly.IsValid:
            try:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            except:
                pass
            continue

        try:
            pts = [p for p in poly]
        except:
            pts = []
        if len(pts) < 3:
            try:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            except:
                pass
            continue

        is_closed = False
        try:
            is_closed = poly.IsClosed
        except:
            try:
                is_closed = pts[0].EpsilonEquals(pts[-1], tolerance)
            except:
                is_closed = False

        if is_closed:
            if len(pts) >= 2 and pts[0].EpsilonEquals(pts[-1], tolerance):
                pts = pts[:-1]

        n_pts = len(pts)
        if n_pts < 3:
            try:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            except:
                pass
            continue

        corner_to_radius = {}
        for (t, r) in t_list:
            try:
                p = curve.PointAt(t)
            except:
                continue
            best_i = -1
            best_d = float('inf')
            start_i = 0 if is_closed else 1
            end_i = n_pts - 1 if is_closed else n_pts - 2
            for i in range(start_i, end_i + 1):
                d = point_distance_sq(p, pts[i])
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                try:
                    r0 = float(r)
                except:
                    r0 = None
                if r0 is None:
                    continue
                if best_i in corner_to_radius:
                    try:
                        corner_to_radius[best_i] = max(float(corner_to_radius[best_i]), r0)
                    except:
                        corner_to_radius[best_i] = r0
                else:
                    corner_to_radius[best_i] = r0

        if not corner_to_radius:
            try:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            except:
                pass
            continue

        A_pts = {}
        B_pts = {}
        arcs = {}

        def prev_index(i):
            return (i - 1 + n_pts) % n_pts if is_closed else (i - 1)

        def next_index(i):
            return (i + 1) % n_pts if is_closed else (i + 1)

        for idx, rad in corner_to_radius.items():
            if not is_closed and (idx <= 0 or idx >= n_pts - 1):
                continue
            i_prev = prev_index(idx)
            i_next = next_index(idx)
            if i_prev < 0 or i_next >= n_pts:
                continue
            data = compute_corner_data(pts[i_prev], pts[idx], pts[i_next], rad)
            if data is None:
                continue
            A_pts[idx], B_pts[idx], arcs[idx] = data

        if not arcs:
            try:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            except:
                pass
            continue

        pieces = []

        def add_line(a, b):
            try:
                if a.DistanceTo(b) > tolerance:
                    ln = rg.Line(a, b).ToNurbsCurve()
                    if ln is not None and ln.IsValid:
                        pieces.append(ln)
            except:
                pass

        if not is_closed:
            for i in range(0, n_pts - 1):
                a = B_pts[i] if i in B_pts else pts[i]
                b = A_pts[i + 1] if (i + 1) in A_pts else pts[i + 1]
                add_line(a, b)
                if (i + 1) in arcs:
                    pieces.append(arcs[i + 1])
        else:
            for i in range(0, n_pts):
                j = (i + 1) % n_pts
                a = B_pts[i] if i in B_pts else pts[i]
                b = A_pts[j] if j in A_pts else pts[j]
                add_line(a, b)
                if j in arcs:
                    pieces.append(arcs[j])

        new_curve = None
        try:
            joined = rg.Curve.JoinCurves(pieces, tolerance)
            if joined and len(joined) > 0 and joined[0].IsValid:
                new_curve = joined[0]
        except:
            new_curve = None
        if new_curve is None:
            try:
                pc = rg.PolyCurve()
                for seg in pieces:
                    try:
                        pc.AppendSegment(seg)
                    except:
                        try:
                            pc.Append(seg)
                        except:
                            pass
                if pc is not None and pc.IsValid:
                    new_curve = pc
            except:
                new_curve = None

        try:
            if new_curve is None or not new_curve.IsValid:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            else:
                result_tree.Add(new_curve, path)
        except Exception as e:
            try:
                print("Fillet at parameters failed for path {}: {}".format(path, e))
            except:
                pass
            try:
                dup = curve.DuplicateCurve() if hasattr(curve, 'DuplicateCurve') else curve
                if dup is not None and dup.IsValid:
                    result_tree.Add(dup, path)
            except:
                pass

    return result_tree

filleted_curves = fillet_at_parameters(curves, params, radii, True)