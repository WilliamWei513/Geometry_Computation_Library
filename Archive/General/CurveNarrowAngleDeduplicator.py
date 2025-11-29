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

def deduplicate_lines_by_angle(lines_tree, min_angle_deg=5.0, max_angle_deg=175.0):
	"""
	Deduplicate lines within each branch by removing later-index lines whose angle to an earlier line is < min_angle_deg or > max_angle_deg.

	Parameters:
		lines_tree: DataTree of lines/curves/line-like objects
		min_angle_deg: float, small-angle threshold in degrees (default 5)
		max_angle_deg: float, large-angle threshold in degrees (default 175)

	Returns:
		DataTree[object]: Deduplicated items, branch structure preserved and empty branches kept.
	"""
	result = DataTree[object]()
	def _path_key(p):
		try:
			return tuple(p.Indices)
		except:
			return tuple()
	for path in sorted(list(lines_tree.Paths), key=_path_key):
		tgt_path = GH_Path(path)
		result.EnsurePath(tgt_path)
		items = list(lines_tree.Branch(path))
		if not items:
			continue
		vecs = []
		for it in items:
			try:
				start = None
				end = None
				if isinstance(it, rg.Line):
					start = it.From
					end = it.To
				else:
					try:
						if hasattr(it, 'PointAtStart') and hasattr(it, 'PointAtEnd'):
							start = it.PointAtStart
							end = it.PointAtEnd
						else:
							c = coerce_to_curve(it)
							if c is not None and hasattr(c, 'IsValid') and c.IsValid:
								start = c.PointAtStart
								end = c.PointAtEnd
					except:
						start = None
						end = None
				if start is not None and end is not None:
					v = end - start
					if not v.IsZero:
						v.Unitize()
						vecs.append(v)
					else:
						vecs.append(None)
				else:
					vecs.append(None)
			except:
				vecs.append(None)
		keep = [True] * len(items)
		import math
		def _clamp(x, lo, hi):
			try:
				if x < lo:
					return lo
				if x > hi:
					return hi
				return x
			except:
				return x
		for i in range(len(items)):
			if not keep[i]:
				continue
			vi = vecs[i]
			if vi is None:
				continue
			for j in range(i + 1, len(items)):
				if not keep[j]:
					continue
				vj = vecs[j]
				if vj is None:
					continue
				try:
					dot = rg.Vector3d.Multiply(vi, vj)
				except:
					dot = 0.0
				dot = _clamp(dot, -1.0, 1.0)
				try:
					ang = math.degrees(math.acos(dot))
				except:
					ang = 0.0 if dot >= 0.0 else 180.0
				if (ang < float(min_angle_deg)) or (ang > float(max_angle_deg)):
					keep[j] = False
		for idx, it in enumerate(items):
			if keep[idx]:
				try:
					result.Add(it, tgt_path)
				except:
					pass
	return result

deduplicated_lines = deduplicate_lines_by_angle(lines)