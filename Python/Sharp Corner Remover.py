import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System
import System.Collections.Generic as scg

def to_curve(obj):
    if isinstance(obj, Rhino.Geometry.Curve):
        return obj
    elif isinstance(obj, System.Guid):
        return rs.coercecurve(obj)
    return None

def remove_fins_by_extension(curve, extend_len="to_be_input", area_ratio=0.05):
    crv = to_curve(curve)
    if not crv:
        return None
    
    # 转成 Polyline
    if isinstance(crv, Rhino.Geometry.PolylineCurve):
        success, polyline = crv.TryGetPolyline()
        if not success:
            return None
    else:
        polyline = crv.ToPolyline(0,0,0,0,0,0,True)  
    
    if not polyline.IsClosed:
        polyline.Add(polyline[0])
    
    # 原封闭面
    base_brep_list = Rhino.Geometry.Brep.CreatePlanarBreps(
        [Rhino.Geometry.PolylineCurve(polyline)], 
        sc.doc.ModelAbsoluteTolerance
    )
    if not base_brep_list:
        return None
    base_brep = base_brep_list[0]
    
    # 炸开线段并延伸
    lines = []
    for i in range(len(polyline)-1):
        seg = Rhino.Geometry.Line(polyline[i], polyline[i+1])
        seg.Extend(extend_len, extend_len)
        lines.append(seg.ToNurbsCurve())
    
    # 用延伸线切割原面
    split_breps = [base_brep]
    for ln in lines:
        temp = []
        cutter = scg.List[Rhino.Geometry.Curve]([ln])  # ✅ 转成 Curve 集合
        for b in split_breps:
            pieces = b.Split(cutter, sc.doc.ModelAbsoluteTolerance)
            if pieces:
                temp.extend(pieces)
            else:
                temp.append(b)
        split_breps = temp
    
    # 按面积过滤
    total_area = sum(b.GetArea() for b in split_breps)
    big_faces = [b for b in split_breps if b.GetArea() >= total_area * area_ratio]
    
    # 输出去掉 fins 后的边界曲线
    result_crvs = []
    for b in big_faces:
        edges = b.DuplicateNakedEdgeCurves(True, False)
        joined = Rhino.Geometry.Curve.JoinCurves(edges, sc.doc.ModelAbsoluteTolerance)
        if joined:
            result_crvs.extend(joined)
    
    return result_crvs

# GH 输入
crv_input = curve
area_ratio_input = area_ratio  # 0.05 = 删除面积占比5%以下的面
major_curves = remove_fins_by_extension(crv_input, extend_len=500, area_ratio=area_ratio_input)