import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System

def to_curve(obj):
    """确保输入是 Rhino.Geometry.Curve"""
    if isinstance(obj, Rhino.Geometry.Curve):
        return obj
    elif isinstance(obj, System.Guid):
        return rs.coercecurve(obj)  # 将 GUID 转为 Curve
    return None

def remove_small_fins(crv, tol=1e-6, min_len=1.0):
    crv = to_curve(crv)
    if not crv:
        return None

    # 1. 找自交点
    events = Rhino.Geometry.Intersect.Intersection.CurveSelf(crv, tol)
    if not events:
        return crv  # 没有自交直接返回

    # 2. 获取分割参数
    params = sorted(set([e.ParameterA for e in events] + [e.ParameterB for e in events]))

    # 3. 分割曲线
    segments = crv.Split(params)
    if not segments:
        return crv

    # 4. 过滤掉小段
    clean_segments = [seg for seg in segments if seg.GetLength() > min_len]

    # 5. 合并成一条曲线
    joined = Rhino.Geometry.Curve.JoinCurves(clean_segments, tol)
    if joined:
        return joined[0] if len(joined) == 1 else joined
    return None

# Grasshopper 输入
crv = curve  # x输入曲线
min_len = fin_length  # 最小保留长度（单位）

result = remove_small_fins(crv, tol=sc.doc.ModelAbsoluteTolerance, min_len=min_len)
cleaned = result
