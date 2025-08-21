import Rhino
import rhinoscriptsyntax as rs
import System


def _coerce_curve(obj):
    if isinstance(obj, Rhino.Geometry.Curve):
        return obj
    try:
        curve = rs.coercecurve(obj, True)
        return curve
    except Exception:
        return None


def CreateClosedPolylineFromSegments(segments_in):
    """
    将一组线段连接起来，并自动处理交点以形成新的闭合多边形。
    
    Args:
        segments_in (list[Rhino.Geometry.Curve or Guid]): 要连接的线段数组（可为曲线或 Rhino 对象 GUID）。
        
    Returns:
        Rhino.Geometry.Curve: 新的闭合多边形曲线，如果失败则返回 None。
    """
    if not segments_in:
        print("错误：无效的输入，请提供线段数组。")
        return None

    # 将任何 GUID 强制转换为 Curve
    coerced_segments = []
    for i, seg in enumerate(segments_in):
        curve = _coerce_curve(seg)
        if not curve:
            print("错误：第 {} 条输入无法转为曲线。".format(i))
            return None
        coerced_segments.append(curve)

    # 找到新的交点并重新创建多边形
    new_points = []
    extended_curves = []  # 保存延长后的线段用于检查
    tol = 0.01 # 默认公差
    try:
        import scriptcontext as sc
        if hasattr(sc.doc, 'ModelAbsoluteTolerance'):
            tol = sc.doc.ModelAbsoluteTolerance
    except Exception:
        pass

    # 首先尝试延长所有线段
    extended_segments = []
    for i, segment in enumerate(coerced_segments):
        # 延长每条线段 - 在两端各延长10000个单位
        extended_segment = segment.Extend(10000, 10000)
        if extended_segment:
            extended_segments.append(extended_segment)
            extended_curves.append(extended_segment)
            print(f"线段 {i} 已延长")
        else:
            print(f"警告：无法延长线段 {i}")
            return None, None

    # 检测所有线段之间的交点
    intersection_points = []
    for i in range(len(extended_segments)):
        for j in range(i + 1, len(extended_segments)):
            current_segment = extended_segments[i]
            other_segment = extended_segments[j]
            
            # 检测两条线段之间的交点
            ccx = Rhino.Geometry.Intersect.Intersection.CurveCurve(
                current_segment, other_segment, tol, tol
            )
            
            if ccx and ccx.Count > 0:
                for k in range(ccx.Count):
                    intersection_point = ccx[k].PointA
                    # 避免重复的交点
                    is_duplicate = False
                    for existing_point in intersection_points:
                        if intersection_point.DistanceTo(existing_point) < tol:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        intersection_points.append(intersection_point)
                        print(f"找到交点：线段 {i} 与线段 {j} 相交")

    # 使用所有交点创建闭合多边形
    if len(intersection_points) >= 3:
        # 按角度排序交点（相对于中心点）
        center = Rhino.Geometry.Point3d()
        for point in intersection_points:
            center += point
        center /= len(intersection_points)
        
        # 按角度排序 - 使用正确的方法
        def get_angle(point):
            vector = point - center
            if vector.Length > 0:
                # 计算与X轴的角度
                angle = Rhino.Geometry.Vector3d.Multiply(vector, 1.0/vector.Length)
                return Rhino.Geometry.Vector3d.VectorAngle(Rhino.Geometry.Vector3d.XAxis, angle)
            return 0.0
        
        sorted_points = sorted(intersection_points, key=get_angle)
        
        # 创建闭合多段线
        polyline = Rhino.Geometry.Polyline(sorted_points)
        # 确保第一个点和最后一个点相同以创建闭合多段线
        if len(sorted_points) > 1 and sorted_points[0].DistanceTo(sorted_points[-1]) > tol:
            polyline.Add(sorted_points[0])
        
        return polyline.ToNurbsCurve(), extended_curves
    else:
        print("错误：找到的交点数量不足，无法形成多边形")
        return None, None


# 获取 GHPython 的输入
segments_in = segments_to_connect  # 假设新的输入变量名是 segments_to_connect

# 调用核心函数并设置输出
closed_curve = None
extended_curves_output = None

if segments_in:
    closed_curve, extended_curves_output = CreateClosedPolylineFromSegments(segments_in)

# 将结果赋值给 GHPython 的输出端口
output_closed_curve = closed_curve
extended_curves_for_check = extended_curves_output  # 新增输出：延长后的线段列表
