def calculate_division_results_general(A, B, C):
    """
    根据通用整除逻辑计算结果。
    
    Args:
        A (int or float): 第一个数。
        B (int or float): 第一个除数。
        C (int or float): 第二个除数。

    Returns:
        tuple: 包含 (outputR, outputL) 的元组。
    """
    # 按照你的新要求，outputR 就是 A 除以 B 的商
    outputM = A // B
    outputR = A % B

    # 首先计算 A 除以 B 的余数
    yushu = A % B
    
    # 然后用余数除以 C，得到的商就是 outputL
    outputS = yushu // C
    
    return (outputM, outputS, outputR)

A = longest_edge_length
B = module_width
C = lot_depth

calculate_division_results_general(A, B, C)

module_count, single_lot_row_count, residue = calculate_division_results_general(A, B, C)
module_count = int(module_count)
single_lot_row_count = int(single_lot_row_count)

print ((module_count,single_lot_row_count,residue))