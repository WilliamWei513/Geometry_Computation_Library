def calculate_division_results_general(A, B, C):
    
    outputM = A // B
    outputR = A % B

    
    yushu = A % B
    
    
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