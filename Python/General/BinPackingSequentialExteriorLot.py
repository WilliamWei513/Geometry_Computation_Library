import Grasshopper as gh

def pack_segments(segment_length, lot_length, lot_width, lot_count):
    # Step 1. 排序
    lot_data = list(zip(lot_length, lot_width, lot_count))
    lot_data.sort(key=lambda x: x[0], reverse=True)
    lot_length, lot_width, lot_count = zip(*lot_data)

    # Step 2. 初始化结果
    tree_result = [[] for _ in segment_length]
    used_counts = [0]*len(lot_width)
    finished = [False]*len(lot_width)

    seg_idx = 0
    current_sum = 0

    # Step 3. 遍历 lot_width
    for i in range(len(lot_width)):
        w = lot_width[i]
        c = lot_count[i]

        for _ in range(c):
            if seg_idx >= len(segment_length):
                break

            if current_sum + w <= segment_length[seg_idx]:
                current_sum += w
                tree_result[seg_idx].append(current_sum)
                used_counts[i]+=1
            else:
                seg_idx += 1
                if seg_idx >= len(segment_length):
                    break
                current_sum = 0
                if current_sum + w <= segment_length[seg_idx]:
                    current_sum += w
                    tree_result[seg_idx].append(current_sum)
                    used_counts[i]+=1

        finished[i] = (used_counts[i] == lot_count[i])

    status = [w if f else "not packed" for w,f in zip(lot_width, finished)]
    return tree_result, status, used_counts, list(lot_length), list(lot_width)


# -----------------------------
# GH 输入输出部分
# -----------------------------
# 输入：segment_tree (Tree Access), lot_length, lot_width, lot_count (List Access)

gh_tree = gh.DataTree[object]()
status_tree = gh.DataTree[object]()
used_counts_tree = gh.DataTree[object]()
sorted_len_tree = gh.DataTree[object]()
sorted_wid_tree = gh.DataTree[object]()

for branch in segment_tree.Paths:
    segs = list(segment_tree.Branch(branch))

    tree_result, status, used_counts, sorted_len, sorted_wid = pack_segments(segs, lot_length, lot_width, lot_count)

    # 每个 segment -> {i;j}
    for i, lst in enumerate(tree_result):
        new_path = branch.AppendElement(i)

        if lst:  # 如果有结果
            for val in lst:
                gh_tree.Add(val, new_path)
        else:    # 如果为空，写入一个 Null 占位
            gh_tree.Add(None, new_path)

    # 其他结果只挂在 branch 上（第一层）
    status_tree.AddRange(status, branch)
    used_counts_tree.AddRange(used_counts, branch)
    sorted_len_tree.AddRange(sorted_len, branch)
    sorted_wid_tree.AddRange(sorted_wid, branch)

# 输出
tree_result = gh_tree          # 每个 segment 的累加结果
status = status_tree      # lot_width 用完/没用完（对应每个segment）
used_counts = used_counts_tree # 实际使用数量（对应每个segment）
sorted_lot_length = sorted_len_tree  # 排序后的 lot_length（对应每个segment）
sorted_lot_width = sorted_wid_tree  # 排序后的 lot_width（对应每个segment）