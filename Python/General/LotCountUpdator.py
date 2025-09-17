# -*- coding: utf-8 -*-
"""
持续更新列表，带最高优先级的重置和自动重置功能

输入:
  - list_in: 外部输入的初始列表 (List Access)
  - index_list: 需要更新的索引列表 (List Access)
  - value_list: 与 index_list 一一对应的新值列表 (List Access)
  - reset: 一个布尔值，用于手动重置列表 (Item Access)
  - button_trigger: 按钮触发信号，只有为True时才执行更新 (Item Access)
  - ID: 一个字符串，用于为每个组件创建唯一的标识符 (Item Access)

输出:
  - updated_list: 更新后的新列表
"""
import scriptcontext as sc
import json

# 检查 ID 输入，如果为空则使用默认值
if ID is None or ID.strip() == "":
    print("警告: ID 输入为空，使用默认 ID。请为每个组件提供一个唯一的 ID。")
    unique_id = "default"
else:
    unique_id = str(ID)

# 使用唯一的 ID 作为键的前缀
list_key = f"{unique_id}_sticky_list"
list_in_hash_key = f"{unique_id}_list_in_hash"

# ----------------- 检查 list_in 是否改变以触发自动重置 -----------------
current_list_in_hash = json.dumps(list_in)
stored_list_in_hash = sc.sticky.get(list_in_hash_key)

needs_reset = reset or (stored_list_in_hash is not None and current_list_in_hash != stored_list_in_hash)

if needs_reset:
    sc.sticky[list_key] = list(list_in)
    sc.sticky[list_in_hash_key] = current_list_in_hash
    print(f"[{unique_id}] 已执行列表重置。")

# ----------------- 常规更新逻辑 -----------------
else:
    if sc.sticky.get(list_key) is None:
        sc.sticky[list_key] = list(list_in)
        sc.sticky[list_in_hash_key] = current_list_in_hash
        print(f"[{unique_id}] 首次运行，列表已初始化。")

    stored_list = sc.sticky.get(list_key)

    if button_trigger:
        # index_list 与 value_list 验证
        if not isinstance(index_list, (list, tuple)) or not isinstance(value_list, (list, tuple)):
            print(f"[{unique_id}] 警告：index_list 与 value_list 需要为列表。")
        elif len(index_list) != len(value_list):
            print(f"[{unique_id}] 警告：index_list 与 value_list 长度不一致。")
        elif not stored_list:
            print(f"[{unique_id}] 警告：列表为空，无法更新。")
        else:
            for raw_idx, val in zip(index_list, value_list):
                try:
                    idx = int(raw_idx)
                except Exception:
                    print(f"[{unique_id}] 警告：索引 {raw_idx} 不是有效整数，跳过。")
                    continue
                if 0 <= idx < len(stored_list):
                    old_value = stored_list[idx]
                    stored_list[idx] = val
                    print(f"[{unique_id}] 已更新索引 {idx}: {old_value} -> {val}")
                else:
                    print(f"[{unique_id}] 警告：输入的索引 {idx} 超出范围。")

updated_list = sc.sticky.get(list_key, list(list_in))
sc.sticky[list_key] = updated_list