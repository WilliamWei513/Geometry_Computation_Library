from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path

def compress_by_initial(processed_tree, initial_tree):
    if processed_tree is None or processed_tree.DataCount == 0:
        return DataTree[object]()

    new_tree = DataTree[object]()
    
    initial_paths = initial_tree.Paths
    flatten_all = False
    if len(initial_paths) == 1:
        flatten_all = True

    if flatten_all:
        
        top_path = GH_Path(0)
        for sp in processed_tree.Paths:
            branch = processed_tree.Branch(sp)
            for item in branch:
                new_tree.Add(item, top_path)
    else:
        path_map = {}
        for sp in initial_paths:
            top_index = sp.Indices[0]
            path_map[top_index] = GH_Path(top_index)

        for sp in processed_tree.Paths:
            branch = processed_tree.Branch(sp)
            if not branch:
                continue
            top_index = sp.Indices[0]
            if top_index in path_map:
                top_path = path_map[top_index]
            else:
                top_path = GH_Path(top_index)
            for item in branch:
                new_tree.Add(item, top_path)

    return new_tree

if processed_tree is not None and initial_tree is not None:
    result = compress_by_initial(processed_tree, initial_tree)
else:
    result = None
