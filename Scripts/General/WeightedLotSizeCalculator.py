import Rhino.Geometry as rg

# lot_length : list[float]
# lot_width : list[float]
# lot_ratio : list[str]   # e.g. ["10%", "5%", "60%", "25%"]
# lot_count : list[float] or float

def parse_ratio(r):
    
    if not r:  
        return 0.0
    r = str(r).strip()
    if r.endswith("%"):
        try:
            return float(r[:-1]) / 100.0
        except:
            return 0.0
    else:
        try:
            return float(r)
        except:
            return 0.0

def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

lot_length = _as_list(globals().get("lot_length"))
lot_width = _as_list(globals().get("lot_width"))
_counts_input = _as_list(globals().get("lot_count"))
_ratios_input = _as_list(globals().get("lot_ratio"))

base_n = min(len(lot_length), len(lot_width))

if len(_counts_input) > 0:
    weights = []
    for c in _counts_input:
        try:
            w = float(c)
            if w < 0:
                w = 0.0
            weights.append(w)
        except:
            weights.append(0.0)
    if len(weights) == 1 and base_n > 1:
        weights = weights * base_n
else:
    weights = [parse_ratio(r) for r in _ratios_input]
    if len(weights) == 1 and base_n > 1:
        weights = weights * base_n


n = min(base_n, len(weights))
lot_length = lot_length[:n]
lot_width = lot_width[:n]
weights = weights[:n]

total_weight = sum(weights)

if total_weight == 0:
    weighted_length = None
    weighted_width = None
else:
    weighted_length = sum(l * w for l, w in zip(lot_length, weights)) / total_weight
    weighted_width  = sum(wd * w for wd, w in zip(lot_width, weights)) / total_weight

# enforce scalar outputs
if isinstance(weighted_length, (list, tuple)):
    weighted_length = weighted_length[0] if len(weighted_length) > 0 else None
if isinstance(weighted_width, (list, tuple)):
    weighted_width = weighted_width[0] if len(weighted_width) > 0 else None


weighted_lot_length = weighted_length
weighted_lot_width = weighted_width