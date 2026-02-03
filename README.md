# Geometry Computation Library

Geometric algorithms for applied CAD computation in Grasshopper.

## Overview

Python functions that replace Grasshopper components, optimized for correctness, robustness, and DataTree fidelity. Designed for Rhino/Grasshopper environments.

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

### Requirements

- Rhino 3D with Grasshopper
- Python (IronPython in Grasshopper environment)
- Standard library: `math`, `json`, `System`

## Testing

Run tests with pytest:

```bash
# exmaple
cd Scripts/GIS
pytest test_LatLonJsonToFeet.py -v
```

## Unique Feature

Functions follow Grasshopper component patterns:
- Accept scalar or DataTree inputs
- Preserve primary input's tree structure
- Return DataTree outputs with path correspondence
- Handle empty branches gracefully

