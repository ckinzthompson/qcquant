# qcquant.py
Quantitative Chemotaxis Quantification (qcquant)
Version: 0.3.0

## Getting Started
``` bash
conda create -n "qcquant" python==3.9
python -m pip install "napari[all]"
python -m pip install numba matplotlib
```

## Guide

<!-- 1. Load flat field image
2. Load data image
3. [optional] Draw a circle around plate to calibrate pixel size
4. Make new Points layer
5. Add two points. First, click center of cells. Second, click background agar.
6. Locate center
7. Calculate radial average
8. Save (raw) radial average -->

<!-- ### Tutorial 1: Radial Average -->
<!-- https://github.com/ckinzthompson/qcquant/assets/17210418/02ba235a-152d-4747-a1e6-29758d60c031 -->

### Tutorial: Calibration
https://github.com/ckinzthompson/qcquant/assets/17210418/d76e4741-a87e-4ba4-9d54-4b667e5decae



## To Do List:
1. Rewrite guide; record updated tutorials
1. Integrate the curve to get growth rate (divide by time) -- need background correction
