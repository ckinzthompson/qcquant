# qcquant
Quantitative Chemotaxis Quantification


## Getting Started
``` bash
conda create -n "qcquant" python==3.9
python -m pip install "napari[all]"
python -m pip install numba matplotlib
```

## Guide
1. Load flat field image
2. Load data image
3. [optional] Draw a circle around plate to calibrate pixel size
4. Make new Points layer
5. Add two points. First, click center of cells. Second, click background agar. 
6. Locate center 
7. Calculate radial average
8. Save (raw) radial average
