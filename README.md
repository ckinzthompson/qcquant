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


## Tutorial
https://github.com/ckinzthompson/qcquant/assets/17210418/77c7c6e9-a633-45f0-9cb1-8a37c469afaf

