## Vision and Image Processing Assignment #3

### Used references:
- Lecture slides
- Forsyth & Ponce - Computer Vision: A Modern Approach, Second Edition

### Used conda installation with python version: `3.8.5`

### Used libraries:
- numpy
- matplotlib
- mayavi version 4.7.2
- scipy
- the provided `ps_utils.py` module

### How to run:
- Install dependencies: 
  - `conda install numpy matplotlib scipy`
  - `conda install -c anaconda pyqt5`
  - `conda install -c conda-forge mayavi`
- Run the main script: `python3 main.py ` without any program arguments to test all the databases one by one, or
- Run `python3 main.py DATASET_NAME` to view the results for a specific database.
  - Valid dataset names: [beethoven, mat_vase, shiny_vase, shiny_vase2, buddha, face]

### Inputs:
##### The .mat files from the data/ directory

### Outputs:
##### matplotlib figures with inputs/albedo/norms and mayavi 3D reconstructions
##### Figures showing the results of processing the following:
- The Beethoven dataset (1 matplotlib, 1 mayavi) 
  -  [PSEUDOINVERSE]
- The mat_vase dataset (1 matplotlib, 1 mayavi) 
  - [PSEUDOINVERSE]
- The shiny_vase dataset (4 matplotlib, 4 mayavi)
  - [PSEUDOINVERSE, RANSAC, PSEUDOINVERSE+SMOOTHING, RANSAC+SMOOTHING]
- The shiny_vase2 dataset (4 matplotlib, 4 mayavi)
  - [PSEUDOINVERSE, RANSAC, PSEUDOINVERSE+SMOOTHING, RANSAC+SMOOTHING]
- The Buddha dataset (4 matplotlib, 4 mayavi)
  - [PSEUDOINVERSE, RANSAC, PSEUDOINVERSE+SMOOTHING, RANSAC+SMOOTHING]
- The face dataset (2 maplotlib, 2 mayavi)
  - [RANSAC+SMOOTHING (200 iters), RANSAC+SMOOTHING (2 iters)]
