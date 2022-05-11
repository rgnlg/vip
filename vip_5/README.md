## Vision and Image Processing Assignment #5

### Used references:
- Lecture slides 
- Forsyth, D., & Ponce, J. (2011). Computer Vision - A Modern Approach, Second Edition. 
- [Chan-Vese implementation](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_chan_vese.html)

### Used conda installation with python version: `3.8.5`

### Used libraries:
- numpy
- matplotlib
- skimage

### How to run:
- Install dependencies:
    - `conda install numpy matplotlib`
    - `conda install -c conda-forge scikit-image`
- `python3 main.py`
  
### Inputs:
##### 5 images in the 'images' folder

### Outputs:
- ##### Figure with k-means segmentation with k = 2
- ##### Figure with Otsu's segmentation
- ##### Figure with denoising and different thresholds (2 passes)
- ##### Figure with denoising and different passes (threshold = 0.5)
- ##### Figure with k-means segmentation with k = 4
- ##### Figure with Chan-Vese implementation
