## Vision and Image Processing Assignment #4

### Used references:
- Lecture slides 
- Forsyth, D., & Ponce, J. (2011). Computer Vision - A Modern Approach, Second Edition. 
- Lowe, D. (2004) Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110. https://doi.org/10.1023/B:VISI.0000029664.99615.94
- https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html


### Used conda installation with python version: `3.8.5`

### Used libraries:
- numpy
- matplotlib
- sklearn
- openCV
- pandas

### How to run:
- Install dependencies:
    - `conda install numpy matplotlib sklearn pandas`
    - `conda install -c conda-forge opencv`
- `python3 main.py` with or without arguments
- Available arguments: `[common, tfidf (default), clean]`
  - `common` performs retrieval based on 'common words' similarity measure
  - `tfidf` is the default and performs retrieval based on TF-IDF vector values and cosine similarity
  - `clean` removes any generated bags.csv, forcing the BoW to be generated again. Can be passed with either `common`, `tfidf`, or by itself
  
### Inputs:
##### The [Caltech 101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)

### Outputs:
- ##### Predictions on test/train images
- ##### 2 matplotlib figures demonstrating image retrieval on a randomly selected image from each experiment
