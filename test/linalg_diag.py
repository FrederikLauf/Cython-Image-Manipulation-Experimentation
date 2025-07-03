import pyximport
pyximport.install()
import cProfile
from linalg import turn_all_towards_grey
import numpy as np
import matplotlib.image as mpimg



    
if __name__ == "__main__":
    
    img = mpimg.imread(r"IMG_2533_02.png").astype(np.double)
    shape = img.shape
    dtype = img.dtype
    new_img = np.empty(shape, dtype)
    grey_vector = np.array([1, 1, 1], dtype=dtype)
    cProfile.run('turn_all_towards_grey(img, new_img, grey_vector)', sort='time')