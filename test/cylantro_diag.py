import cProfile
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_manipulation.cynalg import cylantro


if __name__ == "__main__":
    img = mpimg.imread(r"_data/IMG_2533_02.png").astype(np.double)
    imgplot = plt.imshow(img)
    plt.show()
    shape = img.shape
    dtype = img.dtype
    new_img = np.empty(shape, dtype)
    grey_vector = np.array([1, 1, 1], dtype=dtype)
    cylantro.turn_all_towards_grey(img, new_img, grey_vector)
    cProfile.run('cylantro.turn_all_towards_grey(img, new_img, grey_vector)', sort='time')
    imgplot = plt.imshow(new_img)
    plt.show()