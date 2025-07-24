import cProfile
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_manipulation.cynalg import cylantro


if __name__ == "__main__":
    print('Profiling cylantro.turn_all_towards_other:')
    img = mpimg.imread(r"_data/IMG_2533_02.png").astype(np.double)
    shape = img.shape
    dtype = img.dtype
    new_img = np.empty(shape, dtype)
    grey_vector = np.array([1, 0.8, 0.8], dtype=dtype)
    factor = 0.7
    cProfile.run('cylantro.turn_all_towards_other(img, new_img, grey_vector, factor)', sort='time')

    print('Profiling cylantro.rotate_all_constant:')
    img = mpimg.imread(r"_data/IMG_2533_02.png").astype(np.double)
    shape = img.shape
    dtype = img.dtype
    new_img = np.empty(shape, dtype)
    axis = np.array([0, 1, 0], dtype=dtype)
    angle = - 0.3
    cProfile.run('cylantro.rotate_all_constant(img, new_img, axis, angle)', sort='time')