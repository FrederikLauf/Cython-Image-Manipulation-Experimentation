import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import numba
import logging
import linalg


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)
logger = logging.getLogger("ImageManipulation")

def log_info(text):
    logger.info(text)

def _make_greener2(img, factor):
    img_green = img.copy()
    log_info("Make mask")
    mask_g = np.ma.indices(img_green.shape)[2] == 1
    log_info("Multiply")
    img_green = np.multiply(factor, img_green, where=mask_g)
    return img_green
    
#@numba.jit(numba.float32[:,:,:](numba.float32[:,:,:], numba.float32, numba.float32, numba.float32))
def _multiply_colors(img, red_factor, green_factor, blue_factor):
    img_new = np.ndarray(img.shape, np.float32)
    img_new[:, :, 0] = img[:, :, 0] * red_factor
    img_new[:, :, 1] = img[:, :, 1] * green_factor
    img_new[:, :, 2] = img[:, :, 2] * blue_factor
    # N, M, _ = img_new.shape
    # for i in range(N):
        # for j in range(M):
            # img_new[i, j, 0] = img[i, j, 0] * red_factor
            # img_new[i, j, 1] = img[i, j, 1] * green_factor
            # img_new[i, j, 2] = img[i, j, 2] * blue_factor
    return img_new
    
def turn_to_grey(img):
    img_new = img.astype(np.double)
    N, M, _ = img_new.shape
    grey_vector = np.array([1, 1, 1], dtype=np.double)
    total = N * M
    counter = 0
    for i in range(N):
        for j in range(M):
            vector = img_new[i, j]
            angle = linalg.angle_between_vectors(vector, grey_vector) * 0.5
            new_vector = linalg.rotate_towards_other(vector, grey_vector, angle)
            img_new[i, j] = new_vector
            counter += 1
            print(counter, total)
    return img_new
    
@numba.jit(numba.float32[:,:,:](numba.float32[:,:,:]))
def _make_contrast2(img):
    shape = img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            img[i, j] = np.square(img[i, j])
    return img

def _make_contrast(img):
    poly = np.poly1d([1, 0, 0])
    img = poly(img)
    return img