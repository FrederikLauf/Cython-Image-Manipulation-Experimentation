import numpy as np

def rotation_matrix(angle, axis, i, j):
    c = np.cos(angle)
    s = np.sin(angle)
    if i != j:
        idx = [0, 1, 2]
        idx.remove(i)
        idx.remove(j)
        k = idx[0]
        return (1 - c) * axis[i] * axis[j] - s * epsilon(i, j, k) * axis[k]
    else:
        return (1 - c) * axis[i] * axis[j] + c * delta(i, j)
    
def delta(i, j):
    if i == j:
        return 1
    else:
        return 0
    
def epsilon(i, j, k):
    if (i == 0 and j == 1 and k == 2) or (i == 2 and j == 0 and k == 1) or (i == 1 and j == 2 and k == 0):
        return 1
    elif (i == 0 and j == 2 and k == 1) or (i == 2 and j == 1 and k == 0) or (i == 1 and j == 0 and k == 2):
        return -1
    else:
        return 0

def cross_product(v, w, i):
    cross_i = 0
    for j in range(3):
        for k in range(3):
            cross_i += epsilon(i, j, k) * v [j] * w[k]
    return cross_i
    
def scalar_product(v, w):
    scalar = 0
    for j in range(3):
        scalar += v [j] * w[j]
    return scalar
    

def vector_norm(v):
    norm = 0
    N = len(v)
    for i in range(N):
        norm += v[i] * v[i]
    return np.sqrt(norm)


def angle_between_vectors(v, w):
    denominator = scalar_product(v, w)
    nominator = vector_norm(v) * vector_norm(w)
    if nominator == 0.0:
        return 0.0
    return np.arccos(denominator / nominator)
    
def turn_all_towards_grey(img_in, img_out, grey):
    N = img_in.shape[0]
    M = img_in.shape[1]
    for i in range(N):
        for j in range(M):
            current_v = img_in[i, j]
            angle = 0.5 * angle_between_vectors(current_v, grey)
            img_out[i, j, 0] = angle
            img_out[i, j, 1] = angle
            img_out[i, j, 2] = angle