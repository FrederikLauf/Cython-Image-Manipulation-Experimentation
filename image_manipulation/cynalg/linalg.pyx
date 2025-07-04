## cython: profile=True

from cython cimport cdivision, boundscheck, wraparound
from cpython cimport array
from libc.math cimport sqrt, acos, sin, cos
import numpy as np


cpdef double rotation_matrix_ij(double angle, double[:] axis, int i, int j):
    cdef:
        double c = cos(angle)
        double s = sin(angle)
        int k
    if i != j:
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            k = 2
        elif i == 0 or j == 0:
            k = 1
        else:
            k = 0
        return (1 - c) * axis[i] * axis[j] - s * epsilon(i, j, k) * axis[k]
    else:
        return (1 - c) * axis[i] * axis[j] + c


cpdef rotate_vector(double[:] v, double[:] w, double angle, double[:] axis):
    cdef:
        int i, j
    for i in range(3):
        for j in range(3):
            w[i] = rotation_matrix_ij(angle, axis, i, j) * v[j]


cpdef inline int delta(int i, int j):
    if i == j:
        return 1
    else:
        return 0


cpdef inline int epsilon(int i, int j, int k):
    if i == j or i == k or j == k:
        return 0
    elif (i == 0 and j == 1) or (i == 2 and j == 0) or (i == 1 and j == 2):
        return 1
    else:
        return -1


cpdef double[:] cross_product(double[:] v, double[:] w):
    cdef:
        double[:] cross = v.copy()
    cross[0] = v[1] * w[2] - v[2] * w[1]
    cross[1] = v[2] * w[0] - v[0] * w[2]
    cross[2] = v[0] * w[1] - v[1] * w[0]
    return cross


cpdef inline double scalar_product(double[:] v, double[:] w):
    return v[0] * w[0] + v[1] * w[1] +v[2] * w[2] 


cpdef inline double vector_norm(double[:] v):
    return sqrt(v[0] * v[0] + v[1] * v[1] +v[2] * v[2])


@cdivision(True)
cpdef double[:] normalized_vector(double[:] v):
    cdef:
        double[:] normalized = v.copy()
        int i
        double norm = vector_norm(v)
    if norm == 0.0:
        return normalized
    for i in range(3):
        normalized[i] /= norm
    return normalized


@cdivision(True)
cpdef normalize_vector(double[:] v):
    cdef:
        int i
        double norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm != 0.0:
        for i in range(3):
            v[i] /= norm


@cdivision(True)
cpdef double _angle_between_vectors(double[:] v, double[:] w):
    cdef:
        double nominator = vector_norm(v) * vector_norm(w)
    if nominator == 0.0:
        return 0.0
    return acos(scalar_product(v, w) / nominator)


@cdivision(True)
cpdef double angle_between_vectors(double[:] v, double[:] w):
    cdef:
        double nominator = (sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) *
                            sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]))
    if nominator == 0.0:
        return 0.0
    return acos((v[0] * w[0] + v[1] * w[1] +v[2] * w[2]) / nominator)


cpdef double[:] __rotation_axis_between_vectors(double[:] v, double[:] w):
    cdef:
        double[:] cross = cross_product(v, w)
    normalize_vector(cross)
    return cross


@cdivision(True)
cpdef (double, double, double) _rotation_axis_between_vectors(double[:] v, double[:] w):
    cdef:
        double a0, a1, a2, norm
    a0 = (v[1] * w[2] - v[2] * w[1])
    a1 = (v[2] * w[0] - v[0] * w[2])
    a2 = (v[0] * w[1] - v[1] * w[0])
    norm = sqrt(a0 * a0 + a1 * a1 + a2 * a2)    
    if norm != 0.0:
        a0 /= norm
        a1 /= norm
        a2 /= norm
    return a0, a1, a2


def test_memory_view():
    cdef:
        double[:] original = np.array([0.0, 1.0, 2.0, 3.0])
        double[:] original_2 = original
        double[:] copied_original = original_2.copy()
    print(np.asarray(original), original)
    print(np.asarray(original_2), original_2)
    print(np.asarray(copied_original), copied_original)
    original[0] = -1.0
    print(np.asarray(original), original)
    print(np.asarray(original_2), original_2)
    print(np.asarray(copied_original), copied_original)
    copied_original[-1] = -1.0
    print(np.asarray(original), original)
    print(np.asarray(original_2), original_2)
    print(np.asarray(copied_original), copied_original)