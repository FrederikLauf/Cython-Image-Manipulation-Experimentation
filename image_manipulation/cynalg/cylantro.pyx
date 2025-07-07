## cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False

from cython cimport cdivision
from libc.math cimport sqrt, asin, sin, cos


@cdivision(True)
cpdef rotate_vector_towards_other(double[:] v, double[:] w, double[:] g, double g_norm, double factor):
    cdef:
        double a0, a1, a2, cross_norm, s, c, C, angle
    # cross product of v and g
    a0 = (v[1] * g[2] - v[2] * g[1])
    a1 = (v[2] * g[0] - v[0] * g[2])
    a2 = (v[0] * g[1] - v[1] * g[0])
    if a0 == a1 == a2 == 0.0:  # <=> v=0 or g=0 or v||g
        w[:] = v
    else:
        # rotation axis = normalised cross product
        cross_norm = sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        a0 /= cross_norm
        a1 /= cross_norm
        a2 /= cross_norm
        # angle between v and g
        angle = factor * asin(cross_norm / (sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) * g_norm))
        # rotate
        s = sin(angle)
        c = cos(angle)
        C = (1 - c)
        w[0] = (C * a0 * a0 + c) * v[0] + (C * a0 * a1 - s * a2) * v[1] + (C * a0 * a2 + s * a1) * v[2]
        w[1] = (C * a1 * a0 + s * a2) * v[0] + (C * a1 * a1 + c) * v[1] + (C * a1 * a2 - s * a0) * v[2]
        w[2] = (C * a2 * a0 - s * a1) * v[0] + (C * a2 * a1 + s * a0) * v[1] + (C * a2 * a2 + c) * v[2]


cpdef turn_all_towards_other(double[:,:,:] img_in, double[:,:,:] img_out, double[:] other, double factor):
    cdef:
        int N = img_in.shape[0]
        int M = img_in.shape[1]
        int i, j
        double other_norm = sqrt(other[0] * other[0] + other[1] * other[1] + other[2] * other[2])
    for i in range(N):
        for j in range(M):
            rotate_vector_towards_other(img_in[i, j], img_out[i, j], other, other_norm, factor)