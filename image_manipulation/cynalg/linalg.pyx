## cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False

from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt, asin, sin, cos


@cdivision(True)
cpdef _rotate_vector_ultimate(double[:] v, double[:] w, double[:] g, double g_norm):
    cdef:
        double nominator, a0, a1, a2, norm, s, c, C
        double angle = 0.0
    if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
        w[0] = 0.0
        w[1] = 0.0
        w[2] = 0.0
        return
    # rotation axis
    a0 = (v[1] * g[2] - v[2] * g[1])
    a1 = (v[2] * g[0] - v[0] * g[2])
    a2 = (v[0] * g[1] - v[1] * g[0])
    norm = sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    # if norm != 0.0:
    a0 /= norm
    a1 /= norm
    a2 /= norm
    # angle with grey
    nominator = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) * g_norm
    # if nominator != 0.0:
    angle = 0.5 * asin(norm / nominator)
    # rotate
    s = sin(angle)
    c = cos(angle)
    C = (1 - c)
    w[0] = (C * a0 * a0 + c) * v[0] + (C * a0 * a1 - s * a2) * v[1] + (C * a0 * a2 + s * a1) * v[2]
    w[1] = (C * a1 * a0 + s * a2) * v[0] + (C * a1 * a1 + c) * v[1] + (C * a1 * a2 - s * a0) * v[2]
    w[2] = (C * a2 * a0 - s * a1) * v[0] + (C * a2 * a1 + s * a0) * v[1] + (C * a2 * a2 + c) * v[2]


cpdef turn_all_towards_grey(double[:,:,:] img_in, double[:,:,:] img_out, double[:] grey):
    cdef:
        int N = img_in.shape[0]
        int M = img_in.shape[1]
        int i, j
        double grey_norm = sqrt(grey[0] * grey[0] + grey[1] * grey[1] + grey[2] * grey[2])
    for i in range(N):
        for j in range(M):
            _rotate_vector_ultimate(img_in[i, j], img_out[i, j], grey, grey_norm)