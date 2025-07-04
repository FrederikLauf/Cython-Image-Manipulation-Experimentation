import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_manipulation.cynalg import linalg_pretty as lap
import numpy as np
import array


class TestLinalgPretty:
    
    def test_delta(self):
        assert lap.delta(0, 0) == 1
        assert lap.delta(1, 0) == 0

    def test_epsilon(self):
        assert lap.epsilon(0, 0, 0) == 0
        assert lap.epsilon(0, 1, 2) == 1
        assert lap.epsilon(0, 2, 1) == -1

    def test_delta(self):
        assert lap.delta(0, 0) == 1
        assert lap.delta(1, 0) == 0
    
    def test_epsilon(self):
        assert lap.epsilon(0, 0, 0) == 0
        assert lap.epsilon(0, 1, 2) == 1
        assert lap.epsilon(0, 2, 1) == -1

    def test_cross_product(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        assert lap.cross_product(v, w) == array.array('d', [-3, 6, -3])
        assert lap.cross_product_i(v, w, 0) == -3
        assert lap.cross_product_i(v, w, 1) == 6
        assert lap.cross_product_i(v, w, 2) == -3

    def test_normalized_vector(self):
        v = np.array([3.0, 0.0, 4.0])
        assert lap.normalized_vector(v) == array.array('d', [0.6, 0.0, 0.8])

    def test__rotation_axis_between_vectors(self):
        v = np.array([0.0, 2.0, 0.0])
        w = np.array([1.0, 0.0, 0.0])
        assert lap._rotation_axis_between_vectors(v, w) == array.array('d', [0.0, 0.0, -1.0])

    def test_scalar_product(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        assert lap.scalar_product(v, w) == 32

    def test_vector_norm(self):
        v = np.array([3.0, 0.0, 4.0])
        assert lap.vector_norm(v) == 5

    def test_rotation_matrix_ij(self):
        v = np.array([0.0, 0.0, 1.0])
        assert lap.rotation_matrix_ij(np.pi, v, 0, 0) == -1
        assert lap.rotation_matrix_ij(np.pi, v, 1, 1) == -1
        assert lap.rotation_matrix_ij(np.pi, v, 2, 2) == 1
        assert lap.rotation_matrix_ij(np.pi, v, 0, 1) == 0

    def test_angle_with_grey(self):
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([0.0, 1.0, 1.0])

    def test_angle_between_vectors(self):
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([0.0, 1.0, 1.0])
        assert lap.angle_between_vectors(v, w) == np.pi, 0.25