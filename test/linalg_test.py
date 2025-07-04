import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_manipulation.cynalg import linalg as la


class TestLinalg:

    def test_delta(self):
        assert la.delta(0, 0) == 1
        assert la.delta(1, 0) == 0

    def test_epsilon(self):
        assert la.epsilon(0, 0, 0) == 0
        assert la.epsilon(0, 1, 2) == 1
        assert la.epsilon(0, 2, 1) == -1

    @pytest.mark.skip(reason="not yet implemented")
    def test_cross_product(self):
        # to do
        pass

    @pytest.mark.skip(reason="not yet implemented")
    def test_normalized_vector(self):
        # to do
        pass

    @pytest.mark.skip(reason="not yet implemented")
    def test__rotation_axis_between_vectors(self):
        # to do
        pass

    def test_scalar_product(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        assert la.scalar_product(v, w) == 32

    def test_vector_norm(self):
        v = np.array([3.0, 0.0, 4.0])
        assert la.vector_norm(v) == 5

    def test_rotation_matrix_ij(self):
        v = np.array([0.0, 0.0, 1.0])
        print(la.rotation_matrix_ij(np.pi, v, 0, 1))
        assert la.rotation_matrix_ij(np.pi, v, 0, 0) == -1
        assert la.rotation_matrix_ij(np.pi, v, 1, 1) == -1
        assert la.rotation_matrix_ij(np.pi, v, 2, 2) == 1
        assert np.round(la.rotation_matrix_ij(np.pi, v, 0, 1), decimals=15) == 0

    def test_angle_between_vectors(self):
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([0.0, 1.0, 1.0])
        print(la.angle_between_vectors(v, w) / np.pi)
        assert np.round(la.angle_between_vectors(v, w) / np.pi, decimals=15) == 0.25