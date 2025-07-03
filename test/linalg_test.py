from context.image_manipulation.cynalg.artifacts import linalg
import numpy as np
import array


def verify(expression, expectation):
    res = (eval(expression) == expectation)
    print(expression + " == " + str(expectation) + " (" + str(eval(expression)) +")" + " -> " + str(res))
    

if __name__ == "__main__":
    #  linalg.delta
    verify("linalg.delta(0, 0)", 1)
    verify("linalg.delta(1, 0)", 0)
    
    #  linalg.epsilon
    verify("linalg.epsilon(0, 0, 0)", 0)
    verify("linalg.epsilon(0, 1, 2)", 1)
    verify("linalg.epsilon(0, 2, 1)", -1)
    
    # linalg.cross_product
    # linalg.cross_product_i
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([4.0, 5.0, 6.0])
    verify("linalg.cross_product(v, w)", array.array('d', [-3, 6, -3]))
    verify("linalg.cross_product_i(v, w, 0)", -3)
    verify("linalg.cross_product_i(v, w, 1)", 6)
    verify("linalg.cross_product_i(v, w, 2)", -3)
    
    # linalg.normalized_vector
    v = np.array([3.0, 0.0, 4.0])
    verify("linalg.normalized_vector(v)", array.array('d', [0.6, 0.0, 0.8]))
    
    # linalg.rotation_axis_between_vectors
    v = np.array([0.0, 2.0, 0.0])
    w = np.array([1.0, 0.0, 0.0])
    verify("linalg.rotation_axis_between_vectors(v, w)", array.array('d', [0.0, 0.0, -1.0]))
    
    # linalg.scalar_product
    verify("linalg.scalar_product(v, w)", 32)
    
    # linalg.vector_norm
    v = np.array([3.0, 0.0, 4.0])
    verify("linalg.vector_norm(v)", 5)
    
    # linalg.rotation_matrix
    v = np.array([0.0, 0.0, 1.0])
    verify("linalg.rotation_matrix(np.pi, v, 0, 0)", -1)
    verify("linalg.rotation_matrix(np.pi, v, 1, 1)", -1)
    verify("linalg.rotation_matrix(np.pi, v, 2, 2)", 1)
    verify("linalg.rotation_matrix(np.pi, v, 0, 1)", 0)
    
    # linalg.angle_with_grey
    v = np.array([0.0, 0.0, 1.0])
    w = np.array([0.0, 1.0, 1.0])
    
    # linalg.angle_between_vectors
    verify("linalg.angle_between_vectors(v, w) / np.pi", 0.25)
    
    #linalg.test_memory_view
    linalg.test_memory_view()