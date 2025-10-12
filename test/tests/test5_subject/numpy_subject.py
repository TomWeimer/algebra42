import data.testData as data

import data.testUtils as utils

import numpy as np

from numpy.linalg import det, inv, norm

# ------------------------
# Logging utilities
# ------------------------



def logScalarResult(name, result, write_log_fn):
    write_log_fn("\n" + name)
    x = round(result, ndigits=7)
    write_log_fn(x)

def logStrResult(name, result, write_log_fn):
    write_log_fn("\n" + name)
    write_log_fn(result)    
    
def logResult(name, result, write_log_fn):
    write_log_fn("\n" + name)
    utils.write_content(result, write_log_fn)

def logAddResult(name, elem1, elem2, write_log_fn):
    logResult(name, elem1 + elem2, write_log_fn) 

def logSubResult(name, elem1, elem2, write_log_fn):
    logResult(name, elem1 - elem2, write_log_fn) 

def logProdResult(name, elem1, scalar, write_log_fn):
    logResult(name, elem1 * scalar, write_log_fn) 

# ------------------------
# Test functions
# ------------------------

def test_add_function(write_log_fn):
    logAddResult("vector + vector 1", data.vector_3.matrix, data.vector_4.matrix, write_log_fn)
    logAddResult("vector + vector 2", data.vector_3.matrix, data.vector_3.matrix, write_log_fn)

def test_sub_function(write_log_fn):
    logSubResult("vector - vector 1", data.vector_3.matrix, data.vector_4.matrix, write_log_fn)
    logSubResult("vector - vector 2", data.vector_3.matrix, data.vector_3.matrix, write_log_fn)

def test_prod_function(write_log_fn):
    scalar = 42
    logProdResult("vector * 42 1", data.vector_3.matrix, scalar, write_log_fn)
    logProdResult("vector * 42 2", data.vector_4.matrix, scalar, write_log_fn)

def test_linear_combination(write_log_fn):
    e1 = np.array([1., 0., 0.])
    e2 = np.array([0., 1., 0.])
    e3 = np.array([0., 0., 1.])
    vectors = [e1, e2, e3]
    coefs = [10., -2., 0.5]
    result = sum(a * v for a, v in zip(coefs, vectors))
    logResult("linear comb 1:", result, write_log_fn)

    v1 = np.array([1., 2., 3.])
    v2 = np.array([0., 10., -100.])
    vectors = [v1, v2]
    coefs = [10., -2.]
    result = sum(a * v for a, v in zip(coefs, vectors))
    logResult("linear comb 2:", result, write_log_fn)

def test_lerp(write_log_fn):
    logScalarResult("lerp 1:", 0.0, write_log_fn)
    logScalarResult("lerp 2:", 1.0, write_log_fn)
    logScalarResult("lerp 3:", 0.5, write_log_fn)
    logScalarResult("lerp 4:", 27.3, write_log_fn)
    logResult("lerp 5:", np.array([2.6, 1.3]), write_log_fn)
    logResult("lerp 6:", np.array([[11., 5.5], [16.5, 22.]]), write_log_fn)

def test_dot(write_log_fn):
    u = np.array([0., 0.])
    v = np.array([1., 1.])
    logScalarResult("dot 1:", np.dot(u, v), write_log_fn)

    u = np.array([1., 1.])
    v = np.array([1., 1.])
    logScalarResult("dot 2:", np.dot(u, v), write_log_fn)

    u = np.array([-1., 6.])
    v = np.array([3., 2.])
    logScalarResult("dot 3:", np.dot(u, v), write_log_fn)

def test_norm(write_log_fn):
    u = np.array([0., 0., 0.])
    logStrResult("norms 1:", f"{norm(u, 1)}, {norm(u)}, {norm(u, np.inf)}", write_log_fn)
    u = np.array([1., 2., 3.])
    logStrResult("norms 2:", f"{norm(u, 1)}, {norm(u)}, {norm(u, np.inf)}", write_log_fn)
    u = np.array([-1., -2.])
    logStrResult("norms 3:", f"{norm(u, 1)}, {norm(u)}, {norm(u, np.inf)}", write_log_fn)

def test_angle_cos(write_log_fn):
    def angle(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    tests = [
        (np.array([1.,0.]), np.array([1.,0.])),
        (np.array([1.,0.]), np.array([0.,1.])),
        (np.array([-1.,1.]), np.array([1.,-1.])),
        (np.array([2.,1.]), np.array([4.,2.])),
        (np.array([1.,2.,3.]), np.array([4.,5.,6.]))
    ]

    for i, (u,v) in enumerate(tests, 1):
        logScalarResult(f"angle cos {i}:", angle(u,v), write_log_fn)

def test_cross_product(write_log_fn):
    u = np.array([0.,0.,1.])
    v = np.array([1.,0.,0.])
    logResult("cross product 1:", np.cross(u,v), write_log_fn)

    u = np.array([1.,2.,3.])
    v = np.array([4.,5.,6.])
    logResult("cross product 2:", np.cross(u,v), write_log_fn)

    u = np.array([4.,2.,-3.])
    v = np.array([-2.,-5.,16.])
    logResult("cross product 3:", np.cross(u,v), write_log_fn)

def test_mul(write_log_fn):
    A = np.array([[1.,0.],[0.,1.]])
    v = np.array([4.,2.])
    logResult("mul 1:", A @ v, write_log_fn)

    A = np.array([[2.,0.],[0.,2.]])
    logResult("mul 2:", A @ v, write_log_fn)

    A = np.array([[2.,-2.],[-2.,2.]])
    logResult("mul 3:", A @ v, write_log_fn)

    A = np.array([[1.,0.],[0.,1.]])
    B = np.array([[1.,0.],[0.,1.]])
    logResult("mul 4:", A @ B, write_log_fn)

    A = np.array([[1.,0.],[0.,1.]])
    B = np.array([[2.,1.],[4.,2.]])
    logResult("mul 5:", A @ B, write_log_fn)

    A = np.array([[3.,-5.],[6.,8.]])
    B = np.array([[2.,1.],[4.,2.]])
    logResult("mul 6:", A @ B, write_log_fn)

def test_trace(write_log_fn):
    A = np.array([[1.,0.],[0.,1.]])
    logScalarResult("trace 1:", np.trace(A), write_log_fn)

    A = np.array([[2.,-5.,0.],[4.,3.,7.],[-2.,3.,4.]])
    logScalarResult("trace 2:", np.trace(A), write_log_fn)

    A = np.array([[-2.,-8.,4.],[1.,-23.,4.],[0.,6.,4.]])
    logScalarResult("trace 3:", np.trace(A), write_log_fn)

def test_transpose(write_log_fn):
    A = np.array([[1.,2.],[3.,4.]])
    logResult("transpose 1:", A.T, write_log_fn)

    A = np.array([[2.,-5.,0.],[4.,3.,7.],[-2.,3.,4.]])
    logResult("transpose 2:", A.T, write_log_fn)

    A = np.array([[-2.,-8.,4.],[1.,-23.,4.],[0.,6.,4.]])
    logResult("transpose 3:", A.T, write_log_fn)

# ------------------------
# Row echelon (using scipy LU decomposition)
# ------------------------
def rref(A):
    """Return the reduced row echelon form using LU decomposition as a simple approximation"""
    A = A.astype(float).copy()
    m,n = A.shape
    row = 0
    for col in range(n):
        if row >= m:
            break
        # find pivot
        i_max = np.argmax(np.abs(A[row:,col])) + row
        if A[i_max,col] == 0:
            continue
        # swap
        A[[row,i_max]] = A[[i_max,row]]
        # normalize pivot
        A[row] = A[row] / A[row,col]
        # eliminate below and above
        for r in range(m):
            if r != row:
                A[r] = A[r] - A[r,col]*A[row]
        row += 1
    return A

def test_row_echelon(write_log_fn):
    A = np.eye(3)
    logResult("row_echelon 1:", rref(A), write_log_fn)

    A = np.array([[1.,2.],[3.,4.]])
    logResult("row_echelon 2:", rref(A), write_log_fn)

    A = np.array([[1.,2.],[2.,4.]])
    logResult("row_echelon 3:", rref(A), write_log_fn)

    A = np.array([
        [8.,5.,-2.,4.,28.],
        [4.,2.5,20.,4.,-4.],
        [8.,5.,1.,4.,17.]
    ])
    logResult("row_echelon 4:", rref(A), write_log_fn)

def test_determinant(write_log_fn):
    A = np.array([[1.,-1.],[-1.,1.]])
    logScalarResult("determinant 1:", det(A), write_log_fn)

    A = np.diag([2.,2.,2.])
    logScalarResult("determinant 2:", det(A), write_log_fn)

    A = np.array([[8.,5.,-2.],[4.,7.,20.],[7.,6.,1.]])
    logScalarResult("determinant 3:", det(A), write_log_fn)

    A = np.array([[8.,5.,-2.,4.],[4.,2.5,20.,4.],[8.,5.,1.,4.],[28.,-4.,17.,1.]])
    logScalarResult("determinant 4:", det(A), write_log_fn)

def test_inverse(write_log_fn):
    A = np.eye(3)
    logResult("inverse 1:", inv(A), write_log_fn)

    A = np.diag([2.,2.,2.])
    logResult("inverse 2:", inv(A), write_log_fn)

    A = np.array([[8.,5.,-2.],[4.,7.,20.],[7.,6.,1.]])
    logResult("inverse 3:", inv(A), write_log_fn)

def test_rank(write_log_fn):
    A = np.eye(3)
    logScalarResult("rank 1:", np.linalg.matrix_rank(A), write_log_fn)

    A = np.array([[1.,2.,0.,0.],[2.,4.,0.,0.],[-1.,2.,1.,1.]])
    logScalarResult("rank 2:", np.linalg.matrix_rank(A), write_log_fn)

    A = np.array([[8.,5.,-2.],[4.,7.,20.],[7.,6.,1.],[21.,18.,7.]])
    logScalarResult("rank 3:", np.linalg.matrix_rank(A), write_log_fn)


def run_tests(write_log_fn):
    test_add_function(write_log_fn)
    test_sub_function(write_log_fn)
    test_prod_function(write_log_fn)
    test_linear_combination(write_log_fn)
    test_lerp(write_log_fn)
    test_dot(write_log_fn)
    test_norm(write_log_fn)
    test_angle_cos(write_log_fn)
    test_cross_product(write_log_fn)
    test_mul(write_log_fn)
    test_trace(write_log_fn)
    test_transpose(write_log_fn)
    test_row_echelon(write_log_fn)
    #test_determinant(write_log_fn)
    test_inverse(write_log_fn)
    test_rank(write_log_fn)
