import data.testData as data

import data.testUtils as utils
import numpy as np

def logScalarContent(name, test, write_log_fn: callable):
    write_log_fn("\n" + name)
    utils.write_scalar_content(test, write_log_fn)
    
def logBoolContent(name, isTrue:bool, write_log_fn: callable):
    write_log_fn("\n" + name)
    if (isTrue is True):
        write_log_fn("true")
    else:
        write_log_fn("false")

def logViewContent(name, test, write_log_fn: callable):
    write_log_fn("\n" + name)
    utils.write_content(test, write_log_fn)

def size(ndarray):
    return ndarray.shape



def test_basic_contiguous_view(write_log_fn):
    A = np.arange(16).reshape(4, 4)
    v = A[1:3, 1:4]

    logViewContent("basic view 1", v, write_log_fn)
    
    logBoolContent("basic view size ok", v.shape == (2, 3), write_log_fn)
    logBoolContent("basic view access 1 ok", v[0, 0] == A[1, 1], write_log_fn)
    logBoolContent("basic view access 2 ok", v[1, 2] == A[2, 3], write_log_fn)

    v[0, 0] = -99

    logBoolContent("basic view 1 (modified)", A[1, 1] == -99, write_log_fn)
    


def test_colon_view(write_log_fn):
    A = np.arange(9).reshape(3, 3)
    v = A[:, :]
    
    logViewContent("colon view 1", v, write_log_fn)

    logBoolContent("colon size ok: ", np.array_equal(v, A), write_log_fn)

    logBoolContent("same memory ok: ", np.shares_memory(A, v), write_log_fn)


def test_dimension_dropping(write_log_fn):
    A = np.arange(16).reshape(4, 4)
    row = A[1, :]     # shape (4,)
    col = A[:, 2]     # shape (4,)
    
    logViewContent("row view:", row, write_log_fn)
    logViewContent("col view:", col, write_log_fn)

    logBoolContent("row view size ok:", size(row) == (4,), write_log_fn)
    logBoolContent("col view size ok:", size(col) == (4,), write_log_fn)

    logBoolContent("row view access ok:", row[0] == A[1, 0], write_log_fn)
    logBoolContent("col view access ok:", col[3] == A[3, 2], write_log_fn)


def test_strided_slice(write_log_fn):
    A = np.arange(16).reshape(4, 4)
    v = A[::2, ::2]
    
    logViewContent("strided slice view:", v, write_log_fn)

    logBoolContent("strided slice size ok:", size(v) == (2, 2), write_log_fn)

    logBoolContent("strided slice access ok 1:", v[0, 0] == A[0, 0], write_log_fn)

    logBoolContent("strided slice access ok 2:", v[1, 1] == A[2, 2], write_log_fn)
    
    logBoolContent("strided share memory:", np.shares_memory(A, v), write_log_fn)


def test_view_of_view(write_log_fn):
    A = np.arange(16).reshape(4, 4)
    v1 = A[1:4, 1:4]

    logViewContent("view of view v1:", v1, write_log_fn)
    
    v2 = v1[1:3, 0:2]
    
    logViewContent("view of view v2:", v2, write_log_fn)

    logBoolContent("view v2 size ok:", size(v2) == (2, 2), write_log_fn)

    logBoolContent("view v2 access ok:",  v2[0, 0] == A[2, 1], write_log_fn)
    
    logBoolContent("v2 share memory:", np.shares_memory(A, v2), write_log_fn)


def test_fancy_indexing(write_log_fn):
    A = np.arange(16).reshape(4, 4)
    rows = [0, 2, 3]
    cols = [1, 3]
    fancy = A[rows, :][:, cols]
    
    logViewContent("fancy index: ", fancy, write_log_fn)

    logBoolContent("fancy index is copy: ", not np.shares_memory(A, fancy),  write_log_fn)

    logBoolContent("fancy index size ok: ", size(fancy) == (3, 2), write_log_fn)

    logBoolContent("fancy index access 1 ok: ", fancy[0, 0] == A[0, 1],  write_log_fn)

    logBoolContent("fancy index access 2 ok: ", fancy[2, 1] == A[3, 3],  write_log_fn)


def test_scalar_view(write_log_fn):
    A = np.arange(9).reshape(3, 3)
    s = A[1, 2]
    
    logScalarContent("scalar view: ", s, write_log_fn)

    logBoolContent("scalar content: ", s == 5, write_log_fn)

    logBoolContent("is numeric: ",  np.isscalar(s),  write_log_fn)


def test_transpose_is_view(write_log_fn):
    A = np.arange(9).reshape(3, 3)
    T = A.T
    
    logViewContent("transpose view: ", T, write_log_fn)
    logBoolContent("transpose is view: ",  np.shares_memory(A, T), write_log_fn)
    logBoolContent("transpose size ok: ", size(T) == (3, 3), write_log_fn)
    logBoolContent("transpose access ok: ", T[0, 1] == A[1, 0], write_log_fn)


def test_write_through_behavior(write_log_fn):
    A = np.arange(9).reshape(3, 3)
    v = A[1:, 1:]
    
    logViewContent("write through view: ", v, write_log_fn)
    v[0, 0] = -42

    logBoolContent("write through parent: ", A[1, 1] == -42, write_log_fn)


def test_trivial_view_identity(write_log_fn):
    A = np.arange(9).reshape(3, 3)
    v = A[:, :]
    
    logViewContent("trivial view: ", v, write_log_fn)

    logBoolContent("trivial view is parent: ",  v.base is A, write_log_fn)


def test_shape_and_stride_consistency(write_log_fn):
    A = np.arange(12).reshape(3, 4)
    v = A[::2, :]
    
    logViewContent("v_stride2 view: ", v, write_log_fn)

    logBoolContent("v_stride2 size: ", size(v) == (2, 4), write_log_fn)

    logBoolContent("v_stride2 strides ok: ", v.strides[0] == 2 * A.strides[0], write_log_fn)


def test_error_fancy_index_copy(write_log_fn):
    A = np.arange(16).reshape(4, 4)
    v = A[[0, 1], [2, 3]]  # diagonal fancy indexing
    # fancy indexing returns a copy
    
    logViewContent("v_fancy_diag view: ", v, write_log_fn)
    logBoolContent("v_fancy_diag is copy: ",  not np.shares_memory(A, v), write_log_fn)
    logBoolContent("v_fancy_diag size: ", size(v) == (2,), write_log_fn)


def test_nested_view_does_not_copy(write_log_fn):
    A = np.arange(16).reshape(4, 4)
    v1 = A[1:, :]
    
    
    logViewContent("v_nested1 view: ", v1, write_log_fn)
    
    v2 = v1[:, 2:]

    logViewContent("v_nested2 view: ", v2, write_log_fn)

    logBoolContent("Nested view does not copy view: ", np.shares_memory(A, v2), write_log_fn)
    

def test_mixed_indexing(write_log_fn):
    A = np.arange(27).reshape(3, 3, 3)
    v = A[1, :, 1:]
    
    logViewContent("v_mixed view: ", v, write_log_fn)
    logBoolContent("v_mixed size ok: ",  size(v) == (3, 2), write_log_fn)
    logBoolContent("v_mixed is not copy: ",  np.shares_memory(A, v), write_log_fn)


# def test_boolean_mask_creates_copy(write_log_fn):
#     A = np.arange(9).reshape(3, 3)
#     mask = A > 4
#     v = A[mask]
#     assert not np.shares_memory(A, v)
#     assert v.ndim == 1
#     assert np.all(v > 4)


def run_tests(write_log_fn):
    test_basic_contiguous_view(write_log_fn)
    test_colon_view(write_log_fn)
    test_dimension_dropping(write_log_fn)
    test_strided_slice(write_log_fn)
    test_view_of_view(write_log_fn)
    test_fancy_indexing(write_log_fn)
    test_scalar_view(write_log_fn)
   # test_transpose_is_view(write_log_fn)
    test_write_through_behavior(write_log_fn)
    test_trivial_view_identity(write_log_fn)
    test_shape_and_stride_consistency(write_log_fn)
    test_error_fancy_index_copy(write_log_fn)
    test_nested_view_does_not_copy(write_log_fn)
    test_mixed_indexing(write_log_fn)
   #test_boolean_mask_creates_copy(write_log_fn)
