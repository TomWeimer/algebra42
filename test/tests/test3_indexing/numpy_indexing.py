import data.testData as data

import data.testUtils as utils

import numpy as np

def logIndexing(name, result, write_log_fn):
    write_log_fn("\n" + name)
    utils.write_content(result, write_log_fn)

def test_shape_incompatibility(name, test:data.TestDataArray, write_log_fn, *indices):
    passed = False
    try:
        tmp = test.matrix[*indices]
        logIndexing(name, tmp, write_log_fn)
    except Exception as e:
        passed = isinstance(e, IndexError)
    assert passed

def test_fancy_indexing(name, test:data.TestDataArray, write_log_fn, *indices):
    got = test.matrix[*indices]
    logIndexing(name, got, write_log_fn)
    

def run_tests(write_log_fn):
    str = None
    
    # test_fancy_indexing("Int indexing", data.array_4D_1, write_log_fn, [0, 1], [[0, 1], [1, 0]])
    # test_fancy_indexing("Bool mask 1", data.vector_1, write_log_fn, [True, False, True, True])
    # test_fancy_indexing("Bool mask 2", data.matrix_1, write_log_fn, [[True, False, True], [True, True, False]])
    # test_fancy_indexing("Bool mask 3", data.matrix_1, write_log_fn,  [[False, False, False], [False, False, False]])
    # test_fancy_indexing("Bool mask 4", data.array_3D_1, write_log_fn,[False, True], [False, True], [False, True])
    # test_fancy_indexing("Bool mask 5", data.array_3D_1, write_log_fn, [False, False], [False, False], [False, False])
    # test_fancy_indexing("Bool indexing", data.array_4D_1, write_log_fn, [[True, False], [False, True]], [[False, True], [True, False]])
    # test_shape_incompatibility("Shape invalid", data.array_4D_1, write_log_fn, [0, 1, 2], [[0, 1], [1, 0]])
    # test_fancy_indexing("Mixed indexing 1", data.array_4D_1, write_log_fn, [[True, False], [True, False]], [[0, 1], [0, 1]] )
    # test_shape_incompatibility("Mixed indexing 2", data.array_4D_1, write_log_fn, [[False, False], [False, False]], [[0, 1], [1, 0]])