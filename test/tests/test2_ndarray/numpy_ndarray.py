import data.testData as data

import data.testUtils as utils

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests NDArray (shape, content)                                                  #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


# scalar

# function testScalarNDArray(name, test::TestDataScalar, write_log_fn)
#     logNDArrayContent(name, test.data, write_log_fn)

# exceptions = [
#     (() -> Base.iterate(ndarray), DomainError),
#     (() -> ndarray[1], DomainError)
# ]

# @assert allExceptionPassed(exceptions)
# end


# 'ragged' ndarray:

# function testRaggedArray(name, test::NDArrayData, write_log_fn)

#     logNDArrayContent(name, test, write_log_fn)

#     # f1 = () -> x = NDArray{Int}(test.data)
#     # errorThrown = exceptionPassed(f1, 1, DomainError)

#     # @assert errorThrown
# end


def logNDArrayContent(name, test, write_log_fn: callable):
    write_log_fn("\n" + name)
    utils.write_content(test, write_log_fn)


def logNDArrayScalar(name, test, write_log_fn: callable):
    write_log_fn("\n" + name)
    utils.write_scalar_content(test, write_log_fn)

def ndarray_test_set1(write_log_fn):
    logNDArrayScalar("scalar", data.scalar, write_log_fn)
    logNDArrayContent("vector", data.vector_1, write_log_fn)
    logNDArrayContent("empty", data.empty_vector, write_log_fn)
    logNDArrayContent("matrix 1", data.matrix_1, write_log_fn)
    logNDArrayContent("matrix 2", data.matrix_2, write_log_fn)
    logNDArrayContent("row", data.row_matrix, write_log_fn)
    logNDArrayContent("column", data.column_matrix, write_log_fn)
    logNDArrayContent("array3D 1", data.array_3D_1, write_log_fn)
    logNDArrayContent("array3D 2", data.array_3D_2, write_log_fn)
    logNDArrayContent("ragged array", data.ragged_array, write_log_fn)


# # ---------------------------------------------------------------------------------------------------------------------------------#
# #                                                                                                                                  #
# #                                                       Tests NDArray (slices)                                                     #
# #                                                                                                                                  #
# # ---------------------------------------------------------------------------------------------------------------------------------#


# 'scalar' ndarray
def testScalarSlices(name, test: data.TestDataScalar, indices, write_log_fn):
    logNDArrayScalar(name, test, write_log_fn)


def testNDArraySlices(name, test: data.TestDataArray, indices, write_log_fn):
    sliced = test.matrix[indices]
    logNDArrayContent(name, sliced, write_log_fn)


def ndarray_test_set3(write_log_fn):
    testScalarSlices("scalar", data.scalar, (slice(None),), write_log_fn)
    testNDArraySlices("vector", data.vector_1, (slice(None),), write_log_fn)
    testNDArraySlices("matrix", data.matrix_1, (0, slice(None)), write_log_fn)
    testNDArraySlices("array3D", data.array_3D_1, (0, 0, slice(None)), write_log_fn)
    testNDArraySlices("emptyArray", data.empty_vector, (slice(None),), write_log_fn)
    testNDArraySlices(
        "columnVector", data.column_matrix, (0, slice(None)), write_log_fn
    )
    testNDArraySlices("rowVector", data.row_matrix, (0, slice(None)), write_log_fn)
    testNDArraySlices("raggedArray", data.ragged_array, (slice(None),), write_log_fn)


# # ---------------------------------------------------------------------------------------------------------------------------------#
# #                                                                                                                                  #
# #                                                       Tests NDArray (all sets)                                                   #
# #                                                                                                                                  #
# # ---------------------------------------------------------------------------------------------------------------------------------#


def run_tests(write_log_fn):
    ndarray_test_set1(write_log_fn)
    ndarray_test_set3(write_log_fn)
