using Algebra42
using .TestData
using .TestUtils

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests NDArray (shape, content)                                                  #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function logNDArrayContent(name, data, write_log_fn)
    write_log_fn("\n" * name)
    write_content(data, write_log_fn)
end


# empty ndarray:

function testEmptyNDArray(name, test::NDArrayData, write_log_fn)
    logNDArrayContent(name, test.array, write_log_fn)

    # exceptions = [
    #     (() -> ndarray[1], DomainError),
    #     (() -> ndarray[2], DomainError),
    #     (() -> ndarray[1, :], DomainError),
    #     (() -> begin
    #             for x in ndarray
    #                 print(x)
    #             end
    #         end, DomainError)
    # ]

    # Check iteration
#    nothingHappen = all(x -> false, ndarray)

    #@assert allExceptionPassed(exceptions) && nothingHappen
end

# scalar

function testScalarNDArray(name, test::TestDataScalar, write_log_fn)
    logNDArrayContent(name, fill(test.data), write_log_fn)

    # exceptions = [
    #     (() -> Base.iterate(ndarray), DomainError),
    #     (() -> ndarray[1], DomainError)
    # ]

    # @assert allExceptionPassed(exceptions)
end


# 'ragged' ndarray:

function testRaggedArray(name, test::NDArrayData, write_log_fn)

    logNDArrayContent(name, test, write_log_fn)

    # f1 = () -> x = NDArray{Int}(test.data)
    # errorThrown = exceptionPassed(f1, 1, DomainError)

    # @assert errorThrown
end


function ndarray_test_set1(write_log_fn)
    testScalarNDArray("scalar", scalar, write_log_fn)
    logNDArrayContent("vector", vector_1, write_log_fn)
    testEmptyNDArray("empty", empty_vector,  write_log_fn)
    logNDArrayContent("matrix 1", matrix_1,  write_log_fn)
    logNDArrayContent("matrix 2", matrix_2,  write_log_fn)
    logNDArrayContent("row", row_matrix,  write_log_fn)
    logNDArrayContent("column", column_matrix,  write_log_fn)
    logNDArrayContent("array3D 1", array_3D_1,  write_log_fn)
    logNDArrayContent("array3D 2", array_3D_2,  write_log_fn)
    testRaggedArray("ragged array", ragged_array,  write_log_fn)
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                       Tests NDArray (slices)                                                     #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


# 'scalar' ndarray
function testScalarSlices(name, test::TestDataScalar, indices, write_log_fn)
    logNDArrayContent(name, fill(test.data), write_log_fn)

    # f = () -> ndarray[indices...]

    # @assert exceptionPassed(f, 1, DomainError)
end

# 'normal' ndarray

function testNDArraySlices(name, test::TestDataArray, indices, write_log_fn)
    sliced =  @view test.matrix[indices...]

    logNDArrayContent(name, sliced, write_log_fn)
end

function ndarray_test_set3(write_log_fn)
    testScalarSlices("scalar", scalar, (:,), write_log_fn)
    testNDArraySlices("vector", vector_1, (:,), write_log_fn)
    testNDArraySlices("matrix", matrix_1, (1, :), write_log_fn)
    testNDArraySlices("array3D", array_3D_1, (1, 1, :), write_log_fn)
    testNDArraySlices("emptyArray", empty_vector, (:,), write_log_fn)
    testNDArraySlices("columnVector", column_matrix, (1, :), write_log_fn)
    testNDArraySlices("rowVector", row_matrix, (1, :), write_log_fn)
    testNDArraySlices("raggedArray", ragged_array, (:,), write_log_fn)
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                       Tests NDArray (all sets)                                                   #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function run_tests_julia_ndarray(write_log_fn)
    ndarray_test_set1(write_log_fn)
    ndarray_test_set3(write_log_fn)
end