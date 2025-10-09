using Algebra42
using .TestData
using .TestUtils

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests NDArray (shape, content)                                                  #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


function logNDArrayContent(name, ndarray, write_log_fn)
    write_log_fn("\n" * name)
    write_content(ndarray, write_log_fn)
end


# empty ndarray:

function testEmptyNDArray(name, test::NDArrayData, write_log_fn)
    array = NDArray{Int}(test.array)

    logNDArrayContent(name, array, write_log_fn)

    exceptions = [
        (() -> array[1], DomainError),
        (() -> array[2], DomainError),
        (() -> array[1, :], DomainError)
    ]

    # Check iteration
    nothingHappen = all(x -> false, array)

    @assert allExceptionPassed(exceptions) && nothingHappen
end

# scalar

function testScalarNDArray(name, test::TestDataScalar, write_log_fn)
    array = NDArray{Int}(test.data)

    logNDArrayContent(name, array, write_log_fn)

    exceptions = [
        (() -> Base.iterate(array), DomainError),
        (() -> array[1], DomainError)
    ]

    @assert allExceptionPassed(exceptions)
end


# 'ragged' ndarray:

function testRaggedArray(name, test::NDArrayData, write_log_fn)

    array = NDArray{Any}(test.array)

    logNDArrayContent(name, array, write_log_fn)

    f1 = () -> x = NDArray{Int}(test.array)
    errorThrown = exceptionPassed(f1, 1, DomainError)

    @assert errorThrown
end


function ndarray_test_set1(write_log_fn)
    testScalarNDArray("scalar", scalar, write_log_fn)
    logNDArrayContent("vector", vector_1, write_log_fn)
    testEmptyNDArray("empty", empty_vector, write_log_fn)
    logNDArrayContent("matrix 1", matrix_1, write_log_fn)
    logNDArrayContent("matrix 2", matrix_2,  write_log_fn)
    logNDArrayContent("row", row_matrix,  write_log_fn)
    logNDArrayContent("column", column_matrix,  write_log_fn)
    logNDArrayContent("array3D 1", array_3D_1,  write_log_fn)
    logNDArrayContent("array3D 2", array_3D_2,  write_log_fn)
    testRaggedArray("ragged array", ragged_array,  write_log_fn)
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests NDArray (formats numpy and julia)                                         #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function testBothFormat(data::NDArrayData)
    format1, format2 = NDArray{Int}(data.array), NDArray{Int}(data.matrix)
    passed = size(format1) == size(format2) && compareByIndex(format1, format2)
    return passed
end

function ndarray_test_set2(write_log_fn)
    @assert testBothFormat(vector_1)
    @assert testBothFormat(empty_vector)
    @assert testBothFormat(matrix_1)
    @assert testBothFormat(matrix_2)
    @assert testBothFormat(row_matrix)
    @assert testBothFormat(column_matrix)
    @assert testBothFormat(array_3D_1)
    @assert testBothFormat(array_3D_2)
end

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                       Tests NDArray (slices)                                                     #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


# 'scalar' ndarray
function testScalarSlices(name, test::TestDataScalar, indices, write_log_fn)
    ndarray = NDArray{Int}(test.data)

    logNDArrayContent(name, ndarray, write_log_fn)

    f = () -> ndarray[indices...]

    @assert exceptionPassed(f, 1, DomainError)
end

# 'normal' ndarray

function testNDArraySlices(name, ndarray::NDArray, ndarrayToChange::NDArray, indices, write_log_fn)
    sliced, slicedToChange = ndarray[indices...], ndarrayToChange[indices...]

    logNDArrayContent(name, sliced, write_log_fn)

    if (isempty(sliced))
        return
    end


    idx1, idx2 = nindexes(ndims(ndarray)), nindexes(ndims(sliced))
    mutableOk = mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, 10)
    passByRef = (ndarrayToChange[idx1...] == 10)


    mutableOk || println("mutable: ", mutableOk, " should be ", true)
    passByRef || println("passByRef: ", passByRef, " should be ", true)

    @assert mutableOk && passByRef
end

function testNDArraySlices(name, test::NDArrayData, indices, write_log_fn)
    ndarray = NDArray{Int}(test.array)
    ndarrayToChange = NDArray{Int}(test.array)
    return testNDArraySlices(name, ndarray, ndarrayToChange, indices, write_log_fn)
end

function testRaggedArraySlices(name, test::NDArrayData, indices, write_log_fn)
    ndarray = NDArray{Any}(test.array)
    ndarrayToChange = NDArray{Any}(test.array)
    return testNDArraySlices(name, ndarray, ndarrayToChange, indices, write_log_fn)
end


function mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, val)
    slicedToChange[idx2...] = val
    return ndarrayToChange[idx1...] == val
end


function ndarray_test_set3(write_log_fn)
    testScalarSlices("scalar", scalar, (:,), write_log_fn)
    testNDArraySlices("vector", vector_1, (:,), write_log_fn)
    testNDArraySlices("matrix", matrix_1, (1, :), write_log_fn)
    testNDArraySlices("array3D", array_3D_1, (1, 1, :), write_log_fn)
    testNDArraySlices("emptyArray", empty_vector, (:,), write_log_fn)
    testNDArraySlices("columnVector", column_matrix, (1, :), write_log_fn)
    testNDArraySlices("rowVector", row_matrix, (1, :), write_log_fn)
    testRaggedArraySlices("raggedArray", ragged_array, (:,), write_log_fn)
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                       Tests NDArray (all sets)                                                   #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function run_tests_my_ndarray(write_log_fn)
    ndarray_test_set1(write_log_fn)
    ndarray_test_set2(write_log_fn)
    ndarray_test_set3(write_log_fn)
end