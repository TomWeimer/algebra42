using Algebra42
using .TestData
using .TestUtils

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests NDArray (shape, content)                                                  #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

# empty ndarray:

function testEmptyNDArray(test::NDArrayData)
    ndarray = NDArray{Int}(test.array)

    exceptions = [
        (() -> ndarray[1], DomainError),
        (() -> ndarray[2], DomainError),
        (() -> ndarray[1, :], DomainError),
        (() -> begin
                for x in ndarray
                    print(x)
                end
            end, DomainError)
    ]

    # Check iteration
    nothingHappen = all(x -> false, ndarray)  # will be true if array is empty
    nothingHappen || println("something happened in iteration")

    return allExceptionPassed(exceptions) && nothingHappen && verifyShape(ndarray, test)
end

# scalar:

function testScalarNDArray(test::TestDataScalar)
    ndarray = NDArray{Int}(test.data)

    exceptions = [
        (() -> Base.iterate(ndarray), DomainError),
        (() -> ndarray[1], DomainError)
    ]

    contentOK = (ndarray[()] == test.data)
    contentOK || println("Scalar mismatch: got $(ndarray[()]), expected $(test.data)")

    return allExceptionPassed(exceptions) && verifyShape(ndarray, test) && contentOK
end


# 'normal' ndarray:

function testNDArray(test::NDArrayData)
    ndarray = NDArray{Int}(test.array)
    passed = verifyShape(ndarray, test) && compareArrayContent(ndarray, test)
    passed || println("test: \n", ndarray.content, "\nexpected: \n", test.matrix)
    return passed
end


# 'ragged' ndarray:

function testRaggedArray(test::NDArrayData)
    ndarray = NDArray{Any}(test.array)

    f1 = () -> x = NDArray{Int}(test.data)
    errorThrown = exceptionPassed(f1, 1, DomainError)

    return errorThrown && verifyShape(ndarray, test) && compareArrayContent(ndarray, test)
end


function ndarray_test_set1()
    printTest("scalar", testScalarNDArray(scalar))
    printTest("vector", testNDArray(vector_1))
    printTest("empty", testEmptyNDArray(empty_vector))
    printTest("matrix 1", testNDArray(matrix_1))
    printTest("matrix 2", testNDArray(matrix_2))
    printTest("row", testNDArray(row_matrix))
    printTest("column", testNDArray(column_matrix))
    printTest("array3D 1", testNDArray(array_3D_1))
    printTest("array3D 2", testNDArray(array_3D_2))
    printTest("raggedArray", testRaggedArray(ragged_array))
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests NDArray (formats numpy and julia)                                         #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function testBothFormat(data::NDArrayData)
    format1, format2 = NDArray{Int}(data.array), NDArray{Int}(data.matrix)
    passed = size(format1) == size(format2) && compareByIndex(format1, format2)
    passed || println("format1: \n", format1, "\format2: \n", format2)
    return passed
end

function ndarray_test_set2()
    println("\n Test that both the format used as the input create the same ndarray")
    printTest("vector", testBothFormat(vector_1))
    printTest("empty", testBothFormat(empty_vector))
    printTest("matrix 1", testBothFormat(matrix_1))
    printTest("matrix 2", testBothFormat(matrix_2))
    printTest("row", testBothFormat(row_matrix))
    printTest("column", testBothFormat(column_matrix))
    printTest("array3D 1", testBothFormat(array_3D_1))
    printTest("array3D 2", testBothFormat(array_3D_2))
end

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                       Tests NDArray (slices)                                                     #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


# 'scalar' ndarray
function testScalarSlices(test::TestDataScalar, indices)
    ndarray = NDArray{Int}(test.data)

    f = () -> ndarray[indices...]

    return exceptionPassed(f, 1, DomainError)
end

# 'normal' ndarray

function testNDArraySlices(ndarray::NDArray, ndarrayToChange::NDArray, expected, indices)
    sliced, expectedSliced, slicedToChange = ndarray[indices...], expected[indices...], ndarrayToChange[indices...]

    shapeOk = verifyShape(sliced, expectedSliced)
    contentOk = comparesSlices(ndarray, expected, indices)

    isempty(expected) && return shapeOk && contentOk

    idx1, idx2 = nindexes(ndims(ndarray)), nindexes(ndims(sliced))
    mutableOk = mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, 10)
    passByRef = (ndarrayToChange[idx1...] == 10)


    shapeOk || println("shape mismatch")
    contentOk || println("content mismatch")
    mutableOk || println("mutable: ", mutableOk, " should be ", true)
    passByRef || println("passByRef: ", passByRef, " should be ", true)

    return shapeOk && contentOk && mutableOk && passByRef
end

function testNDArraySlices(test::NDArrayData, indices)
    ndarray = NDArray{Int}(test.array)
    ndarrayToChange = NDArray{Int}(test.array)
    return testNDArraySlices(ndarray, ndarrayToChange, test.matrix, indices)
end

function testRaggedArraySlices(test::NDArrayData, indices)
    ndarray = NDArray{Any}(test.array)
    ndarrayToChange = NDArray{Any}(test.array)
    return testNDArraySlices(ndarray, ndarrayToChange, test.matrix, indices)
end


function mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, val)
    slicedToChange[idx2...] = val
    return ndarrayToChange[idx1...] == val
end


function ndarray_test_set3()
    printstyled("\n   check slices:\n", bold=false, italic=true)
    printTest("scalar", testScalarSlices(scalar, (:,)))
    printTest("vector", testNDArraySlices(vector_1,(:,)))
    printTest("matrix", testNDArraySlices(matrix_1, (1, :)))
    printTest("array3D", testNDArraySlices(array_3D_1, (1, 1, :)))
    printTest("emptyArray", testNDArraySlices(empty_vector, (:,)))
    printTest("columnVector", testNDArraySlices(column_matrix,  (1, :)))
    printTest("rowVector", testNDArraySlices(row_matrix,  (1, :)))
    printTest("raggedArray", testRaggedArraySlices(ragged_array,  (:,)))
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                       Tests NDArray (all sets)                                                   #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function testNdArray()
    println("NDArray Test: ")

    ndarray_test_set1()
    ndarray_test_set2()
    ndarray_test_set3()
end

testNdArray()