using Algebra42
using .TestData
using .TestUtils

function logNDArrayContent(name, ndarray, log)
    log("\n" * name)
    write_content(ndarray, log)
end

function ndarray_test_set1(log, bench)

    # empty ndarray:
    function testEmptyNDArray(name, test::NDArrayData, log)
        array = NDArray{Int}(test.array)

        logNDArrayContent(name, array, log)

        exceptions = [
            (() -> array[1], BoundsError),
            (() -> array[2], BoundsError),
            (() -> array[1, :], BoundsError)
        ]

        # Check iteration
        nothingHappen = all(x -> false, array)

        @assert allExceptionPassed(exceptions) && nothingHappen
    end

    # scalar

    function testScalarNDArray(name, test::TestDataScalar, log)
        array = NDArray{Int}(test.data)

        logNDArrayContent(name, array, log)
        a = array[:]
        @assert exceptionPassed(() -> array[2], 1, BoundsError)

    end

    # 'ragged' ndarray:

    function testRaggedArray(name, test::NDArrayData, log)

        array = NDArray{Any}(test.array)

        logNDArrayContent(name, array, log)

        f1 = () -> x = NDArray{Int}(test.array)
        errorThrown = exceptionPassed(f1, 1, DomainError)

        @assert errorThrown
    end

    function testNDArray(name, data, log)
        if (name == "scalar")
            testScalarNDArray(name, data, log)
        elseif (name == "empty")
            testEmptyNDArray(name, data, log)
        elseif (name == "raggedArray")
            testRaggedArray(name, data, log)
        else
            logNDArrayContent(name, NDArray{Int}(data.array), log)
        end
    end

    function constructor(eltype, matrix)
        NDArray{eltype}(matrix)
    end


    # Define all tests as (name, data, function) tuples
    datasets = [
        ("scalar", scalar),
        ("vector", vector_1),
        ("empty", empty_vector),
        ("matrix 1", matrix_1),
        ("matrix 2", matrix_2),
        ("row", row_matrix),
        ("column", column_matrix),
        ("array3D 1", array_3D_1),
        ("array3D 2", array_3D_2),
        ("raggedArray", ragged_array)
    ]

    for (name, data) in datasets
        testNDArray(name, data, log)
        # Benchmark array creation
        if !(name in ["scalar", "raggedArray"])
            Eltype = eltype(data.matrix)
            bench(name * "-constructor", constructor, args=(Eltype, data.matrix))
        end
    end
end


function ndarray_test_set2(log, bench)

    dataset1 = [
        vector_1,
        empty_vector,
        matrix_1,
        matrix_2,
        array_3D_1,
        array_3D_2,
        column_matrix,
        row_matrix,
    ]

    testBothFormat(data::NDArrayData) = (
        testBothFormat(NDArray{Int}(data.array), NDArray{Int}(data.matrix))
    )

    testBothFormat(data1::AbstractArray{T,1}, data2::AbstractArray{U,N}) where {T,U,N} = (
        testBothFormat(NDArray{Int}(data1), NDArray{Int}(data2))
    )

    function testBothFormat(format1::Algebra42.AbstractNDArray,
        format2::Algebra42.AbstractNDArray)
        return size(format1) == size(format2) && compareByIndex(format1, format2)
    end

    for data in dataset1
        @assert testBothFormat(data)
    end

    dataset2 = [
        ([[1, 2], [3, 4]], [1 2; 3 4]),
        ([[1, 2, 3], [4, 5, 6]], [1 2 3; 4 5 6]),
        ([[1, 2], [3, 4], [5, 6]], [1 2; 3 4; 5 6]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [1 2; 3 4;;; 5 6; 7 8]),
    ]

    for (array, matrix) in dataset2
        @assert testBothFormat(array, matrix)
    end
end


function ndarray_test_set3(log, bench)

    datasets = [
        ("scalar", scalar, (:,)),
        ("vector", vector_1, (:,)),
        ("matrix", matrix_1, (1, :)),
        ("array3D", array_3D_1, (1, 1, :)),
        ("emptyArray", empty_vector, (:,)),
        ("columnVector", column_matrix, (1, :)),
        ("rowVector", row_matrix, (1, :)),
        ("raggedArray", ragged_array, (:,))
    ]

    function mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, val)
        slicedToChange[idx2...] = val
        return ndarrayToChange[idx1...] == val
    end

    function testNDArraySlices(name, ndarray::NDArray, ndarrayToChange::NDArray, indices, log)
        sliced, slicedToChange = ndarray[indices...], ndarrayToChange[indices...]

        logNDArrayContent(name, sliced, log)

        isempty(sliced) && return

        idx1, idx2 = ntuple(_ -> 1, ndims(ndarray)), ntuple(_ -> 1, ndims(sliced))
        mutableOk = mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, 10)
        passByRef = (ndarrayToChange[idx1...] == 10)


        mutableOk || println("mutable: ", mutableOk, " should be ", true)
        passByRef || println("passByRef: ", passByRef, " should be ", true)

        @assert mutableOk && passByRef
    end

    function testNDArraySlices(name, test::NDArrayData, indices, log)
        ndarray = NDArray{Int}(test.array)
        ndarrayToChange = NDArray{Int}(test.array)
        return testNDArraySlices(name, ndarray, ndarrayToChange, indices, log)
    end

    function testRaggedArraySlices(name, test::NDArrayData, indices, log)
        ndarray = NDArray{Any}(test.array)
        ndarrayToChange = NDArray{Any}(test.array)
        return testNDArraySlices(name, ndarray, ndarrayToChange, indices, log)
    end

    function testScalarSlices(name, test::TestDataScalar, indices, log)
        ndarray = NDArray{Int}(test.data)

        logNDArrayContent(name, ndarray, log)
    end

    function logNDArraySlice2(name, data, idx)
        if name == "scalar"
            testScalarSlices(name, data, idx, log)
        elseif name == "raggedArray"
            testRaggedArraySlices(name, data, idx, log)
        else
            testNDArraySlices(name, data, idx, log)
        end
    end


    for (name, data, idx) in datasets
        logNDArraySlice2(name, data, idx)

        if !(name in ["scalar", "raggedArray"])
            ndarray = NDArray{Int}(data.matrix)
            bench(name * "-view", view, args=(ndarray, idx...))
        end
    end

end


function run_tests_my_ndarray(log, bench)
    tests = [ndarray_test_set1, ndarray_test_set2, ndarray_test_set3]

    for test in tests
        test(log, bench)
    end
end