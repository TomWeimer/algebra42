using Algebra42
using .TestData
using .TestUtils

function logNDArrayContent(name, data, log)
    if data isa NDArrayData && name == "scalar"
        data_to_log = fill(data.data)
    elseif data isa NDArrayData && name == "raggedArray"
         data_to_log = data
    elseif data isa NDArrayData
        data_to_log = data.matrix
    else
        data_to_log = data
    end
    log("\n" * name)
    write_content(data_to_log, log)
end


function ndarray_julia_set1(log, bench)
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


    function constructor(src, eltype, shape)
        dest = Array{eltype}(undef, shape...)
        copy!(dest, src)
    end

    # Iterate over datasets
    for (name, data) in datasets
        logNDArrayContent(name, data, log)

        # Benchmark array creation
        if !(name in ["scalar", "raggedArray"])
            shape = size(data.matrix)
            Eltype = eltype(data.matrix)
            bench(name * "-constructor", constructor, args=(data.matrix, Eltype, shape))
        end
    end
end


function ndarray_julia_set2(log, bench)

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

    testNDArraySlices(name, test::TestDataArray, indices, log) = (
        sliced = @view test.matrix[indices...];
        logNDArrayContent(name, sliced, log)
    )

    function logNDArraySlice(name, data, idx)
        if name == "scalar"
            logNDArrayContent(name, data, log)
        else
            testNDArraySlices(name, data, idx, log)
        end
    end

    for (name, data, idx) in datasets
        logNDArraySlice(name, data, idx)

        if !(name in ["scalar", "raggedArray"])
            bench(name * "-view", view, args=(data.matrix, idx...))
        end
    end
end


function run_tests_julia_ndarray(log, bench)
    tests = [ndarray_julia_set1, ndarray_julia_set2]

    for test in tests
        test(log, bench)
    end
end