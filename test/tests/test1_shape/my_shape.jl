using Algebra42
using Test
using .TestData
using .TestUtils

function logShape2(name::AbstractString, ndarray::NDArray, log::Function)
    dim, total_element, shape = ndims(ndarray), length(ndarray), size(ndarray)
    log(name * "\n dim: $dim, total_element: $total_element, shape: $shape")
end

function testShape2(name::AbstractString, testData, log::Function, bench::Function)
    data = testData isa TestDataArray ? testData.array : testData.data

    if (name == "scalar")
        ndarray = NDArray{eltype(data)}(data)
    elseif (name == "raggedArray")
        ndarray = NDArray{Any}(data)
    else
        ndarray = NDArray{eltype(testData.matrix)}(data)
    end
    
    # We log the result:
    logShape2(name, ndarray, log)

    # And benchmark the functions
    if !(name in ["scalar", "raggedArray"])
        bench(name * "-size",   size,    args=(ndarray,))
        bench(name * "-length", length,  args=(ndarray,))
        bench(name * "-ndims",  ndims,   args=(ndarray,))
    end
end

function run_tests_my_shape(log::Function, bench::Function)

    datasets = [
        ("scalar", scalar),
        ("empty", empty_vector),
        ("vector", vector_1),
        ("matrix", matrix_1),
        ("row", row_matrix),
        ("column", column_matrix),
        ("array3D 1", array_3D_1),
        ("array3D 2", array_3D_2),
        ("raggedArray", ragged_array)
    ]

    # verify exception for ragged array
    @assert exceptionPassed(() -> NDArray{Int}(ragged_array.array), 1, DomainError)

    for (name, data) in datasets
        testShape2(name, data, log, bench)
    end
end

