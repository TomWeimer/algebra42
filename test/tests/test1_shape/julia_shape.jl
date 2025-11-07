using Algebra42
using Test
using .TestData
using .TestUtils

function logShape(name::AbstractString, testData::ShapeData, log::Function)
    dim, total_element, shape = testData.dim, testData.total_element, testData.shape
    log(name * "\n dim: $dim, total_element: $total_element, shape: $shape")
end

function testShape(name::AbstractString, data, log::Function, bench::Function)
    # We log the expected values
    logShape(name, data, log)

    # We benchmark base function
    if !(name in ["scalar", "raggedArray"])
        bench(name * "-size",   size,    args=(data.matrix,))
        bench(name * "-length", length,  args=(data.matrix,))
        bench(name * "-ndims",  ndims,   args=(data.matrix,))
    end
end

function run_tests_julia_shape(log::Function, bench::Function)
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

    for (name, data) in datasets
        testShape(name, data, log, bench)
    end
end
