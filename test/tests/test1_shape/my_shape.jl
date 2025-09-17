using Algebra42
using Test
using .TestData
using .TestUtils


function logShape(name::AbstractString, testData::ShapeData, write_log_fn::Function)
    shape = isa(testData, TestDataArray) ? Shape(testData.array) : Shape(testData.data)
    toLog = name * "\n dim: " * string(ndims(shape)) * ", total_element: " * string( length(shape) ) * ", shape: " * string(size(shape))
    write_log_fn(toLog)
end



function logRaggedShape(name::AbstractString, testData::ShapeData, write_log_fn::Function)
    shape = Shape(testData.array, dtype=Any)
    toLog = name * "\n dim: " * string(ndims(shape)) * ", total_element: " * string( length(shape) ) * ", shape: " * string(size(shape))
    write_log_fn(toLog)
end

function run_tests(write_log_fn::Function)
    logShape("scalar", scalar, write_log_fn)
    logShape("empty", empty_vector, write_log_fn)
    logShape("vector", vector_1, write_log_fn)
    logShape("matrix", matrix_1, write_log_fn)
    logShape("row", row_matrix, write_log_fn)
    logShape("column", column_matrix, write_log_fn)
    logShape("array3D 1", array_3D_1, write_log_fn)
    logShape("array3D 2", array_3D_2, write_log_fn)
    logRaggedShape("raggedArray", ragged_array, write_log_fn)

    # verify exception for ragged array
    f = () -> Shape(ragged_array.array)

    @assert exceptionPassed(f, 1, DomainError)
end

