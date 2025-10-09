using Algebra42
using Test
using .TestData
using .TestUtils

function logShape(name::AbstractString, testData::ShapeData, write_log_fn::Function)
    toLog = name * "\n dim: " * string(testData.dim) * ", total_element: " * string(testData.total_element) * ", shape: " * string(testData.shape)
    write_log_fn(toLog)
end

function run_tests_julia_shape(write_log_fn::Function)
    logShape("scalar", scalar, write_log_fn)
    logShape("empty", empty_vector, write_log_fn)
    logShape("vector", vector_1, write_log_fn)
    logShape("matrix", matrix_1, write_log_fn)
    logShape("row", row_matrix, write_log_fn)
    logShape("column", column_matrix, write_log_fn)
    logShape("array3D 1", array_3D_1, write_log_fn)
    logShape("array3D 2", array_3D_2, write_log_fn)
    logShape("raggedArray", ragged_array, write_log_fn)
end
