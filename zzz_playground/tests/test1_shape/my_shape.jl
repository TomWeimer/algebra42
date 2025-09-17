using Algebra42
using Test
using .TestData
using .TestUtils

function testRaggedArrayShape(ragged)
    # We can't obtain the shape of a ragged arrays because it do not have the same number of elements in each dimensions
    # So if we try without telling that it is a nested array then we throw an error
    f = () -> Shape(ragged.array)
    return exceptionPassed(f, 1, DomainError) && verifyShape(ragged, ragged, Any)
end

function testShape()
    println("Shape Test: ")    
    printTest("scalar",         verifyShape(scalar))
    printTest("empty",          verifyShape(empty_vector))
    printTest("vector",         verifyShape(vector_1))
    printTest("matrix",         verifyShape(matrix_1))
    printTest("row",            verifyShape(row_matrix))
    printTest("column",         verifyShape(column_matrix))
    printTest("array3D 1",      verifyShape(array_3D_1))
    printTest("array3D 2",      verifyShape(array_3D_2))
    printTest("raggedArray",    testRaggedArrayShape(ragged_array))
    println()
end

testShape()
