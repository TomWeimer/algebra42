using Algebra42
using Test
using Infiltrator
using .TestUtils

struct ShapeResult
    dim::Int
    length::Int
    dims::Tuple{Vararg{Int}}
end

struct ShapeData
    data
    expected::ShapeResult

    ShapeData(data; dim, length, dims) = new(data, ShapeResult(dim, length, dims))
end

scalar          = ShapeData( 42, dim  = 0, length = 1, dims = (0,))

vector1D        = ShapeData([1, 2, 3, 4], dim = 1, length = 4, dims = (4,))

matrix2D        = ShapeData([[1, 2, 3], [4, 5, 6]], dim = 2, length = 6, dims = (2, 3))

columnVector2D  = ShapeData([[1], [2], [3]], dim = 2, length = 3, dims = (3, 1) )

rowVector2D     = ShapeData( [[1, 2, 3]], dim = 2, length = 3, dims = (1, 3) )

array3D         = ShapeData( [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dim = 3, length = 8, dims = (2, 2, 2) )

array4D         = ShapeData( (2, 3, 4, 5), dim = 4, length = 120, dims =  (2, 3, 4, 5) )

raggedArray     = ShapeData( [[1, 2], [3, 4, 5]], dim = 1, length = 2, dims = (2,))

emptyArray      = ShapeData( [], dim = 1, length = 0, dims = (0,))


function verifyShape(test, expected)
    shape = Shape(test.data);
    if shape.dims != expected.dims
        println("Shape mismatch: got $(shape.dims), expected $(expected.dims)")
    end

    if shape.length != expected.length
        println("Size mismatch: got $(shape.length), expected $(expected.length)")
    end

    if ndims(shape) != expected.dim
        println("Dim mismatch: got $(ndims(shape)), expected $(expected.dim)")
    end

    return shape.dims == expected.dims &&
           shape.length == expected.length &&
           ndims(shape) == expected.dim
end


function verifyShape(test, expected, dtype)
    shape = Shape(test.data; dtype = dtype);
    if shape.dims != expected.dims
        println("Shape mismatch: got $(shape.dims), expected $(expected.dims)")
    end

    if shape.length != expected.length
        println("Size mismatch: got $(shape.length), expected $(expected.length)")
    end

    if ndims(shape) != expected.dim
        println("Dim mismatch: got $(ndims(shape)), expected $(expected.dim)")
    end

    return shape.dims == expected.dims &&
           shape.length == expected.length &&
           ndims(shape) == expected.dim
end

function testShape()
println("Shape Test: ")    
printTest("scalar shape", verifyShape(scalar, scalar.expected))

#2 Simple Vector
printTest("vector1D shape", verifyShape(vector1D, vector1D.expected))

#3 Simple Matrix 2D
printTest("matrix2D shape", verifyShape(matrix2D, matrix2D.expected))

#4 Array 3D
printTest("array3D shape", verifyShape(array3D, array3D.expected))

#5 Empty Array
printTest("emptyArray shape",  verifyShape(emptyArray, emptyArray.expected))

#6 Column Vector
printTest("columnVector2D shape",verifyShape(columnVector2D, columnVector2D.expected))

#7 Row Vector
printTest("rowVector2D shape", verifyShape(rowVector2D, rowVector2D.expected))

#8 Ragged Array ( But wrong dtype )
printTest("raggedArray shape", () -> Shape(raggedArray.data), DomainError)

# 9 Ragged Array no dtype

printTest("raggedArray shape", verifyShape(raggedArray, raggedArray.expected, Any))

println()

end

testShape()