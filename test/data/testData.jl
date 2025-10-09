module TestData

export ShapeData, NDArrayData, TestDataScalar, TestFancyIndexing, TestDataArray, scalar, vector_1, vector_2, vector_3, vector_4, vector_5, empty_vector, matrix_1, matrix_2, column_matrix, row_matrix, array_3D_1, array_3D_2, ragged_array, allData, array_4D_1, e1, e2, e3, v1, v2
abstract type ShapeData end

abstract type NDArrayData <: ShapeData end

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Test Data: scalar                                                               #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

struct TestDataScalar <: NDArrayData
    data::Number
    total_element::Int
    shape::Tuple
    dim::Int

    TestDataScalar(data::Number) = new(data, 1, (), 0)
end

scalar = TestDataScalar(42)

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Test Data: arrays                                                               #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

struct TestDataArray <: NDArrayData
    array::AbstractVector
    matrix::AbstractArray
    total_element::Int
    shape::Tuple
    dim::Int
end

testVector(data; total_element) = TestDataArray(data, total_element, (total_element,), 1)

testVector(data, expected; total_element) = TestDataArray(data, expected, total_element, (total_element,), 1)

testMatrix(data; total_element, row, col) = TestDataArray(data, total_element, (row, col), 2)

testMatrix(data, expected; total_element, row, col) = TestDataArray(data, expected, total_element, (row, col), 2)

testRaggedArray(data, expected; total_element) = TestDataArray(data, expected, total_element, (total_element,), 1)

testRaggedArray(data; total_element) = TestDataArray(data, total_element, (total_element,), 1)

# 3d array
testArray(data, expected; total_element, blocks, row, col) = TestDataArray(data, expected, total_element, (blocks, row, col), 3)

testArray(data; total_element, blocks, row, col) = TestDataArray(data, total_element, (blocks, row, col), 3)

testArray(expected) = TestDataArray(expected=expected, total_element=Base.prod(size(expected)), shape=size(expected), dim=ndims(expected))

# nd array

testArray(data, expected, dimensions::Vararg{Int}; total_element) = TestDataArray(data, expected, total_element, (dimensions...), length(dimensions))

testArray(data, dimensions::Vararg{Int}; total_element) = TestDataArray(data, total_element, (dimensions...), length(dimensions))

function TestDataArray(data, total_element, shape, dim)

    litteral = dim == 1 ? Vector(1:total_element) : reshape(1:total_element, shape...)
    return TestDataArray(data, litteral, total_element, shape, dim)
end

function TestDataArray(data, litteral, total_element, shape, dim)

    # We look if the two format contains the same element at the same index
    function get_nested(nested, I::Tuple)
        x = nested
        for i in I
            x = x[i]
        end
        return x
    end

    same_content = true

    for indices in Iterators.product(shape)
        if (litteral[indices...] != get_nested(data, indices))
            println("Comparison byIndex failed at index: ", indices)
            println(" Nested Array format was : ", data)
            println(" Matrix Litteral format was : ", litteral)
            same_content = false
            break
        end
    end

    @assert same_content == true

    return TestDataArray(data, litteral, total_element, shape, dim)
end


function TestDataArray(;expected, total_element, shape, dim)
    return TestDataArray([], expected, total_element, shape, dim)
end


# vector:

vector_1 = testVector([1, 2, 3, 4], total_element=4)
vector_2 = testVector([10, 20, 30, 40, 50], [10, 20, 30, 40, 50],  total_element=5)

vector_3 = testVector([1, 1, 1, 1, 1], [1, 1, 1, 1, 1],  total_element=5)
vector_4 = testVector([2, 2, 2, 2, 2], [2, 2, 2, 2, 2],  total_element=5)
vector_5 = testVector([2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2],  total_element=6)

e1 = testVector([1, 0, 0], [1, 0, 0], total_element=3)
e2 = testVector([0, 1, 0], [0, 1, 0], total_element=3)
e3 = testVector([0, 0, 1], [0, 0, 1], total_element=3)

v1 = testVector([1, 2, 3], [1, 2, 3],  total_element=3)
v2 = testVector([0, 10, -100], [0, 10, -100],  total_element=3)

empty_vector = testVector([], total_element=0)

# matrix:

matrix_1 = testMatrix([
        [1, 3, 5],
        [2, 4, 6]], total_element=6, row=2, col=3)

matrix_2 = testMatrix([
    [1, 4, 7], 
    [2, 5, 8], 
    [3, 6, 9]], total_element=9, row=3, col=3)

column_matrix = testMatrix([
        [1],
        [2],
        [3]], total_element=3, row=3, col=1)

row_matrix = testMatrix([
        [1, 2, 3]], total_element=3, row=1, col=3)

# array n-th dimension:

array_3D_1 = testArray([
        [
            [1, 3],
            [2, 4]
        ],
        [
            [5, 7],
            [6, 8]
        ]
    ], total_element=8, blocks=2, row=2, col=2)

array_3D_2 = testArray([
        [
            [1, 4],
            [2, 5],
            [3, 6]
        ], [
            [7, 10],
            [8, 11],
            [9, 12]
        ], [
            [13, 16],
            [14, 17],
            [15, 18]
        ], [
            [19, 22],
            [20, 23],
            [21, 24]
        ]], total_element=24, blocks=4, row=3, col=2)

array_4D_1 = testArray(reshape(1:16, 2, 2, 2, 2))


# ragged array

ragged_array = testRaggedArray([[1, 2], [3, 4, 5]], [[1, 2], [3, 4, 5]], total_element=2)

allData = [
    ("scalar", scalar),
    ("vector_1", vector_1),
    ("vector_2", vector_2),
    ("vector_3", vector_3),
    ("vector_4", vector_4),
    ("vector_5", vector_5),
    ("e1", e1),
    ("e2", e2),
    ("e3", e3),
    ("v1", v1),
    ("v2", v2),
    ("empty_vector", empty_vector),
    ("matrix_1", matrix_1),
    ("matrix_2", matrix_2),
    ("column_matrix", column_matrix),
    ("row_matrix", row_matrix),
    ("array_3D_1", array_3D_1),
    ("array_3D_2", array_3D_2),
    ("ragged_array", ragged_array)
    
]

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Test Data: create and check data                                                #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


end # TestData


#my_1d_array     =  NDArray{Int}([10, 20, 30, 40, 50])

# my_2d_array     =  NDArray{Int}([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# my_3d_array     =  NDArray{Int}([
# [
#     [1, 2],
#     [3, 4],
#     [5, 6]
# ],

# [
#     [7, 8],
#     [9, 10],
#     [11, 12]
# ],

# [
#     [13, 14],
#     [15, 16],
#     [17, 18]
# ],

# [
#     [19, 20],
#     [21, 22],
#     [23, 24]
# ]])

# my_3d_array2 =  NDArray{Int}(1:24, 4, 3, 2)
