module TestData

export ShapeData, NDArrayData, TestDataScalar, TestFancyIndexing, TestDataArray, scalar, vector_1,
    vector_2, vector_3, vector_4, vector_5, empty_vector, matrix_1, matrix_2, column_matrix,
    row_matrix, array_3D_1, array_3D_2, ragged_array, allData, e1, e2, e3, v1, v2


# ═════════════════════════════════════ Abstract Data Types ══════════════════════════════════════ #

abstract type ShapeData end

abstract type NDArrayData <: ShapeData end

# ═══════════════════════════════════════ Scalar Test Data ═══════════════════════════════════════ #

struct TestDataScalar <: NDArrayData
    data::Number
    total_element::Int
    shape::Tuple
    dim::Int

    TestDataScalar(data::Number) = new(data, 1, (), 0)
end

scalar = TestDataScalar(42)

# ═══════════════════════════════════════ Array Test Data ════════════════════════════════════════ #

struct TestDataArray <: NDArrayData
    array::AbstractVector
    matrix::AbstractArray
    total_element::Int
    shape::Tuple
    dim::Int
end

# ──── vector constructors ─────────────────────────────────────────────────────────────────────── #

testVector(data; total_element) = 
    TestDataArray(data, total_element, (total_element,), 1)

testVector(data, expected; total_element) = 
    TestDataArray(data, expected, total_element, (total_element,), 1)

# ──── matrix constructor ──────────────────────────────────────────────────────────────────────── #
            
testMatrix(data; total_element, row, col) = 
    TestDataArray(data, total_element, (row, col), 2)

testMatrix(data, expected; total_element, row, col) = 
    TestDataArray(data, expected, total_element, (row, col), 2)

# ──── ragged array constructor ────────────────────────────────────────────────────────────────── #
            
testRaggedArray(data, expected; total_element) = 
    TestDataArray(data, expected, total_element, (total_element,), 1)

testRaggedArray(data; total_element) = 
    TestDataArray(data, total_element, (total_element,), 1)

# ──── other array constructors ────────────────────────────────────────────────────────────────── #

testArray(data, expected; total_element, blocks, row, col) = 
    TestDataArray(data, expected, total_element, (row, col, blocks), 3)

testArray(data; total_element, blocks, row, col) = 
    TestDataArray(data, total_element, (row, col, blocks), 3)

testArray(expected) = TestDataArray(
    expected=expected, 
    total_element=prod(size(expected)), shape=size(expected), dim=ndims(expected)
)

# N dimensions
testArray(data, expected, dimensions::Vararg{Int}; total_element) = 
    TestDataArray(data, expected, total_element, (dimensions...), length(dimensions))

    testArray(data, dimensions::Vararg{Int}; total_element) = 
    TestDataArray(data, total_element, (dimensions...), length(dimensions))


# ──── catchall TestDataArray constructor ──────────────────────────────────────────────────────── #

function TestDataArray(data, total_element, shape, dim)
    litteral = dim == 1 ? Vector(1:total_element) : reshape(collect(1:total_element), shape...)
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


function TestDataArray(; expected, total_element, shape, dim)
    return TestDataArray([], expected, total_element, shape, dim)
end


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                            Test Datas                                            #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #


# ──── vectors ─────────────────────────────────────────────────────────────────────────────────── #
            
vector_1 = testVector([1, 2, 3, 4], total_element=4)
vector_2 = testVector([10, 20, 30, 40, 50], [10, 20, 30, 40, 50], total_element=5)

vector_3 = testVector([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], total_element=5)
vector_4 = testVector([2, 2, 2, 2, 2], [2, 2, 2, 2, 2], total_element=5)
vector_5 = testVector([2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], total_element=6)

e1 = testVector([1, 0, 0], [1, 0, 0], total_element=3)
e2 = testVector([0, 1, 0], [0, 1, 0], total_element=3)
e3 = testVector([0, 0, 1], [0, 0, 1], total_element=3)

v1 = testVector([1, 2, 3], [1, 2, 3], total_element=3)
v2 = testVector([0, 10, -100], [0, 10, -100], total_element=3)

empty_vector = testVector([], total_element=0)

# ──── matrix ──────────────────────────────────────────────────────────────────────────────────── #
            
matrix_1 = testMatrix([[1, 2, 3], [4, 5, 6]], [1 2 3; 4 5 6], total_element=6, row=2, col=3)

matrix_2 = testMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1 2 3; 4 5 6; 7 8 9],
    total_element=9, row=3, col=3)

column_matrix = testMatrix([
        [1],
        [2],
        [3]], Base.reshape([1, 2, 3], 3, 1), total_element=3, row=3, col=1)

row_matrix = testMatrix([
        [1, 2, 3]], Base.reshape([1, 2, 3], 1, 3), total_element=3, row=1, col=3)

# ──── n dimensional array ─────────────────────────────────────────────────────────────────────── #

array_3D_1 = testArray([
        [[1, 2],
            [3, 4]],
        [[5, 6],
            [7, 8]],
    ], [1 2; 3 4;;; 5 6; 7 8], total_element=8, blocks=2, row=2, col=2)

array_3D_2 = testArray(
    # in numpy this is (4 block, 3 row, 2 col )
    [
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
        [[13, 14], [15, 16], [17, 18]],
        [[19, 20], [21, 22], [23, 24]]
    ],

    # in julia this is ( 3 row, 2 col, 4 block )
    [1 2; 3 4; 5 6;;; 7 8; 9 10; 11 12;;; 13 14; 15 16; 17 18;;; 19 20; 21 22; 23 24],
    total_element=24,
    blocks=4,
    row=3,
    col=2
)

# ──── ragged array ────────────────────────────────────────────────────────────────────────────── #
            
ragged_array = testRaggedArray([[1, 2], [3, 4, 5]], [[1, 2], [3, 4, 5]], total_element=2)

# ═══════════════════════════════════════════ all data ═══════════════════════════════════════════ #
            

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
    ("ragged_array", ragged_array)]


end # TestData