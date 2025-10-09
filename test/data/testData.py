import numpy as np

class TestDataScalar:
    def __init__(self, data, total_element, shape, dim):
        self.data = data
        self.total_element = total_element
        self.shape = shape
        self.dim = dim

def testScalar(data):
    return TestDataScalar(data, 1, (), 0)

class TestDataArray:

    def __init__(self, array, matrix, total_element, shape, dim):
        self.array = array
        self.matrix = matrix
        self.total_element = total_element
        self.shape = shape
        self.dim = dim

    @classmethod
    def fromArray(cls, data, total_element, shape, dim):
        if dim == 1:
            litteral = np.arange(1, total_element + 1)  # 1D arrat
        else:
            litteral = np.arange(1, total_element + 1).reshape(
                shape, order='F'
            )  # reshape to shape tuple
        return cls(data, litteral, total_element, shape, dim)

    @classmethod
    def fromExpected(cls, expected, total_element=None, data=None):
        if total_element is None:
            total_element = np.prod(expected.shape)
        if data is None:
            data = []
        return cls(data, expected, total_element, expected.shape, len(expected.shape))
    
    @classmethod
    def fromRaggedArray(cls, data, total_element):
        array = np.array(data, dtype=object)
        return cls(data, array, total_element, (total_element,), 1)

    @classmethod
    def fromVector(cls, data, total_element):
        array = np.arange(1, total_element + 1).reshape((total_element,))
        return cls(data, array, total_element, array.shape, 1)

    @classmethod
    def fromMatrix(cls, data, total_element, row, col):
        array = np.arange(1, total_element + 1).reshape((row, col), order='F')
        return cls(data, array, total_element, array.shape, 2)

    @classmethod
    def fromNArray(cls, data, total_element, *indices):
        array = np.arange(1, total_element + 1).reshape(order='F', *indices)
        return cls(data, array, total_element, array.shape, len(indices))


scalar = testScalar(42)


vector_1 = TestDataArray.fromVector([1, 2, 3, 4], total_element=4)

vector_2 = TestDataArray.fromExpected(
    data=[10, 20, 30, 40, 50], expected=np.array([10, 20, 30, 40, 50]), total_element=5
)

e1 = TestDataArray.fromExpected(data=[1, 0, 0], expected=np.array([1, 0, 0]), total_element=3)

e2 = TestDataArray.fromExpected(data=[0, 1, 0], expected=np.array([0, 1, 0]), total_element=3)

e3 = TestDataArray.fromExpected(data=[0, 0, 1], expected=np.array([0, 0, 1]), total_element=3)

v1 = TestDataArray.fromExpected(data=[1, 2, 3], expected=np.array([1, 2, 3]), total_element=3)
v2 = TestDataArray.fromExpected(data=[0, 10, -100], expected=np.array([0, 10, -100]), total_element=3)

vector_3 =  TestDataArray.fromExpected( data=[1, 1, 1, 1, 1], expected=np.array([1, 1, 1, 1, 1]),  total_element=5)
vector_4 =  TestDataArray.fromExpected( data=[2, 2, 2, 2, 2], expected=np.array([2, 2, 2, 2, 2]),  total_element=5)
vector_5 =  TestDataArray.fromExpected( data=[2, 2, 2, 2, 2, 2], expected=np.array([2, 2, 2, 2, 2, 2]),  total_element=6)

empty_vector = TestDataArray.fromVector([], total_element=0)

matrix_1 = TestDataArray.fromMatrix(
    data=[[1, 3, 5], [2, 4, 6]], total_element=6, row=2, col=3
)

matrix_2 = TestDataArray.fromMatrix(
    [[1, 4, 7], [2, 5, 8], [3, 6, 9]], total_element=9, row=3, col=3
)

column_matrix = TestDataArray.fromMatrix([[1], [2], [3]], total_element=3, row=3, col=1)

row_matrix = TestDataArray.fromMatrix([[1, 2, 3]], total_element=3, row=1, col=3)

# array n-th dimension:
array_3D_1 = TestDataArray.fromNArray([[[1, 3], [2, 4]], [[5, 7], [6, 8]]], 8, 2, 2, 2)

array_3D_2 = TestDataArray.fromNArray(
    [
        [[1, 4], 
         [2, 5], 
         [3, 6]],
        [[7, 10], [8, 11], [9, 12]],
        [[13, 16], [14, 17], [15, 18]],
        [[19, 22], [20, 23], [21, 24]],
    ],
    24,
    4,
    3,
    2,
)

array_4D_1 = TestDataArray.fromExpected(np.arange(1, 17).reshape(2, 2, 2, 2, order='F'))


# ragged array
ragged_array = TestDataArray.fromRaggedArray([[1, 2], [3, 4, 5]], total_element=2)

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
