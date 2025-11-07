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
        expected = np.array(data)
        return cls(data, expected, total_element, expected.shape, 1)

    @classmethod
    def fromMatrix(cls, data, total_element, row, col):
        expected = np.arange(1, total_element + 1).reshape(row, col)
        return cls(data, expected, total_element, expected.shape, 2)

    @classmethod
    def fromNArray(cls, data, total_element, *indices):
        expected = np.arange(1, total_element + 1).reshape(*indices)
        return cls(data, expected, total_element, expected.shape, len(indices))


scalar = testScalar(42)


vector_1 = TestDataArray.fromVector([1, 2, 3, 4], total_element=4)

vector_2 = TestDataArray.fromExpected(
    data=[10, 20, 30, 40, 50], expected=np.array([10, 20, 30, 40, 50]), total_element=5
)

e1 = TestDataArray.fromExpected(
    data=[1, 0, 0], expected=np.array([1, 0, 0]), total_element=3
)

e2 = TestDataArray.fromExpected(
    data=[0, 1, 0], expected=np.array([0, 1, 0]), total_element=3
)

e3 = TestDataArray.fromExpected(
    data=[0, 0, 1], expected=np.array([0, 0, 1]), total_element=3
)

v1 = TestDataArray.fromExpected(
    data=[1, 2, 3], expected=np.array([1, 2, 3]), total_element=3
)
v2 = TestDataArray.fromExpected(
    data=[0, 10, -100], expected=np.array([0, 10, -100]), total_element=3
)

vector_3 = TestDataArray.fromExpected(
    data=[1, 1, 1, 1, 1], expected=np.array([1, 1, 1, 1, 1]), total_element=5
)
vector_4 = TestDataArray.fromExpected(
    data=[2, 2, 2, 2, 2], expected=np.array([2, 2, 2, 2, 2]), total_element=5
)
vector_5 = TestDataArray.fromExpected(
    data=[2, 2, 2, 2, 2, 2], expected=np.array([2, 2, 2, 2, 2, 2]), total_element=6
)

empty_vector = TestDataArray.fromVector([], total_element=0)


matrix_1 = TestDataArray.fromExpected(
    data=[[1, 2, 3], [4, 5, 6]],  # manually rewritten in row-major
    expected=np.array([[1, 2, 3], [4, 5, 6]]),
    total_element=6,
)
matrix_2 = TestDataArray.fromExpected(
    data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # row-major
    expected=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    total_element=9,
)

column_matrix = TestDataArray.fromExpected(
    data=[[1], [2], [3]],
    expected=np.array([[1], [2], [3]]),
    total_element=3,
)

row_matrix = TestDataArray.fromExpected(
    data=[[1, 2, 3]],  # already row-major
    expected=np.array([[1, 2, 3]]),
    total_element=3,
)

array_3D_1 = TestDataArray.fromExpected(
    data=[
        [[1, 5], [2, 6]],
        [[3, 7], [4, 8]],
    ],
    expected=np.array(
        [
            [[1, 5], [2, 6]],
            [[3, 7], [4, 8]],
        ]
    ),
    total_element=8,
)

array_3D_2 = TestDataArray.fromExpected(
    data=[
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
        [[13, 14], [15, 16], [17, 18]],
        [[19, 20], [21, 22], [23, 24]],
    ],
    expected=np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
            [[19, 20], [21, 22], [23, 24]],
        ],
        order="F",
    ).transpose((1, 2, 0)),
    total_element=24,
)


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
    ("ragged_array", ragged_array),
]
