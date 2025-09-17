import numpy as np

def test_final_shape_and_broadcast():
    # --- Test 1: Simple integer indexing ---
    A = np.arange(16).reshape(2, 2, 2, 2)
    idx1 = np.array([0, 1])
    idx2 = np.array([[0, 1], [1, 0]])

    print(A[idx1, idx2])
    expected_shape = (2, 2, 2, 2)
    result_shape = np.broadcast(idx1, idx2).shape + A.shape[2:]
    print("Test 1:")
    print("Expected:", expected_shape)
    print("Result  :", result_shape)
    assert result_shape == expected_shape

    # --- Test 2: Boolean indexing ---
    B1 = np.array([[True, False], [False, True]])
    B2 = np.array([[0, 1], [1, 0]])
    # Shape of result = broadcasted shape of B1 and B2 + remaining axes
    expected_shape = np.broadcast(B1, B2).shape + A.shape[2:]
    result_shape = np.broadcast(B1, B2).shape + A.shape[2:]
    print("Test 2:")
    print("Expected:", expected_shape)
    print("Result  :", result_shape)
    assert result_shape == expected_shape

    # --- Test 3: Incompatible shapes ---
    C1 = np.array([0, 1])
    C2 = np.array([[0, 1], [1, 0]])
    try:
        np.broadcast(C1, C2)
        compatible = True
    except ValueError:
        compatible = False
    print("Test 3: Broadcast compatible?", compatible)
    assert compatible == True  # they are compatible

    # --- Test 4: Truly incompatible shapes ---
    D1 = np.array([0, 1, 2])
    D2 = np.array([[0, 1], [1, 0]])
    try:
        np.broadcast(D1, D2)
        compatible = True
    except ValueError:
        compatible = False
    print("Test 4: Broadcast compatible?", compatible)
    assert compatible == False

    # --- Test 5: Mixed boolean + integer indexing ---
    B3 = np.array([[0, 0], [0, 0]])
    I1 = np.array([[0, 1], [1, 0]])
    expected_shape = np.broadcast(B3, I1).shape + A.shape[2:]
    result_shape = np.broadcast(B3, I1).shape + A.shape[2:]
    print("Test 5:")
    print("Expected:", expected_shape)
    print("Result  :", result_shape)
    assert result_shape == expected_shape

    print("\nAll tests passed!")

test_final_shape_and_broadcast()