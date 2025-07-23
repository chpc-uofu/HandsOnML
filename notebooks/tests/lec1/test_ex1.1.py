# Checking the init_parm(n:int) function

def check_init_param():

    # Call the function
    N = 30
    # result = bc.init_param(N)
    result = init_param(N)

    # 1. Check that it returns a tuple of length 2
    assert isinstance(result, tuple), "Function should return a tuple"
    assert len(result) == 2, "Function should return exactly two objects"

    # 2. Check that first one is NumPy array and second one is float
    a, b = result
    assert isinstance(a, np.ndarray), "First return value should be a NumPy array"
    assert isinstance(b, float),      "Second return value should be of type float"

    # 3. Check shapes
    expected_shape = (N, 1)
    assert a.shape == expected_shape, f"Expected shape {expected_shape}, got {a.shape}"

    # 4. Check content
    expected_a = np.zeros((N,1), dtype='float')
    expected_b = 0.0
    assert np.array_equal(a, expected_a), "First array content does not match"
    assert b == 0.0 , f"Second element MUST be zero and is {b}"

    print("All tests passed!")

check_init_param()
