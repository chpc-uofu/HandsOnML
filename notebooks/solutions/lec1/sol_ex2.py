# Solution for ex2: Sigmoid activation function

def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.
    
    Args:
        Z (np.ndarray): The input value(s).
    
    Returns:
        np.ndarray: The sigmoid of the input value(s).
    """
    return 1.0 / (1.0 + np.exp(-Z))
