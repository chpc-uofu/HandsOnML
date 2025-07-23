# Solution for ex2.1
def calcgrad(X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the gradients of the cost function with respect to W and b.
    
    Args:
        X (np.ndarray): The training data (features)  -> shape(n, m) 
                        where n is #features & m is #samples.
        Y (np.ndarray): The training labels (targets) -> shape(1, m)
                        where m is #samples.
        A (np.ndarray): The activation matrix          -> shape(1, m).
        
    Returns:
        Tuple[np.ndarray, float]: 
          A tuple containing the gradients with respect to W and b.
    """

    num_samples = X.shape[1]
    dZ = A - Y
    dW = np.dot(X, dZ.T) / num_samples
    db = np.sum(dZ) / num_samples
    return dW, db

