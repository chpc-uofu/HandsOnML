# Solution for Ex. 4: Backpropagation

def calcgrad(X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the gradients of the cost function with respect to W and b.
    
    Args:
        X (np.ndarray): The training data (features)  -> shape(m,n) 
                        where m is #samples & n is #features 
        Y (np.ndarray): The training labels (targets) -> shape(m,1)
                        where m is #samples
        A (np.ndarray): The activation matrix         -> shape(m,1).
        
    Returns:
        Tuple[np.ndarray, float]: 
          A tuple containing the gradients with respect to W and b.
    """
    num_samples = X.shape[0]
    dZ = A - Y
    dW = np.dot(X.T, dZ) / num_samples
    db = np.sum(dZ) / num_samples
    return dW, db
