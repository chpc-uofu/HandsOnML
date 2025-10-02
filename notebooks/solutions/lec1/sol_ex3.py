# Solution for Ex. 3: Forward Propagation
def forward(X: np.ndarray, Y: np.ndarray,
            W: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """
    Perform the forward pass of the binary classifier.
 
    Args:
        X (np.ndarray): The training data (features)  -> shape(m, n) 
                        where m is #samples & n is #features.
        Y (np.ndarray): The training labels (targets) -> shape(m,1)  
                        where m is #samples.
        W (np.ndarray): The weight vector             -> shape(n,1)
        b (float)     : The bias term                 -> float
    
    Returns:
        Tuple[np.ndarray, float]: 
          A tuple containing the activation matrix and the cost.
    """
    num_samples = X.shape[0]
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    cost = -np.sum(Y * np.log(A) + \
                    (1.0 - Y) * np.log(1.0 - A)) /num_samples
    return A, cost
