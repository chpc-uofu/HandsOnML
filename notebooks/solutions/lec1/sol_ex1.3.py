# Solution for ex1.3
def forward(X: np.ndarray, Y: np.ndarray,
            W: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """
    Perform the forward pass of the binary classifier.
    
    Args:
        X (np.ndarray): The training data (features)  -> shape(n, m) 
                        where n is #features & m is #samples.
        Y (np.ndarray): The training labels (targets) -> shape(1, m)
                        where m is #samples.
        W (np.ndarray): The weight vector             -> shape(n,1 ).
        b (float)     : The bias term                 -> float
    
    Returns:
        Tuple[np.ndarray, float]: 
          A tuple containing the activation matrix and the cost.
    """
    num_samples = X.shape[1]
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    cost = -np.sum(Y * np.log(A) + \
                    (1.0 - Y) * np.log(1.0 - A)) /num_samples
    return A, cost
