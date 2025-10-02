# Solution for Ex. 7: Predict labels
def predict_labels(X: np.ndarray, W: np.ndarray, b: float) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        X (np.ndarray): The input data (features)  -> shape(m,n) 
                        where m is #samples & n is #.
        W (np.ndarray): The weight vector          -> shape(n,1)
        b (float)     : The bias term              -> float
    
    Returns:
        np.ndarray: The predicted labels (0 or 1).
    """
    A = sigmoid(np.dot(X,W) + b)
    return np.where(A >= 0.5, 1, 0)
