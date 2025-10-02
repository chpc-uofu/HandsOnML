# Solution Ex. 8: Accuracy 

def accuracy(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of the predictions.
    
    Args:
        Y_true (np.ndarray): The true labels      -> shape(m,1)
        Y_pred (np.ndarray): The predicted labels -> shape(m,1)
    
    Returns:
        float: The accuracy as a percentage.
    """
    return np.mean(Y_true == Y_pred) * 100

