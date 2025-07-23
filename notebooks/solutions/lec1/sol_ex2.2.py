# Ex2.2 : Update function
def update(Weights: np.ndarray, bias: float,
           dWeight: np.ndarray, dbias:float,
           lr:float) -> Tuple[np.ndarray, float]:

    """
    Update the parameters using the gradients and learning rate.    

    Args:
        Weights (np.ndarray): The weight vector             -> shape(n, 1).
        bias (float)        : The bias term                 -> float
        dWeight (np.ndarray): The gradient of the cost w.r.t. W -> shape(n, 1).
        dbias (float)       : The gradient of the cost w.r.t. b -> float
        lr (float)          : The learning rate.    

    Returns:
        Tuple[np.ndarray, float]: 
          A tuple containing the updated weight and bias.
    """
    Weights -= lr * dWeight
    bias -= lr * dbias
    return Weights, bias
