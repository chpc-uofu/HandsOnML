# Solution Ex.6 : Training (complete)
def train_model(X: np.ndarray, Y: np.ndarray,
                num_epochs: int, lr: float) -> Tuple[List[float],np.ndarray, float]:
    """
    Train the binary classifier using gradient descent.

    Args:
        X (np.ndarray): The training data (features)  -> shape(m,n) 
                        where m is #samples & n is #features.
        Y (np.ndarray): The training labels (targets) -> shape(m,1)
                        where m is #samples.
        W (np.ndarray): The weight vector             -> shape(n,1).
        b (float)     : The bias term                 -> float
        num_epochs (int): The number of epochs to train.
        lr (float)    : The learning rate.   

    Returns:
        Tuple[np.ndarray, float]:
          A tuple containing the final weight and bias after training.
    """
    lstCost = []
    W, b = init_param(X.shape[1])
    # Loop over the number of epochs
    for i in range(num_epochs):
        A, cost = forward(X, Y, W, b)
        lstCost.append(cost)
        dW, db = calcgrad(X, Y, A)
        W, b = update(W, b, dW, db, lr)
    return lstCost, W, b
