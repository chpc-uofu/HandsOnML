import numpy as np
from typing import Tuple, List, Dict, Any

### Exercise 1

def init_param(n: int) -> Tuple[np.ndarray, float]:
    """
    Initialize the parameters (bias, weight) for the Binary Classifier.
    
    Args:
        n (int): The number of features.
    
    Returns:
        Tuple[np.ndarray, float]: A tuple containing the initialized weight (column vector)
                                 & the initialized bias (scalar).
    """
    return (np.zeros((n, 1),dtype='float'), 0.0)

def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.
    
    Args:
        Z (np.ndarray): The input value(s).
    
    Returns:
        np.ndarray: The sigmoid of the input value(s).
    """
    return 1.0 / (1.0 + np.exp(-Z))

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

### Exercise 2:

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

def train_model(X: np.ndarray, Y: np.ndarray, 
                num_epochs: int, lr: float) -> Tuple[List[float],np.ndarray, float]:
    """
    Train the binary classifier using gradient descent.
    
    Args:
        X (np.ndarray): The training data (features)  -> shape(n, m) 
                        where n is #features & m is #samples.
        Y (np.ndarray): The training labels (targets) -> shape(1, m)
                        where m is #samples.
        W (np.ndarray): The weight vector             -> shape(n, 1).
        b (float)     : The bias term                 -> float
        num_epochs (int): The number of epochs to train.
        lr (float)    : The learning rate.    
    
    Returns:
        Tuple[np.ndarray, float]: 
          A tuple containing the final weight and bias after training.
    """
    lstCost = []
    W, b = init_param(X.shape[0])
    # Loop over the number of epochs
    for i in range(num_epochs):
        A, cost = forward(X, Y, W, b)
        lstCost.append(cost)
        dW, db = calcgrad(X, Y, A)
        W, b = update(W, b, dW, db, lr)
    return lstCost, W, b

def predict_labels(X: np.ndarray, W: np.ndarray, b: float) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        X (np.ndarray): The input data (features)  -> shape(n, m) 
                        where n is #features & m is #samples.
        W (np.ndarray): The weight vector             -> shape(n, 1).
        b (float)     : The bias term                 -> float
    
    Returns:
        np.ndarray: The predicted labels (0 or 1).
    """
    A = sigmoid(np.dot(W.T, X) + b)
    return np.where(A >= 0.5, 1, 0)


def accuracy(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of the predictions.
    
    Args:
        Y_true (np.ndarray): The true labels.
        Y_pred (np.ndarray): The predicted labels.
    
    Returns:
        float: The accuracy as a percentage.
    """
    return np.mean(Y_true == Y_pred) * 100


     
