from typing import Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


### Exercise 1: Initialize the parameters
def init_param(n: int) -> Tuple[np.ndarray, float]:
    """
    Initialize the parameters (weights, bias) for the Binary Classifier.
    
    Args:
        n (int): The number of features/dim. of the input vector
    
    Returns:
        Tuple[np.ndarray, float]: A tuple containing
           - initialized weight -> shape (n,1)
           - initialized bias   -> scalar

    """
    return (np.zeros((n, 1),dtype='float'), 0.0)


### Exercise 2: Sigmoid activation function
def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.
    
    Args:
        Z (np.ndarray): The input vector.
    
    Returns:
        np.ndarray: The sigmoid of the input value(s).
    """
    return 1.0 / (1.0 + np.exp(-Z))


### Exercise 3: Forward Propagation
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


### Exercise 4: Backward Propagation
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


### Exercise 5: Update of the parameters
def update(Weights: np.ndarray, bias: float, 
           dWeight: np.ndarray, dbias:float, 
           lr:float) -> Tuple[np.ndarray, float]:

    """
    Update the parameters using the gradients and learning rate.    

    Args:
        Weights (np.ndarray): The weight vector              -> shape(n, 1).
        bias (float)        : The bias term                  -> float
        dWeight (np.ndarray): The grad. of the cost w.r.t. W -> shape(n, 1).
        dbias (float)       : The grad. of the cost w.r.t. b -> float
        lr (float)          : The learning rate.    

    Returns:
        Tuple[np.ndarray, float]: 
          A tuple containing the updated weight and bias.
    """
    Weights -= lr * dWeight
    bias -= lr * dbias
    return Weights, bias


# Exercise 6: Training (complete)
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


# Exercise 7: Predict labels
def predict_labels(X: np.ndarray, W: np.ndarray, b: float) -> np.ndarray:
    """
    Make predictions using the trained model.
    
    Args:
        X (np.ndarray): The input data (features)  -> shape(m,n) 
                        where m is #samples & n is #features.
        W (np.ndarray): The weight vector          -> shape(n,1).
        b (float)     : The bias term              -> float
    
    Returns:
        np.ndarray: The predicted labels (0 or 1).
    """
    A = sigmoid(np.dot(X,W) + b)
    return np.where(A >= 0.5, 1, 0)


# Exercise 8: Obtain accuracy
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



if __name__ == "__main__":

    # Generate the data set (fix random set)
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    print(f"Original data set:")
    print(f"  X.shape:{X.shape}")
    print(f"  y.shape:{y.shape}")

    test_ratio = 0.30
    X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=test_ratio, random_state=42)
    print(f"\nSplitting the data set -> test ratio:{test_ratio:4.2f}")
    print(f"  Training data:")
    print(f"    X_train.shape :: {X_train.shape}")
    print(f"    y_train.shape :: {y_train.shape}")
    print(f"  Test data:")
    print(f"    X_test.shape  :: {X_test.shape}")
    print(f"    y_test.shape  :: {y_test.shape}") 

    # Visualization of the training set
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu')  
    # plt.title("Synthetic (Training) Data (2 Features)")
    # plt.xlabel(r"$x_1$")
    # plt.ylabel(r"$x_2$")
    # plt.show()

    # TRAIN the model
    print(f"\nStart Training (Using own code)")
    lstCost, W, b = train_model(X=X_train, Y=y_train[:, np.newaxis], num_epochs=20000, lr=0.05)
    y_pred = predict_labels(X_train, W,b)
    acc = accuracy(y_pred, y_train[:,np.newaxis])  
    print(f"  Final cost: {lstCost[-1]:.4f}")
    print(f"  Weights   : {W.ravel()}")
    print(f"  Bias      : {b:.4f}") 
    print(f"  Accuracy  : {acc:.4f}")
    print(f"Training completed")

    # Comparison with SKLEARN API (Without L2 Regularization)
    skmodel = LogisticRegression(penalty=None, max_iter=100000, tol=1.E-12).fit(X_train,y_train)
    print(f"\nStart Training LogisticRegression (sklearn) without L2 (Training Set)")
    print(f"  coef:{skmodel.coef_}")
    print(f"  intercept:{skmodel.intercept_}")
    print(f"  score:{skmodel.score(X_train,y_train):8.4f}")
    print(f"Training completed")

    # TEST the model
    print(f"\nTest (Using own code)")
    y_pred = predict_labels(X_test, W,b)
    acc = accuracy(y_pred, y_test[:,np.newaxis])
    print(f"  Accuracy:{acc:.4f}")

    # Comparison with SKLEARN API (Without L2 Regularization)
    print(f"\nTest LogisticRegression (sklearn) without L2 (Test Data Set)")
    print(f"    score:{skmodel.score(X_test,y_test):8.4f}")

    

