# Checking the FORWARD function
import numpy as np 
import numpy.random as rnd

def check_forward():

    UPLIMIT = 1.E-10
    # Gen. Random matrices
    rng = rnd.default_rng(seed=42)
    X = rng.random((5,100))
    Y = rng.random((1,100))
    Y = np.where(Y>0.5,1.0,0.0)
    W = rng.random((5,1))
    b = 0.3

    # Result in situ
    n = X.shape[1]
    Z = np.dot(W.T, X) + b
    A = 1.0/(1.0+np.exp(-Z))
    cost = -np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A)) /n

    # Result from code
    A_code, cost_code = forward(X,Y,W,b)  

    # Differences
    frob_diff = np.sqrt(np.sum((A_code-A)**2))
    cost_diff = np.abs(cost-cost_code)

    assert frob_diff<UPLIMIT, "Frob. Norm (A) TOO large"  
    assert cost_diff<UPLIMIT, "Cost diff. TOO large"  

    print("All tests passed!")

check_forward()
