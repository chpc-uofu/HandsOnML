# Test Ex. 7: Check the prediction of labels
import numpy as np 
import numpy.random as rnd

def check_predict_labels():

    # Gen. Random matrices
    rng = rnd.default_rng(seed=42)
    m = 100
    X = rng.random((m,5))
    W = rng.random((5,1))
    b = 0.3

    Z = np.dot(X,W) + b
    A = 1.0/(1.0+np.exp(-Z))
    Y_corr = np.where(A>=0.5, 1,0)

    # Result from the function
    Y_code = predict_labels(X,W,b)

    # Different labels?
    assert np.all(Y_code == Y_corr), "Some Labels are diff."  

    print("All tests passed!")

check_predict_labels()
