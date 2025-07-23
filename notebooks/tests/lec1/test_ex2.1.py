# Checking the calcgrad function
import numpy as np 
import numpy.random as rnd

def check_calcgrad():

    UPLIMIT = 1.E-10
    # Gen. Random matrices
    rng = rnd.default_rng(seed=42)
    X = rng.random((5,100))
    Y = rng.random((1,100))
    Y = np.where(Y>0.5,1.0,0.0)
    A = rng.random((1,100))

    num_samples = X.shape[1]
    dZ = A - Y
    dW_corr = np.dot(X, dZ.T)/num_samples  
    db_corr = np.sum(dZ)/num_samples

    # Result from code
    dW_code, db_code = calcgrad(X,Y, A)

    # Differences
    frob_diff = np.sqrt(np.sum((dW_corr - dW_code)**2))
    db_diff = np.abs(db_corr - db_code)

    assert frob_diff<UPLIMIT, "Frob. Norm (d(dW) TOO large"  
    assert db_diff<UPLIMIT, "Diff. db TOO large"  

    print("All tests passed!")

check_calcgrad()
