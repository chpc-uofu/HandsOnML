# Checking the update function
import numpy as np 
import numpy.random as rnd

def check_update():

    UPLIMIT = 1.E-10
    # Gen. Random matrices
    rng = rnd.default_rng(seed=42)
    N = 100
    W = rng.random((N,1))
    dW = 0.25*rng.random((N,1))
    b, db = 0.3, 0.05
    lr = 0.05

    W_corr = W - lr*dW
    b_corr = b - lr*db

    # Result from the function
    W_code, b_code = update(W,b,dW,db,lr)

    # Differences
    frob_diff = np.sqrt(np.sum((W_corr - W_code)**2))
    b_diff = np.abs(b_corr - b_code)

    assert frob_diff<UPLIMIT, "Frob. Norm (d(dW) TOO large"  
    assert b_diff<UPLIMIT, "Diff. b TOO large"  

    print("All tests passed!")

check_update()
