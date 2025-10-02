# Test for Ex. 2: Checking the sigmoid(function)
import numpy as np 
import numpy.random as rnd

def check_sigmoid():

    UPLIMIT = 1.E-10
    # Gen. random matrix
    rng = rnd.default_rng(seed=42)
    Z = rng.random((8,10))

    # Result in situ
    corr = 1.0/(1.0+np.exp(-Z))

    # Invoke result from code
    res = sigmoid(Z)
    frob_diff = np.sqrt(np.sum((corr-res)**2))

    assert frob_diff<UPLIMIT, "Frob. Norm TOO large"  

    print("All tests passed!")

check_sigmoid()
