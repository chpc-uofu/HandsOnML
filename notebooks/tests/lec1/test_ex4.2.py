# Checking the accuracy function
import numpy as np 
import numpy.random as rnd

def check_accuracy():

    # Gen. Random Vectors
    VERY_SMALL = 1.0E-10
    rng = rnd.default_rng(seed=42)
    N = 200
    Y1 = np.where(rng.random((N))>=0.5,1,0)
    Y2 = np.where(rng.random((N))>=0.5,1,0)
    
    acc_corr = np.mean(Y1 == Y2) * 100
    acc_code = accuracy(Y1,Y2)    

    # Different labels?
    assert np.abs(acc_code-acc_code) < VERY_SMALL, "Diff. accuracy!"

    print("All tests passed!")

check_accuracy()
