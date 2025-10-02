import numpy as np
from typing import Tuple, List, Dict, Any

### Exercise 1: Initialize the parameters

def init_param(n: int) -> Tuple[np.ndarray, float]:
    """
    Initialize the parameters (weight, bias) for the Binary Classifier.
    
    Args:
        n (int): The number of features/dim. of the input vector
    
    Returns:
        Tuple[np.ndarray, float]: A tuple containing 
           - initialized weight -> shape (n,1)
           - initialized bias   -> scalar
       
    """
    return (np.zeros((n, 1),dtype='float'), 0.0)
