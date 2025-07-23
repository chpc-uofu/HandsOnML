import numpy as np
from typing import Tuple, List, Dict, Any

### Exercise 1.1:

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
