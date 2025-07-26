# numpy implementation

import numpy as np


def s4(x: np.ndarray, derivative: bool = False, k: float = 5.0) -> np.ndarray:
    """
    S4 hybrid activation function with smooth transition between softsign and sigmoid.
    
    Uses weighted combination: a * softsign(x) + (1-a) * sigmoid(x)
    where a = sigmoid(k*x) is the weighting factor.
    
    For x << 0: behaves like sigmoid (smoother gradients)
    For x >> 0: behaves like softsign (bounded output, stable gradients)
    
    Args:
        x: Input array
        derivative: If True, returns derivative; if False, returns function value
        k: Steepness parameter for transition (higher k = sharper transition)
        
    Returns:
        Function values or derivatives of same shape as input
        
    Mathematical form:
        f(x) = sigmoid(k*x) * softsign(x) + (1 - sigmoid(k*x)) * sigmoid(x)
        
    Properties:
        - Smooth transition between activation types
        - Bounded output approximately in [-1, 1]
        - Stable gradients for large |x|
        - Differentiable everywhere
    """
    # Precompute common exponentials to avoid redundant calculations
    exp_neg_kx = np.exp(-k * x)
    exp_neg_x = np.exp(-x)
    
    # Weighting factor a = sigmoid(k*x)
    a = 1 / (1 + exp_neg_kx)
    
    if not derivative:
        # Precompute abs(x) and 1 + abs(x) for softsign
        abs_x = np.abs(x)
        one_plus_abs_x = 1 + abs_x
        
        # Vectorized computation
        softsign = x / one_plus_abs_x
        sigmoid = 1 / (1 + exp_neg_x)
        
        return a * softsign + (1 - a) * sigmoid
    
    else:
        # Derivative computation with optimized shared calculations
        one_minus_a = 1 - a
        
        # da/dx = k * a * (1-a) - reuse one_minus_a
        da_dx = k * a * one_minus_a
        
        # Softsign and its derivative
        abs_x = np.abs(x)
        one_plus_abs_x = 1 + abs_x
        softsign = x / one_plus_abs_x
        d_softsign = 1 / (one_plus_abs_x * one_plus_abs_x)  # Avoid **2
        
        # Sigmoid and its derivative - reuse exp_neg_x
        sigmoid = 1 / (1 + exp_neg_x)
        d_sigmoid = sigmoid * (1 - sigmoid)
        
        return (da_dx * (softsign - sigmoid) + 
                a * d_softsign + 
                one_minus_a * d_sigmoid)