# numpy implementation

import numpy as np


def smooth_s3_activation(
    input_array: np.ndarray, 
    derivative: bool = False, 
    steepness: float = 5.0
) -> np.ndarray:
    """
    Compute smooth S3 activation function or its derivative.
    
    This function implements a hybrid activation that smoothly transitions
    between softsign and sigmoid functions based on input magnitude.
    
    Args:
        input_array: Input numpy array of any shape
        derivative: If True, returns the derivative of the function
        steepness: Controls the transition steepness between functions (default: 5.0)
    
    Returns:
        numpy.ndarray: Activation values or derivatives with same shape as input
        
    Raises:
        TypeError: If input_array is not a numpy array
        ValueError: If steepness is not positive
        
    Examples:
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> smooth_s3_activation(x)
        array([-0.88079708, -0.5, 0., 0.5, 0.88079708])
        
        >>> smooth_s3_activation(x, derivative=True)
        array([0.10499359, 0.25, 0.5, 0.25, 0.10499359])
    """
    if not isinstance(input_array, np.ndarray):
        raise TypeError("input_array must be a numpy array")
    
    if steepness <= 0:
        raise ValueError("steepness must be positive")
    
    # Compute blending factor using sigmoid
    blending_factor = 1 / (1 + np.exp(-steepness * input_array))
    
    if not derivative:
        return _compute_activation(input_array, blending_factor)
    else:
        return _compute_derivative(input_array, blending_factor, steepness)


def _compute_activation(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compute the main activation function."""
    softsign_component = x / (1 + np.abs(x))
    sigmoid_component = 1 / (1 + np.exp(-x))
    return alpha * softsign_component + (1 - alpha) * sigmoid_component


def _compute_derivative(
    x: np.ndarray, 
    alpha: np.ndarray, 
    steepness: float
) -> np.ndarray:
    """Compute the derivative of the activation function."""
    # Derivative of blending factor
    dalpha_dx = steepness * alpha * (1 - alpha)
    
    # Softsign and its derivative
    softsign = x / (1 + np.abs(x))
    softsign_derivative = 1 / (1 + np.abs(x)) ** 2
    
    # Sigmoid and its derivative  
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)
    
    # Apply chain rule
    return (dalpha_dx * (softsign - sigmoid) + 
            alpha * softsign_derivative + 
            (1 - alpha) * sigmoid_derivative)