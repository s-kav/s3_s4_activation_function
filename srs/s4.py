# numba (CPU), CUDA (GPU), and Legacy compatibility implementation
# version 2.0
# s4.py
"""
S4 Hybrid Activation Function - Accelerated Implementation

This module provides a high-performance implementation of the S4 activation function
using Numba for CPU (JIT) and GPU (CUDA) acceleration.

--------------------------------------------------------------------------------
-- USER GUIDE & ENVIRONMENT SETUP --
--------------------------------------------------------------------------------
This implementation relies on the Numba library. Due to the complexity of CUDA
dependencies, environments like Google Colab can be challenging to configure.
For maximum stability and compatibility, we strongly recommend using Conda.

**Recommended Setup (using Conda):**

1. Install Conda or Miniconda.
2. Create a dedicated, clean environment. This command installs Python and all
   necessary Numba/CUDA components that are guaranteed to be compatible:

   conda create -n s4_env -c conda-forge python=3.11 numba cudatoolkit=11.8

3. Activate the environment:

   conda activate s4_env

4. Run your Python script within this environment.

--------------------------------------------------------------------------------
-- USAGE --
--------------------------------------------------------------------------------
from s4 import S4Activation, s4
import numpy as np

# Object-oriented interface (recommended)
activator = S4Activation(device='auto')
x = np.linspace(-10, 10, 1000)
y = activator(x)
dy = activator(x, derivative=True)

# Functional interface (convenient)
y = s4(x, device='auto')
"""


import numpy as np
import math

# --- Graceful Numba/CUDA Import ---
# This allows the module to be imported and used even if Numba is not installed.
# It will fall back to a pure NumPy implementation.
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
    GPU_AVAILABLE = cuda.is_available()
except ImportError:
    print("Warning: Numba not found. Falling back to a slower NumPy implementation.")
    NUMBA_AVAILABLE = False
    GPU_AVAILABLE = False

    # Define dummy decorators so the code doesn't crash on import
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    class DummyCuda:
        def __getattr__(self, name):
            def dummy_func(*args, **kwargs):
                raise NotImplementedError("CUDA is not available.")
            return dummy_func
    cuda = DummyCuda()

# ============================================================================
# Pure NumPy Implementation (Fallback)
# ============================================================================

def s4_numpy(x: np.ndarray, derivative: bool = False, k: float = 5.0) -> np.ndarray:
    """Pure NumPy implementation of S4. Used as a fallback if Numba is unavailable."""
    exp_neg_kx = np.exp(-k * x)
    exp_neg_x = np.exp(-x)
    a = 1.0 / (1.0 + exp_neg_kx)
    
    if not derivative:
        abs_x = np.abs(x)
        one_plus_abs_x = 1.0 + abs_x
        softsign = x / one_plus_abs_x
        sigmoid = 1.0 / (1.0 + exp_neg_x)
        return a * softsign + (1.0 - a) * sigmoid
    else:
        one_minus_a = 1.0 - a
        da_dx = k * a * one_minus_a
        abs_x = np.abs(x)
        one_plus_abs_x = 1.0 + abs_x
        softsign = x / one_plus_abs_x
        d_softsign = 1.0 / (one_plus_abs_x * one_plus_abs_x)
        sigmoid = 1.0 / (1.0 + exp_neg_x)
        d_sigmoid = sigmoid * (1.0 - sigmoid)
        return (da_dx * (softsign - sigmoid) + a * d_softsign + one_minus_a * d_sigmoid)

# ============================================================================
# Numba CPU JIT Implementation
# ============================================================================

@jit(nopython=True, fastmath=True)
def s4_cpu_kernel(x: np.ndarray, derivative: bool = False, k: float = 5.0) -> np.ndarray:
    """
    S4 hybrid activation function - CPU optimized with Numba JIT.
    
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
    out = np.empty_like(x)
    for i in range(x.size):
        x_val = x[i]
        exp_neg_kx = math.exp(-k * x_val)
        exp_neg_x = math.exp(-x_val)
        a = 1.0 / (1.0 + exp_neg_kx)
        if not derivative:
            abs_x = math.fabs(x_val)
            one_plus_abs_x = 1.0 + abs_x
            softsign = x_val / one_plus_abs_x
            sigmoid = 1.0 / (1.0 + exp_neg_x)
            out[i] = a * softsign + (1.0 - a) * sigmoid
        else:
            one_minus_a = 1.0 - a
            da_dx = k * a * one_minus_a
            abs_x = math.fabs(x_val)
            one_plus_abs_x = 1.0 + abs_x
            softsign = x_val / one_plus_abs_x
            d_softsign = 1.0 / (one_plus_abs_x * one_plus_abs_x)
            sigmoid = 1.0 / (1.0 + exp_neg_x)
            d_sigmoid = sigmoid * (1.0 - sigmoid)
            out[i] = (da_dx * (softsign - sigmoid) + a * d_softsign + one_minus_a * d_sigmoid)
    return out


# ============================================================================
# Numba CUDA (GPU) Implementation
# ============================================================================

if GPU_AVAILABLE:
    @cuda.jit(device=True)
    def s4_cuda_device_function(x_val: np.ndarray, derivative: bool = False, k: float = 5.0) -> np.ndarray:
        """A CUDA device function that computes the S4 value for a single number."""
        exp_neg_kx = math.exp(-k * x_val)
        exp_neg_x = math.exp(-x_val)
        a = 1.0 / (1.0 + exp_neg_kx)
        if not derivative:
            abs_x = math.fabs(x_val)
            one_plus_abs_x = 1.0 + abs_x
            softsign = x_val / one_plus_abs_x
            sigmoid = 1.0 / (1.0 + exp_neg_x)
            return a * softsign + (1.0 - a) * sigmoid
        else:
            one_minus_a = 1.0 - a
            da_dx = k * a * one_minus_a
            abs_x = math.fabs(x_val)
            one_plus_abs_x = 1.0 + abs_x
            softsign = x_val / one_plus_abs_x
            d_softsign = 1.0 / (one_plus_abs_x * one_plus_abs_x)
            sigmoid = 1.0 / (1.0 + exp_neg_x)
            d_sigmoid = sigmoid * (1.0 - sigmoid)
            return (da_dx * (softsign - sigmoid) + a * d_softsign + one_minus_a * d_sigmoid)

    @cuda.jit
    def s4_cuda_kernel(x_array, out_array, derivative, k):
        """
        CUDA kernel for S4 activation function.
        
        Each thread processes one element of the input array.
        
        Args:
            x: Input array (device memory)
            result: Output array (device memory)
            derivative: Boolean flag for derivative computation
            k: Steepness parameter
        """
        idx = cuda.grid(1)
        if idx < x_array.size:
            out_array[idx] = s4_cuda_device_function(x_array[idx], derivative, k)

# ============================================================================
# HIGH-LEVEL BACKEND WRAPPERS
# These functions provide a clean interface to the low-level kernels.
# ============================================================================

def s4_cpu(x: np.ndarray, derivative: bool = False, k: float = 5.0) -> np.ndarray:
    """High-level wrapper for CPU execution, choosing between Numba and NumPy."""
    if NUMBA_AVAILABLE:
        return s4_cpu_kernel(x, derivative=derivative, k=k)
    return s4_numpy(x, derivative=derivative, k=k)

def s4_gpu(x: np.ndarray, derivative: bool = False, k: float = 5.0) -> np.ndarray:
    """High-level wrapper for GPU execution, handling all CUDA memory management."""
    if not GPU_AVAILABLE:
        raise RuntimeError("s4_gpu called, but CUDA is not available.")
    
    # Ensure input is a numpy array and has the correct type
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    
    d_x = cuda.to_device(x)
    d_out = cuda.device_array_like(d_x)
    threads_per_block = 256
    blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block
    s4_cuda_kernel[blocks_per_grid, threads_per_block](d_x, d_out, derivative, k)
    return d_out.copy_to_host()

# ============================================================================
# Unified Interface with Automatic Device Selection
# ============================================================================

class S4Activation:
    """
    S4 hybrid activation function with automatic CPU/GPU selection.
    
    This class provides a unified interface for the S4 activation function
    with automatic backend selection based on hardware availability and
    array size.
    
    Examples:
        >>> import numpy as np
        >>> s4 = S4Activation(device='auto')
        >>> x = np.random.randn(1000, 512)
        >>> y = s4(x)
        >>> dy = s4(x, derivative=True)
    """
    def __init__(self, device: str = 'auto', k: float = 5.0, gpu_threshold: int = 10000):
        """
        Initialize S4 activation function.
        
        Args:
            device: Device selector - 'auto', 'cpu', or 'gpu'
                   'auto' automatically selects based on availability and size.
            k: Steepness parameter for transition (default: 5.0).
            gpu_threshold: Minimum array size to prefer GPU (when device='auto').
        """
        self.k = k
        self.gpu_threshold = gpu_threshold
        self.device = device
        self._gpu_available = GPU_AVAILABLE  # Use the globally checked flag
        
        if device not in ['auto', 'cpu', 'gpu']:
            raise ValueError("Device must be one of 'auto', 'cpu', or 'gpu'")
        if device == 'gpu' and not self._gpu_available:
            raise RuntimeError("GPU device requested but CUDA is not available.")
    
    def _select_backend(self, x: np.ndarray) -> str:
        """Automatically select computation backend based on hardware and array size."""
        if self.device == 'cpu':
            return 'cpu'
        if self.device == 'gpu':
            return 'gpu'
        # 'auto' mode
        if self._gpu_available and x.size >= self.gpu_threshold:
            return 'gpu'
        return 'cpu'
    
    def __call__(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        """
        Compute S4 activation function or its derivative.
        
        Args:
            x: Input array of any shape.
            derivative: If True, returns derivative; otherwise, returns function value.
            
        Returns:
            Output array of same shape as input.
        """
        backend = self._select_backend(x)
        
        if backend == 'gpu':
            return s4_gpu(x, derivative=derivative, k=self.k)
        else:
            return s4_cpu(x, derivative=derivative, k=self.k)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of S4 activation."""
        return self(x, derivative=False)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Backward pass of S4 activation (derivative)."""
        return self(x, derivative=True)
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_available
    
    def get_backend_info(self, x: np.ndarray) -> dict:
        """Get information about which backend would be used for a given input."""
        backend = self._select_backend(x)
        return {
            'backend': backend, 'gpu_available': self._gpu_available,
            'array_size': x.size, 'array_shape': x.shape,
            'gpu_threshold': self.gpu_threshold, 'device_setting': self.device
        }

# ============================================================================
# Convenience Functions (Legacy API compatibility)
# ============================================================================

def s4(x: np.ndarray, derivative: bool = False, k: float = 5.0, 
       device: str = 'auto') -> np.ndarray:
    """
    S4 hybrid activation function with automatic device selection.
    
    Convenience function that provides a simple interface similar to the
    original numpy implementation but with automatic CPU/GPU acceleration.
    
    Args:
        x: Input array
        derivative: If True, returns derivative; if False, returns function value
        k: Steepness parameter for transition (default: 5.0)
        device: Device selector - 'auto', 'cpu', or 'gpu'
        
    Returns:
        Function values or derivatives of same shape as input
        
    Examples:
        >>> import numpy as np
        >>> x = np.random.randn(1000, 512)
        >>> y = s4(x)  # Auto-selects best device
        >>> dy = s4(x, derivative=True)
        >>> y_gpu = s4(x, device='gpu')  # Force GPU
    """
    activator = S4Activation(device=device, k=k)
    return activator(x, derivative=derivative)