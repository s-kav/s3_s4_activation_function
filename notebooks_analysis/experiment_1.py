# experiment 1


## short version

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Unified smooth_s3 function (with derivative flag)
def smooth_s3_np(x: np.ndarray, derivative: bool = False, k: float = 5.0) -> np.ndarray:
    a = 1 / (1 + np.exp(-k * x))
    if not derivative:
        return a * (x / (1 + np.abs(x))) + (1 - a) * (1 / (1 + np.exp(-x)))
    else:
        da_dx = k * a * (1 - a)
        softsign = x / (1 + np.abs(x))
        d_softsign = 1 / (1 + np.abs(x))**2
        sigmoid = 1 / (1 + np.exp(-x))
        d_sigmoid = sigmoid * (1 - sigmoid)
        return da_dx * (softsign - sigmoid) + a * d_softsign + (1 - a) * d_sigmoid

# Define Mish and Swish and their derivatives
def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

def d_mish(x):
    sp = np.exp(x)
    omega = 4 * (x + 1) + 4 * sp + sp**2 + 2 * x * sp + 2 * x * sp**2
    delta = (2 + 2 * sp + sp**2)**2
    return np.exp(x) * omega / delta

def swish(x, beta=1):
    return x * sigmoid(beta * x)

def d_swish(x, beta=1):
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2. Mathematical description will be given below in text

# 3. Animate smooth_s3 for different k
x = np.linspace(-10, 10, 500)

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-10, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_title('smooth_s3(x) for varying k')
ax.set_xlabel('x')
ax.set_ylabel('smooth_s3(x)')

def init():
    line.set_data([], [])
    return line,

def animate(k):
    y = smooth_s3_np(x, derivative=False, k=k)
    line.set_data(x, y)
    ax.set_title(f'smooth_s3(x) with k={k:.2f}')
    return line,

ani = FuncAnimation(fig, animate, frames=np.linspace(0.5, 10, 60), init_func=init,
                    blit=True, interval=100)
plt.close(fig)

# 4. Plot smooth_s3', Mish' and Swish'
dy_smooth_s3 = smooth_s3_np(x, derivative=True, k=5.0)
dy_mish = d_mish(x)
dy_swish = d_swish(x)

plt.figure(figsize=(10, 6))
plt.plot(x, dy_smooth_s3, label="s4'", color='blue')
plt.plot(x, dy_mish, label="Mish'", color='green')
plt.plot(x, dy_swish, label="Swish'", color='red')
plt.title("Derivatives of s4, Mish and Swish")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

## full version

"""
Advanced Activation Function Analysis and Visualization Suite.

This module provides comprehensive tools for analyzing, comparing, and
visualizing activation functions and their derivatives. It includes
implementations of Smooth S3, Mish, Swish, and other modern activation
functions with proper mathematical foundations and visualization capabilities.

Features:
- Multiple activation function implementations
- Derivative calculations with numerical stability
- Interactive animations for parameter exploration
- Comparative analysis and plotting
- Extensible architecture for custom functions

Author: Sergii Kavun
Version: 1.0.0
Date: 2025
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes


@dataclass
class ActivationConfig:
    """
    Configuration parameters for activation function analysis.
    
    Attributes:
        x_range: Tuple defining the input range (start, end)
        num_points: Number of points for function evaluation
        figure_size: Tuple defining matplotlib figure size
        animation_frames: Number of frames for parameter animations
        animation_interval: Milliseconds between animation frames
        grid_enabled: Whether to show grid in plots
        legend_enabled: Whether to show legend in plots
    """
    x_range: Tuple[float, float] = (-10.0, 10.0)
    num_points: int = 500
    figure_size: Tuple[int, int] = (10, 6)
    animation_frames: int = 60
    animation_interval: int = 100
    grid_enabled: bool = True
    legend_enabled: bool = True


@dataclass
class FunctionMetadata:
    """
    Metadata container for activation functions.
    
    Attributes:
        name: Human-readable function name
        description: Mathematical description
        parameters: Dictionary of function parameters
        color: Default color for plotting
        line_style: Default line style for plotting
    """
    name: str
    description: str
    parameters: Dict[str, float] = field(default_factory=dict)
    color: str = 'blue'
    line_style: str = '-'


class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.
    
    This class defines the interface that all activation function
    implementations must follow, ensuring consistency and extensibility.
    """
    
    def __init__(self, metadata: FunctionMetadata):
        """
        Initialize activation function with metadata.
        
        Args:
            metadata: Function metadata container
        """
        self.metadata = metadata
        self._validate_parameters()
    
    @abstractmethod
    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute forward pass of activation function.
        
        Args:
            x: Input array
            **kwargs: Additional parameters
            
        Returns:
            Output array after activation
        """
        pass
    
    @abstractmethod
    def derivative(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute derivative of activation function.
        
        Args:
            x: Input array
            **kwargs: Additional parameters
            
        Returns:
            Derivative array
        """
        pass
    
    def _validate_parameters(self) -> None:
        """Validate function parameters - override in subclasses."""
        pass
    
    def __call__(
        self, 
        x: np.ndarray, 
        derivative: bool = False, 
        **kwargs
    ) -> np.ndarray:
        """
        Unified interface for function evaluation.
        
        Args:
            x: Input array
            derivative: Whether to compute derivative
            **kwargs: Additional parameters
            
        Returns:
            Function output or derivative
        """
        self._validate_input(x)
        
        if derivative:
            return self.derivative(x, **kwargs)
        else:
            return self.forward(x, **kwargs)
    
    def _validate_input(self, x: np.ndarray) -> None:
        """
        Validate input array.
        
        Args:
            x: Input array to validate
            
        Raises:
            TypeError: If input is not numpy array
            ValueError: If input contains invalid values
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if not np.isfinite(x).all():
            warnings.warn("Input contains non-finite values", RuntimeWarning)


class SmoothS3Activation(ActivationFunction):
    """
    Smooth S3 activation function implementation.
    
    The Smooth S3 function is a hybrid activation that smoothly transitions
    between softsign and sigmoid functions based on input magnitude.
    
    Mathematical form:
    f(x) = α(x) * softsign(x) + (1 - α(x)) * sigmoid(x)
    where α(x) = sigmoid(k * x) is the blending factor.
    """
    
    def __init__(self, steepness: float = 5.0):
        """
        Initialize Smooth S3 activation function.
        
        Args:
            steepness: Controls transition steepness between functions
        """
        metadata = FunctionMetadata(
            name="Smooth S3",
            description="Hybrid softsign-sigmoid activation with smooth blending",
            parameters={"steepness": steepness},
            color="blue",
            line_style="-"
        )
        super().__init__(metadata)
        self.steepness = steepness
    
    def _validate_parameters(self) -> None:
        """Validate steepness parameter."""
        if self.steepness <= 0:
            raise ValueError("Steepness parameter must be positive")
    
    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute Smooth S3 forward pass.
        
        Args:
            x: Input array
            **kwargs: Additional parameters (steepness override)
            
        Returns:
            Activated output array
        """
        k = kwargs.get('steepness', self.steepness)
        
        # Compute blending factor using sigmoid
        with np.errstate(over='warn', invalid='warn'):
            blending_factor = 1.0 / (1.0 + np.exp(-k * x))
        
        # Compute component functions
        softsign_component = x / (1.0 + np.abs(x))
        sigmoid_component = 1.0 / (1.0 + np.exp(-x))
        
        # Blend components
        return (blending_factor * softsign_component + 
                (1.0 - blending_factor) * sigmoid_component)
    
    def derivative(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute Smooth S3 derivative.
        
        Args:
            x: Input array
            **kwargs: Additional parameters (steepness override)
            
        Returns:
            Derivative array
        """
        k = kwargs.get('steepness', self.steepness)
        
        # Compute blending factor and its derivative
        with np.errstate(over='warn', invalid='warn'):
            blending_factor = 1.0 / (1.0 + np.exp(-k * x))
        blending_derivative = k * blending_factor * (1.0 - blending_factor)
        
        # Compute component functions and derivatives
        softsign = x / (1.0 + np.abs(x))
        softsign_derivative = 1.0 / np.power(1.0 + np.abs(x), 2)
        
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        sigmoid_derivative = sigmoid * (1.0 - sigmoid)
        
        # Apply chain rule
        return (blending_derivative * (softsign - sigmoid) +
                blending_factor * softsign_derivative +
                (1.0 - blending_factor) * sigmoid_derivative)


class MishActivation(ActivationFunction):
    """
    Mish activation function implementation.
    
    Mish is a smooth, non-monotonic activation function defined as:
    f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    """
    
    def __init__(self):
        """Initialize Mish activation function."""
        metadata = FunctionMetadata(
            name="Mish",
            description="Self-regularized non-monotonic activation",
            parameters={},
            color="green",
            line_style="-"
        )
        super().__init__(metadata)
    
    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute Mish forward pass.
        
        Args:
            x: Input array
            **kwargs: Additional parameters (unused)
            
        Returns:
            Activated output array
        """
        # Use log1p for numerical stability: ln(1 + e^x)
        with np.errstate(over='warn', invalid='warn'):
            softplus = np.log1p(np.exp(x))
        return x * np.tanh(softplus)
    
    def derivative(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute Mish derivative using stable formulation.
        
        Args:
            x: Input array
            **kwargs: Additional parameters (unused)
            
        Returns:
            Derivative array
        """
        with np.errstate(over='warn', invalid='warn'):
            exp_x = np.exp(x)
            
            # Numerator: ω = 4(x+1) + 4e^x + e^(2x) + 2xe^x + 2xe^(2x)
            omega = (4.0 * (x + 1.0) + 4.0 * exp_x + 
                    np.power(exp_x, 2) + 2.0 * x * exp_x + 
                    2.0 * x * np.power(exp_x, 2))
            
            # Denominator: δ = (2 + 2e^x + e^(2x))^2
            delta = np.power(2.0 + 2.0 * exp_x + np.power(exp_x, 2), 2)
            
            return exp_x * omega / delta


class SwishActivation(ActivationFunction):
    """
    Swish activation function implementation.
    
    Swish (also known as SiLU) is defined as:
    f(x) = x * sigmoid(βx) where β is a learnable parameter.
    """
    
    def __init__(self, beta: float = 1.0):
        """
        Initialize Swish activation function.
        
        Args:
            beta: Scaling parameter for sigmoid component
        """
        metadata = FunctionMetadata(
            name="Swish",
            description="Self-gated activation function",
            parameters={"beta": beta},
            color="red",
            line_style="-"
        )
        super().__init__(metadata)
        self.beta = beta
    
    def _validate_parameters(self) -> None:
        """Validate beta parameter."""
        if not np.isfinite(self.beta):
            raise ValueError("Beta parameter must be finite")
    
    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute Swish forward pass.
        
        Args:
            x: Input array
            **kwargs: Additional parameters (beta override)
            
        Returns:
            Activated output array
        """
        beta = kwargs.get('beta', self.beta)
        
        with np.errstate(over='warn', invalid='warn'):
            sigmoid_component = 1.0 / (1.0 + np.exp(-beta * x))
        
        return x * sigmoid_component
    
    def derivative(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute Swish derivative.
        
        Args:
            x: Input array
            **kwargs: Additional parameters (beta override)
            
        Returns:
            Derivative array
        """
        beta = kwargs.get('beta', self.beta)
        
        with np.errstate(over='warn', invalid='warn'):
            sigmoid_val = 1.0 / (1.0 + np.exp(-beta * x))
        
        return sigmoid_val + beta * x * sigmoid_val * (1.0 - sigmoid_val)


class ActivationAnalyzer:
    """
    Comprehensive analysis suite for activation functions.
    
    This class provides methods for comparing activation functions,
    creating visualizations, and generating parameter sweep animations.
    """
    
    def __init__(self, config: ActivationConfig = None):
        """
        Initialize activation analyzer.
        
        Args:
            config: Configuration parameters for analysis
        """
        self.config = config or ActivationConfig()
        self.functions: Dict[str, ActivationFunction] = {}
        self.x_values = np.linspace(
            self.config.x_range[0],
            self.config.x_range[1],
            self.config.num_points
        )
    
    def add_function(
        self, 
        name: str, 
        function: ActivationFunction
    ) -> None:
        """
        Add activation function to analysis suite.
        
        Args:
            name: Unique identifier for the function
            function: ActivationFunction instance
        """
        if not isinstance(function, ActivationFunction):
            raise TypeError("Function must be ActivationFunction instance")
        
        self.functions[name] = function
    
    def compare_functions(
        self,
        derivative: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> Figure:
        """
        Create comparative plot of all registered functions.
        
        Args:
            derivative: Whether to plot derivatives
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.functions:
            raise ValueError("No functions registered for comparison")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Plot each function
        for name, func in self.functions.items():
            try:
                y_values = func(self.x_values, derivative=derivative)
                ax.plot(
                    self.x_values,
                    y_values,
                    label=f"{func.metadata.name}{'′' if derivative else ''}",
                    color=func.metadata.color,
                    linestyle=func.metadata.line_style,
                    linewidth=2
                )
            except Exception as error:
                warnings.warn(f"Failed to plot {name}: {error}", RuntimeWarning)
        
        # Configure plot appearance
        title_suffix = "Derivatives" if derivative else "Functions"
        ax.set_title(f"Activation {title_suffix} Comparison")
        ax.set_xlabel("Input (x)")
        ax.set_ylabel(f"f{'′' if derivative else ''}(x)")
        
        if self.config.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        if self.config.legend_enabled:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_parameter_animation(
        self,
        function_name: str,
        parameter_name: str,
        parameter_range: Tuple[float, float],
        derivative: bool = False,
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Create parameter sweep animation for a specific function.
        
        Args:
            function_name: Name of registered function
            parameter_name: Name of parameter to animate
            parameter_range: Tuple of (min_value, max_value)
            derivative: Whether to animate derivative
            save_path: Optional path to save animation
            
        Returns:
            FuncAnimation object
        """
        if function_name not in self.functions:
            raise KeyError(f"Function '{function_name}' not registered")
        
        function = self.functions[function_name]
        
        # Create figure and setup
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        line, = ax.plot([], [], linewidth=2, color=function.metadata.color)
        
        # Set axis limits
        ax.set_xlim(self.config.x_range)
        
        # Estimate y-limits by sampling parameter range
        sample_params = np.linspace(
            parameter_range[0], 
            parameter_range[1], 
            10
        )
        y_samples = []
        for param_val in sample_params:
            kwargs = {parameter_name: param_val}
            try:
                y_vals = function(self.x_values, derivative=derivative, **kwargs)
                y_samples.extend(y_vals)
            except Exception:
                continue
        
        if y_samples:
            y_margin = 0.1 * (max(y_samples) - min(y_samples))
            ax.set_ylim(min(y_samples) - y_margin, max(y_samples) + y_margin)
        else:
            ax.set_ylim(-2, 2)  # Default range
        
        # Configure plot labels
        suffix = "′" if derivative else ""
        ax.set_xlabel("Input (x)")
        ax.set_ylabel(f"{function.metadata.name}{suffix}(x)")
        
        if self.config.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        # Animation functions
        def init():
            line.set_data([], [])
            return line,
        
        def animate(param_value):
            kwargs = {parameter_name: param_value}
            try:
                y_values = function(
                    self.x_values, 
                    derivative=derivative, 
                    **kwargs
                )
                line.set_data(self.x_values, y_values)
                ax.set_title(
                    f"{function.metadata.name}{suffix} "
                    f"({parameter_name}={param_value:.2f})"
                )
            except Exception as error:
                warnings.warn(f"Animation frame failed: {error}", RuntimeWarning)
            
            return line,
        
        # Create parameter sequence
        param_sequence = np.linspace(
            parameter_range[0],
            parameter_range[1],
            self.config.animation_frames
        )
        
        # Create animation
        animation = FuncAnimation(
            fig,
            animate,
            frames=param_sequence,
            init_func=init,
            blit=True,
            interval=self.config.animation_interval,
            repeat=True
        )
        
        if save_path:
            animation.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to: {save_path}")
        
        return animation
    
    def generate_analysis_report(self) -> str:
        """
        Generate comprehensive text report of registered functions.
        
        Returns:
            Formatted analysis report string
        """
        if not self.functions:
            return "No functions registered for analysis."
        
        report = ["Activation Function Analysis Report"]
        report.append("=" * 50)
        report.append(f"Analysis range: {self.config.x_range}")
        report.append(f"Number of evaluation points: {self.config.num_points}")
        report.append("")
        
        for name, func in self.functions.items():
            report.append(f"Function: {func.metadata.name}")
            report.append(f"Description: {func.metadata.description}")
            
            if func.metadata.parameters:
                report.append("Parameters:")
                for param, value in func.metadata.parameters.items():
                    report.append(f"  - {param}: {value}")
            
            # Compute basic statistics
            try:
                y_vals = func(self.x_values, derivative=False)
                y_prime = func(self.x_values, derivative=True)
                
                report.append(f"Value range: [{np.min(y_vals):.4f}, {np.max(y_vals):.4f}]")
                report.append(f"Derivative range: [{np.min(y_prime):.4f}, {np.max(y_prime):.4f}]")
                
                # Check for zero crossings
                zero_crossings = np.sum(np.diff(np.signbit(y_vals)))
                report.append(f"Approximate zero crossings: {zero_crossings}")
                
            except Exception as error:
                report.append(f"Analysis failed: {error}")
            
            report.append("-" * 30)
        
        return "\n".join(report)


def create_standard_analysis_suite() -> ActivationAnalyzer:
    """
    Create analyzer with standard activation functions.
    
    Returns:
        ActivationAnalyzer with pre-registered functions
    """
    config = ActivationConfig(
        x_range=(-10.0, 10.0),
        num_points=500,
        figure_size=(12, 8)
    )
    
    analyzer = ActivationAnalyzer(config)
    
    # Add standard functions
    analyzer.add_function("smooth_s3", SmoothS3Activation(steepness=5.0))
    analyzer.add_function("mish", MishActivation())
    analyzer.add_function("swish", SwishActivation(beta=1.0))
    
    return analyzer


def demonstrate_usage():
    """
    Demonstrate comprehensive usage of the activation analysis suite.
    """
    print("Activation Function Analysis Suite Demonstration")
    print("=" * 60)
    
    # Create standard analyzer
    analyzer = create_standard_analysis_suite()
    
    # Generate analysis report
    print(analyzer.generate_analysis_report())
    
    # Create comparative plots
    print("\nGenerating comparative plots...")
    analyzer.compare_functions(
        derivative=False,
        save_path="activation_functions_comparison.png"
    )
    
    analyzer.compare_functions(
        derivative=True,
        save_path="activation_derivatives_comparison.png"
    )
    
    # Create parameter animation
    print("\nCreating parameter sweep animation...")
    animation = analyzer.create_parameter_animation(
        function_name="smooth_s3",
        parameter_name="steepness",
        parameter_range=(0.5, 10.0),
        derivative=False,
        save_path="smooth_s3_parameter_sweep.gif"
    )
    
    return analyzer, animation


if __name__ == "__main__":
    # Run demonstration
    analyzer, animation = demonstrate_usage()
    
    # Keep animation reference to prevent garbage collection
    plt.show()

#############################################
# Professional approach replacing your original code
analyzer = create_standard_analysis_suite()

# Generate comparison plots (replaces your manual plotting)
analyzer.compare_functions(derivative=True, save_path="derivatives_comparison.png")

# Create parameter animation (replaces your FuncAnimation code)
animation = analyzer.create_parameter_animation(
    function_name="smooth_s3",
    parameter_name="steepness", 
    parameter_range=(0.5, 10.0)
)

# Generate comprehensive analysis report
print(analyzer.generate_analysis_report())