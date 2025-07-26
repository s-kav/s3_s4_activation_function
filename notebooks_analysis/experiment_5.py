# experiment 5

# Author: Sergii Kavun
# Version: 1.0.0
# Date: 2025


## short version

import numpy as np
import matplotlib.pyplot as plt

# Define activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def softsign(x):
    return x / (1 + np.abs(x))

def d_softsign(x):
    return 1 / (1 + np.abs(x))**2

def alpha(x, k=5):
    """Smooth switch function using a sigmoid-like gate"""
    return 1 / (1 + np.exp(-k * x))

def s3(x):
    """Original hard switch: sigmoid for x<0, softsign for x>=0"""
    return np.where(x <= 0, sigmoid(x), softsign(x))

def d_s3(x):
    return np.where(x <= 0, d_sigmoid(x), d_softsign(x))

def smooth_s3(x, k=5):
    a = alpha(x, k)
    return a * softsign(x) + (1 - a) * sigmoid(x)

def d_smooth_s3(x, k=5):
    a = alpha(x, k)
    da_dx = k * a * (1 - a)
    return da_dx * (softsign(x) - sigmoid(x)) + a * d_softsign(x) + (1 - a) * d_sigmoid(x)

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

# Generate values
x = np.linspace(-10, 10, 1000)

# Compute values
y_s3 = s3(x)
y_smooth_s3 = smooth_s3(x)
y_mish = mish(x)
y_swish = swish(x)

dy_s3 = d_s3(x)
dy_smooth_s3 = d_smooth_s3(x)
dy_mish = d_mish(x)
dy_swish = d_swish(x)

# Plot activations
plt.figure(figsize=(12, 6))
plt.plot(x, y_s3, label='s3 (hard switch)', linestyle='--')
plt.plot(x, y_smooth_s3, label='smooth_s3 (soft switch)')
plt.plot(x, y_mish, label='Mish')
plt.plot(x, y_swish, label='Swish')
plt.title('Activation Functions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot derivatives
plt.figure(figsize=(12, 6))
plt.plot(x, dy_s3, label="s3'", linestyle='--')
plt.plot(x, dy_smooth_s3, label="smooth_s3'")
plt.plot(x, dy_mish, label="Mish'")
plt.plot(x, dy_swish, label="Swish'")
plt.title('Derivatives of Activation Functions')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


## full version

#!/usr/bin/env python3
"""
Extended Activation Functions Analysis and Visualization Module.

This module implements various activation functions including sigmoid, softsign,
s3 (hard switch), smooth_s3 (soft switch), Mish, and Swish with their derivatives,
along with comprehensive visualization capabilities for neural network research.

Mathematical Background:
- s3: Hard switch between sigmoid (x≤0) and softsign (x>0)
- smooth_s3: Soft switch using sigmoid-gated combination
- All functions preserve mathematical properties for gradient flow

Author: Neural Network Research Team
Date: 2025
License: MIT
Version: 2.0.0
"""

from typing import Union, Tuple, Optional, List
import warnings
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class BaseActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Forward pass of the activation function."""
        pass
    
    @abstractmethod
    def derivative(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Derivative of the activation function."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the activation function."""
        pass


class ActivationFunctionLibrary:
    """
    Comprehensive library of activation functions and their derivatives.
    
    This class implements various activation functions commonly used in
    neural networks, with emphasis on mathematical accuracy and numerical
    stability.
    """
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        
        σ(x) = 1 / (1 + exp(-x))
        
        Args:
            x: Input array of any shape
            
        Returns:
            Sigmoid function values with same shape as input
            
        Note:
            Numerically stable implementation with overflow protection
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Clip to prevent overflow
            x_clipped = np.clip(x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x_clipped))
    
    @staticmethod
    def d_sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid activation function.
        
        σ'(x) = σ(x) * (1 - σ(x))
        
        Args:
            x: Input array of any shape
            
        Returns:
            Sigmoid derivative values with same shape as input
        """
        s = ActivationFunctionLibrary.sigmoid(x)
        return s * (1.0 - s)
    
    @staticmethod
    def softsign(x: np.ndarray) -> np.ndarray:
        """
        Softsign activation function.
        
        softsign(x) = x / (1 + |x|)
        
        Args:
            x: Input array of any shape
            
        Returns:
            Softsign function values with same shape as input
            
        Note:
            More numerically stable than tanh for large values
        """
        return x / (1.0 + np.abs(x))
    
    @staticmethod
    def d_softsign(x: np.ndarray) -> np.ndarray:
        """
        Derivative of softsign activation function.
        
        softsign'(x) = 1 / (1 + |x|)²
        
        Args:
            x: Input array of any shape
            
        Returns:
            Softsign derivative values with same shape as input
        """
        denominator = 1.0 + np.abs(x)
        return 1.0 / (denominator ** 2)
    
    @staticmethod
    def alpha(x: np.ndarray, k: float = 5.0) -> np.ndarray:
        """
        Smooth switch function using sigmoid-like gate.
        
        α(x, k) = 1 / (1 + exp(-k * x))
        
        Args:
            x: Input array of any shape
            k: Steepness parameter controlling transition sharpness
            
        Returns:
            Alpha function values with same shape as input
            
        Raises:
            ValueError: If k is non-positive
        """
        if k <= 0:
            raise ValueError("Parameter k must be positive")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_scaled = np.clip(k * x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x_scaled))
    
    @staticmethod
    def s3(x: np.ndarray) -> np.ndarray:
        """
        S3 activation function with hard switch.
        
        s3(x) = {sigmoid(x)  if x ≤ 0
                {softsign(x) if x > 0
        
        Args:
            x: Input array of any shape
            
        Returns:
            S3 function values with same shape as input
            
        Note:
            Non-differentiable at x=0, causing gradient issues in backpropagation
        """
        return np.where(
            x <= 0, 
            ActivationFunctionLibrary.sigmoid(x), 
            ActivationFunctionLibrary.softsign(x)
        )
    
    @staticmethod
    def d_s3(x: np.ndarray) -> np.ndarray:
        """
        Derivative of S3 activation function.
        
        s3'(x) = {σ'(x)       if x ≤ 0
                 {softsign'(x) if x > 0
        
        Args:
            x: Input array of any shape
            
        Returns:
            S3 derivative values with same shape as input
            
        Warning:
            Undefined at x=0; implementation uses left derivative
        """
        return np.where(
            x <= 0, 
            ActivationFunctionLibrary.d_sigmoid(x), 
            ActivationFunctionLibrary.d_softsign(x)
        )
    
    @staticmethod
    def smooth_s3(x: np.ndarray, k: float = 5.0) -> np.ndarray:
        """
        Smooth S3 activation function with soft switch.
        
        smooth_s3(x, k) = α(x, k) * softsign(x) + (1 - α(x, k)) * σ(x)
        
        Args:
            x: Input array of any shape
            k: Transition sharpness parameter (default: 5.0)
            
        Returns:
            Smooth S3 function values with same shape as input
            
        Raises:
            ValueError: If k is non-positive
        """
        if k <= 0:
            raise ValueError("Parameter k must be positive")
            
        a = ActivationFunctionLibrary.alpha(x, k)
        sigmoid_vals = ActivationFunctionLibrary.sigmoid(x)
        softsign_vals = ActivationFunctionLibrary.softsign(x)
        
        return a * softsign_vals + (1.0 - a) * sigmoid_vals
    
    @staticmethod
    def d_smooth_s3(x: np.ndarray, k: float = 5.0) -> np.ndarray:
        """
        Derivative of smooth S3 activation function.
        
        Args:
            x: Input array of any shape
            k: Transition sharpness parameter (default: 5.0)
            
        Returns:
            Smooth S3 derivative values with same shape as input
            
        Raises:
            ValueError: If k is non-positive
        """
        if k <= 0:
            raise ValueError("Parameter k must be positive")
            
        a = ActivationFunctionLibrary.alpha(x, k)
        da_dx = k * a * (1.0 - a)
        
        sigmoid_vals = ActivationFunctionLibrary.sigmoid(x)
        softsign_vals = ActivationFunctionLibrary.softsign(x)
        d_sigmoid_vals = ActivationFunctionLibrary.d_sigmoid(x)
        d_softsign_vals = ActivationFunctionLibrary.d_softsign(x)
        
        return (da_dx * (softsign_vals - sigmoid_vals) + 
               a * d_softsign_vals + 
               (1.0 - a) * d_sigmoid_vals)
    
    @staticmethod
    def mish(x: np.ndarray) -> np.ndarray:
        """
        Mish activation function.
        
        Mish(x) = x * tanh(ln(1 + exp(x))) = x * tanh(softplus(x))
        
        Args:
            x: Input array of any shape
            
        Returns:
            Mish function values with same shape as input
            
        Note:
            Self-regularizing and smooth activation with good properties
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Use log1p for numerical stability
            softplus = np.log1p(np.exp(np.clip(x, -50, 50)))
            return x * np.tanh(softplus)
    
    @staticmethod
    def d_mish(x: np.ndarray) -> np.ndarray:
        """
        Derivative of Mish activation function.
        
        Args:
            x: Input array of any shape
            
        Returns:
            Mish derivative values with same shape as input
            
        Note:
            Complex derivative involving exponential and hyperbolic functions
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            x_clipped = np.clip(x, -50, 50)
            sp = np.exp(x_clipped)
            
            omega = (4.0 * (x_clipped + 1.0) + 4.0 * sp + 
                    sp**2 + 2.0 * x_clipped * sp + 2.0 * x_clipped * sp**2)
            delta = (2.0 + 2.0 * sp + sp**2)**2
            
            return sp * omega / delta
    
    @staticmethod
    def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """
        Swish activation function (also known as SiLU).
        
        Swish(x, β) = x * σ(β * x)
        
        Args:
            x: Input array of any shape
            beta: Scaling parameter (default: 1.0)
            
        Returns:
            Swish function values with same shape as input
            
        Raises:
            ValueError: If beta is zero
            
        Note:
            β=1 gives standard Swish/SiLU; discovered by Google
        """
        if beta == 0:
            raise ValueError("Parameter beta cannot be zero")
            
        return x * ActivationFunctionLibrary.sigmoid(beta * x)
    
    @staticmethod
    def d_swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """
        Derivative of Swish activation function.
        
        Swish'(x, β) = σ(βx) + βx * σ(βx) * (1 - σ(βx))
        
        Args:
            x: Input array of any shape
            beta: Scaling parameter (default: 1.0)
            
        Returns:
            Swish derivative values with same shape as input
            
        Raises:
            ValueError: If beta is zero
        """
        if beta == 0:
            raise ValueError("Parameter beta cannot be zero")
            
        s = ActivationFunctionLibrary.sigmoid(beta * x)
        return s + beta * x * s * (1.0 - s)


class ActivationAnalyzer:
    """
    Comprehensive analysis and visualization toolkit for activation functions.
    
    This class provides methods for comparing activation functions,
    analyzing their properties, and creating publication-quality visualizations.
    """
    
    def __init__(self, 
                 x_range: Tuple[float, float] = (-10.0, 10.0),
                 num_points: int = 1000,
                 k_smooth: float = 5.0,
                 beta_swish: float = 1.0):
        """
        Initialize the activation analyzer.
        
        Args:
            x_range: Domain range for analysis (min, max)
            num_points: Number of evaluation points
            k_smooth: Parameter for smooth_s3 function
            beta_swish: Parameter for Swish function
        """
        self.x_range = x_range
        self.num_points = num_points
        self.k_smooth = k_smooth
        self.beta_swish = beta_swish
        
        self.x = np.linspace(x_range[0], x_range[1], num_points)
        self.functions = ActivationFunctionLibrary()
        
        # Pre-compute function values for efficiency
        self._compute_all_values()
    
    def _compute_all_values(self) -> None:
        """Pre-compute all function and derivative values."""
        # Function values
        self.y_s3 = self.functions.s3(self.x)
        self.y_smooth_s3 = self.functions.smooth_s3(self.x, self.k_smooth)
        self.y_mish = self.functions.mish(self.x)
        self.y_swish = self.functions.swish(self.x, self.beta_swish)
        self.y_sigmoid = self.functions.sigmoid(self.x)
        self.y_softsign = self.functions.softsign(self.x)
        
        # Derivative values
        self.dy_s3 = self.functions.d_s3(self.x)
        self.dy_smooth_s3 = self.functions.d_smooth_s3(self.x, self.k_smooth)
        self.dy_mish = self.functions.d_mish(self.x)
        self.dy_swish = self.functions.d_swish(self.x, self.beta_swish)
        self.dy_sigmoid = self.functions.d_sigmoid(self.x)
        self.dy_softsign = self.functions.d_softsign(self.x)
    
    def plot_activation_functions(self, 
                                figsize: Tuple[int, int] = (14, 8),
                                include_base: bool = False) -> Figure:
        """
        Create comprehensive comparison plot of activation functions.
        
        Args:
            figsize: Figure size tuple (width, height)
            include_base: Whether to include sigmoid and softsign base functions
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors and styles
        colors = {
            's3': '#FF6B6B',
            'smooth_s3': '#4ECDC4', 
            'mish': '#45B7D1',
            'swish': '#96CEB4',
            'sigmoid': '#FFEAA7',
            'softsign': '#DDA0DD'
        }
        
        # Plot main activation functions
        ax.plot(self.x, self.y_s3, label='S3 (hard switch)', 
               color=colors['s3'], linestyle='--', linewidth=2.5, alpha=0.8)
        ax.plot(self.x, self.y_smooth_s3, 
               label=f'Smooth S3 (k={self.k_smooth})', 
               color=colors['smooth_s3'], linewidth=2.5)
        ax.plot(self.x, self.y_mish, label='Mish', 
               color=colors['mish'], linewidth=2.5)
        ax.plot(self.x, self.y_swish, 
               label=f'Swish (β={self.beta_swish})', 
               color=colors['swish'], linewidth=2.5)
        
        # Optionally include base functions
        if include_base:
            ax.plot(self.x, self.y_sigmoid, label='Sigmoid', 
                   color=colors['sigmoid'], linestyle=':', linewidth=2, alpha=0.7)
            ax.plot(self.x, self.y_softsign, label='Softsign', 
                   color=colors['softsign'], linestyle=':', linewidth=2, alpha=0.7)
        
        # Styling
        ax.set_title('Activation Functions Comparison', fontsize=18, fontweight='bold')
        ax.set_xlabel('Input (x)', fontsize=14)
        ax.set_ylabel('Output f(x)', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(self.x_range)
        
        # Add zero lines for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        plt.tight_layout()
        return fig
    
    def plot_derivatives_comparison(self, 
                                  figsize: Tuple[int, int] = (14, 8),
                                  include_base: bool = False) -> Figure:
        """
        Create comprehensive comparison plot of activation function derivatives.
        
        Args:
            figsize: Figure size tuple (width, height)
            include_base: Whether to include sigmoid and softsign derivatives
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors matching the activation functions plot
        colors = {
            's3': '#FF6B6B',
            'smooth_s3': '#4ECDC4',
            'mish': '#45B7D1', 
            'swish': '#96CEB4',
            'sigmoid': '#FFEAA7',
            'softsign': '#DDA0DD'
        }
        
        # Plot derivatives
        ax.plot(self.x, self.dy_s3, label="S3' (hard switch)", 
               color=colors['s3'], linestyle='--', linewidth=2.5, alpha=0.8)
        ax.plot(self.x, self.dy_smooth_s3, 
               label=f"Smooth S3' (k={self.k_smooth})", 
               color=colors['smooth_s3'], linewidth=2.5)
        ax.plot(self.x, self.dy_mish, label="Mish'", 
               color=colors['mish'], linewidth=2.5)
        ax.plot(self.x, self.dy_swish, 
               label=f"Swish' (β={self.beta_swish})", 
               color=colors['swish'], linewidth=2.5)
        
        # Optionally include base function derivatives
        if include_base:
            ax.plot(self.x, self.dy_sigmoid, label="Sigmoid'", 
                   color=colors['sigmoid'], linestyle=':', linewidth=2, alpha=0.7)
            ax.plot(self.x, self.dy_softsign, label="Softsign'", 
                   color=colors['softsign'], linestyle=':', linewidth=2, alpha=0.7)
        
        # Styling
        ax.set_title('Activation Function Derivatives Comparison', 
                    fontsize=18, fontweight='bold')
        ax.set_xlabel('Input (x)', fontsize=14)
        ax.set_ylabel("Derivative f'(x)", fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(self.x_range)
        
        # Add zero lines for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        plt.tight_layout()
        return fig
    
    def plot_combined_analysis(self, 
                             figsize: Tuple[int, int] = (16, 10)) -> Figure:
        """
        Create combined subplot showing both functions and derivatives.
        
        Args:
            figsize: Figure size tuple (width, height)
            
        Returns:
            Matplotlib Figure object with subplots
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Colors for consistency
        colors = {
            's3': '#FF6B6B',
            'smooth_s3': '#4ECDC4',
            'mish': '#45B7D1',
            'swish': '#96CEB4'
        }
        
        # Top subplot: Activation functions
        ax1.plot(self.x, self.y_s3, label='S3 (hard switch)', 
                color=colors['s3'], linestyle='--', linewidth=2.5, alpha=0.8)
        ax1.plot(self.x, self.y_smooth_s3, 
                label=f'Smooth S3 (k={self.k_smooth})', 
                color=colors['smooth_s3'], linewidth=2.5)
        ax1.plot(self.x, self.y_mish, label='Mish', 
                color=colors['mish'], linewidth=2.5)
        ax1.plot(self.x, self.y_swish, 
                label=f'Swish (β={self.beta_swish})', 
                color=colors['swish'], linewidth=2.5)
        
        ax1.set_title('Activation Functions', fontsize=16, fontweight='bold')
        ax1.set_ylabel('f(x)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Bottom subplot: Derivatives
        ax2.plot(self.x, self.dy_s3, label="S3'", 
                color=colors['s3'], linestyle='--', linewidth=2.5, alpha=0.8)
        ax2.plot(self.x, self.dy_smooth_s3, label=f"Smooth S3'", 
                color=colors['smooth_s3'], linewidth=2.5)
        ax2.plot(self.x, self.dy_mish, label="Mish'", 
                color=colors['mish'], linewidth=2.5)
        ax2.plot(self.x, self.dy_swish, label="Swish'", 
                color=colors['swish'], linewidth=2.5)
        
        ax2.set_title('Derivatives', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Input (x)', fontsize=12)
        ax2.set_ylabel("f'(x)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        plt.tight_layout()
        return fig
    
    def analyze_function_properties(self) -> dict:
        """
        Analyze mathematical properties of all activation functions.
        
        Returns:
            Dictionary containing analysis results for each function
        """
        analysis = {}
        
        functions_data = {
            'S3': self.y_s3,
            'Smooth S3': self.y_smooth_s3,
            'Mish': self.y_mish,
            'Swish': self.y_swish
        }
        
        derivatives_data = {
            'S3': self.dy_s3,
            'Smooth S3': self.dy_smooth_s3,
            'Mish': self.dy_mish,
            'Swish': self.dy_swish
        }
        
        for name in functions_data:
            y = functions_data[name]
            dy = derivatives_data[name]
            
            analysis[name] = {
                'range': (float(np.min(y)), float(np.max(y))),
                'monotonic': bool(np.all(dy >= 0) or np.all(dy <= 0)),
                'max_derivative': float(np.max(dy)),
                'min_derivative': float(np.min(dy)),
                'zero_crossings': int(len(np.where(np.diff(np.signbit(y)))[0])),
                'inflection_points': int(len(np.where(np.diff(np.signbit(
                    np.gradient(dy))))[0]))
            }
        
        return analysis
    
    def print_analysis_summary(self) -> None:
        """Print a comprehensive analysis summary of all functions."""
        analysis = self.analyze_function_properties()
        
        print("=" * 80)
        print("ACTIVATION FUNCTIONS ANALYSIS SUMMARY")
        print("=" * 80)
        
        for name, props in analysis.items():
            print(f"\n{name.upper()}:")
            print(f"  Range: [{props['range'][0]:.3f}, {props['range'][1]:.3f}]")
            print(f"  Monotonic: {props['monotonic']}")
            print(f"  Derivative range: [{props['min_derivative']:.3f}, "
                  f"{props['max_derivative']:.3f}]")
            print(f"  Zero crossings: {props['zero_crossings']}")
            print(f"  Inflection points: {props['inflection_points']}")
        
        print("\n" + "=" * 80)
        print("MATHEMATICAL NOTES:")
        print("- S3 has discontinuous derivative at x=0")
        print("- Smooth S3 provides differentiable alternative to S3")
        print("- Mish exhibits self-regularizing properties")
        print("- Swish/SiLU is widely used in modern architectures")
        print("=" * 80)


def main() -> None:
    """
    Main function demonstrating the extended activation functions analysis.
    
    This function creates comprehensive visualizations and analysis of
    various activation functions commonly used in neural networks.
    """
    print("Initializing Extended Activation Functions Analysis...")
    
    # Create analyzer with custom parameters
    analyzer = ActivationAnalyzer(
        x_range=(-10.0, 10.0),
        num_points=1000,
        k_smooth=5.0,
        beta_swish=1.0
    )
    
    # Generate visualizations
    print("Creating activation functions comparison plot...")
    functions_fig = analyzer.plot_activation_functions(
        figsize=(14, 8), 
        include_base=False
    )
    plt.show()
    
    print("Creating derivatives comparison plot...")
    derivatives_fig = analyzer.plot_derivatives_comparison(
        figsize=(14, 8), 
        include_base=False
    )
    plt.show()
    
    print("Creating combined analysis plot...")
    combined_fig = analyzer.plot_combined_analysis(figsize=(16, 10))
    plt.show()
    
    # Print comprehensive analysis
    print("\nGenerating mathematical analysis...")
    analyzer.print_analysis_summary()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Features implemented:")
    print("✓ PEP 8 compliant formatting")
    print("✓ Comprehensive docstrings with mathematical formulas")
    print("✓ Complete type hints and error handling")
    print("✓ Object-oriented modular design")
    print("✓ Numerical stability improvements")
    print("✓ Publication-quality visualizations")
    print("✓ Mathematical properties analysis")
    print("✓ Extensible architecture for new functions")
    print("="*60)


if __name__ == "__main__":
    main()