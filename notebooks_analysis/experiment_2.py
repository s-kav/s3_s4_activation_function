# experiment 2

## short version
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

%%time
for _ in range(10000):
  x = np.logspace(0, 100, 1000) # logspace
  der = np.random.choice([False, True])
  res = s3_act_function(x, der)
CPU times: user 2.22 s, sys: 5.75 ms, total: 2.22 s
Wall time: 2.28 s

# ======================================

%%time
for _ in range(10000):
  x = np.logspace(0, 100, 1000) # logspace
  der = np.random.choice([False, True])
  res = s4_act_function(x, der)
CPU times: user 1.35 s, sys: 2.91 ms, total: 1.36 s
Wall time: 1.36 s

## full version

"""
Activation Function Performance Benchmarking Module.

This module provides comprehensive benchmarking tools for comparing
different activation function implementations with proper timing,
visualization, and statistical analysis.

Author: Sergii Kavun
Version: 1.0.0
Date: 2025
"""

import time
from typing import Callable, Tuple, List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class BenchmarkConfig:
    """
    Configuration class for benchmark parameters.
    
    Attributes:
        iterations: Number of benchmark iterations
        data_size: Size of input data arrays
        logspace_start: Start value for logspace generation
        logspace_end: End value for logspace generation
        random_seed: Seed for reproducible random generation
    """
    iterations: int = 10_000
    data_size: int = 1_000
    logspace_start: float = 0.0
    logspace_end: float = 100.0
    random_seed: Optional[int] = 42


@dataclass
class BenchmarkResult:
    """
    Container for benchmark execution results.
    
    Attributes:
        function_name: Name of benchmarked function
        total_time: Total execution time in seconds
        average_time: Average time per iteration
        iterations: Number of iterations performed
        cpu_time: CPU time (if available)
        wall_time: Wall clock time
    """
    function_name: str
    total_time: float
    average_time: float
    iterations: int
    cpu_time: Optional[float] = None
    wall_time: Optional[float] = None


class ActivationBenchmark:
    """
    Comprehensive benchmarking suite for activation functions.
    
    This class provides methods to benchmark activation functions,
    generate performance reports, and create visualizations.
    """
    
    def __init__(self, config: BenchmarkConfig = None):
        """
        Initialize the benchmark suite.
        
        Args:
            config: Benchmark configuration parameters
        """
        self.config = config or BenchmarkConfig()
        self.results: Dict[str, BenchmarkResult] = {}
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    @contextmanager
    def timer(self):
        """
        Context manager for high-precision timing.
        
        Yields:
            dict: Dictionary containing timing information
        """
        timing_info = {}
        
        # Record start times
        start_time = time.time()
        start_cpu = time.process_time()
        
        try:
            yield timing_info
        finally:
            # Calculate elapsed times
            timing_info['wall_time'] = time.time() - start_time
            timing_info['cpu_time'] = time.process_time() - start_cpu
            timing_info['total_time'] = timing_info['wall_time']
    
    def generate_test_data(self) -> Tuple[np.ndarray, bool]:
        """
        Generate standardized test data for benchmarking.
        
        Returns:
            tuple: (input_array, derivative_flag)
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.config.data_size <= 0:
            raise ValueError("Data size must be positive")
        
        if self.config.logspace_start >= self.config.logspace_end:
            raise ValueError("Logspace start must be less than end")
        
        # Generate logarithmically spaced input data
        input_array = np.logspace(
            self.config.logspace_start,
            self.config.logspace_end,
            self.config.data_size
        )
        
        # Random derivative flag
        derivative_flag = np.random.choice([False, True])
        
        return input_array, derivative_flag
    
    def benchmark_function(
        self,
        function: Callable,
        function_name: str,
        verbose: bool = True
    ) -> BenchmarkResult:
        """
        Benchmark a single activation function.
        
        Args:
            function: The activation function to benchmark
            function_name: Name identifier for the function
            verbose: Whether to print progress information
            
        Returns:
            BenchmarkResult: Comprehensive timing results
            
        Raises:
            TypeError: If function is not callable
            RuntimeError: If benchmarking fails
        """
        if not callable(function):
            raise TypeError(f"'{function_name}' is not callable")
        
        if verbose:
            print(f"Benchmarking {function_name}...")
        
        try:
            with self.timer() as timing:
                # Perform benchmark iterations
                for iteration in range(self.config.iterations):
                    input_data, derivative_flag = self.generate_test_data()
                    result = function(input_data, derivative_flag)
                    
                    # Optional progress reporting
                    if verbose and iteration % (self.config.iterations // 10) == 0:
                        progress = (iteration / self.config.iterations) * 100
                        print(f"  Progress: {progress:.1f}%")
            
            # Create result object
            benchmark_result = BenchmarkResult(
                function_name=function_name,
                total_time=timing['total_time'],
                average_time=timing['total_time'] / self.config.iterations,
                iterations=self.config.iterations,
                cpu_time=timing['cpu_time'],
                wall_time=timing['wall_time']
            )
            
            # Store results
            self.results[function_name] = benchmark_result
            
            if verbose:
                self._print_timing_summary(benchmark_result)
            
            return benchmark_result
            
        except Exception as error:
            raise RuntimeError(
                f"Benchmarking failed for {function_name}: {error}"
            ) from error
    
    def _print_timing_summary(self, result: BenchmarkResult) -> None:
        """
        Print formatted timing summary.
        
        Args:
            result: Benchmark result to display
        """
        print(f"\n=== {result.function_name} Timing Summary ===")
        print(f"Total iterations: {result.iterations:,}")
        print(f"Total time: {result.total_time:.4f} seconds")
        print(f"Average time per call: {result.average_time:.6f} seconds")
        
        if result.cpu_time is not None:
            print(f"CPU time: {result.cpu_time:.4f} seconds")
        if result.wall_time is not None:
            print(f"Wall time: {result.wall_time:.4f} seconds")
        print("=" * 40)
    
    def compare_functions(
        self,
        functions: Dict[str, Callable],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple activation functions.
        
        Args:
            functions: Dictionary mapping names to functions
            verbose: Whether to show detailed output
            
        Returns:
            pd.DataFrame: Comparison results table
        """
        if verbose:
            print("Starting comprehensive function comparison...\n")
        
        # Benchmark all functions
        for name, func in functions.items():
            self.benchmark_function(func, name, verbose)
        
        # Create comparison DataFrame
        comparison_data = []
        for result in self.results.values():
            comparison_data.append({
                'Function': result.function_name,
                'Total Time (s)': result.total_time,
                'Avg Time (s)': result.average_time,
                'Iterations': result.iterations,
                'CPU Time (s)': result.cpu_time,
                'Wall Time (s)': result.wall_time
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Total Time (s)')
        
        if verbose:
            self._print_comparison_summary(comparison_df)
        
        return comparison_df
    
    def _print_comparison_summary(self, comparison_df: pd.DataFrame) -> None:
        """
        Print formatted comparison summary.
        
        Args:
            comparison_df: DataFrame with comparison results
        """
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 60)
        print(comparison_df.to_string(index=False, float_format='%.6f'))
        
        # Calculate speedup ratios
        if len(comparison_df) >= 2:
            fastest_time = comparison_df.iloc[0]['Total Time (s)']
            print(f"\nSpeedup Analysis (relative to fastest):")
            print("-" * 40)
            
            for _, row in comparison_df.iterrows():
                speedup = row['Total Time (s)'] / fastest_time
                print(f"{row['Function']}: {speedup:.2f}x")
    
    def create_performance_plot(
        self,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Create performance comparison visualization.
        
        Args:
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        if not self.results:
            raise ValueError("No benchmark results available for plotting")
        
        # Prepare data for plotting
        function_names = list(self.results.keys())
        total_times = [result.total_time for result in self.results.values()]
        avg_times = [result.average_time for result in self.results.values()]
        
        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total time comparison
        bars1 = ax1.bar(function_names, total_times, color='skyblue', alpha=0.7)
        ax1.set_title('Total Execution Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Activation Functions')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars1, total_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s',
                    ha='center', va='bottom')
        
        # Average time comparison
        bars2 = ax2.bar(function_names, avg_times, color='lightcoral', alpha=0.7)
        ax2.set_title('Average Time Per Call Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xlabel('Activation Functions')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, avg_time in zip(bars2, avg_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{avg_time:.6f}s',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
    
    def export_results(self, filepath: str) -> None:
        """
        Export benchmark results to CSV file.
        
        Args:
            filepath: Path for the output CSV file
        """
        if not self.results:
            raise ValueError("No results to export")
        
        # Create DataFrame from results
        export_data = []
        for result in self.results.values():
            export_data.append({
                'function_name': result.function_name,
                'total_time_seconds': result.total_time,
                'average_time_seconds': result.average_time,
                'iterations': result.iterations,
                'cpu_time_seconds': result.cpu_time,
                'wall_time_seconds': result.wall_time
            })
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(filepath, index=False)
        print(f"Results exported to: {filepath}")


# Example usage and demonstration functions
def example_benchmark_usage():
    """
    Demonstrate proper usage of the benchmarking suite.
    
    This function shows how to use the ActivationBenchmark class
    for comparing different activation function implementations.
    """
    print("Activation Function Benchmarking Example")
    print("=" * 50)
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        iterations=10_000,
        data_size=1_000,
        logspace_start=0.0,
        logspace_end=100.0,
        random_seed=42
    )
    
    # Initialize benchmark suite
    benchmark = ActivationBenchmark(config)
    
    # Example function definitions (replace with actual implementations)
    def s3_act_function(x: np.ndarray, derivative: bool) -> np.ndarray:
        """Example S3 activation function."""
        # Placeholder implementation
        return np.tanh(x) if not derivative else 1 - np.tanh(x)**2
    
    def s4_act_function(x: np.ndarray, derivative: bool) -> np.ndarray:
        """Example S4 activation function."""
        # Placeholder implementation - optimized version
        return np.where(x > 0, x, 0.01 * x)  # Leaky ReLU example
    
    # Define functions to compare
    functions_to_compare = {
        's3_act_function': s3_act_function,
        's4_act_function': s4_act_function
    }
    
    # Perform comparison
    try:
        results_df = benchmark.compare_functions(functions_to_compare)
        
        # Create visualization
        benchmark.create_performance_plot(
            save_path='activation_performance_comparison.png'
        )
        
        # Export results
        benchmark.export_results('benchmark_results.csv')
        
        return results_df
        
    except Exception as error:
        print(f"Benchmark failed: {error}")
        return None


# Jupyter notebook compatibility functions
def create_jupyter_timing_cell() -> str:
    """
    Generate Jupyter-compatible timing code.
    
    Returns:
        str: Formatted code for Jupyter notebook cells
    """
    jupyter_code = '''
# Jupyter Notebook Cell 1: S3 Function Benchmarking
%%time
config = BenchmarkConfig(iterations=10_000, data_size=1_000)
benchmark = ActivationBenchmark(config)

for _ in range(config.iterations):
    x = np.logspace(0, 100, 1000)  # logarithmically spaced data
    derivative_flag = np.random.choice([False, True])
    result = s3_act_function(x, derivative_flag)

# Jupyter Notebook Cell 2: S4 Function Benchmarking  
%%time
for _ in range(config.iterations):
    x = np.logspace(0, 100, 1000)  # logarithmically spaced data
    derivative_flag = np.random.choice([False, True])
    result = s4_act_function(x, derivative_flag)
'''
    return jupyter_code


if __name__ == "__main__":
    # Run example benchmark
    example_results = example_benchmark_usage()
    
    # Display Jupyter code
    print("\nJupyter Notebook Compatible Code:")
    print("-" * 40)
    print(create_jupyter_timing_cell())