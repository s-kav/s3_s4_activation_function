# experiment 3

# Author: Sergii Kavun
# Version: 1.0.0
# Date: 2025

import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
import pandas as pd
from tqdm import tqdm  # For progress bar

# Range of iteration counts
iterations = range(1, 100_000, 1_000)

# Lists to store execution times (in seconds)
times_old = []
times_new = []
speedups = []


def generate_input():
    """
    Generate input data to avoid overhead in timeit.
    
    Returns:
        tuple: (x array, derivative boolean flag)
    """
    x = np.logspace(0, 100, 1000)
    derivative_flag = np.random.choice([False, True])
    return x, derivative_flag


def validate_functions():
    """
    Validate that both functions exist and are callable.
    
    Returns:
        tuple: (s4_old_available, s4_available)
    """
    try:
        s4_old(generate_input()[0], generate_input()[1])
        s4_old_available = True
    except NameError:
        print("Warning: Function 's4_old' is not defined. Skipping its evaluation.")
        s4_old_available = False
    
    try:
        s4(generate_input()[0], generate_input()[1])
        s4_available = True
    except NameError:
        print("Warning: Function 's4' is not defined. Skipping its evaluation.")
        s4_available = False
    
    return s4_old_available, s4_available


def benchmark_function(func, iterations_list, function_name):
    """
    Benchmark a function across different iteration counts.
    
    Args:
        func: Function to benchmark
        iterations_list: List of iteration counts
        function_name: Name of function for logging
    
    Returns:
        list: Execution times for each iteration count
    """
    execution_times = []
    print(f"Benchmarking {function_name}...")
    
    for iteration_count in tqdm(iterations_list):
        try:
            # Optimization: measure for adjusted_n repetitions and scale
            adjusted_iterations = max(iteration_count // 10, 1)  # Minimum 1
            measured_time = timeit(
                stmt=lambda: func(*generate_input()),
                number=adjusted_iterations
            ) * (iteration_count / adjusted_iterations)  # Scale to full count
            
            execution_times.append(measured_time)
            print(f"n={iteration_count}: {measured_time:.4f} sec")
            
        except Exception as error:
            print(f"Error at n={iteration_count} for {function_name}: {error}")
            execution_times.append(np.nan)
    
    return execution_times


def calculate_speedups(old_times, new_times):
    """
    Calculate speedup ratios between old and new implementations.
    
    Args:
        old_times: List of execution times for old function
        new_times: List of execution times for new function
    
    Returns:
        list: Speedup ratios (old_time / new_time)
    """
    speedup_ratios = []
    
    for old_time, new_time in zip(old_times, new_times):
        if (new_time > 0 and 
            not np.isnan(old_time) and 
            not np.isnan(new_time)):
            speedup_ratios.append(old_time / new_time)
        else:
            speedup_ratios.append(np.nan)
    
    return speedup_ratios


def save_results(iterations_list, old_times, new_times, speedup_ratios):
    """
    Save benchmark results to CSV file.
    
    Args:
        iterations_list: List of iteration counts
        old_times: Execution times for old function
        new_times: Execution times for new function
        speedup_ratios: Calculated speedup ratios
    """
    results_dataframe = pd.DataFrame({
        'iterations': iterations_list,
        'time_old': old_times,
        'time_new': new_times,
        'speedup': speedup_ratios
    })
    
    results_dataframe.to_csv('performance_comparison.csv', index=False)
    print("Results saved to 'performance_comparison.csv'")


def plot_execution_times(iterations_list, old_times, new_times):
    """
    Create and save execution time comparison plot.
    
    Args:
        iterations_list: List of iteration counts
        old_times: Execution times for old function
        new_times: Execution times for new function
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_list, old_times, 
             label='s4_old', marker='o', color='red')
    plt.plot(iterations_list, new_times, 
             label='s4', marker='o', color='blue')
    
    plt.xlabel('Number of iterations (n)')
    plt.ylabel('Execution time (seconds)')
    plt.yscale('log')  # Logarithmic Y-axis scale
    plt.title('Comparison of execution time of s4_old and s4')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_comparison.png')
    plt.show()


def plot_speedup(iterations_list, speedup_ratios):
    """
    Create and save speedup comparison plot.
    
    Args:
        iterations_list: List of iteration counts
        speedup_ratios: Calculated speedup ratios
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_list, speedup_ratios, 
             label='Speedup (s4_old / s4)', marker='o', color='green')
    
    plt.xlabel('Number of iterations (n)')
    plt.ylabel('Speedup (times)')
    plt.yscale('log')  # Logarithmic Y-axis for large differences
    plt.title('Speedup of s4 compared to s4_old')
    plt.legend()
    plt.grid(True)
    plt.savefig('speedup.png')
    plt.show()


def main():
    """
    Main function to execute the complete benchmark analysis.
    """
    # Validate function availability
    s4_old_available, s4_available = validate_functions()
    
    # Benchmark s4_old function
    if s4_old_available:
        times_old = benchmark_function(s4_old, iterations, 's4_old')
    else:
        times_old = [np.nan] * len(iterations)
    
    # Benchmark s4 function
    if s4_available:
        times_new = benchmark_function(s4, iterations, 's4')
    else:
        times_new = [np.nan] * len(iterations)
    
    # Calculate speedup ratios
    speedups = calculate_speedups(times_old, times_new)
    
    # Display average speedup statistics
    if speedups and not all(np.isnan(speedups)):
        average_speedup = np.nanmean(speedups)
        print(f"\nAverage speedup (s4 faster than s4_old): {average_speedup:.2f}x")
    
    # Save results to CSV
    save_results(iterations, times_old, times_new, speedups)
    
    # Generate visualization plots
    plot_execution_times(iterations, times_old, times_new)
    plot_speedup(iterations, speedups)


# Execute benchmark if script is run directly
if __name__ == "__main__":
    main()