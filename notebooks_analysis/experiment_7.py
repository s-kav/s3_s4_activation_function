# experiment 7

# Author: Sergii Kavun
# Version: 1.0.0
# Date: 2025

###################################################################################
#        1st step
!pip install -q condacolab
import condacolab
condacolab.install()
###################################################################################
# IMPORTANT: Restarting kernel
###################################################################################
#        2nd step
!conda create -n my_env -c conda-forge python=3.11 numba cudatoolkit=11.8 -y
###################################################################################
#        3rd step
import os
import sys
import numba
import numpy as np
import math
import timeit

print("--- Environment Verification ---")
print(f"Python version: {sys.version}")
print(f"Numba version: {numba.__version__}")
# This command now runs INSIDE the clean environment and should find the correct compiler
os.system('which ptxas')
os.system('ptxas --version')
print("------------------------------\n")

from numba import jit, cuda
GPU_AVAILABLE = cuda.is_available()
###################################################################################
#        4th step

%%writefile main.py
import numpy as np
import timeit
import sys

# --- Import from the local s4.py file ---
try:
    from s4 import S4Activation, s4, s4_numpy
except ImportError:
    print("FATAL ERROR: Could not import from s4.py.")
    print("Please make sure 's4.py' is in the same directory as 'main.py'.")
    sys.exit(1)

def run_demo_and_benchmark():
    """Demonstrates the usage of the S4Activation module and runs a benchmark."""
    print("--- S4 Activation Function Demo & Benchmark ---")

    # --- 1. Demonstration of the S4Activation Class ---
    print("\n--- 1. Class-based API Demo ---")
    
    s4_auto = S4Activation(device='auto')
    
    print(f"Is GPU available according to the activator? {s4_auto.is_gpu_available}")

    small_array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    print(f"\nInput: {small_array}")
    print(f"s4(x): {s4_auto(small_array)}")
    print(f"s4'(x): {s4_auto(small_array, derivative=True)}")

    large_array = np.ones(s4_auto.gpu_threshold + 1)
    print("\nBackend selection logic demo:")
    print(f"Info for small array (size={small_array.size}): {s4_auto.get_backend_info(small_array)}")
    print(f"Info for large array (size={large_array.size}): {s4_auto.get_backend_info(large_array)}")

    # --- 2. Demonstration of the s4() convenience function ---
    print("\n--- 2. Functional API Demo ---")
    y_cpu = s4(small_array, device='cpu')
    print(f"s4(x) with device='cpu' forced: {y_cpu}")

    # --- 3. Performance Benchmark ---
    print("\n--- 3. Performance Benchmark (on a very large array) ---")
    
    benchmark_data = np.linspace(-10, 10, 10_000_000, dtype=np.float64)
    
    backend_info = s4_auto.get_backend_info(benchmark_data)
    print(f"Benchmarking with backend: '{backend_info['backend']}'")
    
    print("Performing a warm-up run (this includes compilation time)...")
    _ = s4_auto(benchmark_data)
    print("Warm-up complete.")
    
    number_of_runs = 100
    
    # --- CORRECTED PART ---
    # Create a dictionary of the local variables that timeit needs to see.
    timeit_namespace = {
        "s4_auto": s4_auto,
        "benchmark_data": benchmark_data,
        "s4_numpy": s4_numpy  # Also include the numpy function for its benchmark
    }

    # Time the accelerated version using the custom namespace
    accelerated_time = timeit.timeit(
        's4_auto(benchmark_data)',
        globals=timeit_namespace, # Pass the dictionary with our local variables
        number=number_of_runs
    )

    # Time the pure NumPy version using the same namespace
    numpy_time = timeit.timeit(
        's4_numpy(benchmark_data)',
        globals=timeit_namespace, # Pass the same dictionary
        number=number_of_runs
    )

    print(f"\nBaseline (NumPy): {numpy_time:.4f} seconds for {number_of_runs} runs.")
    print(f"Accelerated Path: {accelerated_time:.4f} seconds for {number_of_runs} runs.")
    
    if accelerated_time > 0 and 'numba' in sys.modules:
        speedup = numpy_time / accelerated_time
        print(f"\nSpeedup: {speedup:.2f}x")
    
    print("\n--- Benchmark Complete ---")

if __name__ == '__main__':
    run_demo_and_benchmark()
###################################################################################
#        5th step
# !conda run -n my_env python main.py


