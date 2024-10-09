import os
import time
import shutil
from functools import wraps

TRAIN_DIR = './data/archive/train_zip/train'
TEST_DIR = './data/archive/test_zip/test'

OUTPUT_DIR = './output'

AUGMENTED_DIR = f'{OUTPUT_DIR}/augmented'

"""
Benchmark CSV Column Keys
"""
BENCHMARK_DIR = f'{OUTPUT_DIR}/benchmark'
BENCHMARK_RESULTS_CSV = f"{BENCHMARK_DIR}/results.csv"

IMAGE_DIMENSION = "Image Dimension"
NUMBER_OF_IMAGES = "Number of Images"
BATCH_SIZE = "Batch Size"
AUGMENTATION_METHOD = "Augmentation Method"
TIME_SEC = "Time (sec)"


def time_exec(func):
    """
    Decorator that wraps a function to measure and return the execution time.
    
    :param func: The function to be wrapped and timed.
    :param verbose: If True (default), print the execution time. If False, suppress output.
    :return: Tuple of (result, time_taken)
    """
    @wraps(func)
    def wrapper(*args, verbose=True, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        total_time = end_time - start_time

        if verbose:
            print(f"time_exec {func.__name__}: {total_time:.3f} seconds")
        
        return result, total_time
    
    return wrapper


def ensure_empty_dir(dir_path):
    """Ensures that the specified directory exists and is empty.

    Args:
        dir_path (str): The path to the directory.

    Raises:
        OSError: If an error occurs during directory creation or removal.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def load_kernel_code(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CUDA kernel file not found: {file_path}")
    
    with open(file_path, 'r') as kernel_file:
        kernel_code = kernel_file.read()
    
    return kernel_code
