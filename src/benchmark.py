import os
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt

from util import *

SCRIPT_PATH = 'src/multiclass_cnn.py'
AUGMENTATION_METHODS = ['pillow', 'cuda']


def run_benchmark(benchmark_values, static, x_axis, command_template):
    """
    Run benchmarks based on varying and fixed values for image processing.

    Args:
        benchmark_values (list): List of varying values (dimensions or image counts).
        static (int): Fixed value for the other parameter (image dimension or number of images).
        x_axis (str): The x_axis for plotting results.
        filename (str): The filename for saving benchmark results.
    """
    for augment in AUGMENTATION_METHODS:
        for value in benchmark_values:

            command = command_template.format(SCRIPT_PATH=SCRIPT_PATH, augment=augment, value=value, static=static, BENCHMARK_DIR=BENCHMARK_DIR)
            
            print(f'Executing: {command}')
            subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    plot_benchmark_result(x_axis=x_axis)


def plot_benchmark_result(x_axis):

    filename = x_axis.replace(' ', '_').lower()
    results_df = pd.read_csv(BENCHMARK_RESULTS_CSV)
    print(results_df)

    os.rename(BENCHMARK_RESULTS_CSV,f'{BENCHMARK_DIR}/results_{filename}.csv')

    plt.figure(figsize=(10, 6))

    pivot_df = results_df.pivot(index=x_axis, columns=AUGMENTATION_METHOD, values=TIME_SEC)

    for method in AUGMENTATION_METHODS:
        plt.plot(pivot_df.index, pivot_df[method], marker='o', label=method)

    plt.title(f'Data Augmentation Execution Time by {x_axis}', fontsize=14, fontweight='bold')
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(TIME_SEC, fontsize=14)
    plt.xticks(pivot_df.index)
    plt.grid()
    plt.legend(title="Data Augmentation Methods", loc='upper left', fontsize=14)
    plt.savefig(f"{BENCHMARK_DIR}/{filename}_benchmark.png", format='png')


if __name__ == "__main__":

    start_time = datetime.now()
    print(f'\nBenchmark Start Time: {start_time.strftime("%H:%M:%S")}\n')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_empty_dir(dir_path=BENCHMARK_DIR)

    results = []
    image_dimensions = [32, 64, 128, 256, 512, 1024]
    NUM_IMG_STATIC = 240
    run_benchmark(
        benchmark_values=image_dimensions,
        static=NUM_IMG_STATIC,
        x_axis=IMAGE_DIMENSION,
        command_template=(
            "python3 {SCRIPT_PATH} --benchmark --augment={augment} --img_dim={value} --num_img={static} >> {BENCHMARK_DIR}/benchmark_log.log"
        )
    )

    results = []
    image_nums = [100, 200, 400, 800, 1600, 3200]
    IMG_DIM_STATIC = 200
    run_benchmark(
        benchmark_values=image_nums,
        static=IMG_DIM_STATIC,
        x_axis=NUMBER_OF_IMAGES,
        command_template=(
            "python3 {SCRIPT_PATH} --benchmark --augment={augment} --img_dim={static} --num_img={value} >> {BENCHMARK_DIR}/benchmark_log.log"
        )
    )


    end_time = datetime.now()
    duration = end_time - start_time
    formatted_duration = str(timedelta(seconds=duration.total_seconds()))
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'\nBenchmark End Time: {end_time.strftime("%H:%M:%S")}')
    print(f'BenchmarkDuration: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n')