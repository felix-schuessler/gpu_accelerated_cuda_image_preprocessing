#!/usr/bin/env bash

echo "clean outputs..."

make clean

echo "\nRunning the program...\nLogfiles will be written to ./output/"

python3 src/multiclass_cnn.py > ./output/raw_log.log
python3 src/multiclass_cnn.py --augment=pillow > ./output/pillow_log.log
python3 src/multiclass_cnn.py --augment=cuda > ./output/cuda_log.log

python3 src/benchmark.py
