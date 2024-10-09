# Variables
ENV_NAME=tfcuda

# Create and activate conda environment
create-env:
	conda env create -f tfcuda.yml

# Install dependencies via pip
install:
	pip3 install -r requirements.txt

# Clean up build files
clean:
	rm -rf __pycache__ *.pyc .pytest_cache
	rm -rf ./output/*

# Run the main Python script
run:
	python3 src/benchmark.py

# Default action is to show the options
help:
	@echo "Makefile for Python Project"
	@echo "Available commands:"
	@echo "  create-env   Create the conda environment (recommended)"
	@echo "  install      Install dependencies via pip"
	@echo "  clean        Remove temporary files"
	@echo "  run          Run the main Python script"
