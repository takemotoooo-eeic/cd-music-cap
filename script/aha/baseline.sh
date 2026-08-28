#!/bin/bash
#SBATCH --job-name=aha-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

uv run python run_benchmark.py -o gemma-eval -s af3 -t aha-jl
uv run python run_benchmark.py -o gemma-eval -s qwen -t aha-jl
