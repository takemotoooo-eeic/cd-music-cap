#!/bin/bash
#SBATCH --job-name=mmau-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

uv run python run_benchmark.py -o gemma-eval -s af3 -t mmau-test-mini-jl
uv run python run_benchmark.py -o gemma-eval -s qwen -t mmau-test-mini-jl
