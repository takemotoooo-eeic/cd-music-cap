#!/bin/bash
#SBATCH --job-name=acb-aad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

uv run python run_benchmark.py -o gemma-eval -s af3-aad -t audiocapbench-jl --model_config config/aad_10.yaml
uv run python run_benchmark.py -o gemma-eval -s qwen-aad -t audiocapbench-jl --model_config config/aad_10.yaml
