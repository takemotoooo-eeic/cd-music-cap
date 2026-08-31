#!/bin/bash
#SBATCH --job-name=mmau-avs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

uv run python run_benchmark.py -o gemma-eval -s qwen-avs -t mmau-test-mini-jl --model_config config/avs_qwen.yaml
uv run python run_benchmark.py -o gemma-eval -s af3-avs -t mmau-test-mini-jl --model_config config/avs_af3.yaml
