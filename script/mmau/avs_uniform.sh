#!/bin/bash
#SBATCH --job-name=mmau-avs-uniform
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

uv run python run_benchmark.py -o uniform -s qwen-avs -t mmau-test-mini-jl --model_config config/avs_uniform.yaml
uv run python run_benchmark.py -o uniform -s af3-avs -t mmau-test-mini-jl --model_config config/avs_uniform.yaml
