#!/bin/bash
#SBATCH --job-name=mmau-avs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

uv run python run_collect_activations.py -o wavcaps-audioset-sl -s qwen --model_config config/activation_qwen.yaml
