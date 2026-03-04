#!/bin/bash

#SBATCH --job-name=xgb_tune_pnx_v2

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=04:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/xgb_tune_pnx_v2_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/xgb_tune_pnx_v2_%j.err



set -euo pipefail

module load python/3.10

source /nfs/turbo/si-acastel/mimic-project/venv/bin/activate



python /nfs/turbo/si-acastel/mimic-project/scripts/tune_xgb_pneumothorax_v2_randomsearch.py
