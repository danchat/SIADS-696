#!/bin/bash

#SBATCH --job-name=ehr_baselines

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=02:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/ehr_baselines_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/ehr_baselines_%j.err



set -euo pipefail

module load python/3.10

source /nfs/turbo/si-acastel/mimic-project/venv/bin/activate



python /nfs/turbo/si-acastel/mimic-project/scripts/train_ehr_baselines.py
