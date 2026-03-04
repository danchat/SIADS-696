#!/bin/bash
#SBATCH --job-name=mimiciv_dl
#SBATCH --account=siads696w26_class
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/%x_%j.out
#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/%x_%j.err

/nfs/turbo/si-acastel/mimic-project/download_mimiciv.sh

