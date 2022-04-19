#!/bin/bash
#SBATCH --account=def-encaenia-ab   # replace this with your own account
#SBATCH --mem=10G     # memory; default unit is megabytes
#SBATCH --time=3:00:00             # time (DD-HH:MM)
python R04batch_sum_np2.py

