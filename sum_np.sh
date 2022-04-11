#!/bin/bash
#SBATCH --account=def-encaenia-ab   # replace this with your own account
#SBATCH --mem=50048M      # memory; default unit is megabytes
#SBATCH --time=6:00:00             # time (DD-HH:MM)

python sum_outputs_50years3_np.py
