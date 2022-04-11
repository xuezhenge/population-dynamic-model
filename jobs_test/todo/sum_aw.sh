#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate
cd /home/$USER/scratch/eco_model/population-dynamic-model
python sum_outputs_50years3_np.py
