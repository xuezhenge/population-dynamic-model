#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate
cd /home/$USER/scratch/eco_model/population-dynamic-model
python file_length.py --alter=aw-44