# population-dynamic-model
code for "Predator-prey interactions in a warming world: warmer and more seasonal climate decreases the effect of predation"

# How to run the code for testing
- Step 1: Dowload three files:
  - create_folders_alter_aw.py
  - model.py
  - 'outputs' folder

- Step 2: 
`
python create_folders_alter_aw.py --case=0 --alter=aw00 --year=2080 --Ratio=200
`

- Step 3: 
`
python model.py --start_idx=100 --end_idx=102 --num_cores=1 --alter=aw00 --case=0 --Ratio=200
`

#you can change the start_idx and end_idx, they just represent the index of different a and w in the "a_w_changes_2020_2080.csv"
