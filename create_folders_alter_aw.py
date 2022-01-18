import os
import argparse

# how to run this code
# python create_folders.py --species=ladybird
parser = argparse.ArgumentParser()
parser.add_argument('--species', type=str,
    default='aphid', help="aphid or ladybird")
parser.add_argument('--Ratio', type=int,
    default=200, help="N_prey/N_predator")
parser.add_argument("--case", type=int, 
    default="0", help="1 or 2 or 3 or 4")
parser.add_argument("--alter", type=str, 
    default="aw00", help="w2 or w4 or a-4 or a4 or aw-42 or aw-44 or aw42 or aw44")
parser.add_argument("--year", type=str, 
    default="2080", help="2020 or 2080")


args = parser.parse_args()
case = args.case
alter = args.alter
year = args.year
Ratio = args.Ratio

def main(args):
    fold_dir = f"exports_case{case}_{alter}_{year}_r{Ratio}_sc3"
    scenarios = ['test']
    for scenario in scenarios:
        for folder_name in ["data", "plotAnp", "plotAp", "plotH", "plotAH"]:
            folder = os.path.join(
                fold_dir, "{}_{}".format(scenario, folder_name))
            os.makedirs(folder)

if __name__ == '__main__':
    main(args)
