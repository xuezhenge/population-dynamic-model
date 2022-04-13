import os
import pandas as pd
import numpy as np
# import tqdm
# import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--alter', type=str,
    default="aw00", help="..")

args = parser.parse_args()
alter = args.alter
cases = [0,1,2,3,4,5,6,7,8]
for case in cases:
    data_dir = f'../outputs/exports_case{case}_{alter}_50years/eco_data'
    files = os.listdir(data_dir)
    no_files = len(files)
    print(f'{alter} -- case{case}: {no_files} files')
    i = 0
    for file in files:
        i = i + 1
        print(i)
        file_dir = os.path.join(data_dir,file)
        data = pd.read_csv(file_dir)
        data_len = len(data)
        if data_len != 36500*50:
            print(file,data_len)
