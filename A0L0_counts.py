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
    df = pd.read_csv(f'../outputs/sum_csv/AAPs_sum_{alter}_50years/AAPs_case{case}.csv')
    df = df[df["Class"] == 'A0L0']
    df = df[args.start_idx:args.end_idx]
    a = np.array(df.a)
    len = len(a)
    print(alter, case, a)
