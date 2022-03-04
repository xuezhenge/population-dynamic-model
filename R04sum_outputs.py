import os
from pandas import read_csv
import pandas as pd
import numpy as np
import tqdm
import csv
import random
import argparse
from sympy import symbols, Eq, solve
import sympy
import math

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int,
    default='0', help="1,2,3 or 4")
parser.add_argument('--alter', type=str,
    default='aw00', help="w3 or a3 etc")

args = parser.parse_args()
case = args.case
alter = args.alter

def get_outputs(file,data_dir):
    loc = file.split(".csv")[0]
    a_name,a,w_name,w = loc.split("_")
    a = float(a)
    w = float(w)
    file_dir = os.path.join(data_dir,file)
    df = read_csv(file_dir, header=None)
    data_np_p = df.iloc[:,1:4]
    data_np_p_sum = data_np_p.sum(axis=0).tolist()
    #row_AAP = [a] + [w] + data_np_p_sum + [N_prey,N_predator]
    row_AAP = [a] + [w] + data_np_p_sum
    return [row_AAP]

#rowname = ['a','w','Anp','Ap','L','N_prey','N_predator']
rowname = ['a','w','Anp','Ap','L']

def writer_csv(rows, filename, rowname = rowname, bool_continue=False):
    with open(filename, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rowname)
        writer.writerows(rows)

def main():
    print(alter,case)
    data_dir = f'../outputs/exports_case{case}_{alter}/eco_data'
    files = os.listdir(data_dir)
    dump_dir = f'../outputs/AAPs_sum_{alter}'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir) 
    # creat output files
    fn_AAP = f"AAPs_case{case}.csv"
    fn_AAP = os.path.join(dump_dir, fn_AAP)
    if os.path.exists(fn_AAP):
         print("file exist! continue!")
         return

    rows_AAP = []
    files = os.listdir(data_dir)
    i = 0
    for file in tqdm.tqdm(files):
        row_AAP = get_outputs(file,data_dir)
        rows_AAP += row_AAP
    writer_csv(rows_AAP,filename = fn_AAP)

if __name__ == '__main__':
    main()
	
