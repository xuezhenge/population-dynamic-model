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
    default='1', help="1,2,3 or 4")
parser.add_argument('--alter', type=str,
    default='aw00', help="w3 or a3 etc")
parser.add_argument('--year', type=str,
    default='2080', help="2080")
parser.add_argument('--Ratio', type=int,
    default='200', help="1,2,3 or 4")

args = parser.parse_args()
case = args.case
year = args.year
alter = args.alter
Ratio = args.Ratio
    
def get_date(a,w):
    # the date when T < Tmin:
    t = symbols('t')
    Temp = Eq(-a*sympy.cos(2*sympy.pi*t/365) + w, 10)
    sol = solve(Temp)
    date = math.ceil(sol[1])
    import pdb;pdb.set_trace()
    return date
    
def get_outputs(file,data_dir):
    loc = file.split(".csv")[0]
    a_name,a,w_name,w = loc.split("_")
    a = float(a)
    w = float(w)
    file_dir = os.path.join(data_dir,file)
    df = read_csv(file_dir, header=None)
    data_np_p = df.iloc[:,1:4]
    ind = list(set(df.index[df.iloc[:,3] > 0]))
    threshold_date = get_date(a,w)
    if ind == []:
        N_predator = 0
        N_prey = 0
    elif data_np_p[2][int(threshold_date-1)] < 1:
        # if the Aden < 1 by the end of the growing season
        N_predator = 0
        N_prey = 0
    else: 
        N_predator= df.iloc[:,3][ind].to_numpy()[0]
        N_prey= df.iloc[:,2][ind].to_numpy()[0]
    data_np_p_sum = data_np_p.sum(axis=0).tolist()
    row_AAP = [a] + [w] + data_np_p_sum + [N_prey,N_predator]
    return [row_AAP]


rowname = ['a','w','Anp','Ap','L','N_prey','N_predator']

def writer_csv(rows, filename, rowname = rowname, bool_continue=False):
    with open(filename, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rowname)
        writer.writerows(rows)

def main():
    print(alter,case,year)
    data_dir = f'exports_case{case}_{alter}_{year}_r{Ratio}_sc3/test_data'
    files = os.listdir(data_dir)
    dump_dir = f'./outputs/AAPs_test_summary_{alter}_r{Ratio}_sc3'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir) 
    # creat output files
    fn_AAP = f"AAPs_case{case}_{year}.csv"
    fn_AAP = os.path.join(dump_dir, fn_AAP)
    if os.path.exists(fn_AAP):
         print("file exist! continue!")
         return

    rows_AAP = []
    files = os.listdir(data_dir)
    for file in tqdm.tqdm(files):
        row_AAP = get_outputs(file,data_dir)
        rows_AAP += row_AAP
    writer_csv(rows_AAP,filename = fn_AAP)


if __name__ == '__main__':
    main()
