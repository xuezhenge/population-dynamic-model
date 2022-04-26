import os
from shutil import get_archive_formats
from pandas import read_csv
import pandas as pd
import numpy as np
import tqdm
import csv
from joblib import Parallel, delayed
import argparse
from sympy import symbols, Eq, solve
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
# import sympy
# import math
import matplotlib.pyplot as plt
import glob
# reference blog
# https://www.machinelearningplus.com/time-series/time-series-analysis-python/
# https://towardsdatascience.com/time-series-decomposition-in-python-8acac385a5b2


def get_data(df,year):
    df = df.rename({'Unnamed: 0': 'dt'}, axis=1)
    start_ind = 36500*20
    end_ind = 36500*year
    data = df.iloc[start_ind:end_ind,:]
    return data

def get_sum(df):
    #import pdb;pdb.set_trace()
    data = df.iloc[:,1:3]
    # yearly mean + dt = 0.01*0.1
    data_sum = data.sum(axis=0)*0.01*0.1
    data_sum = data_sum.tolist()
    return data_sum

rowname = ['a','w','Aden', 'Lden']
def writer_csv(rows, filename, rowname = rowname, bool_continue=False):
    with open(filename, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rowname)
        writer.writerows(rows)

def out_csv(i,idxs):
    idx = idxs[i]
    print(idx)
    alter = idx[0]
    case = idx[1]
    data_dir = f'../outputs/exports_case{case}_{alter}_30years_np/eco_data'
    files = os.listdir(data_dir)
    dump_dir = f'../outputs/AAPs_sum_{alter}_30years_np'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir) 
    # creat output files
    fn_AAP = f"AAPs_case{case}.csv"
    fn_AAP = os.path.join(dump_dir, fn_AAP)
    if os.path.exists(fn_AAP):
         print("file exist! continue!")
         return

    rows = []
    files = os.listdir(data_dir)
    for file in tqdm.tqdm(files):
            if file == '.DS_Store':
                continue
            loc = file.split(".csv")[0]
            a_name,a,w_name,w = loc.split("_")
            a = float(a)
            w = float(w)
            print(a,w)
            file_dir = os.path.join(data_dir,file)
            df = read_csv(file_dir, header=0)
            data = get_data(df,year=30)
            data_sum = get_sum(data)
            row = [a] + [w] + data_sum
            print(row)
            rows += [row]
    #import pdb;pdb.set_trace()
    writer_csv(rows,filename = fn_AAP)
    print('Done!!!')
    
cases = [0,1,2,3,4,5,6,7,8]
#alters = ['aw00','aw04','aw08','aw-40','aw-44','aw-48','aw40','aw44','aw48']
alters = ['aw04','aw08','aw-40','aw-44','aw-48','aw40','aw44','aw48']

idxs = []
for alter in alters:
    for case in cases:
        idx = [alter,case]
        idxs += [idx]
idxs = [['aw48',2],['aw48',3],['aw48',4],['aw48',5],['aw48',6],['aw48',7],['aw48',8],['aw44',4],['aw48',1]]
num_idxs = len(idxs)
num_cores = 1
for i in np.arange(num_idxs) :
    processed_list = Parallel(n_jobs=num_cores)(delayed(out_csv)(i,idxs) for i in range(num_idxs))

