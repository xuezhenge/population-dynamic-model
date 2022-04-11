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
    mid_ind = 36500*30
    end_ind = 36500*year
    data1 = df.iloc[start_ind:mid_ind,:]
    data2 = df.iloc[mid_ind:end_ind,:]
    return data1,data2

def get_sum(df):
    #import pdb;pdb.set_trace()
    data = df.iloc[:,1:3]
    # yearly mean + dt = 0.01*0.1
    data_sum = data.sum(axis=0)*0.01*0.1
    data_sum = data_sum.tolist()
    return data_sum

def get_born(df):
    # calculate A born and L born
    data = df.iloc[:,3:5]
    # yearly mean
    data_born = data.sum(axis=0)*0.1
    data_born = data_born.tolist()
    return data_born

def get_average(df):
    dataA = df.Aden_dt.to_numpy()
    dataL = df.Lden_dt.to_numpy()
    # reshape array into 36500 rows x 10 columns, and transpose the result
    reshaped_dataA = dataA.reshape(10,36500)
    reshaped_dataL = dataL.reshape(10,36500)
    #calculate Amin and Lmin in each year
    Amin_val = np.min(reshaped_dataA,axis=1)
    Lmin_val = np.min(reshaped_dataL,axis=1)
    #calculate Amax and Lmax in each year
    Amax_val = np.max(reshaped_dataA,axis=1)
    Lmax_val = np.max(reshaped_dataL,axis=1)
    #calculate averaged [A_ave = (Amin + Amax)/2 and  L_ave =  (Amin + Amax)/2] across 10 years
    A_ave = np.mean((Amin_val + Amax_val)/2)
    L_ave = np.mean((Lmin_val + Lmax_val)/2)
    return [A_ave,L_ave]

def get_decomposed_coef(data):
    # time series decomposition
    data.set_index('dt',inplace=True)
    data.index=pd.to_datetime(data.index)
    #drop null values
    data.dropna(inplace=True)
    result=seasonal_decompose(data['Lden_dt'], model='Additive', period=36500)
    result_trend = result.trend
    #drop null values
    result_trend.dropna(inplace=True)
    # import pdb;pdb.set_trace()
    trend = result_trend.to_numpy()
    dt = np.arange(len(trend)).reshape((-1, 1))
    model = LinearRegression().fit(dt, trend)
    # yearly trend of the decomposed trend
    coef_ = model.coef_*36500
    coef_ = coef_[0]
    return coef_
    
def get_peak(data):
    #import pdb;pdb.set_trace()
    data = data.Lden_dt.to_numpy()
    # reshape array into 36500 rows x 10 columns, and transpose the result
    reshaped_data = data.reshape(20,36500)
    peak_val = np.max(reshaped_data,axis=1)
    year = np.arange(1,20).reshape((-1, 1))
    Lpeak = peak_val[-1]
    print(peak_val)
    return Lpeak

rowname = ['a','w','Aden', 'Lden']
def writer_csv(rows, filename, rowname = rowname, bool_continue=False):
    with open(filename, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rowname)
        writer.writerows(rows)

def out_csv(i,idxs):
    idx = idxs[i]
    alter = idx[0]
    case = idx[1]
    data_dir = f'../outputs/exports_case{case}_{alter}_50years_np/eco_data'
    files = os.listdir(data_dir)
    dump_dir = f'../outputs/AAPs_sum_{alter}_50years_np'
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
    files = files[800:1161]
    for file in tqdm.tqdm(files):
            if file == '.DS_Store':
                continue
            loc = file.split(".csv")[0]
            a_name,a,w_name,w = loc.split("_")
            a = float(a)
            w = float(w)
            file_dir = os.path.join(data_dir,file)
            df = read_csv(file_dir, header=0)
            if len(df) == 1825000:
                data1,data2 = get_data(df,year=50)
                data_sum = get_sum(data1)
                row = [a] + [w] + data_sum
                rows += [row]
            else:
                row = [a] + [w] + ['nan'] + ['nan']
                print(row)
                rows += [row]
    #import pdb;pdb.set_trace()
    writer_csv(rows,filename = fn_AAP)
    print('Done!!!')

idxs = [['aw00',0]]
out_csv(0,idxs)
