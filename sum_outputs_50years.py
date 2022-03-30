import os
from pandas import read_csv
import pandas as pd
import numpy as np
import tqdm
import csv
# import random
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

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int,
    default='0', help="1,2,3 or 4")
parser.add_argument('--alter', type=str,
    default='aw00', help="w3 or a3 etc")

args = parser.parse_args()
case = args.case
alter = args.alter

def get_sum(df):
    #import pdb;pdb.set_trace()
    data = df.iloc[:,0:2]
    # yearly mean + dt = 0.01*0.1
    data_sum = data.sum(axis=0)*0.01*0.05
    data_sum = data_sum.tolist()
    return data_sum

def get_born(df):
    # calculate A born and L born
    data = df.iloc[:,2:4]
    # yearly mean
    data_born = data.sum(axis=0)*0.05
    data_born = data_born.tolist()
    return data_born

def get_data(df,year):
    # drop the data when ladybird < 1
    df = df.rename({'Unnamed: 0': 'dt'}, axis=1)
    # Lden_dt = df.Lden_dt.to_list()
    # L_ind = [n for n,i in enumerate(Lden_dt) if i>1]
    # if len(L_ind) == 0:
    #     end_ind = 36500*7
    #     Lend = 0
    # else:
    # # for decomposition, must have 2 complete cycles requires 73000 observations.
    #     end_ind = np.max(L_ind)
    #     Lend = df.Lden_dt[end_ind]
    # if end_ind < 36500*7 or end_ind == 'NA':
    #     end_ind = 36500*7
    #     L_end = 0
    #import pdb;pdb.set_trace()
    start_ind = 36500*30
    end_ind = 36500*year
    data = df.iloc[start_ind:end_ind,:]
    return data

def get_decomposed_coef(data):
    # time series decomposition
    data.set_index('dt',inplace=True)
    data.index=pd.to_datetime(data.index)
    #drop null values
    data.dropna(inplace=True)
    result=seasonal_decompose(data['Lden_dt'], model='Additive', period=36500)
    #data.plot()
    # result.plot()
    # plt.show()
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



def get_sample_coef(data,year):
    #import pdb;pdb.set_trace()
    # get the yearly peak value of Ladybird population and calculate the changing rate of the peak value
    # start_ind = 36500*5
    # end_ind = 36500*year
    # data = df.iloc[start_ind:end_ind,:]
    data = data.Lden_dt.to_numpy()
    # reshape array into 36500 rows x 10 columns, and transpose the result
    reshaped_data = data.reshape(year-30,36500)
    peak_val = np.max(reshaped_data,axis=1)
    year = np.arange(1,year-30+1).reshape((-1, 1))
    model = LinearRegression().fit(year, peak_val)
    coef_ = model.coef_
    coef_ = coef_[0]
    Lpeak = peak_val[-1]
    print(peak_val)
    return coef_,Lpeak
    


#rowname = ['a','w','Anp','Ap','L','N_prey','N_predator']
rowname = ['a','w','Aden', 'Lden', 'Aborn', 'Lborn','decomposed_coef','sampled_coef', 'Lpeak','YorN']

def writer_csv(rows, filename, rowname = rowname, bool_continue=False):
    with open(filename, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rowname)
        writer.writerows(rows)

def main():
    print(alter,case)
    data_dir = f'../outputs_cc/exports_case{case}_{alter}_50years/eco_data'
    files = os.listdir(data_dir)
    dump_dir = f'../outputs_cc/AAPs_sum_{alter}_50years'
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
    
    i = 0
    for file in tqdm.tqdm(files):
        if file == '.DS_Store':
            continue
        # a = 7
        # w = 9
        # file = f'a_{a}_w_{w}.csv'
        
        loc = file.split(".csv")[0]
        a_name,a,w_name,w = loc.split("_")
        a = float(a)
        w = float(w)
        file_dir = os.path.join(data_dir,file)
        df = read_csv(file_dir, header=0)
        print(a,w,len(df))
        # import pdb;pdb.set_trace()
        data = get_data(df,year=50)
        sample_coef,Lpeak = get_sample_coef(data,year=50)
        decomposed_coef = get_decomposed_coef(data)
        data_sum = get_sum(data)
        data_born = get_born(data)
        if decomposed_coef > 0 or Lpeak>=50000:
            YorN = 'Y'
        else:
            YorN = 'N'
        #row = [a] + [w] + [0] + [0] + [0] + [0] + [0] + [0]
        row = [a] + [w] + data_sum + data_born + [decomposed_coef] + [sample_coef] + [Lpeak] + [YorN]
        rows += [row]
    import pdb;pdb.set_trace()
    writer_csv(rows,filename = fn_AAP)

if __name__ == '__main__':
    main()
	
