import os
import pandas as pd
import numpy as np
# import tqdm
# import csv

cases = [0,1,2,3,4,5,6,7,8]
alters = ['aw00','aw04','aw08','aw-40','aw-44','aw-48','aw40','aw44','aw48']
for alter in alters:
    for case in cases:
        data_dir = f'../outputs/exports_case{case}_{alter}_50years/eco_data'
        files = os.listdir(data_dir)
        no_files = len(files)
        print(f'{alter} -- case{case}: {no_files} files')
        for file in files:
            import pdb;pdb.set_trace()
            file_dir = os.path.join(data_dir,file)
            data = pd.read_csv(file_dir)
            data_len = len(data)
            if data_len != 36500*50:
                print(file,data_len)

    
