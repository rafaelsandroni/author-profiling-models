

import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import os

#%%


#%%

def getDatasets(task = None, file_type = '_', dataset_name = None):
    if task == None:
        return
    
    root = '/home/rafael/drive/Data/Dataframe/'
    columns = {'dataset_name': 0,'path': 0,'task': 0,'training': 0,'test': 0}
    datasets_name = ['b5post', 'esic', 'brmoral', 'enblogs', 'brblogset', 'pan13_en','pan13_es']
    datasets = pd.DataFrame(columns=columns)
    # mount dataset dataframe
    for ds in datasets_name:    
        c = columns
        c['dataset_name'] = ds
        c['task'] = task
        c['training'] = None
        c['test'] = None
        for (walk_path, walk_subdir, walk_files) in os.walk(root+ds):                
            c['path'] = walk_path
            for file in walk_files:            
                if 'training' in file and task in file and file_type in file:
                    c['training'] = file
                elif 'test' in file and task in file and file_type in file:
                    c['test'] = file            

        datasets = datasets.append(c, ignore_index=True)

    # check if datasets task exists
    datasets = datasets[pd.notnull(datasets['training'])]
    
    if dataset_name != None:
        datasets = datasets[datasets.dataset_name.str.lower() == dataset_name.lower()]
    
    return datasets



