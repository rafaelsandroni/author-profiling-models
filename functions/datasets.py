

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

def getDatasets(task = None, file_type = '_', dataset_name = None, root = None):
    if task == None:
        return
    
    root = root or '/home/rafael/drive/Data/Dataframe/'
    print("loading from", root)
    columns = {'dataset_name': 0,'path': 0,'task': 0,'training': 0,'test': 0}
    datasets_name = ['b5post', 'esic', 'brmoral', 'enblogs', 'brblogset', 'pan13_en','pan13_es','smscorpus']
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


def loadTrainTest(task, dataset_name, root, lang = "pt"):
    
    task = task.lower()
    dataset_name = dataset_name.lower()

    extension = "df"

    train_filename = "{0}_{1}_training_{2}.csv".format(task, lang, extension)
    test_filename = "{0}_{1}_test_{2}.csv".format(task, lang, extension)

    root += "/"
    root += dataset_name
    root += "/"

    df_training = pd.read_csv(root + training_filename)
    df_test = pd.read_csv(root + test_filename)

    # load training and test dataframes
    X_train = df_training['text'].values
    y_train = df_training[task].values

    X_test = df_test['text'].values
    y_test = df_test[task].values

    return (X_train, X_test, y_train, y_test)
