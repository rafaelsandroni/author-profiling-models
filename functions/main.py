from functions.datasets import getDatasets
from functions.grid_baseline1 import reglog
from functions.grid_baseline1 import mlp
from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def labelEncoder(y):
    le = preprocessing.LabelEncoder()
    le.fit(y)

    # print('>> classes', list(le.classes_))

    return (le.transform(y), len(le.classes_))

def run(task = None, dataset = None):
    if task == None:
        return False
    
    results = []
    
    datasets = getDatasets(task,'df', dataset)

    for i in datasets.iterrows():    
        name = i[1]['dataset_name']
        label = task
        print("Dataset: {0} and task: {1}".format(name, label))
        print()
        ds_path = i[1]['path']

        # load training and test dataframes
        training_path = ds_path + '/' + i[1]['training']
        test_path = ds_path + '/' +  i[1]['test']
        
        df_training = pd.read_csv(training_path)#, usecols=cols)
        df_test = pd.read_csv(test_path)#, usecols=cols)

        print('training set: ', df_training.shape)        
        print(df_training.groupby([label]).size())
        print()
        print('test set: ', df_test.shape)        
        print(df_test.groupby([label]).size())
        print()
        # df_training.groupby([label]).size().plot(kind='bar',title='Corpus '+name+' training dataset classes distributions')
        
        X_train = df_training['text'].values
        y_train, _ = labelEncoder(df_training[label].values)

        X_test = df_test['text'].values
        y_test, n_classes = labelEncoder(df_test[label].values)    
        
        del(df_test)
        del(df_training)
        
        print("n_classes: {0}".format(n_classes)
              
        (acc, f1, cm) = grid(X_train, X_test, y_train, y_test, n_classes)
        print()
        
        del(X_train)
        del(X_test)
        del(y_train)
        del(y_test)
        
        print('\n+', '-'*60,'+\n')            
        pass
