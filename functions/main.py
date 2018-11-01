from functions.datasets import getDatasets
#from functions.grid_baseline1 import reglog
from functions.grid_baseline2 import Model
from functions.plot import ROC
from functions.metrics import evaluator, reportPath
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
import datetime
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

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
    
        X_train = df_training['text'].values
        y_train, _ = labelEncoder(df_training[label].values)

        X_test = df_test['text'].values
        y_test, n_classes = labelEncoder(df_test[label].values)    
        
        report_path = reportPath(task, name)
        f = open(report_path, 'a')
        print("Datetime {0}".format(datetime.datetime.now()), file=f)
        print("Dataset: {0} and task: {1}".format(name, label), file=f)
        print(file=f)
        print('training set: ', df_training.shape, file=f)        
        print(df_training.groupby([label]).size(), file=f)
        print(file=f)
        print('test set: ', df_test.shape, file=f)        
        print(df_test.groupby([label]).size(), file=f)
        print(file=f)
        print("n_classes: {0}".format(n_classes), file=f)        
        
        del(df_test)
        del(df_training)
        
        # (acc, f1, cm) = mlp(X_train, X_test, y_train, y_test, n_classes)
        clf = Model(task, name, f)
        
        clf.mlp(X_train, y_train, X_test, y_test, n_classes)
        
        f.close()
        
        #del(Model)
        del(X_train)
        del(X_test)
        del(y_train)
        del(y_test)
        pass
