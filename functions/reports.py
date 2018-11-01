import pandas as pd
import os

def showReports(task, model_name = 'baseline1', name = 'report.csv'):    
    for root, folder, files in os.walk('./'):
        if len(files) > 0 and model_name in root and name in files and task == root.split('/')[1]:
            task = root.split('/')[1]
            dataset_name = root.split('/')[2]            
            print(task, dataset_name)
            display(pd.read_csv(root + '/' + name, index_col=0))    
