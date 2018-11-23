import sklearn

print(sklearn.__version__)

import numpy as np
import pandas as pd
from time import time

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# loading custom functions
from Models.functions.datasets import getDatasets, loadTrainTest
from Models.functions.metrics import evaluator
from Models.functions.plot import ROC, plot_confusion_matrix
from Models.functions.preprocessing import clean, labelEncoder
from Models.functions.utils import checkFolder, listProblems

from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

# Synthetic Minority Oversampling Technique (SMOTE)
def oversampling(X, y):    
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)    
    #return X, y
    return X_resampled, y_resampled        

def getBestParams(task, dataset_name):
    baseline = 'baseline1'
    dataset_name = dataset_name.strip().lower()
    task = task.strip().lower()
    
    # load excel params
    baseline1 = pd.read_excel('./best_params.xlsx', baseline)
    
    baseline1['Task'] = baseline1['Task'].str.lower()
    baseline1['Name'] = baseline1['Name'].str.lower()
    
    best_params = baseline1[(baseline1['Name'] == dataset_name) & (baseline1['Task'] == task)]
    
    
    if len(best_params) < 1: return dict(
                clf__C =  1428.5715142857143,
                clf__penalty =  'l2',
                vect__max_df =  0.8,
                vect__max_features =  None,
                vect__stop_words = None)
    
    max_features = best_params['max features'].values[0]
    
    model_params = {
                    'vect__max_features': max_features if max_features != 'None' and not pd.isnull(max_features) else None,
                    'vect__max_df': best_params['max df'].values[0] if not pd.isnull(best_params['max df'].values[0]) else 1,
                    'clf__C': best_params['C'].values[0] if not pd.isnull(best_params['C'].values[0]) else 1000.0, 
                    'clf__penalty': best_params['P'].values[0] if not pd.isnull(best_params['P'].values[0]) else 'l2'
                    }
    
    return model_params

def model(X, y, n_classes, classes_name, params):
    
    # pipeline.set_params(**params)    
    vect = TfidfVectorizer(max_features=params.get('vect__max_features'), max_df=params.get('vect__max_df'))
    
    K = StratifiedKFold(n_splits=10)
    
    t0 = time()
    
    predicted_y = []
    expected_y = []    
    score_y = []
    
    for train_index, test_index in K.split(X, y):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        X_train = vect.fit_transform(X_train)
        X_test = vect.transform(X_test)
        
        X_train, y_train = oversampling(X_train, y_train)
        X_test, y_test = oversampling(X_test, y_test)

        clf = LogisticRegression(C=params.get('clf__C'), penalty=params.get('clf__penalty'), solver='liblinear')
        
        clf.fit(X_train, y_train)
        
        predicted_y.extend(clf.predict(X_test))
        expected_y.extend(y_test)
        score_y.extend(clf.predict_proba(X_test))

    # print("done in %0.2fs and %0.1fmin" % ((time() - t0), ((time() - t0) / 60) ))
    # print()
    
    report = pd.DataFrame(classification_report(expected_y, predicted_y, digits=5, target_names=classes_name, output_dict=True))
    report = report.transpose()
    
    return (
        report, 
        np.asarray(expected_y),
        np.asarray(predicted_y),
        np.asarray(score_y)
        )

def run(task, dataset_name, root, lang):    
    
    directory = './Reports/' + task + '/' + dataset_name + '_' + lang + '/'
    checkFolder(directory)
    
    X, _, y, _ = loadTrainTest(task=task, dataset_name=dataset_name, root=root, lang=lang)
    y, n_classes, classes_name = labelEncoder(y)    

    # clean text
    X = X.apply(clean, lang=lang)
    
    params = getBestParams(task, dataset_name)    
    print("params: ", params)

    report, expected_y, predicted_y, score_y = model(X, y, n_classes, classes_name, params)

    # TODO: compute ROC score after processing, with expected_y and score_y
    # get ROC
    # roc_c = ROC(expected_y, score_y, n_classes, task, dataset_name+'_'+lang, classes_name)
    # report['roc'] = list(roc_c.values()) + [roc_c['macro']] * 2

    # compute accuracy
    accuracy = accuracy_score(expected_y, predicted_y)
    report['accuracy'] = [accuracy] * (n_classes + 3)

    # compute confusion matrix
    c_matrix = confusion_matrix(expected_y, predicted_y)
    plot_confusion_matrix(c_matrix, classes_name, task, dataset_name+'_'+lang, True)
    cm = pd.DataFrame(c_matrix, columns=classes_name, index=classes_name)
    
    report.to_csv(directory + 'report.csv')
    cm.to_csv(directory + 'confusion_matrix.csv')
    np.save(directory + '/expected_y.numpy', expected_y)
    np.save(directory + '/predicted_y.numpy', predicted_y)
    np.save(directory + '/score_y.numpy', score_y)
        
    print("F-fold 10", task, dataset_name, lang, display(report))
    print()

    pass




import multiprocessing as mp
import random
import string

if __name__ == '__main__':

    g_root = root = sys.argv[1]
    g_task = task = sys.argv[2]
    g_dataset_name = dataset_name = sys.argv[3]
    g_lang = root = sys.argv[4]

    print("PARAMS")
    print(sys.argv)

    if g_task is not None:
    	run(g_task, g_dataset_name, g_root, g_lang)
    else:
        args = []
        problems = listProblems()
        print("############################################")
        print(" RUNNING {0} PROBLEMS".format(len(problems)))

        # create a list of tasks
        for task, dataset_name, lang in problems:
            args.append([task, dataset_name, g_root, lang])

        # Define an output queue
        output = mp.Queue()

        # Setup a list of processes that we want to run
        processes = [mp.Process(target=run, args=(x[0], x[1], x[2], x[3])) for x in args]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = [output.get() for p in processes]

        print("###############################")
        print("FINISHED")
