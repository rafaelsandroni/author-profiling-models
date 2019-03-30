
# coding: utf-8

# In[1]:


# In[1]:

import numpy as np
import pandas as pd
import pickle

from Models.functions.plot import plot_history, full_multiclass_report, plot_confusion_matrix
from Models.functions.preprocessing import clean, labelEncoder
from Models.functions.datasets import loadTrainTest
from Models.functions.utils import checkFolder, listProblems

from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix

import itertools
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE, ADASYN
import collections, numpy
import gc
from time import time, sleep

results_dataframe = "/reports_grid/results.csv"
try:
    results = pd.read_csv(results_dataframe)
except:
    results = pd.DataFrame()

# In[4]:

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif


def create_model(filters = [100], kernel_size = [50], strides = [100], 
                 dropout_rate = 0.5, pool_size = [5], dense_units = 512, max_len = 1000, n_classes = 2):

    model = Sequential()

    # conv 1
    model.add(Conv1D(filters = filters[0], 
                     kernel_size = kernel_size[0],
                     strides = strides[0], 
                     activation = 'relu', 
                     input_shape = (max_len, 1)))
                     #activity_regularizer = regularizers.l2(0.2)))

    # pooling layer 1
    
    model.add(MaxPooling1D(pool_size = pool_size[0], strides = 1))
    model.add(Activation('relu'))
    """
    model.add(Conv1D(filters = filters[1], 
                     kernel_size = kernel_size[1],
                     strides = strides[0], 
                     activation = 'relu',
                     activity_regularizer = regularizers.l2(0.2)))
    
    model.add(MaxPooling1D(pool_size = pool_size[1], strides = 1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters = filters[2], 
                     kernel_size = kernel_size[2],
                     strides = strides[0], 
                     activation = 'relu',
                     activity_regularizer = regularizers.l2(0.2)))
    
    model.add(MaxPooling1D(pool_size = pool_size[2], strides = 1))
    model.add(Activation('relu'))
    """
    model.add(Flatten())
    
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(units = dense_units, activation = 'relu'))
    model.add(Dense(units = n_classes, activation = 'softmax'))

    #TODO: test others foss functions: https://keras.io/losses/
    model.compile(optimizer = 'adadelta', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model


# In[5]:


def garbage_collection(): 
    gc.collect()
    print("gargabe colletion...")
    sleep(3)

# In[6]:

task = "gender"
dataset_name = "brmoral"
lang = "pt"
root = "/home/rafael/Dataframe/"

# Synthetic Minority Oversampling Technique (SMOTE)
def oversampling(X, y):
    try:
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    except:
        X_resampled, y_resampled = X, y
        
    return X_resampled, y_resampled
    # return X, y

def train_val_metrics(histories):
    print('Training: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'loss'), get_avg(histories, 'acc')))
    print('Validation: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'val_loss'), get_avg(histories, 'val_acc')))

def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)



def run(task, dataset_name, root, lang):

    histories = []
    test_loss = []
    test_accs = []

    predicted_y = []
    predicted_y_proba = []
    expected_y = []

    directory='./Reports_kbest/'+task+'/'+dataset_name+'_'+lang+'/'
    checkFolder(directory)

    X, _, y, _ = loadTrainTest(task, dataset_name, root, lang)

    X = X.apply(clean, lang=lang)
    X = X.values # mandatory for pan13
    
    y, n_classes, classes_names = labelEncoder(y)    

    max_length = np.max([len(x.split(" ")) for x in X])
    mean_length = np.mean([len(x.split(" ")) for x in X])
    median_length = np.median([len(x.split(" ")) for x in X])

    if mean_length < 30:
        mean_length = 30
    elif mean_length > 1500 and median_length < 1500:
        mean_length = median_length
    elif mean_length > 1500 and median_length > 1500:
        mean_length = 1300

    print("max: ", max_length, " / mean: ", mean_length, " / median: ", median_length)
    
    K = StratifiedKFold(n_splits=3)
    idx = 0
    for train_index, test_index in K.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 
        vect = None
        vect = TfidfVectorizer(max_features=50000) #int(mean_length))        
        X_train = vect.fit_transform(X_train).toarray()
        X_test = vect.transform(X_test).toarray()

        X_train, y_train = oversampling(X_train, y_train)
        X_test,  y_test  = oversampling(X_test, y_test)

        # KBEST
        kvalue = 30000
        k_best_func = f_regression#chi2, #, f_classif
        # feature selection
        sel = SelectKBest(k_best_func,k=kvalue)
        ft = sel.fit(X_train, y_train)
        X_train = ft.transform(X_train)
        X_test = ft.transform(X_test)


        y_train = to_categorical(y_train, n_classes)
        y_test  = to_categorical(y_test, n_classes)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # validation
        validation_split = 0.1
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_split) 

        params = dict(
            filters = [50],
            kernel_size = [3],
            strides = [1],
            dropout_rate = 0.15,
            pool_size = [2],
            epochs = 100,
            batch_size = 12
        )        
        """

        
        vectors_filename = r'/home/rafael/GDrive/Embeddings/word2vec/'+ ds_name +'_sg_'+ str(params['embedding_dim']) +'dim.model'        
        embedding_type = 1
        embedding_matrix = create_embeddings(vect, params['max_num_words'], params['max_seq_length'], name=dataset_name, embedding_dim=params['embedding_dim'], filename=vectors_filename, type=embedding_type, return_matrix=True)        
        """

        ## create the model with the best params found
        #model = KerasClassifier(build_fn=create_model,
        model = None
        model = create_model(
                                max_len=X_train.shape[1],
                                n_classes=n_classes,
                                filters=params['filters'],
                                kernel_size=params['kernel_size'],
                                strides=params['strides'],                        
                                dropout_rate=params['dropout_rate'],
                                pool_size=params['pool_size']
                            )

        ## Then train it and display the results
        history = model.fit(X_train,
                            y_train,                            
                            validation_data=(X_val, y_val),                            
                            verbose = 1,
                            batch_size=params['batch_size'],                                
                            epochs=params['epochs'],
                            callbacks=[
                                #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01),
                                EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=0)
                        ])        
        
        y_pred_proba = model.predict(X_test, batch_size=params['batch_size'])
        predicted_y_proba.extend(y_pred_proba)

        binary = False # True if len(classes) < 3 else False
        # 1. Transform one-hot encoded y_test into their class number
        if not binary:
            y_test = np.argmax(y_test,axis=1)
        
        # 2. Predict classes and stores 
        #y_pred = model.predict(X_test, batch_size=params['batch_size'])        
        y_pred = y_pred_proba.argmax(axis=1)
        
        predicted_y.extend(y_pred)
        expected_y.extend(y_test)
        histories.append(history.history)
        garbage_collection()

    del X, y, model, vect

    expected_y = np.array(expected_y)
    predicted_y = np.array(predicted_y)
    predicted_y_proba = np.array(predicted_y)

    np.save(directory + '/expected.numpy', expected_y)
    np.save(directory + '/predicted.numpy', predicted_y)
    np.save(directory + '/predicted_proba.numpy', predicted_y_proba)
    with open(directory + '/histories.pkl', 'wb') as f:
        pickle.dump(histories, f)

    # metrics    
    train_val_metrics(histories)

    # plot_history(histories, directory)
    
    # y_pred = model.predict(x, batch_size=batch_size)

    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(expected_y,predicted_y)))    
    print("F1-Score : "+ str(f1_score(expected_y,predicted_y,average="macro")))    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    report = pd.DataFrame(
        classification_report(expected_y, predicted_y, digits=3, target_names=classes_names, output_dict=True)
    )
    report = report.transpose()
    accuracy = accuracy_score(expected_y, predicted_y)
    report['accuracy'] = [accuracy] * (n_classes + 3)    
    report.to_csv(directory + '/report.csv')
    print(report)

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(expected_y,predicted_y)    
    np.save(directory + "/confusion_matrix", np.array(cnf_matrix))    
    plot_confusion_matrix(cnf_matrix, classes=classes_names, directory=directory, normalize=True)

    # 6. Clean memory
    garbage_collection()
    gc.collect()

    print("+"+"-"*50+"+")
    print()
        
import sys
if __name__ == '__main__':
    print(sys.argv)

    run_all = True

    #try:
    #task = sys.argv[1] or None    
    g_root              = sys.argv[1] or None

    filter_dataset_name = sys.argv[2] or None

    try:
        filter_task         = sys.argv[3] or None
    except:
        filter_task = None
    #lang = sys.argv[4] or None

    #run(task, dataset_name, g_root, lang)
    #except:
    #    run_all = True
    #    pass
    
    if run_all == True:
        args = []
        problems = listProblems(filter_dataset_name, filter_task)
        print("############################################")
        print(" RUNNING {0} PROBLEMS".format(len(problems)))

        # create a list of tasks
        for task, dataset_name, lang in problems:

            #args.append([task, dataset_name, g_root, lang])
            print(" Dataset: ",dataset_name," / Task:",task," / Lang:",lang)
            run(task, dataset_name, g_root, lang)
    

    

