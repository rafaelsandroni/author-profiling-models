
# coding: utf-8

# In[1]:


# In[1]:

import numpy as np
import pandas as pd

from Models.functions.plot import plot_history, full_multiclass_report
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




def create_model_simple(filters = [100], kernel_size = [50], strides = [100], 
                 dropout_rate = 0.5, pool_size = [5], dense_units = 512, max_len = 1000, n_classes = 2):

    model = Sequential()

    # conv 1
    model.add(Conv1D(filters = filters[0], 
                     kernel_size = kernel_size[0],
                     strides = strides[0], 
                     activation = 'relu', 
                     input_shape = (max_len, 1) ))

    # pooling layer 1
    
    model.add(MaxPooling1D(pool_size = pool_size[0], strides = 1))
    model.add(Activation('relu'))

    model.add(Flatten())
    
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(units = dense_units, activation = 'relu'))
    model.add(Dense(units = n_classes, activation = 'softmax'))

    #TODO: test others foss functions: https://keras.io/losses/
    model.compile(optimizer = 'adadelta', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model


def create_model(filters = [100], kernel_size = [50], strides = [100], 
                 dropout_rate = 0.5, pool_size = [5], dense_units = 512, max_len = 1000, n_classes = 2):

    model = Sequential()

    # conv 1
    model.add(Conv1D(filters = filters[0], 
                     kernel_size = kernel_size[0],
                     strides = strides[0], 
                     activation = 'relu', 
                     input_shape = (max_len, 1) ))

    # pooling layer 1
    
    model.add(MaxPooling1D(pool_size = pool_size[0], strides = 1))
    model.add(Activation('relu'))
    
    model.add(Conv1D(filters = filters[1], 
                     kernel_size = kernel_size[1],
                     strides = strides[0], 
                     activation = 'relu'))
    
    model.add(MaxPooling1D(pool_size = pool_size[1], strides = 1))
    model.add(Activation('relu'))

    model.add(Conv1D(filters = filters[2], 
                     kernel_size = kernel_size[2],
                     strides = strides[0], 
                     activation = 'relu'))
    
    model.add(MaxPooling1D(pool_size = pool_size[2], strides = 1))
    model.add(Activation('relu'))

    model.add(Flatten())
    
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
        
    model.add(Dense(units = dense_units, activation = 'relu'))
    model.add(Dense(units = n_classes, activation = 'softmax'))

    #TODO: test others foss functions: https://keras.io/losses/
    # sparse_categorical_entropy
    model.compile(optimizer = 'adadelta', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model


# In[5]:

def get_results(model, y_espected, y_predicted):

    config = model.get_config()

    row = {}

    conv_layers = np.sum([1 if i['class_name'] == "Conv1D" else 0 for i in config])
    pooling_layers = np.sum([1 if i['class_name'] == "MaxPooling1D" else 0 for i in config])

    row.update({ '_accuracy': accuracy_score(y_espected, y_predicted) })
    row.update({ '_f1-score': f1_score(y_espected, y_predicted,average='weighted')})
    row.update({ 'conv_layers': conv_layers })
    row.update({ 'pooling_layers': pooling_layers })

    _, _, fscore, support = precision_recall_fscore_support(y_espected, y_predicted)

    [row.update({'_fscore_class_'+str(i[0]): i[1]}) for i in enumerate(fscore)]
    [row.update({'_support_class_'+str(i[0]): i[1]}) for i in enumerate(support)]

    idx = 1
    for i in config:
        if i['class_name'] == "Conv1D":
            j = str(idx)
            row.update({
                'filters_'+j: i['config']['filters'],
                'strides_'+j: i['config']['strides'],
                'kernel_size_'+j: i['config']['kernel_size'],
                'activation_'+j: i['config']['activation']
            })
        pass
    return row


def vectorizer(X, max_features=None): return TfidfVectorizer(max_features=int(max_features)).fit_transform(X).toarray()

def garbage_collection(): 
    gc.collect()
    sleep(2)

# In[6]:

task = "gender"
dataset_name = "brblogset"
lang = "pt"
root = "/home/rafael/GDrive/Data/Dataframe/"

# Synthetic Minority Oversampling Technique (SMOTE)
def oversampling(X, y):
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    return X_resampled, y_resampled
    # return X, y

def run(task, dataset_name, root, lang):
    
    directory='./Reports/'+task+'/'+dataset_name+'/'
    checkFolder(directory)

    X, _, y, _ = loadTrainTest(task, dataset_name, root, lang)

    X = X.apply(clean, lang=lang)

    y, n_classes, classes_names = labelEncoder(y)    

    max_length = np.max([len(x.split(" ")) for x in X])
    mean_length = np.mean([len(x.split(" ")) for x in X])
    median_length = np.median([len(x.split(" ")) for x in X])

    print("max: ", max_length, " / mean: ", mean_length, " / median: ", median_length)

    X_tfidf = vectorizer(X, max_features=mean_length)

    X_resampled, y_resampled = oversampling(X_tfidf, y)

    print(collections.Counter(y), collections.Counter(y_resampled))

    garbage_collection() 
    #### Split train and test

    # In[9]:
    y_resampled = to_categorical(y_resampled, n_classes)

    X_resampled = np.reshape(X_resampled, (X_resampled.shape[0], X_resampled.shape[1], 1))

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2)
    # validation
    validation_split = 0.2
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_split)

    garbage_collection() 

    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    

    print("X", X_train.shape, X_test.shape, X_val.shape)
    print("y", y_train.shape, y_test.shape, y_val.shape)

    print(y_train[3])

    params = dict(
        filters = [128, 128, 128],
        kernel_size = [5, 5, 5],
        strides = [3, 3, 3],
        dropout_rate = 0.4,
        pool_size = [8, 8, 4],
        epochs = 50
    )
    
    ## create the model with the best params found
    model = create_model(
                        max_len=X_train.shape[1],
                        n_classes=n_classes,
                        filters=params['filters'],
                        kernel_size=params['kernel_size'],
                        strides=params['strides'],                        
                        dropout_rate=params['dropout_rate'],
                        pool_size=params['pool_size']
                        )
    model.summary()

    ## Then train it and display the results
    history = model.fit(X_train,
                        y_train,
                        epochs=params['epochs'],
                        validation_data=(X_val, y_val),
                        #batch_size=params['batch_size'],
                        verbose = 1,
                        callbacks=[
                            #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01),
                            EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
                    ])

    # save y_predicted


    plot_history(history, directory=directory)

    full_multiclass_report(model,
                        X_test,
                        y_test,
                        classes=classes_names,
                        directory=directory,
                        plot=True
                        )
                        #batch_size=32,
                        #binary= )
    batch_size = 32
    y_pred = model.predict(X_test, batch_size=batch_size)

    np.save(directory + "/y_predicted", y_pred)
    np.save(directory + "/y_expected", y_test)
    
    # results = results.append(pd.DataFrame(get_results(model)), ignore_index=True)

    # results.to_csv(results_dataframe)
        
import sys
if __name__ == '__main__':
    print(sys.argv)

    task = None
    run_all = False

    #try:
    task = sys.argv[1] or None
    dataset_name = sys.argv[2] or None
    g_root = sys.argv[3] or None
    lang = sys.argv[4] or None

    run(task, dataset_name, g_root, lang)
    #except:
    #    run_all = True
    #    pass
    """
    if run_all == True:
        args = []
        problems = listProblems()
        print("############################################")
        print(" RUNNING {0} PROBLEMS".format(len(problems)))

        # create a list of tasks
        for task, dataset_name, lang in problems:
            #args.append([task, dataset_name, g_root, lang])
            run(task, dataset_name, g_root, lang)
    """

    

