
# coding: utf-8

# In[1]:


# In[1]:

import numpy as np
import pandas as pd
import pickle
import os
from Models.functions.plot import plot_history, full_multiclass_report, plot_confusion_matrix
from Models.functions.preprocessing import clean, labelEncoder
from Models.functions.datasets import loadTrainTest
from Models.functions.utils import checkFolder, listProblems

from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding, multiply
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.recurrent import GRU
from keras import regularizers
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.optimizers import Adadelta, RMSprop, SGD, Adam


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

# In[4]:

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import layers
from Models.functions.cnn_model import build_cnn1

from Models.functions.transform import tokenizer_pad_sequence
from Models.functions.vectors import create_embeddings, train_vectors


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras



from keras.layers import merge, concatenate, multiply, dot
from keras.layers.core import *
from keras.layers.recurrent import LSTM, GRU
from keras.models import *

from attention_utils import get_activations, get_data_recurrent

INPUT_DIM = 2
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

def attention_3d_block(inputs, input_dim):
    # inputs.shape = (batch_size, time_steps, input_dim)
    print(inputs.shape)
    
    # input_dim = int(inputs.shape[2])
    
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, ))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if False:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = concatenate([inputs, a_probs], axis=1, name='attention_mul')
    # output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    output_attention_mul = multiply([inputs, a_probs], axes=1, name='attention_mul', normalize=False)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    
    return output_attention_mul

def model_attention_applied_after_lstm(embedding_layer, max_seq_length, units = 32, n_classes = 2):
    # old: inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    inputs = Input(shape=(max_seq_length,), dtype='int32')  
    
    emb_layer = embedding_layer(inputs)

    lstm_units = units
    lstm_out = LSTM(lstm_units, return_sequences=True)(emb_layer)
    #attention_mul = attention_3d_block(lstm_out, max_seq_length)
    #attention_mul = Flatten()(attention_mul)
    #dropout = Dropout(0.12)(attention_mul)
    flatten = Flatten()(lstm_out)
    dropout = Dropout(0.12)(flatten)
    deep = Dense(512, activation='relu')(dropout)
    output = Dense(n_classes, activation='softmax')(deep)
    model = Model(input=[inputs], output=output)
    return model

def create_rnn(embedding_layer, max_seq_length, units, n_classes):
    print(max_seq_length)
    
    m = model_attention_applied_after_lstm(embedding_layer, max_seq_length, units, n_classes)
    return m

"""
if __name__ == '__main__':

    N = 300000
    # N = 300 -> too few = no training
    inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)

    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm()

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(300):
        testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=2).squeeze()
        print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()
"""



def garbage_collection(): 
    gc.collect()
    print("gargabe colletion...")
    sleep(3)

# In[6]:

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
    return (get_avg(histories, 'loss'), get_avg(histories, 'val_loss'), get_avg(histories, 'acc'),get_avg(histories, 'val_acc'))

def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)
    
def run(task, dataset_name, root, lang, params = None, report_version = None):

    if params == None:
        params = dict(
            features_maps = [100,10,10],
            kernel_size = [3,4,5],
            strides = [1,1,1],
            dropout_rate = 0.5,
            epochs = 2,
            batch_size = 32,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = None,
        )
    
    histories = []
    test_loss = []
    test_accs = []

    predicted_y = []
    predicted_y_proba = []
    expected_y = []

    if report_version != None:
        directory = r'Reports'+ str(report_version) +'/'+task+'/'+dataset_name+'_'+lang+'/'
    else:    
        directory = r'Reports/'+task+'/'+dataset_name+'_'+lang+'/'

    checkFolder(directory)

    X, _, y, _ = loadTrainTest(task, dataset_name, root, lang)
    print("original len(X)", len(X))

    # small sample
    if len(X) > 5000:
        # X, _, y, _ = train_test_split(X, y, train_size = 5000)
        print("sample len(X)", len(X))

    X = X.apply(clean, lang=lang)
    X = X.values # mandatory for pan13

    y, n_classes, classes_names = labelEncoder(y)    
    params['n_classes'] = n_classes

    max_length = np.max([len(x.split(" ")) for x in X])
    mean_length = np.mean([len(x.split(" ")) for x in X])
    median_length = np.median([len(x.split(" ")) for x in X])

    if mean_length < 50:
        mean_length = 50
    
    MAX_NUM_WORDS = params['max_num_words']

    if params['max_seq_length'] == None:
        MAX_SEQ_LENGTH = int(mean_length)
    else:
        MAX_SEQ_LENGTH = params['max_seq_length']

    print("max: ", max_length, " / mean: ", mean_length, " / median: ", median_length, " / DEFINED: ", MAX_SEQ_LENGTH)

    # if word vectors is not created for the dataset
    #if not os.path.exists('/home/rafael/GDrive/Embeddings/'+dataset_name):
    if not os.path.exists(r'C:/Users/Rafael Sandroni/Google Drive/Mestrado/Data/Embeddings/'+dataset_name):
        
        #train_vectors(X, name=dataset_name, embedding_dim=params['embedding_dim'])
        pass

    K = StratifiedKFold(n_splits=3)
    idx = 0

    _, _, _, vect_all  = tokenizer_pad_sequence(X, MAX_NUM_WORDS,  MAX_SEQ_LENGTH)    

    # 0. Define cross validation KFolds
    for train_index, test_index in K.split(X, y):
        # avoid two or more loops

        # 1. Define train and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]        
        vect = None        
        # 2. Oversampling
        #X_test,  y_test  = oversampling(X_test, y_test)

        # 3. Categorical labels
        #y_train = to_categorical(y_train, n_classes)
        #y_test  = to_categorical(y_test, n_classes)
        
        # 4. Define validation set
        validation_split = 0.1
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_split)

        # 5. Create matrix of words and pad sequences
        X_train, _MAX_NUM_WORDS, _MAX_SEQ_LENGTH, vect  = tokenizer_pad_sequence(X_train, MAX_NUM_WORDS,  MAX_SEQ_LENGTH)    
        X_val, _, _, _                                  = tokenizer_pad_sequence(X_val,  _MAX_NUM_WORDS, _MAX_SEQ_LENGTH, vect)
        X_test, _, _, _                                 = tokenizer_pad_sequence(X_test, _MAX_NUM_WORDS, _MAX_SEQ_LENGTH, vect)
        
        X_train, y_train = oversampling(X_train, y_train)
        X_test, y_test = oversampling(X_test, y_test)
        X_val, y_val = oversampling(X_val, y_val)

        y_train = to_categorical(y_train, n_classes)
        y_test = to_categorical(y_test, n_classes)
        y_val = to_categorical(y_val, n_classes)

        # 6. Create the embedding layer from trained vectors

        # 7. Update params
        params['max_seq_length'] = _MAX_SEQ_LENGTH
        if params['max_num_words'] == None:
            params['max_num_words'] = _MAX_NUM_WORDS + 1000
        
        # vectors_filename = '\home/rafael/GDrive/Embeddings/fasttext_skip_s100.txt'
        # if params['embedding_type'] is not None and params['embedding_type'] == 1:
        # LINUX
        if lang == 'es' and dataset_name == 'pan13':
            ds_name = 'pan13_train_es'
        else:
            ds_name = dataset_name

        # vectors_filename = r'/home/rafael/GDrive/Embeddings/fasttext/'+ ds_name +'_sg_'+ str(params['embedding_dim']) +'dim.model'
        # vectors_filename = r'/home/rafael/GDrive/Embeddings/en_wordvectors/wiki-news-300d-1M.vec'
        # vectors_filename = r'/home/rafael/GDrive/Embeddings/nilc/fasttext_pt_skip_s'+ str(params['embedding_dim']) +r'.txt'        
        # WINDOWS
        #vectors_filename = r'C:/Users/Rafael Sandroni/Google Drive/Mestrado/Data/Embeddings/fasttext/'+dataset_name+r'_sg_'+ str(params["embedding_dim"]) + 'dim.model'
        vectors_filename = r'C:/Users/Rafael Sandroni/Google Drive/Mestrado/Data/Embeddings/word2vec/'+dataset_name+r'_sg_'+ str(params["embedding_dim"]) + 'dim.model'
        # vectors_filename = r'/home/rafael/GDrive/Embeddings/nilc/fasttext_pt_skip_s'+ str(params['embedding_dim']) +r'.txt'        
        embedding_type = 1

        embedding_layer = create_embeddings(vect_all, params['max_num_words'], params['max_seq_length'], name=dataset_name, embedding_dim=params['embedding_dim'], filename=vectors_filename, type=embedding_type)

        # 8. Create the CNN model with the best params        
        model = None        
        #build_cnn1
        model = create_rnn(embedding_layer=embedding_layer,
                           max_seq_length=params['max_seq_length'],
                           units=params['units'],
                           n_classes=params['n_classes']
                          )
        
        if params['optimizer'] == 'adadelta':
            optimizer = Adadelta(lr=params['lr'])
        elif params['optimizer'] == 'rmsprop':
            optimizer = RMSprop(lr=params['lr'])
        elif params['optimizer'] == 'sgd':
            optimizer = SGD(lr=params['lr'])
        else:
            optimizer = Adam(lr=params['lr'])


        model.compile(
                loss='categorical_crossentropy',
                optimizer=params['optimizer'],
                metrics=['accuracy']
        )

        model.summary()

        # 8. Train the model
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
        
        # 9. Get predict probabilistic results
        y_pred_proba = model.predict(X_test, batch_size=params['batch_size'])
        predicted_y_proba.extend(y_pred_proba)

        binary = False # True if len(classes) < 3 else False
        # 10. Transform one-hot encoded y_test into their class number (get max value index from each instance)
        if not binary:
            y_test = np.argmax(y_test,axis=1)
        
        # 11. Get predict classes
        #y_pred = model.predict_classes(X_test, batch_size=params['batch_size'])
        y_pred = np.argmax(y_pred_proba,axis=1)

        # 12. Add results
        predicted_y.extend(y_pred)
        expected_y.extend(y_test)
        histories.append(history.history)

        # 13. Clean cache
        garbage_collection()

    # 13. Clean cache
    del X, y, model, vect
    
    # 14. Transform results to numpy instance (Fortran)
    expected_y = np.array(expected_y)
    predicted_y = np.array(predicted_y)
    predicted_y_proba = np.array(predicted_y)

    # 15. Store results
    np.save(directory + '/expected.numpy', expected_y)
    np.save(directory + '/predicted.numpy', predicted_y)
    np.save(directory + '/predicted_proba.numpy', predicted_y_proba)
    with open(directory + '/histories.pkl', 'wb') as f:
        pickle.dump(histories, f)

    # 16. Show metrics from training/validation model performance    
    loss, val_loss, _, _ = train_val_metrics(histories)

    # plot_history(histories, directory)
    
    # y_pred = model.predict(x, batch_size=batch_size)

    # 17. Print accuracy score
    acc = accuracy_score(expected_y,predicted_y)
    print("Accuracy : "+ str(acc))    
    f1 = f1_score(expected_y,predicted_y,average="macro")
    print("F1-Score : "+ str(f1))
    print("")
    
    # 18. Classification report
    print("Classification Report")
    report = pd.DataFrame(
        classification_report(expected_y, predicted_y, digits=3, target_names=classes_names, output_dict=True)
    )
    report = report.transpose()
    accuracy = accuracy_score(expected_y, predicted_y)
    report['accuracy'] = [accuracy] * (n_classes + 3)    
    report.to_csv(directory + '/report.csv')
    print(report)

    # 19. Plot confusion matrix
    cnf_matrix = confusion_matrix(expected_y,predicted_y)    
    np.save(directory + "/confusion_matrix", np.array(cnf_matrix))    
    plot_confusion_matrix(cnf_matrix, classes=classes_names, directory=directory, normalize=True)

    # 20. Clean
    garbage_collection()
    gc.collect()

    print("+"+"-"*50+"+")
    print()
    return acc, f1, cnf_matrix, loss, val_loss
        
import sys
if __name__ == '__main__':
    print(sys.argv)

    run_all = True

    #try:
    #task = sys.argv[1] or None    
    g_root = sys.argv[1] or None

    filter_dataset_name = sys.argv[2] or None

    try:
        filter_task = sys.argv[3] or None
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
            acc, f1, cnf_matrix = run(task, dataset_name, g_root, lang)
    

    

