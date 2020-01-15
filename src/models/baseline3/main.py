from Models.functions.datasets import loadTrainTest
from Models.functions.plot import ROC, plot_confusion_matrix
from Models.functions.cnn_model import build_cnn, build_dnn, build_simple_cnn
from Models.functions.plot import plot_history, full_multiclass_report
from Models.functions.preprocessing import clean, labelEncoder, oversampling
from Models.functions.utils import checkFolder, listProblems
from Models.functions.transform import transform
from Models.functions.vectors import train_vectors, create_embeddings

import sys

import keras, os, pickle, re, sklearn, string, tensorflow
print('Keras version: \t\t%s' % keras.__version__)
print('Scikit version: \t%s' % sklearn.__version__)
print('TensorFlow version: \t%s' % tensorflow.__version__)

import numpy as np 
from numpy import zeros, newaxis

import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.layers import Embedding, Dense
from keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk


#from tqdm import tqdm
#tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils
from gensim.models import KeyedVectors

def create_model(emb_layer = None, max_num_words = None, max_seq_length = None):
    
    # CNN
    model = build_cnn(
            embedding_layer=emb_layer,
            num_words=max_num_words or MAX_NUM_WORDS,
            embedding_dim=EMBEDDING_DIM,
            filter_sizes=FILTER_SIZES,
            feature_maps=FEATURE_MAPS,
            max_seq_length=max_seq_length or MAX_SEQ_LENGTH,
            dropout_rate=DROPOUT_RATE
    )

    optimizer = Adadelta(clipvalue=3)

    model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
    )
    return model

def nn(X, y, directory='./Reports/tmp/'):    

    histories = []
    test_loss = []
    test_accs = []

    predicted_y = []
    expected_y = []

    emb_layer = None

    #K = StratifiedKFold(n_splits=3)

    #for train_index, test_index in K.split(X, y):

        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)    

    X_train, _MAX_NUM_WORDS, _MAX_SEQ_LENGTH, vect = transform(X_train, MAX_NUM_WORDS, MAX_SEQ_LENGTH)    
    X_test, _, _, _ = transform(X_test, _MAX_NUM_WORDS, _MAX_SEQ_LENGTH, vect)

    if True:
        X_train, y_train = oversampling(X_train, y_train)
        X_test,  y_test  = oversampling(X_test,  y_test)

    # validation
    validation_split = 0.1
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_split)

    # create embedding layer
    embedding_layer = create_embeddings(X, _MAX_NUM_WORDS, _MAX_SEQ_LENGTH, vect)

    # create model
    model = KerasClassifier(build_fn=create_model, 
                        emb_layer=embedding_layer,
                        max_num_words=_MAX_NUM_WORDS,
                        max_seq_length=_MAX_SEQ_LENGTH,
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=0,
                        validation_data=(X_val, y_val),
                        callbacks=[#ModelCheckpoint('model-%i.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01),
                            EarlyStopping(monitor='val_loss', min_delta=0.005, patience=4, verbose=1)
                        ])
    # model fitting
    history = model.fit(X_train, y_train)    

    # predict probabilities
    y_predicted_proba = model.predict(X_test, verbose=1)

    # save train/val plots
    plot_history(histories, directory=directory)
    
    # save metrics
    full_multiclass_report(model,
                        X_test,
                        y_test,
                        classes=classes_names,
                        directory=directory,
                        plot=True
                        )
                        #batch_size=32,
                        #binary= )
    
    np.save(directory + "/history", history)
    np.save(directory + "/y_predicted_proba", y_predicted_proba)
    np.save(directory + "/y_test", y_test)

    return expected_y, predicted_y, score_y, histories

def run(task, dataset_name, root, lang):
    
    directory='./Reports/'+task+'/'+dataset_name+'/'
    checkFolder(directory)

    X, _, y, _ = loadTrainTest(task, dataset_name, root, lang)

    X = X.apply(clean, lang=lang)

    y, n_classes, classes_names = labelEncoder(y)    

    max_length = np.max([len(x.split(" ")) for x in X])
    mean_length = np.mean([len(x.split(" ")) for x in X])
    median_length = np.median([len(x.split(" ")) for x in X])
    
    if not os.path.exists('/home/rafael/Embeddings/'+dataset_name):
        train_vectors(X, y)

    # cnn model
    nn(X, y, directory=directory)
    """
    (expected_y, predicted_y, score_y, histories) = nn(X, y)
    
    # save model
    
    checkFolder(directory)
    
    with open(directory + '/histories_cnn1.pkl', 'wb') as f:
        pickle.dump(histories, f)
        
    # save arrays        
    np.save(directory + '/expected_cnn1.numpy', expected_y)
    np.save(directory + '/predicted_cnn1.numpy', predicted_y)
    np.save(directory + '/score_cnn1.numpy', score_y)
    
    evaluate(expected_y, predicted_y, score_y, histories, classes_name, n_classes, task, dataset_name)
    """

def train_val_metrics(histories):
    print('Training: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'loss'), get_avg(histories, 'acc')))
    print('Validation: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'val_loss'), get_avg(histories, 'val_acc')))

def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)
    
if __name__ == '__main__':    

    global MAX_NUM_WORDS, MAX_SEQ_LENGTH, g_task, g_dataset_name, g_root
    run_all = True

    g_root = root = sys.argv[1]
    """
    g_task = task = sys.argv[1]
    g_dataset_name = dataset_name = sys.argv[2]
    g_lang = dataset_name = sys.argv[3]
    """
    

    # EMBEDDING
    MAX_NUM_WORDS  = None
    EMBEDDING_DIM  = 10
    MAX_SEQ_LENGTH = None
    USE_EMBEDDINGS = True

    # MODEL
    FILTER_SIZES   = [1,2,3]
    FEATURE_MAPS   = [10,10,10]
    DROPOUT_RATE   = 0.5

    # LEARNING
    BATCH_SIZE     = 100
    NB_EPOCHS      = 1000
    RUNS           = 5
    VAL_SIZE       = 0.2
    
    
    if run_all == True:
        args = []
        problems = listProblems()
        print("############################################")
        print(" RUNNING {0} PROBLEMS".format(len(problems)))

        # create a list of tasks
        for task, dataset_name, lang in problems:
            #args.append([task, dataset_name, g_root, lang])
            print("dataset:",dataset_name,"task:",task,"lang:",lang)
            run(task, dataset_name, g_root, lang)
