from Models.functions.datasets import getDatasets
from Models.functions.plot import ROC, plot_confusion_matrix
from Models.functions.preprocessing import clean
from Models.functions.cnn_model import build_cnn, build_dnn

import sys

import keras, os, pickle, re, sklearn, string, tensorflow
print('Keras version: \t\t%s' % keras.__version__)
print('Scikit version: \t%s' % sklearn.__version__)
print('TensorFlow version: \t%s' % tensorflow.__version__)

import numpy as np
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

from nltk.corpus import stopwords

# Preprocessing

def labelEncoder(y):
    le = LabelEncoder()
    le.fit(y)

    return (le.transform(y), len(le.classes_), list(le.classes_))

def checkFolder(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def length(text):
    """
    Calculate the maximum document length
    """
    result = [len(x.split()) for x in text]
    return np.min(result), np.max(result), np.mean(result)


def transform_onehot(text, num_words = None, tmp = None, vectorizer = None):

    if vectorizer == None:
        vectorizer = TfidfVectorizer(min_df=0, max_df=0.9, max_features=num_words)
        X_onehot = vectorizer.fit_transform(text)
    else:
        X_onehot = vectorizer.transform(text)

    if num_words == None:
        num_words = len(vectorizer.get_feature_names())

    return X_onehot, num_words, None, vectorizer


def create_model(emb_layer = None, max_num_words = None, max_seq_length = None):
    
    model = build_dnn(
            num_words=max_num_words or MAX_NUM_words
    )

    optimizer = Adadelta(lr=1e-4)

    model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
    )
    return model

def nn(X, y):    

    histories = []
    test_loss = []
    test_accs = []

    predicted_y = []
    expected_y = []

    K = StratifiedKFold(n_splits=2)

    for train_index, test_index in K.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, _MAX_NUM_WORDS, _MAX_SEQ_LENGTH, vect = transform_onehot(X_train, MAX_NUM_WORDS, None, None)

        X_test, _, _, _ = transform_onehot(X_test, _MAX_NUM_WORDS, _MAX_SEQ_LENGTH, vect)
      
        model = KerasClassifier(build_fn=create_model, 
                            max_num_words=_MAX_NUM_WORDS,
                            max_seq_length=_MAX_SEQ_LENGTH,
                            epochs=NB_EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=1,
                            #validation_data=(X_val, y_val),
                            validation_split=0.1,
                            callbacks=[#ModelCheckpoint('model-%i.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
                                #ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=4, min_lr=0.01),
                                EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
                            ])

        history = model.fit(X_train, y_train)
        histories.append(history.history)

        y_pred = model.predict(X_test, verbose=1)
        predicted_y.extend(y_pred)
        expected_y.extend(y_test)

    train_val_metrics(histories)

    expected_y = np.asarray(expected_y)
    score_y = np.asarray(predicted_y) # probabilistics    
    predicted_y = np.asarray(predicted_y).round() # estimated
    
    return expected_y, predicted_y, score_y, histories
    
def run(task, dataset_name = None, root = None):

    datasets = getDatasets(task,'df', dataset_name, root)
    for i in datasets.iterrows():

        dataset_name = i[1]['dataset_name']
        print("Task {0} and Dataset {1}".format(task, dataset_name))
        label = task
        ds_path = i[1]['path']

        # load training and test dataframes
        training_path = ds_path + '/' + i[1]['training']        

        df_training = pd.read_csv(training_path)#, usecols=cols)        

        df_training['text'] = df_training['text'].apply(clean)
        X = df_training['text'].values
        y, n_classes, classes_name = labelEncoder(df_training[label].values)
        
        # cnn model
        (expected_y, predicted_y, score_y, histories) = nn(X, y)
        
        # save model
        directory = './Reports/'+task+'/'+dataset_name+'/'
        
        checkFolder(directory)
        
        with open(directory + '/histories_cnn1.pkl', 'wb') as f:
            pickle.dump(histories, f)
            
        # save arrays        
        np.save(directory + '/expected_cnn1.numpy', expected_y)
        np.save(directory + '/predicted_cnn1.numpy', predicted_y)
        np.save(directory + '/score_cnn1.numpy', score_y)
        
        evaluate(expected_y, predicted_y, score_y, classes_name, n_classes, task, dataset_name)

        
def evaluate(expected_y, predicted_y, score_y, classes_name, n_classes, task, dataset_name):

      
    
    # compute ROC curve
    try:
        roc_c = ROC(expected_y, score_y, n_classes, task, dataset_name, classes_name)
        report['roc'] = list(roc_c.values()) + [roc_c['macro']] * 3
    except:
        pass

    # compute accuracy
    accuracy = accuracy_score(expected_y, predicted_y)
    report['accuracy'] = [accuracy] * (n_classes + 3)

    # compute confusion matrix
    c_matrix = confusion_matrix(expected_y, predicted_y)
    print("confusion-matrix")
    print(c_matrix)
    plot_confusion_matrix(c_matrix, classes_name, task, dataset_name, True)
    cm = pd.DataFrame(c_matrix, columns=classes_name, index=classes_name)

    directory = './Reports/' + task + '/' + dataset_name + '/'
    report.to_csv(directory + 'report.csv')
    cm.to_csv(directory + 'confusion_matrix.csv')
    
    print(report)
    print()

def train_val_metrics(histories):
    print('Training: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'loss'), get_avg(histories, 'acc')))
    print('Validation: \t%0.4f loss / %0.4f acc' % (get_avg(histories, 'val_loss'), get_avg(histories, 'val_acc')))

def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)
    
if __name__ == '__main__':    

    task = sys.argv[1]
    dataset_name = sys.argv[2]
    root = sys.argv[3]

    global MAX_NUM_WORDS, MAX_SEQ_LENGTH

    # EMBEDDING
    MAX_NUM_WORDS  = 5000 #15000
    EMBEDDING_DIM  = 300
    MAX_SEQ_LENGTH = 3200 #200
    USE_GLOVE      = False

    # MODEL
    FILTER_SIZES   = [3,4,5]
    FEATURE_MAPS   = [10,10,10]
    DROPOUT_RATE   = 0.5

    # LEARNING
    BATCH_SIZE     = 16
    NB_EPOCHS      = 40
    RUNS           = 5
    VAL_SIZE       = 0.2
    
    print(task, dataset_name, root)

    run(task, dataset_name, root)
