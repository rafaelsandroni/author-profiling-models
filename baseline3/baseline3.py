
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

# In[4]:

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

#from tqdm import tqdm
#tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils


def train_vectors(X, y):

    all_x_w2v = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(X)]

    cores = multiprocessing.cpu_count()
    # CBOW
    """
    model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

    %%time
    for epoch in range(30):
        model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
        model_ug_cbow.alpha -= 0.002
        model_ug_cbow.min_alpha = model_ug_cbow.alpha
    """
    #SKIPGRAM

    model_ug_sg = Word2Vec(sg=1, size=EMBEDDING_DIM, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    model_ug_sg.build_vocab([x.words for x in all_x_w2v])

    for epoch in range(30):
        model_ug_sg.train(utils.shuffle([x.words for x in all_x_w2v]), total_examples=len(all_x_w2v), epochs=1)
        model_ug_sg.alpha -= 0.002
        model_ug_sg.min_alpha = model_ug_sg.alpha

    #model_ug_cbow.save('/content/w2v_model_ug_cbow.word2vec')
    model_ug_sg.save(EMB_DIR+'/'+g_dataset_name+'_w2v_model_ug_sg_'+str(EMBEDDING_DIM)+'.word2vec')


def create_embeddings(text, max_num_words, max_seq_length, tokenizer):

    print('training embeddings...')

    #model_ug_cbow = KeyedVectors.load('/content/w2v_model_ug_cbow.word2vec')
    model_ug_sg = KeyedVectors.load(EMB_DIR + '/'+g_dataset_name+'_w2v_model_ug_sg_'+str(EMBEDDING_DIM)+'.word2vec')

    print("Vocab keys", len(model_ug_sg.wv.vocab.keys()))

    embeddings_index = {}
    for w in model_ug_sg.wv.vocab.keys():
        #embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
        embeddings_index[w] = model_ug_sg.wv[w]

    print('Found %s word vectors.' % len(embeddings_index))
    
    num_words = max_num_words
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("weights", len(embedding_matrix))

    return Embedding(input_dim=max_num_words, output_dim=EMBEDDING_DIM,
                     input_length=max_seq_length,
                     weights=[embedding_matrix],
                     trainable=True
                    )

# In[5]:


def transform(text, max_num_words = None, max_seq_length = None, tokenizer = None):


    if tokenizer == None:
        tokenizer = Tokenizer(num_words=max_num_words)
        tokenizer.fit_on_texts(text)

    sequences = tokenizer.texts_to_sequences(text)

    _, max_length, mean_length = length(text)
    word_index = tokenizer.word_index

    # MAX_SEQ_LENGTH = np.max(arr_length)
    if max_seq_length == None:
        max_seq_length = int(mean_length)

    if max_num_words == None:
        max_num_words = len(word_index)

    result = [len(x.split()) for x in text]
    print('Text informations:')
    print('max length: %i / min length: %i / mean length: %i / limit length: %i' % (np.max(result), np.min(result), np.mean(result), max_seq_length))
    print('vocabulary size: %i / limit: %i' % (len(word_index), max_num_words))

    # Padding all sequences to same length of `max_seq_length`
    X = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

    return X, max_num_words, max_seq_length, tokenizer
    

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

def garbage_collection(): 
    gc.collect()
    sleep(2)

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

    # In[9]:a

    y_resampled = to_categorical(y_resampled, n_classes)

    X_resampled = np.reshape(X_resampled, (X_resampled.shape[0], X_resampled.shape[1], 1))

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2)
    # validation
    validation_split = 0.1
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_split)

    garbage_collection() 

    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    

    print("X", X_train.shape, X_test.shape, X_val.shape)
    print("y", y_train.shape, y_test.shape, y_val.shape)

    params = dict(
        filters = [10, 10, 10],
        kernel_size = [3,4,5],
        strides = [1, 1, 1],
        dropout_rate = 0.4,
        pool_size = [4, 4, 4],
        epochs = 100,
        batch_size = 32
    )
    
    ## create the model with the best params found
    model = KerasClassifier(build_fn=create_model, 
                            emb_layer=embedding_layer,
                            max_num_words=_MAX_NUM_WORDS,
                            max_seq_length=mean_length,
                            epochs=params['epochs'],
                            verbose=0,
                            callbacks=[#ModelCheckpoint('model-%i.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
                                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01),
                                EarlyStopping(monitor='val_loss', min_delta=0.005, patience=4, verbose=1)
                            ])
    model.summary()

    ## Then train it and display the results
    history = model.fit(X_train,
                        y_train,
                        epochs=params['epochs'],
                        validation_data=(X_val, y_val),
                        batch_size=params['batch_size'],
                        verbose = 1,
                        callbacks=[
                            #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01),
                            EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
                    ])

    # save y_predicted


    plot_history(history, directory=directory)
    
    batch_size = 32
    y_predicted_proba = model.predict(X_test, batch_size=batch_size)

    full_multiclass_report(model,
                        X_test,
                        y_test,
                        classes=classes_names,
                        directory=directory,
                        plot=True
                        )
                        #batch_size=32,
                        #binary= )
    
    np.save(directory + "/y_predicted_proba", y_predicted_proba)
    np.save(directory + "/y_test", y_test)
    # results = results.append(pd.DataFrame(get_results(model)), ignore_index=True)

    # results.to_csv(results_dataframe)
        
import sys
if __name__ == '__main__':
    print(sys.argv)

    run_all = True

    EMB_DIR = '/home/rafael/Embeddings/'
    #try:
    #task = sys.argv[1] or None
    #dataset_name = sys.argv[2] or None
    g_root = sys.argv[1] or None
    #lang = sys.argv[4] or None

    #run(task, dataset_name, g_root, lang)
    #except:
    #    run_all = True
    #    pass

    # EMBEDDING
    MAX_NUM_WORDS  = None
    EMBEDDING_DIM  = 10
    MAX_SEQ_LENGTH = None
    USE_EMBEDDINGS = True

    # MODEL
    FILTER_SIZES   = [2,3,4]
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

