from Models.functions.datasets import getDatasets
from Models.functions.plot import ROC, plot_confusion_matrix

import keras, os, pickle, re, sklearn, string, tensorflow
# print('Keras version: \t\t%s' % keras.__version__)
# print('Scikit version: \t%s' % sklearn.__version__)
# print('TensorFlow version: \t%s' % tensorflow.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.layers import Embedding
from keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import cnn_model
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Preprocessing

def labelEncoder(y):
    le = LabelEncoder()
    le.fit(y)

    return (le.transform(y), len(le.classes_), list(le.classes_))

def clean(doc):
    """
    Cleaning a document by several methods:
        - Lowercase
        - Removing whitespaces
        - Removing numbers
        - Removing stopwords
        - Removing punctuations
        - Removing short words
    """
    stop_words = set(stopwords.words('portuguese'))
    
    # Lowercase
    doc = doc.lower()
    # Remove numbers
    doc = re.sub(r"[0-9]+", "", doc)
    # Split in tokens
    tokens = doc.split()
    # Remove Stopwords
    tokens = [w for w in tokens if not w in stop_words]
    # Remove punctuation
    tokens = [w.translate(str.maketrans('', '', string.punctuation)) for w in tokens]
    # Tokens with less then two characters will be ignored
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)


def read_files(path):
    documents = list()
    # Read in all files in directory
    if os.path.isdir(path):
        for filename in os.listdir(path):
            with open('%s/%s' % (path, filename)) as f:
                doc = f.read()
                doc = clean_doc(doc)
                documents.append(doc)
    
    # Read in all lines in a txt file
    if os.path.isfile(path):        
        with open(path, encoding='iso-8859-1') as f:
            doc = f.readlines()
            for line in doc:
                documents.append(clean_doc(line))
    return documents
    
def checkFolder(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def max_length(lines):
    """
    Calculate the maximum document length
    """
    return max([len(s.split()) for s in lines])


def transform(text):

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    length = max_length(text)
    word_index = tokenizer.word_index

    # result = [len(x.split()) for x in text]
    # print('Text informations:')
    # print('max length: %i / min length: %i / mean length: %i / limit length: %i' % (np.max(result),
    #                                                                                 np.min(result),
    #                                                                                 np.mean(result),
    #                                                                                 MAX_SEQ_LENGTH))
    # print('vocabulary size: %i / limit: %i' % (len(word_index), MAX_NUM_WORDS))

    # Padding all sequences to same length of `MAX_SEQ_LENGTH`
    X = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
    return X
    

def create_model(emb_layer):
    
    model = cnn_model.build_cnn(
            embedding_layer=emb_layer,
            num_words=MAX_NUM_WORDS,
            embedding_dim=EMBEDDING_DIM,
            filter_sizes=FILTER_SIZES,
            feature_maps=FEATURE_MAPS,
            max_seq_length=MAX_SEQ_LENGTH,
            dropout_rate=DROPOUT_RATE
    )
    
    model.compile(
            loss='binary_crossentropy',
            optimizer=Adadelta(clipvalue=3),
            metrics=['accuracy']
    )
    return model
    
def cnn1(X, y):    

    histories = []
    test_loss = []
    test_accs = []

    predicted_y = []
    expected_y = []

    emb_layer = None
    #if USE_GLOVE:
        #emb_layer = create_glove_embeddings()
    model = create_model(emb_layer)
    #K = StratifiedKFold(n_splits=2)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    history = model.fit(
        X_train, y_train,
        epochs=NB_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[ModelCheckpoint('model-%i.h5', monitor='val_loss',
                                   verbose=0, save_best_only=True, mode='min'),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01),
                   #EarlyStopping(monitor='val_loss', min_delta=0.1, patience=4, verbose=1)
                  ]
    )

    score = model.evaluate(X_val, y_val, verbose=1)
    test_loss.append(score[0])
    test_accs.append(score[1])

    y_pred = model.predict(X_val, verbose=1)

    predicted_y.extend(y_pred)
    expected_y.extend(y_val)

    histories.append(history.history)
    
    expected_y = np.asarray(expected_y)
    score_y = np.asarray(predicted_y) # probabilistics
    predicted_y = np.asarray(predicted_y).round() # estimated
    
    return expected_y, predicted_y, score_y, histories
    
def run(task, dataset_name = None):

    datasets = getDatasets(task,'df', dataset_name)
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
        
        X = transform(X)
        
        # cnn model
        (expected_y, predicted_y, score_y, histories) = cnn1(X, y)
        
        # save model
        directory = './Reports/'+task+'/'+dataset_name+'/'
        
        checkFolder(directory)
        
        with open(directory + '/history_cnn1.pkl', 'wb') as f:
            pickle.dump(histories, f)
            
        # save arrays        
        np.save(directory + '/expected_cnn1.numpy', expected_y)
        np.save(directory + '/predicted_cnn1.numpy', predicted_y)
        
        evaluate(expected_y, predicted_y, score_y, classes_name, n_classes, task, dataset_name)

        
def evaluate(expected_y, predicted_y, score_y, classes_name, n_classes, task, dataset_name):

    # report
    report = pd.DataFrame(
        classification_report(expected_y, predicted_y, digits=5, target_names=classes_name, output_dict=True)
    )
    report = report.transpose()    
    
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
    plot_confusion_matrix(c_matrix, classes_name, task, dataset_name, True)
    cm = pd.DataFrame(c_matrix, columns=classes_name, index=classes_name)

    directory = './Reports/' + task + '/' + dataset_name + '/'
    report.to_csv(directory + 'report.csv')
    cm.to_csv(directory + 'confusion_matrix.csv')    
    
    print(report)
    
    
    
    
if __name__ == '__main__':    


    # EMBEDDING
    MAX_NUM_WORDS  = 50000 #15000
    EMBEDDING_DIM  = 300
    MAX_SEQ_LENGTH = 3200 #200
    USE_GLOVE      = False

    # MODEL
    FILTER_SIZES   = [3,4,5]
    FEATURE_MAPS   = [10,10,10]
    DROPOUT_RATE   = 0.5

    # LEARNING
    BATCH_SIZE     = 20
    NB_EPOCHS      = 40
    RUNS           = 5
    VAL_SIZE       = 0.2
    
    # run('relig')
    
    # run('education')
    
    # run('professional')
    
    # run('region')
    
    # run('polit')
    
    run('age', 'enblogs')
    
    # run('gender')
