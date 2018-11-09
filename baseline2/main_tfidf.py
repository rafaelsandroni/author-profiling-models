
# coding: utf-8

# In[1]:


#get_ipython().magic('matplotlib inline')

from Models.functions.plot import ROC, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


from Models.functions.datasets import getDatasets

from Models.functions.preprocessing import clean
import pandas as pd
from nltk.corpus import stopwords
import re, string
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from Models.functions.plot import ROC, plot_confusion_matrix


# In[3]:


from bs4 import BeautifulSoup

def labelEncoder(y):
    le = LabelEncoder()
    le.fit(y)

    return (le.transform(y), len(le.classes_), list(le.classes_))

from nltk.corpus import stopwords
import re, string
import numpy as np


# In[4]:


task = 'age'
dataset_name = 'brblogset'
MAX_FEATURES = 5000
datasets = getDatasets(task,'df', dataset_name)
for i in datasets.iterrows():

    name = i[1]['dataset_name']
    label = task
    ds_path = i[1]['path']

    # load training and test dataframes
    training_path = ds_path + '/' + i[1]['training']        
    #test_path = ds_path + '/' + i[1]['test']      

    df_training = pd.read_csv(training_path)#, usecols=cols)        
    #df_test = pd.read_csv(test_path)#, usecols=cols)        

    df_training['text'] = df_training['text'].apply(clean)
    #df_test['text'] = df_test['text'].apply(clean)
    X = df_training['text'].values
    y, n_classes, classes_name = labelEncoder(df_training[label].values)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
vec = TfidfVectorizer(max_features=MAX_FEATURES).fit(X_train)
X_train = vec.transform(X_train)
X_test = vec.transform(X_test)


import cnn_model

# EMBEDDING
MAX_NUM_WORDS  = 50000 #15000
EMBEDDING_DIM  = 300
MAX_SEQ_LENGTH = X_train.shape[1]# or 3200 #200
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

def create_model():
    
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


# In[ ]:


#import cnn_model
#from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

histories = []
test_loss = []
test_accs = []

predicted_y = []
expected_y = []

#K = StratifiedKFold(n_splits=2)

X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

emb_layer = None
#if USE_GLOVE:
    #emb_layer = create_glove_embeddings()


model = KerasClassifier(build_fn=create_model, 
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=0,
                        #validation_data=(X_val, y_val),
                        validation_split=0.2,
                        callbacks=[#ModelCheckpoint('model-%i.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.01),
                            EarlyStopping(monitor='val_loss', min_delta=0.1, patience=4, verbose=1)
                        ])

history = model.fit(X_train, y_train)


# # Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=classes_name))
print()
cm = confusion_matrix(y_test, y_pred)
print(cm)
plot_confusion_matrix(cm, classes_name)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=classes_name))
print()
cm = confusion_matrix(y_test, y_pred)
print(cm)
plot_confusion_matrix(cm, classes_name)
y_score = model.predict_proba(X_test)
# ROC(y_test, y_score, n_classes, None, None, classes_name)
