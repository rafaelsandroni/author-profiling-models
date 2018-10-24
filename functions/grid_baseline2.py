import pandas as pd
#import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# from sklearn.learning_curve import validation_curve
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline

from time import time

# custom
from functions.datasets import getDatasets
from functions.metrics import evaluator, reportPath

import nltk
from nltk.corpus import stopwords

pt_stopwords = stopwords.words('portuguese')
en_stopwords = stopwords.words('english')
all_stopwords = en_stopwords + pt_stopwords

from sklearn.pipeline import Pipeline
from gensim.sklearn_api import W2VTransformer
import gensim

from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin

from gensim.test.utils import common_texts
from gensim.sklearn_api import W2VTransformer

class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model, dim = 100):
        self.word2vec = []
        self.dim = dim
        
        if model != None:
            self.word2vec = dict(zip(model.wv.index2word, model.wv.syn0))
            self.dim = model.wv.vector_size

    def fit(self, X, y):
        model = gensim.models.Word2Vec(X, size=self.dim, window=5, min_count=1, workers=4)
        self.word2vec = dict(zip(model.wv.index2word, model.wv.syn0))        
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model, dim = 100):
        
        self.word2vec = []
        self.dim = dim
        
        if model != None:
            self.word2vec = dict(zip(model.wv.index2word, model.wv.syn0))
            self.dim = model.wv.vector_size
        
        self.word2weight = None

    def fit(self, X, y):
        model = gensim.models.Word2Vec(X, size=self.dim, window=5, min_count=1, workers=4)
        self.word2vec = dict(zip(model.wv.index2word, model.wv.syn0))
        
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

class Model:

    def __init__(self, task, dataset_name, f):
        self.task = task
        self.dataset_name = dataset_name
        self.f = f
        self.n_classes = 0
    
    def tokens(self, X):
        return np.asarray([nltk.word_tokenize(x) for x in X])
        
    def word2vec(self, X_tokens_list, dim = 100):        
        return gensim.models.Word2Vec(X_tokens_list, size=dim, window=5, min_count=1, workers=4)
        
    def mlp(self, X_train, y_train, X_test, y_test, n_classes):
    
        self.n_classes = n_classes
        
        print("MLPClassifier w/ Embeddings")
        print("MLPClassifier w/ Embeddings", file=self.f)

        # classifier    
        clf = MLPClassifier()    

        # w2v 
        X_train = self.tokens(X_train)
        X_test = self.tokens(X_test)
        
        w2v = self.word2vec(X_train)
        
        # params
        params_grid = dict(
                clf__activation = ['tanh','relu'],
                clf__solver = ['lbfgs','adam'],
                clf__max_iter = [200,500,1000],
                clf__hidden_layer_sizes = [(5,2),(30, 2),(50,10)],
                clf__early_stopping = [True],
                clf__alpha = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                w2v__dim = [10, 50, 100])
                #clf__penalty = ['l1','l2'],                  
                # w2v__word2vec = [dict(zip(w2v.wv.index2word, w2v.wv.syn0))])
                #vect__max_features = [None, 3000])            

        # What is the vector representation of the word 'graph'?
        #wordvecs = model.fit(common_texts).transform(['graph', 'system'])
        
        # pipeline
        pipeline = Pipeline([
            # ('vect', CountVectorizer(stop_words=pt_stopwords)),            
            # ('tfidf', TfidfTransformer()),
            # ('smote', SMOTE()),
            ('w2v', TfidfEmbeddingVectorizer(model=w2v)),
            #('word2vec', W2VTransformer(size=10, min_count=1, seed=1)),
            # ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('clf', clf),
        ])

        return self.grid(X_train, y_train, X_test, y_test, pipeline, params_grid)

    def grid(self, X_train, y_train, X_test, y_test, pipeline = None, params_grid = None):

        if pipeline == None or params_grid == None:
            print("Pipeline not defined") if pipeline == None else 0
            print("Params not defined") if params_grid == None else 0
            return

        gridsearch = GridSearchCV(pipeline, params_grid, scoring='accuracy')    

        print("Performing grid search...", file=self.f)    
        print("Pipeline steps:", [name for name, _ in pipeline.steps], file=self.f)    
        t0 = time()

        gridsearch.fit(X_train, y_train)
        
        print("done in %0.2fs and %0.1fmin" % ((time() - t0), ((time() - t0) / 60) ), file=self.f)
        print(file=self.f)
        print("done in %0.2fs and %0.1fmin" % ((time() - t0), ((time() - t0) / 60) ))

        print("Best score: %0.3f" % gridsearch.best_score_, file=self.f)
        
        print("Best score: %0.3f" % gridsearch.best_score_)
        
        print("Best parameters set:", file=self.f)
        print("Best parameters set:")
        try:
            best_parameters = gridsearch.best_estimator_.get_params()
            for param_name in sorted(params_grid.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]), file=self.f)
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        except:
            pass

        evaluator(gridsearch, X_test, y_test, self.n_classes, self.task, self.dataset_name, self.f)
