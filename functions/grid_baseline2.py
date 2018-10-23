import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# from sklearn.learning_curve import validation_curve
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# REPLACED from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

from time import time

# custom
from functions.datasets import getDatasets


import nltk

from nltk.corpus import stopwords

pt_stopwords = stopwords.words('portuguese')
en_stopwords = stopwords.words('english')
all_stopwords = en_stopwords + pt_stopwords

def roc(y_test, y_score, n_classes = 2):
    
    y_test = label_binarize(y_test, classes=[x for x in range(n_classes)])

    print(y_test.shape, y_score.shape)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # fpr[i], tpr[i], _ = roc_curve(y_test, y_score, pos_label=1)
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


#print(list(le.inverse_transform([1, 2])))

def reglog(X_train, X_test, y_train, y_test, n_classes):
    # classifier
    clf = LogisticRegression(verbose=1)
    
    # params
    params_grid = dict(
            clf__C = np.linspace(1e-4, 1e4, num=8),
            clf__penalty = ['l1','l2'],                  
            vect__max_df = [0.8, 0.9, 1.0],
            vect__max_features = [None, 50, 300, 1000, 3000])
            # vect__ngram_range = [(1, 1), (3, 5)],          
    
    # pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words=pt_stopwords)),
        ('tfidf', TfidfTransformer()),
        ('smote', SMOTE()),
        ('clf', clf),
    ])    
        
    return grid(X_train, X_test, y_train, y_test, n_classes, pipeline, params_grid)

def grid(X_train, X_test, y_train, y_test, n_classes, pipeline = None, params_grid = None):
    
    if pipeline == None or params_grid == None:
        print("Pipeline not defined") if pipeline == None else 0
        print("Params not defined") if params_grid == None else 0
        return
    
    grid_search = GridSearchCV(pipeline, params_grid, scoring='accuracy')
    
    #print('best params', best_model.best_params_)
    #print('best scores', best_model.best_score_)
    print("Performing grid search...")    
    print("Pipeline steps:", [name for name, _ in pipeline.steps])    
    t0 = time()
    
    grid_search.fit(X_train, y_train)
    print("done in %0.2fs and %0.1fmin" % ((time() - t0), ((time() - t0) / 60) ))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    y_score = grid_search.decision_function(X_test)
    
    print()
    y_pred = grid_search.predict(X_test) 
    print()
    # Saving accuracy score in table
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average='weighted')
    #cm = result_table.iloc[j,3] = str(confusion_matrix(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    #result_table.iloc[j,4] = classification_report(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    print('Accuracy', accuracy_score(y_test,y_pred))
    print('F1_score', f1_score(y_test,y_pred,average='weighted'))       
    print(confusion_matrix(y_test, y_pred))
    
    # plot curve ROC
    try:
        print("ROC", y_test.shape, y_score.shape)
        roc(y_test, y_score, n_classes)
    except:
        print("Error plotting ROC curve")
        pass
    
    #return result_table
    return (acc, f1, cm)

#pd.DataFrame(results).to_csv('/home/rafael/drive/Models/Reports/baseline1-reglog-tfidf.csv')