from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from functions.plot import ROC
import pickle
import os

def checkFolder(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def saveModel(model, directory, filename):
    try:
        pickle.dump(model, open(directory + '/' + filename, 'wb'))
    except:
        pass

def loadModel(directory, filename):    
    loaded_model = pickle.load(open(directory + '/' + filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    return loaded_model

def reportPath(task, dataset_name):
    directory = './Reports/'+ task + '/' 
    filename = directory + dataset_name + '_report.txt'
    return filename
    
def evaluator(model, X_test, y_test, n_classes, task, dataset_name, f):
    
    directory = './Reports/'+ task + '/' 
    filename = dataset_name + '_model.pickle'
    checkFolder(directory)
    
    saveModel(model, directory, filename)
    
    print("Test evaluation")   
     
    #y_score = grid_search.decision_function(X_test)    
    
    y_pred = model.predict(X_test)
    
    #f = open(reportPath(task, dataset_name), 'a')
    print("Test evaluation", file=f)   
    print(classification_report(y_test, y_pred), file=f)    
    print("confusion matrix", file=f)
    print(confusion_matrix(y_test, y_pred), file=f)        
    #f.close()
    
    print(classification_report(y_test, y_pred))    
    print(confusion_matrix(y_test, y_pred))        
    
    y_score = model.predict_proba(X_test)
    ROC(y_test, y_score, n_classes, task, dataset_name)
    
    