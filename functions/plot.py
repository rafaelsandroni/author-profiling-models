from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
from Models.functions.utils import checkFolder

def ROC(y_test, y_score, n_classes, task = None, dataset_name = None, classes_name = None):
    
    if classes_name is None:
        classes_name = [i for i in n_classes]
        
    # Compute ROC curve and ROC area for each class
    y_test_bkp = y_test
    y_test = label_binarize(y_test, classes=[x for x in range(n_classes)])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # n_classes = 1 if n_classes == 2 else n_classes

    for i in range(n_classes):    
        if n_classes > 2:
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:,i])
        else:
            fpr[i], tpr[i], _ = roc_curve(y_test_bkp, y_score[:,i])

        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                 lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(classes_name[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Receiver operating characteristic')
        plt.legend(loc="lower right")
        if task != None and dataset_name != None:
            directory = './Reports/'+ task + '/' + dataset_name + '/'        
            checkFolder(directory)
            filename =  'ROC_curve_class_'+ str(classes_name[i]) +'.pdf'
            filename = directory + filename
        
            plt.savefig(filename)        
        else:
            #plt.show()
            pass
            
        plt.gcf().clear()



    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2

    # Plot all ROC curves
    plt.figure()
    #plt.plot(fpr["micro"], tpr["micro"],
             #label='micro-average ROC curve (area = {0:0.2f})'
                   #''.format(roc_auc["micro"]),
             #color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes_name[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - Receiver operating characteristic')
    plt.legend(loc="lower right")  
    
    if task != None and dataset_name != None:
        directory = './Reports/'+ task + '/' + dataset_name + '/'
        checkFolder(directory)
        filename =  'ROC_curve.pdf'
        filename = directory + filename

        plt.savefig(filename)
    else:
        #plt.show()
        pass
        
    plt.gcf().clear()
    
    return roc_auc


def plot_confusion_matrix(cm, classes, directory = '/', normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()    

    plt.savefig(directory + 'confusion_matrix.pdf')
    plt.gcf().clear()



def plot_history(history, directory = ''):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## check directory folder
    checkFolder(directory)

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Erro de treinamento (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Erro de validação (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Erro')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    plt.legend()
    plt.savefig(directory + '/loss.pdf')
    #plt.show()
    #plt.gcf().clear()

    ## F2
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Acurácia de treinamento (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Acurácia de validação (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(directory + '/accuracy.pdf')

    #plt.show()
    plt.gcf().clear()
    

    
## multiclass or binary report
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           directory='',
                           plot=True,
                           batch_size=32,
                           binary=False):

    binary = True if len(classes) < 3 else False
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    c_r = classification_report(y_true,y_pred,digits=5)
    print(c_r)
    pd.DataFrame(classification_report(y_true,y_pred,digits=5,output_dict=True)).transpose().to_csv(directory +'/classification_report.csv')

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    np.save(directory + "/confusion_matrix", np.array(cnf_matrix))
    plot_confusion_matrix(cnf_matrix, classes=classes, directory=directory, normalize=True)
    

