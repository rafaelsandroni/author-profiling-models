from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import os

def checkFolder(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory)

def ROC(y_test, y_score, n_classes, task, dataset_name):

    # Compute ROC curve and ROC area for each class
    y_test_bkp = y_test
    y_test = label_binarize(y_test, classes=[x for x in range(n_classes)])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = 1 if n_classes == 2 else n_classes

    for i in range(n_classes):    
        if n_classes > 2:
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:,i])
        else:
            fpr[i], tpr[i], _ = roc_curve(y_test_bkp, y_score[:,1])

        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                 lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Receiver operating characteristic')
        plt.legend(loc="lower right")
        
        directory = './Reports/'+ task + '/' + dataset_name + '/'
        
        checkFolder(directory)
        filename =  dataset_name + '_ROC_curve_class_'+ str(i) +'.pdf'
        filename = directory + filename
        
        plt.savefig(filename)        

    if n_classes > 2:

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
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - Receiver operating characteristic')
        plt.legend(loc="lower right")  
        
        directory = './Reports/'+ task + '/' + dataset_name + '/'
        checkFolder(directory)
        filename =  dataset_name + '_ROC_curve.pdf'
        filename = directory + filename
        
        plt.savefig(filename)
        
    return roc_auc
