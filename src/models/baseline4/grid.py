import pandas as pd
import baseline4
from Models.functions.utils import listProblems
import copy

filter_task = ['education']#,'profession','region']#None#['gender','age']
filter_dataset_name = "brblogset"
#g_root = r"C:/Users/Rafael Sandroni/Google Drive/Mestrado/Data/Dataframe/"
g_root = r"/home/rafael/Dataframe/"
g_lang = "pt"

report_version = '_grid'

if report_version is None:
    rp_file = r"Results/"+ filter_dataset_name+".csv"
else:    
    rp_file = r"Grid/"+ filter_dataset_name+"_"+ report_version +".csv"

tunning = 'kbest'

#brmoral (turned on age task)
""" Results parameters
brmoral: (short texts and small data)
TWO CLASSES:
    kernel_size = [[7,8]]
    features_maps = [[50,50]]
    batch_size = 8
    max_num_words = 500
    max_seq_length = 302
    dense_units = 512

THREE CLASSES:
    kernel_size = [[7,8]]
    features_maps = [[50,50]]
    batch_size = 8
    max_num_words = 500
    max_seq_length = 302
    dense_units = 512

b5post:
params = dict(
            features_maps = [10,10,10],
            kernel_size = [3,4,5],
            strides = [1,1,1],
            dropout_rate = 0.2,
            epochs = 100,
            batch_size = 2,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = 1000,
            dense_units = [128],
            optimizer = None,
            pool_size = [1,1,1]
        )

smscorpus: (too short texts)

pan13: (too large data)

esic: (too large data)
"""
params = dict(
            features_maps = [50, 50],
            kernel_size = [3,4],
            strides = [1,1,1],
            dropout_rate = 0.2,
            epochs = 1000,
            batch_size = 32,
            embedding_dim = 100,
            max_seq_length = 10000,
            max_num_words = None,
            dense_units = [512],
            optimizer = None,
            pool_size = [1,1,1],
            lr = 0.00
        )

#max_num_words = [ 20, 500 ]
#strides = [1]
#embedding_dim = [100]
# set params
optimizer = ['rmsprop']#,'sgd']#'adadelta','adam','sgd','rmsprop']
lr = [1e-4, 1e-5]

list_params = []

for i in range(len(optimizer)):        
    for j in range(len(lr)):
        #for emb in range(len(embedding_dim)):
        params1 = copy.deepcopy(params)
        params1["optimizer"] = optimizer[i]
        params1["lr"] = lr[j]
        list_params.append(params1)

print("params", len(list_params))
print(list_params)

import os
if __name__ == '__main__':

    if os.path.exists(rp_file):
        rp = pd.read_csv(rp_file)
    else:
        rp = pd.DataFrame({"v": [], "tunning": [], "n": [], "dataset": [], "task": [], "params": [], "acc": [], "f1": [], "cm": []})

    problems = listProblems(filter_dataset_name, filter_task)
    print("############################################")
    print(" RUNNING {0} PROBLEMS".format(len(problems)))

    for task, dataset_name, lang in problems:
        if lang != g_lang: continue

        print(" Dataset: ",dataset_name," / Task:",task," / Lang:",lang)
        for n in range(len(list_params)):
            print(n, list_params[n])
            parameters = list_params[n]
            import baseline4
            report_version = '_grid_'+parameters['optimizer']

            acc, f1, cm, loss, val_loss = baseline4.run(task, dataset_name, g_root, lang, parameters, report_version)

            baseline4 = None

            a = {
                "v": report_version,
                "tunning": tunning,
                "n": n,
                "dataset": dataset_name,
                "task": task,
                "lang": lang,
                "params": parameters,
                "acc": acc,                
                "f1": f1,
                "cm": cm,
                "loss": loss,
                "val_loss": val_loss,
                "embeddings": "nilc-word2vec-50dim"
            }
            rp = rp.append(a, ignore_index=True)
    
            rp.to_csv(rp_file, index=False)


