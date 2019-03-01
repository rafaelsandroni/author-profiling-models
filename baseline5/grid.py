import pandas as pd
import baseline5
from Models.functions.utils import listProblems
import copy

filter_task = ['education']#None#['age','education']
filter_dataset_name = 'esic'#None#'esic'#,'esic','b5post','brmoral']
#g_root = r"C:/Users/Rafael Sandroni/Google Drive/Mestrado/Data/Dataframe/"
g_root = r"/home/rafael/Dataframe/"
#g_lang = "pt"
report_version = '_grid'
tunning = 'dot => multiply'

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

enblogs: (long texts and large data)
    kernel_size = 
    features_maps = 
    batch_size = 
    max_num_words = 
    max_seq_length = 
    dense_units = 

smscorpus: (too short texts)

pan13: (too large data)

esic: (too large data)
"""
params = dict(
            features_maps = [10,10],
            kernel_size = [3,4],
            strides = [1],
            dropout_rate = 0.2,
            epochs = 100,
            batch_size = 2,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = None,
            dense_units = [128],
            optimizer = None,
            pool_size = [3,3,3]
        )

list_params = []
list_params.append(params)

"""
#max_num_words = [ 20, 500 ]
kernel_size = [[2,3,4],[3,4,5],[3,4],[2,4]]
features_maps1 = [[10],[50]]
features_maps2 = [[10,10],[50,50]]
features_maps3 = [[10,10,10],[50,50,50]]
max_num_words = [1000, 15000]
max_seq_length = [None]
#strides = [1]
#embedding_dim = [100]
# set params
list_params = []
#list_params.append(params)

for i in range(len(kernel_size)):

    if len(kernel_size[i]) == 1:
        features_maps = features_maps1
    elif len(kernel_size[i]) == 2:
        features_maps = features_maps2
    elif len(kernel_size[i]) == 3:
        features_maps = features_maps3

    for j in range(len(features_maps)):
        for words in range(len(max_num_words)):
            for seq in range(len(max_seq_length)):
                #for emb in range(len(embedding_dim)):
                params1 = copy.deepcopy(params)
                params1["kernel_size"] = kernel_size[i]
                #params1["embedding_dim"] = embedding_dim[emb]
                params1["features_maps"] = features_maps[j]
                params1["max_num_words"] = max_num_words[words]    
                params1["max_seq_length"] = max_seq_length[seq]
                list_params.append(params1)
    
print("params", len(list_params))
print(list_params)
"""

import os
if __name__ == '__main__':


    problems = listProblems(filter_dataset_name, filter_task)
    print("############################################")
    print(" RUNNING {0} PROBLEMS".format(len(problems)))

    for task, dataset_name, lang in problems:
        #if lang != g_lang: continue
        if dataset_name == 'enblog': continue

        if report_version is None:
            rp_file = r"Results/"+ dataset_name+".csv"
        else:    
            rp_file = r"Grid/"+ dataset_name+"_"+ report_version +".csv"

        if os.path.exists(rp_file):
            rp = pd.read_csv(rp_file)
        else:
            rp = pd.DataFrame({"v": [], "tunning": [], "n": [], "dataset": [], "task": [], "params": [], "acc": [], "f1": [], "cm": []})

        print(" Dataset: ",dataset_name," / Task:",task," / Lang:",lang)
        for n in range(len(list_params)):
            print(n, list_params[n])
            parameters = list_params[n]
            import baseline5
            
            acc, f1, cm, loss, val_loss = baseline5.run(task, dataset_name, g_root, lang, parameters, report_version)

            baseline5 = None

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
                "embeddings": "nilc-fasttext-100dim"
            }
            rp = rp.append(a, ignore_index=True)
    
            rp.to_csv(rp_file, index=False)


