import pandas as pd
import baseline3_new
from Models.functions.utils import listProblems
import copy

filter_task = None
filter_dataset_name = "pan13"
g_root = "/home/rafael/Dataframe/"
g_lang = "en"
report_version = '_grid'
rp_file = "./Grid/"+ filter_dataset_name+"_"+report_version+".csv"

#brmoral (turned on age task)
params = dict(
            features_maps = [100,100],
            kernel_size = [15,15],
            strides = [2,2],
            dropout_rate = 0.5,
            epochs = 100,
            batch_size = 32,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = 20,
            optimizer = None
        )
params_pan13 = dict(
        features_maps = [100,100,100],
        kernel_size = [15,15,15],
        strides = [1,1,1],
        dropout_rate = 0.5,
        epochs = 100,
        batch_size = 32,
        embedding_dim = 100,
        max_seq_length = None,
        max_num_words = 1000,
        optimizer = None
        )
#max_num_words = [ 20, 500 ]
features_maps = [1]
kernel_size = [[15,15,15],[4,5,6],[7,8,9],[10,15,20]]

# set params
list_params = []
tunning = "kernel_size"
for i in range(len(kernel_size)):
    params1 = copy.deepcopy(params_pan13)
    #params1["features_maps"] = features_maps[i]    
    #params1["max_num_words"] = max_num_words[j]
    params1["kernel_size"] = kernel_size[i]
    list_params.append(params1)
    
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

            acc, f1, cm = baseline3_new.run(task, dataset_name, g_root, lang, parameters, report_version)
            a = {
                "v": report_version,
                "tunning": tunning,
                "n": n,
                "dataset": dataset_name,
                "task": task,
                "params": parameters,
                "acc": acc,                
                "f1": f1,
                "cm": cm
            }
            rp = rp.append(a, ignore_index=True)
    
            rp.to_csv(rp_file, index=False)


