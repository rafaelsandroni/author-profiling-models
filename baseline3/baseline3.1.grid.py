import pandas as pd
import baseline3_new
from Models.functions.utils import listProblems

filter_task = None# "age"
filter_dataset_name = "b5post"
g_root = "/home/rafael/Dataframe/"
lang = "pt"
report_version = 'Grid1.0'

params1 = dict(
            features_maps = [100,100],
            kernel_size = [3,3],
            strides = [1,1],
            dropout_rate = None,#0.3,
            epochs = 100,
            batch_size = 32,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = 10
        )

params2 = dict(
            features_maps = [100,100],
            kernel_size = [3,3],
            strides = [1,1],
            dropout_rate = None,#0.3,
            epochs = 100,
            batch_size = 32,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = 50
        )
params3 = dict(
            features_maps = [100,100],
            kernel_size = [3,3],
            strides = [1,1],
            dropout_rate = None,#0.3,
            epochs = 100,
            batch_size = 32,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = 500
        )

list_params = [params1, params2, params3]

if __name__ == '__main__':

    rp = pd.DataFrame({"v": [], "n": [], "dataset": [], "task": [], "params": [], "acc": [], "f1": [], "cm": []})

    problems = listProblems(filter_dataset_name, filter_task)
    print("############################################")
    print(" RUNNING {0} PROBLEMS".format(len(problems)))

    for task, dataset_name, lang in problems:

        print(" Dataset: ",dataset_name," / Task:",task," / Lang:",lang)
        print(" Params: ", params)
        for n in range(len(n_params)):

            params = list_params[n]

            acc, f1, cm = 1,1,1#baseline3_new.run(task, dataset_name, g_root, lang, params, report_version)
            a = {
                "v": report_version,
                "n": n,
                "dataset": dataset_name,
                "task": task,
                "params": params,
                "acc": acc,                
                "f1": f1,
                "cm": cm
            }
            rp.append(a)

    rp.to_csv("results_grid_"+ report_version)


