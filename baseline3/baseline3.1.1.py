import baseline3_new
from Models.functions.utils import listProblems

filter_task = None# "age"
filter_dataset_name = "b5post"
g_root = "/home/rafael/Dataframe/"
lang = "pt"
report_version = '1.1'

params = dict(
            features_maps = [100,100],
            kernel_size = [3,3],
            strides = [1,1],
            dropout_rate = None,#0.3,
            epochs = 100,
            batch_size = 32,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = 50000#None
)

if __name__ == '__main__':


    problems = listProblems(filter_dataset_name, filter_task)
    print("############################################")
    print(" RUNNING {0} PROBLEMS".format(len(problems)))

    for task, dataset_name, lang in problems:

        print(" Dataset: ",dataset_name," / Task:",task," / Lang:",lang)
        baseline3_new.run(task, dataset_name, g_root, lang, params, report_version)

