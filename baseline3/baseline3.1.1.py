import baseline3_new.run as run


task = "gender"
dataset_name = "pan13"
g_root = "/home/rafael/Dataframe/"
lang = "en"
report_version = '1.1'

params = dict(
            features_maps = [10],
            kernel_size = [3],
            strides = [1],
            dropout_rate = 0.3,
            epochs = 100,
            batch_size = 32,
            embedding_dim = 100,
            max_seq_length = None,
            max_num_words = None,
        )

run(task, dataset_name, g_root, lang, params, report_version)
