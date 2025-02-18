import os
import sys
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool


# np.random.seed(2018)
# random.seed(2018)
# tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    # setting working directory
    root_folder = './'

    # Reading the config file.
    conf = Configurator(root_folder + "NeuRec.properties", default_section="hyperparameters")
    
    # Setting the seed for the random number generator.
    seed = conf["seed"]
    print('seed=', seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Setting the GPU to use.
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]
    dataset = Dataset(conf)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]

    with tf.compat.v1.Session(config=config) as sess:
        if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.general_recommender." + recommender)

        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.social_recommender." + recommender)
            
        else:
            my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.compat.v1.global_variables_initializer())
        # tf.compat.v1.summary.FileWriter("output", sess.graph)
        model.train_model()
