import os, glob
import numpy as np
from ..config import config

class DataGenerator():

    def __init__(self, data_path, k_fold, batch_size, in_len, out_len):

        self.k_fold = k_fold
        self.batch_size = batch_size
        self.in_len = in_len
        self.out_len = out_len
        self.current_k = 1
        self.files = sorted([file for file 
            in glob.glob(os.path.join(config['DATA_PATH'],'*/*.bin'))])
        self.n_files = len(self.files) - config['IN_LEN'] - config['OUT_LEN'] + 1
        self.current_idx = int(self.n_files / 2)
        self.n_test = int(self.n_files / 5)
        self.n_each = int(self.n_files - self.current_idx - self.n_test)
        self.shuffle()

    def set_k(self, k):
        self.current_k = min(k, self.k_fold)

    def get_train(self, i):
        
        #Clean previous by del
        #get new one and return
        return None, None

    def shuffle(self):
        self.indices = np.arange(self.current_idx)
        np.random.shuffle(self.indices)
        
    def n_train_batch(self):
        return int(np.ceil(self.current_idx/self.batch_size))

    def size(self):
        None