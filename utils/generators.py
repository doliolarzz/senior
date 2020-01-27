import os, glob
import numpy as np
from ..config import config

class DataGenerator():

    def __init__(self, data, k_fold, batch_size, in_len, out_len, windows_size):

        self.k_fold = k_fold
        self.batch_size = batch_size
        self.in_len = in_len
        self.out_len = out_len
        self.current_k = 1
        self.data = torch.from_numpy(data.astype(np.float32)).to(config['DEVICE'])
        self.slided_data = self.data.unfold(0, windows_size, 1).permute(3, 0, 1, 2)[:, :, None, :]
        self.n_files = self.slided_data.shape[1]
        self.set_k(1)
        self.n_test = int(self.n_files / 5)
        self.n_val = self.n_files - int(self.n_files / 2) - self.n_test
        self.last_train = None
        self.last_val = None

    def set_k(self, k):
        self.current_k = max(min(k, self.k_fold), 1)
        self.current_idx = int(self.n_files / 2) + int(self.n_val * (k - 1) / self.k_fold)
        self.shuffle()

    def get_train(self, i):

        if self.last_train is not None:
            del self.last_train[0]
            del self.last_train[1]
            torch.cuda.empty_cache()

        idx = self.train_indices[i * self.batch_size : min((i+1) * self.batch_size, self.train_indices)]
        self.last_train = (self.slided_data[:self.in_len, idx], self.slided_data[self.in_len:, idx])
        return self.last_train

    def get_val(self, i):
        
        if self.last_val is not None:
            del self.last_val[0]
            del self.last_val[1]
            torch.cuda.empty_cache()

        idx = self.val_indices[i * self.batch_size : min((i+1) * self.batch_size, self.val_indices)]
        self.last_val = (self.slided_data[:self.in_len, idx], self.slided_data[self.in_len:, idx])
        return self.val_indices

    def shuffle(self):
        self.train_indices = np.arange(self.current_idx)
        np.random.shuffle(self.train_indices)

        val_size = int(self.n_val / self.k_fold)
        if self.k_fold == self.current_k:
            val_size += self.n_val % self.k_fold
        self.val_indices = np.arange(val_size) + self.current_idx
        
    def n_train_batch(self):
        return int(np.ceil(self.current_idx/self.batch_size))

    def n_val_batch(self):
        return int(np.ceil(self.current_idx/self.batch_size))