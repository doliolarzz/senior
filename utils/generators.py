import os, glob
import torch
import cv2
import numpy as np
from global_config import global_config
import itertools
from utils.units import mm_dbz, get_crop_boundary_idx
np.random.seed(42)

class DataGenerator():

    def __init__(self, data_path, k_fold, batch_size, in_len, out_len, windows_size, config):

        self.config = config
        h_pos = [i for i in range(0, global_config['DATA_HEIGHT'] - global_config['IMG_SIZE'], global_config['STRIDE'])]
        h_pos.append(global_config['DATA_HEIGHT'] - global_config['IMG_SIZE'])
        w_pos = [i for i in range(0, global_config['DATA_WIDTH'] - global_config['IMG_SIZE'], global_config['STRIDE'])]
        w_pos.append(global_config['DATA_WIDTH'] - global_config['IMG_SIZE'])
        self.hw_pos = [[h, w] for h in h_pos for w in w_pos]
        self.n_hw_pos = len(self.hw_pos)

        self.k_fold = k_fold
        self.batch_size = batch_size
        self.in_len = in_len
        self.out_len = out_len
        self.windows_size = windows_size
        self.current_k = 1
        self.files = sorted([file for file in glob.glob(data_path)])
        self.n_files = len(self.files) - windows_size + 1
        self.n_test = int(self.n_files / 5)
        self.n_val = self.n_files - int(self.n_files / 2) - self.n_test
        self.set_k(1)
        self.last_train = None
        self.last_val = None

    def set_k(self, k):
        self.current_k = max(min(k, self.k_fold), 1)
        self.current_idx = int(self.n_files / 2) + int(self.n_val * (k - 1) / self.k_fold)
        self.shuffle()

    def get_data(self, indices):
        sliced_data = np.zeros((self.windows_size, len(indices), global_config['IMG_SIZE'], global_config['IMG_SIZE']), dtype=np.float32)
        for i, [idx, ch] in enumerate(indices):
            for j in range(self.windows_size):
                h, w = self.hw_pos[ch]
                sliced_data[j, i] = np.fromfile(self.files[idx + j], dtype=np.float32).reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))[h : h + global_config['IMG_SIZE'], w : w + global_config['IMG_SIZE']]
                
        return mm_dbz(sliced_data)

    def get_train(self, i):

        if self.last_train is not None:
            del self.last_train
            torch.cuda.empty_cache()
            
        idx = self.train_indices[i * self.batch_size : min((i+1) * self.batch_size, self.train_indices.shape[0])]
        self.last_train = torch.from_numpy(self.get_data(idx)).to(self.config['DEVICE'])
        return self.last_train[:self.in_len,:,None], self.last_train[self.in_len:,:,None]

    def get_val(self, i):
        
        if self.last_val is not None:
            del self.last_val
            torch.cuda.empty_cache()

        idx = self.val_indices[i * self.batch_size : min((i+1) * self.batch_size, self.val_indices.shape[0])]
        self.last_val = torch.from_numpy(self.get_data(idx)).to(self.config['DEVICE'])
        return self.last_val[:self.in_len,:,None], self.last_val[self.in_len:,:,None]

    def shuffle(self):
        idx_train = np.arange(self.current_idx)
        idx_train = np.setdiff1d(idx_train, global_config['MISSINGS'])
        idx_hw = np.arange(self.n_hw_pos)
        self.train_indices = np.hstack([np.repeat(idx_train[:,None], idx_hw.shape[0], axis=0), np.tile(idx_hw, idx_train.shape[0])[:,None]])
        np.random.shuffle(self.train_indices)

        val_size = int(self.n_val / self.k_fold)
        if self.k_fold == self.current_k:
            val_size += self.n_val % self.k_fold
        idx_val = np.arange(val_size) + self.current_idx
        idx_val = np.setdiff1d(idx_val, global_config['MISSINGS'])
        idx_hw = np.arange(self.n_hw_pos)
        self.val_indices = np.hstack([np.repeat(idx_val[:,None], idx_hw.shape[0], axis=0), np.tile(idx_hw, idx_val.shape[0])[:,None]])
        
    def n_train_batch(self):
        return int(np.ceil(self.train_indices.shape[0]/self.batch_size))

    def n_val_batch(self):
        return int(np.ceil(self.val_indices.shape[0]/self.batch_size))