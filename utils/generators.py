import os, glob
import torch
import cv2
import numpy as np
from config import config
import itertools
from utils.units import mm_dbz, get_crop_boundary_idx
np.random.seed(42)

missings = [ 1630,  1629,  1628,  1627,  1626,  1625,  1624,  1623,  1622,
        1621,  1620,  1619,  1618,  1617,  1616,  1615,  1614,  1613,
        1612,  1611,  1610,  1609,  1608,  1607,  1701,  1700,  1699,
        1698,  1697,  1696,  1695,  1694,  1693,  1692,  1691,  1690,
        1689,  1688,  1687,  1686,  1685,  1684,  1683,  1682,  1681,
        1680,  1679,  1678, 13101, 13100, 13099, 13098, 13097, 13096,
       13095, 13094, 13093, 13092, 13091, 13090, 13089, 13088, 13087,
       13086, 13085, 13084, 13083, 13082, 13081, 13080, 13079, 13078,
       17421, 17420, 17419, 17418, 17417, 17416, 17415, 17414, 17413,
       17412, 17411, 17410, 17409, 17408, 17407, 17406, 17405, 17404,
       17403, 17402, 17401, 17400, 17399, 17398, 19444, 19443, 19442,
       19441, 19440, 19439, 19438, 19437, 19436, 19435, 19434, 19433,
       19432, 19431, 19430, 19429, 19428, 19427, 19426, 19425, 19424,
       19423, 19422, 19421, 20388, 20387, 20386, 20385, 20384, 20383,
       20382, 20381, 20380, 20379, 20378, 20377, 20376, 20375, 20374,
       20373, 20372, 20371, 20370, 20369, 20368, 20367, 20366, 20365,
       42051, 42050, 42049, 42048, 42047, 42046, 42045, 42044, 42043,
       42042, 42041, 42040, 42039, 42038, 42037, 42036, 42035, 42034,
       42033, 42032, 42031, 42030, 42029, 42028]

strides = 120
input_size = config['IMG_SIZE']
h_pos = [i for i in range(0, config['DATA_HEIGHT'] - input_size, strides)]
h_pos.append(config['DATA_HEIGHT'] - input_size)
w_pos = [i for i in range(0, config['DATA_WIDTH'] - input_size, strides)]
w_pos.append(config['DATA_WIDTH'] - input_size)
hw_pos = [[h, w] for h in h_pos for w in w_pos]
n_hw_pos = len(hw_pos)

class DataGenerator():

    def __init__(self, data_path, k_fold, batch_size, in_len, out_len, windows_size):

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
        sliced_data = np.zeros((self.windows_size, len(indices), input_size, input_size), dtype=np.float32)
        for i, [idx, ch] in enumerate(indices):
            for j in range(self.windows_size):
                h, w = hw_pos[ch]
                sliced_data[j, i] = np.fromfile(self.files[idx + j], dtype=np.float32).reshape((config['DATA_HEIGHT'], config['DATA_WIDTH']))[h : h + input_size, w : w + input_size]
                
        return mm_dbz(sliced_data)

    def get_train(self, i):

        if self.last_train is not None:
            del self.last_train
            torch.cuda.empty_cache()
            
        idx = self.train_indices[i * self.batch_size : min((i+1) * self.batch_size, self.train_indices.shape[0])]
        self.last_train = torch.from_numpy(self.get_data(idx)).to(config['DEVICE'])
        return self.last_train[:self.in_len,:,None], self.last_train[self.in_len:,:,None]

    def get_val(self, i):
        
        if self.last_val is not None:
            del self.last_val
            torch.cuda.empty_cache()

        idx = self.val_indices[i * self.batch_size : min((i+1) * self.batch_size, self.val_indices.shape[0])]
        self.last_val = torch.from_numpy(self.get_data(idx)).to(config['DEVICE'])
        return self.last_val[:self.in_len,:,None], self.last_val[self.in_len:,:,None]

    def shuffle(self):
        idx_train = np.arange(self.current_idx)
        idx_train = np.setdiff1d(idx_train, missings)
        idx_hw = np.arange(n_hw_pos)
        self.train_indices = np.hstack([np.repeat(idx_train[:,None], idx_hw.shape[0], axis=0), np.tile(idx_hw, idx_train.shape[0])[:,None]])
        np.random.shuffle(self.train_indices)

        val_size = int(self.n_val / self.k_fold)
        if self.k_fold == self.current_k:
            val_size += self.n_val % self.k_fold
        idx_val = np.arange(val_size) + self.current_idx
        idx_val = np.setdiff1d(idx_val, missings)
        idx_hw = np.arange(n_hw_pos)
        self.val_indices = np.hstack([np.repeat(idx_val[:,None], idx_hw.shape[0], axis=0), np.tile(idx_hw, idx_val.shape[0])[:,None]])
        
    def n_train_batch(self):
        return int(np.ceil(self.train_indices.shape[0]/self.batch_size))

    def n_val_batch(self):
        return int(np.ceil(self.val_indices.shape[0]/self.batch_size))