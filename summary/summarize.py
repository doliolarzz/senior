import sys, os, glob
sys.path.insert(0, '../')
import torch
import numpy as np
from global_config import global_config
from conv_test import conv_test
import pandas as pd

start_pred_files = ['20190630_2000.bin']
start_pred_crop = [[31, 37, 127, 142]]

for i, sfile, scrop in enumerate(zip(start_pred_files, start_pred_crop)):
    for f in glob.glob('/home/warit/senior/experiments/conv_logs/logs_*'):
        
        model_path = f + '/model_f1_i5000.pth'
        in_len, out_len, batch_size, multitask= f.split('/')[-1].split('_')[1:][:4]
        [rmse, rmse_rain, rmse_non_rain], csi, csi_multi = \
            conv_test(model_path, sfile, in_len, out_len, batch_size, multitask, crop=scrop)
        #Should be in Jupyter
    