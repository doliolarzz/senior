import sys
sys.path.insert(0, '../')
import torch
import os, glob
import numpy as np
import cv2
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from config import config
from utils.predictor import prepare_testing, get_data, generate_weight_train_data
from utils.evaluators import fp_fn_image_csi, cal_rmse_all
from utils.visualizers import make_gif_color, rainfall_shade
from utils.units import mm_dbz, dbz_mm
from collections import OrderedDict
from models.convLSTM import ConvLSTM
from weight_data_model import train_weight_model

batch_size = config['BATCH_SIZE']
IN_LEN = config['IN_LEN']
OUT_LEN = config['OUT_LEN']

convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1),
    ]
]

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(config['DEVICE'])
forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(config['DEVICE'])
model = EF(encoder, forecaster).to(config['DEVICE'])
model.load_state_dict(
    torch.load('/home/warit/senior/experiments/logs_in5_out1/model_f1_i4000.pth', map_location='cuda'))

train_weight_model(model, 10, crop=None, epochs=10, learning_rate=1e-6)