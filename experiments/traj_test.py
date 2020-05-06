import argparse
import datetime
import os, sys
import os.path as osp
sys.path.insert(0, '../')
import torch
import yaml

from collections import OrderedDict
from models.trajGRU import TrajGRU
from utils.generators import DataGenerator
from global_config import global_config
from summary.test import test
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF

save_dir = '/home/warit/senior/experiments/traj_logs/logs_4_10_2_False_05021847'
config = {
    'DEVICE': torch.device('cuda:0'),
    'CAL_DEVICE': torch.device('cuda:3'),
    'IN_LEN': 4,
    'OUT_LEN': 10,
    'BATCH_SIZE': 2,
    'SCALE': 0.2,
}

batch_size = 1

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 132, 102), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1), config=config),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 44, 34), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1), config=config),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 22, 17), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1), config=config)
    ]
]

forecaster_params = [
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
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 22, 17), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1), config=config),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 44, 34), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1), config=config),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 132, 102), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1), config=config)
    ]
]

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)

encoder = Encoder(encoder_params[0], encoder_params[1]).to(config['DEVICE'])
forecaster = Forecaster(forecaster_params[0], forecaster_params[1], config=config).to(config['DEVICE'])
encoder_forecaster = EF(encoder, forecaster).to(config['DEVICE'])

weight_path = save_dir + '/model_25500.pth'
encoder_forecaster.load_state_dict(torch.load(weight_path, map_location='cuda'))
test(encoder_forecaster, data_loader, config, save_dir, crop=None)