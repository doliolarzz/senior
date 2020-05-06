import argparse
import datetime
import os, sys, glob
import os.path as osp
sys.path.insert(0, '..')
import torch
import yaml

from collections import OrderedDict
from models.convLSTM import ConvLSTM
from utils.generators import DataGenerator
from global_config import global_config
from summary.test_case import test
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from summary.case import case
from tqdm import tqdm

save_dir = './case_result'
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
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 132, 102),
                kernel_size=3, stride=1, padding=1, config=config),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 44, 34),
                kernel_size=3, stride=1, padding=1, config=config),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 22, 17),
                kernel_size=3, stride=1, padding=1, config=config),
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
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 22, 17),
                kernel_size=3, stride=1, padding=1, config=config),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 44, 34),
                kernel_size=3, stride=1, padding=1, config=config),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 132, 102),
                kernel_size=3, stride=1, padding=1, config=config),
    ]
]

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(config['DEVICE'])
forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1], config=config).to(config['DEVICE'])
encoder_forecaster = EF(encoder, forecaster).to(config['DEVICE'])

weight_path = '/home/warit/senior/experiments/conv_logs/logs_4_10_2_False_05032140/model_25500.pth'
encoder_forecaster.load_state_dict(torch.load(weight_path, map_location='cuda'))

files = sorted([file for file in glob.glob(global_config['DATA_PATH'])])
for i in tqdm(case):
    file_name = i[0]
    crop = i[1]
    sp = save_dir + '/' + file_name[:-4]
    if not os.path.exists(sp):
        os.makedirs(sp)
    test(encoder_forecaster, data_loader, config, sp, files, file_name, crop=crop)