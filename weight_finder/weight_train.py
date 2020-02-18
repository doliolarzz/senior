import sys, os
import numpy as np
from argparse import ArgumentParser
sys.path.insert(0, '../')
import torch
from torch.optim import lr_scheduler
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from utils.train import k_train
from global_config import global_config
from collections import OrderedDict
from models.convLSTM import ConvLSTM
from weight_data_model import train_weight_model

if __name__ == "__main__":

    config = {
        'DEVICE': torch.device('cuda:2'),
        'IN_LEN': 5,
        'OUT_LEN': 1,
        'BATCH_SIZE': 1,
    }

    k_fold = 1
    batch_size = config['BATCH_SIZE']
    max_iterations = 1
    test_iteration_interval = 1000
    test_and_save_checkpoint_iterations = 1000
    LR_step_size = 1000
    gamma = 0.7
    LR = 1e-4
    mse_loss = torch.nn.MSELoss().to(config['DEVICE'])

    convlstm_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],

        [
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96),
                    kernel_size=3, stride=1, padding=1, config=config),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                    kernel_size=3, stride=1, padding=1, config=config),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
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
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                    kernel_size=3, stride=1, padding=1, config=config),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                    kernel_size=3, stride=1, padding=1, config=config),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96),
                    kernel_size=3, stride=1, padding=1, config=config),
        ]
    ]

    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(config['DEVICE'])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1], config=config).to(config['DEVICE'])
    model = EF(encoder, forecaster).to(config['DEVICE'])
    model.load_state_dict(
        torch.load('/home/warit/senior/experiments/conv_logs/logs_5_1_4_True_02_17_11_35/model_f1_i5000.pth'))

    w = train_weight_model(model, 500, crop=None, epochs=2, learning_rate=1e-2, config=config)
    np.savez('learnt_weight.npz', w=w)