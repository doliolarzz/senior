import sys
sys.path.insert(0, '../')
import torch
from torch.optim import lr_scheduler
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from utils.train import k_train
from config import config
from collections import OrderedDict
from models.convLSTM import ConvLSTM
# from net_params import convlstm_encoder_params, convlstm_forecaster_params

k_fold = 1
batch_size = config['BATCH_SIZE']
max_iterations = 2
test_iteration_interval = 1000
test_and_save_checkpoint_iterations = 1000
LR_step_size = 1000
gamma = 0.7
LR = 1e-4
mse_loss = torch.nn.MSELoss().to(config['DEVICE'])
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

def get_model_set():
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(config['DEVICE'])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(config['DEVICE'])
    encoder_forecaster = EF(encoder, forecaster).to(config['DEVICE'])
    optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)
    return encoder_forecaster, optimizer, exp_lr_scheduler

k_train(k_fold, get_model_set, mse_loss, 
            batch_size, max_iterations, multitask=True)