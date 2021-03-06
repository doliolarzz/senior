import sys, os
from argparse import ArgumentParser
sys.path.insert(0, '../')
import torch
from torch.optim import lr_scheduler
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from utils.trainer import Trainer
from global_config import global_config
from collections import OrderedDict
from models.trajGRU import TrajGRU
from utils.generators import DataGenerator

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--in', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batchsize', required=True)
    parser.add_argument('--multitask', required=True)
    args = vars(parser.parse_args())

    if not os.path.exists('./traj_logs'):
        os.makedirs('./traj_logs')

    logs_name = './traj_logs/logs_' + args['name']
    multitask = True if args['multitask'].lower() == 'true' else False
    config = {
        'DEVICE': torch.device(args['device']),
        'IN_LEN': int(args['in']),
        'OUT_LEN': int(args['out']),
        'BATCH_SIZE': int(args['batchsize']),
        'SCALE': 0.2,
    }

    batch_size = config['BATCH_SIZE']
    max_iterations = 4
    test_iteration_interval = 1000
    test_and_save_checkpoint_iterations = 1000
    LR_step_size = 1000
    LR = 1e-4
    mse_loss = torch.nn.MSELoss().to(config['DEVICE'])
    mae_loss = torch.nn.L1Loss().to(config['DEVICE'])

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
    optim = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)

    trainer = Trainer(
        config=config,
        model=encoder_forecaster,
        optimizer=optim,
        data_loader=data_loader,
        save_dir=logs_name
    )
    trainer.train()