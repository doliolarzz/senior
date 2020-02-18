import sys, os
from argparse import ArgumentParser
sys.path.insert(0, '../')
import torch
from torch.optim import lr_scheduler
from utils.train_fcn import k_train_fcn
from global_config import global_config
from models.fcn import FCN8s

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--in', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batchsize', required=True)
    parser.add_argument('--mode', required=True)
    args = vars(parser.parse_args())

    if not os.path.exists('./fcn_logs'):
        os.makedirs('./fcn_logs')

    logs_name = './fcn_logs/logs_' + args['name']
    config = {
        'DEVICE': torch.device(args['device']),
        'IN_LEN': int(args['in']),
        'OUT_LEN': int(args['out']),
        'BATCH_SIZE': int(args['batchsize']),
    }

    k_fold = 1
    batch_size = config['BATCH_SIZE']
    max_iterations = 1
    LR_step_size = 1000
    gamma = 0.7
    LR = 1e-4

    def get_model_set():
        model = FCN8s(n_class=4, n_channel=5).to(config['DEVICE'])
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)
        return model, optimizer, exp_lr_scheduler

    k_train_fcn(k_fold, get_model_set, batch_size, max_iterations, save_dir=logs_name, mode=args['mode'], config=config)