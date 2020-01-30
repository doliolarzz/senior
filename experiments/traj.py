import sys
sys.path.insert(0, '../')
import torch
from torch.optim import lr_scheduler
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from utils.train import k_train
from config import config
from net_params import encoder_params, forecaster_params

k_fold = 1
batch_size = config['BATCH_SIZE']
max_iterations = 2
test_iteration_interval = 1000
test_and_save_checkpoint_iterations = 1000
LR_step_size = 20000
gamma = 0.7
LR = 1e-4
mse_loss = torch.nn.MSELoss().to(config['DEVICE'])

def get_model_set():
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(config['DEVICE'])
    forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(config['DEVICE'])
    encoder_forecaster = EF(encoder, forecaster).to(config['DEVICE'])
    optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)
    return encoder_forecaster, optimizer, exp_lr_scheduler

k_train(k_fold, get_model_set, mse_loss, 
            batch_size, max_iterations)