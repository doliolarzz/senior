import sys
sys.path.insert(0, '../')
import torch
from torch.optim import lr_scheduler
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from utils.train import k_train
from config import config
from net_params import convlstm_encoder_params, convlstm_forecaster_params

k_fold = 3
batch_size = 1
max_iterations = 100000
test_iteration_interval = 1000
test_and_save_checkpoint_iterations = 1000
LR_step_size = 20000
gamma = 0.7

LR = 1e-4

mse_loss = torch.nn.MSELoss().to(config['DEVICE'])

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(config['DEVICE'])

forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(config['DEVICE'])

encoder_forecaster = EF(encoder, forecaster).to(config['DEVICE'])

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

k_train(k_fold, encoder_forecaster, optimizer, mse_loss, exp_lr_scheduler, 
            batch_size, max_iterations)