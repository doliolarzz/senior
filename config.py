import torch
from models.model import activation

config = {
    'DATA_PATH': '/media/doliolarzz/Ubuntu_data/wni_data/201807/*/*.bin',
    'DEVICE': torch.device('cuda'),
    'IN_LEN': 6,
    'OUT_LEN': 18,
    'BATCH_SIZE': 2,
    'RNN_ACT_TYPE': activation('leaky', negative_slope=0.2, inplace=True),
}