import torch
from models.model import activation

config = {
    'DATA_PATH': '/home/warit/201807/*/*.bin',
    'DEVICE': torch.device('cuda:1'),
    'IN_LEN': 6,
    'OUT_LEN': 18,
    'BATCH_SIZE': 2,
    'RNN_ACT_TYPE': activation('leaky', negative_slope=0.2, inplace=True),
    'IMG_SIZE': 480,
}