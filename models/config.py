import torch
from model import activation

config = {
    'DATA_PATH': './data',
    'DEVICE': torch.device('cuda'),
    'IN_LEN': 6,
    'OUT_LEN': 18,
    'RNN_ACT_TYPE': activation('leaky', negative_slope=0.2, inplace=True),
}