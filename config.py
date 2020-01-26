import torch
from model import activation

config = {
    'DATA_PATH': '/media/doliolarzz/Ubuntu_data/wni_data/201807/',
    'DEVICE': torch.device('cuda'),
    'IN_LEN': 6,
    'OUT_LEN': 18,
    'RNN_ACT_TYPE': activation('leaky', negative_slope=0.2, inplace=True),
}