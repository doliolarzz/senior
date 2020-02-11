import torch
from models.model import activation

config = {
    'DATA_PATH': '/data/wni_dataset/*/*/*.bin',
    'DEVICE': torch.device('cuda:0'),
    'IN_LEN': 5,
    'OUT_LEN': 18,
    'BATCH_SIZE': 2,
    'RNN_ACT_TYPE': activation('leaky', negative_slope=0.2, inplace=True),
    'IMG_SIZE': 480,
}