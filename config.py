import torch
from models.model import activation

config = {
    'DATA_PATH': '/mnt/raid0/wni_dataset/*/*/*.bin',
    'TEST_PATH': '/media/doliolarzz/Ubuntu_data/test/*.bin',
    'DEVICE': torch.device('cuda:0'),
    'IN_LEN': 5,
    'OUT_LEN': 1,
    'OUT_TARGET_LEN': 18,
    'BATCH_SIZE': 4,
    'RNN_ACT_TYPE': activation('leaky', negative_slope=0.2, inplace=True),
    'IMG_SIZE': 480,
    'STRIDE': 120,
    'DATA_WIDTH': 2560,
    'DATA_HEIGHT': 3360,
    'lAT_MIN': 20.005,
    'LAT_MAX': 47.9958,
    'LON_MIN': 118.006,
    'LON_MAX': 149.994,
}
