import sys
sys.path.insert(0, '../')
import glob
import os
from utils.units import mm_dbz, dbz_mm
from utils.visualizers import make_gif_color, rainfall_shade
from utils.evaluators import fp_fn_image_csi, cal_rmse_all
from utils.predictor import prepare_testing, get_data
from global_config import global_config
from models.model import EF
from models.forecaster import Forecaster
from models.encoder import Encoder
import cv2
import numpy as np
import torch
from collections import OrderedDict
from models.convLSTM import ConvLSTM

config = {
    'DEVICE': torch.device('cuda:0'),
    'IN_LEN': 5,
    'OUT_LEN': 1,
    'BATCH_SIZE': 4,
}

batch_size = config['BATCH_SIZE']

convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1, config=config),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1, config=config),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1, config=config),
    ]
]

convlstm_forecaster_params = [
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
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1, config=config),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1, config=config),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1, config=config),
    ]
]

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(
    config['DEVICE'])
forecaster = Forecaster(
    convlstm_forecaster_params[0], convlstm_forecaster_params[1], config=config).to(config['DEVICE'])
model = EF(encoder, forecaster).to(config['DEVICE'])
model.load_state_dict(
    torch.load('/home/doliolarzz/Desktop/senior_trained/in5_out1.pth', map_location='cuda'))

data = get_data('20190630_2000.bin', crop=[31, 37, 127, 142], config=config)
data = mm_dbz(data)

weight = global_config['MERGE_WEIGHT']

pred, label = prepare_testing(data, model, weight=weight, config=config)
pred = np.maximum(dbz_mm(pred), 0)
label = dbz_mm(label)

print('CSI: ', fp_fn_image_csi(pred, label))
rmse, rmse_rain, rmse_non_rain = cal_rmse_all(pred, label)
print('rmse_all', rmse)
print('rmse_rain', rmse_rain)
print('rmse_non_rain', rmse_non_rain)


path = 'imgs_conv_case_1/'
try:
    os.makedirs(path)
except:
    pass
for i in range(pred.shape[0]):
    cv2.imwrite(path+str(i)+'.png',
                cv2.cvtColor(np.array(pred[i]/60*255, dtype=np.uint8), cv2.COLOR_GRAY2BGR))

# cv2.imwrite('conv_pred_1.png', rainfall_shade(pred[-1]))
# cv2.imwrite('conv_label_1.png', rainfall_shade(data[-1]))
