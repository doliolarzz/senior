import sys
sys.path.insert(0, '../')
import torch
import os, glob
import numpy as np
import cv2
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from config import config
from net_params_BN import convlstm_encoder_params, convlstm_forecaster_params
from utils.predictor import prepare_testing, get_data
from utils.evaluators import fp_fn_image_csi, cal_rmse_all
from utils.visualizers import make_gif_color, rainfall_shade
from utils.units import mm_dbz, dbz_mm

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(config['DEVICE'])
forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(config['DEVICE'])
model = EF(encoder, forecaster).to(config['DEVICE'])
model.load_state_dict(
    torch.load('/home/doliolarzz/Desktop/senior_trained/in5_out18.pth', map_location='cuda'))

data = get_data('20190630_2000.bin', crop=[31, 37, 127, 142])
data = mm_dbz(data)

weight = np.load('../utils/weight.npz')['w'] + 1e-3

pred, label = prepare_testing(data, model, weight=weight)
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
    cv2.imwrite(path+str(i)+'.png', cv2.cvtColor(np.array(pred[i]/60*255,dtype=np.uint8), cv2.COLOR_GRAY2BGR))

# cv2.imwrite('conv_pred_1.png', rainfall_shade(pred[-1]))
# cv2.imwrite('conv_label_1.png', rainfall_shade(data[-1]))