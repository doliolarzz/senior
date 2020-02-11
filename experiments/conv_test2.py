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
from net_params import convlstm_encoder_params, convlstm_forecaster_params
from utils.test import test
from utils.evaluators import fp_fn_image_csi, cal_rmse_all
from utils.visualizers import make_gif_color, rainfall_shade
from utils.units import mm_dbz, dbz_mm

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(config['DEVICE'])
forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(config['DEVICE'])
model = EF(encoder, forecaster).to(config['DEVICE'])
model.load_state_dict(
    torch.load('/home/doliolarzz/Desktop/senior/trained/conv7000.pth', map_location='cuda'))

path = '/media/doliolarzz/Ubuntu_data/test/*.bin'
files = sorted([file for file in glob.glob(path)])
try:
    idx = next(i for i,f in enumerate(files) if os.path.basename(f) == '20190908_1320.bin')
except:
    idx = -1
    print('not found')

def get_crop_boundary_idx(height, width, lat_min, lat_max, lon_min, lon_max, crop_lat1, crop_lat2, crop_lon1, crop_lon2):
    if (crop_lat1 < lat_min) | (crop_lat2 > lat_max) | (crop_lon1 < lon_min) | (crop_lon2> lon_max) :
        print("Crop boundary is out of bound.")
        return
    
    lat_min_idx = round((crop_lat1 - lat_min)/(lat_max - lat_min)*height) # min row idx, num rows = height
    lat_max_idx = round((crop_lat2 - lat_min)/(lat_max - lat_min)*height) # max row idx, num rows = height
    lon_min_idx = round((crop_lon1 - lon_min)/(lon_max - lon_min)*width)  # min col idx, num cols = height
    lon_max_idx = round((crop_lon2 - lon_min)/(lon_max - lon_min)*width)  # max col idx, num cols = height    
    return height - lat_max_idx, height - lat_min_idx, lon_min_idx, lon_max_idx

height, width = (3360, 2560)
lat_min = 20.005
lat_max = 47.9958
lon_min = 118.006
lon_max = 149.994
crop_lat1 = 32
crop_lat2 = 39
crop_lon1 = 136
crop_lon2 = 143
h1, h2, w1, w2 = get_crop_boundary_idx(height, width, lat_min, lat_max, lon_min, lon_max, crop_lat1, crop_lat2, crop_lon1, crop_lon2)

data = np.zeros((config['IN_LEN']+config['OUT_LEN'], h2 - h1 + 1, w2 - w1 + 1), dtype=np.float32)
for i, file in enumerate(files[idx:idx+config['IN_LEN']+config['OUT_LEN']]):
    data[i, :] = np.fromfile(file, dtype=np.float32).reshape((height, width))[h1 : h2 + 1, w1 : w2 + 1]
data = mm_dbz(data)
pred = test(data[:config['IN_LEN']], model)
data = dbz_mm(data)

print('CSI: ', fp_fn_image_csi(pred[-1], data[-1]))
# print('RMSE: ', np.sqrt(np.mean(np.square(data[-1] - pred[-1]))))
rmse, rmse_rain, rmse_non_rain = cal_rmse_all(pred[-1], data[-1])
print('rmse_all', rmse)
print('rmse_rain', rmse_rain)
print('rmse_non_rain', rmse_non_rain)

cv2.imwrite('conv_pred_2.png', rainfall_shade(pred[-1]))
cv2.imwrite('conv_label_2.png', rainfall_shade(data[-1]))