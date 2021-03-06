import sys
sys.path.insert(0, '../')
import glob
import os
from utils.units import mm_dbz, dbz_mm
from utils.visualizers import make_gif_color, rainfall_shade, make_gif
from utils.evaluators import fp_fn_image_csi, cal_rmse_all, fp_fn_image_csi_muti_reg
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

def conv_test(model_path, start_pred_fn, test_case, in_len, out_len, batch_size, multitask, crop=None):

    config = {
        'DEVICE': torch.device('cuda:0'),
        'IN_LEN': in_len,
        'OUT_LEN': out_len,
        'BATCH_SIZE': batch_size,
    }

    convlstm_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],

        [
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 168, 126),
                    kernel_size=3, stride=1, padding=1, config=config),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 56, 42),
                    kernel_size=3, stride=1, padding=1, config=config),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 28, 21),
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
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 28, 21),
                    kernel_size=3, stride=1, padding=1, config=config),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 56, 42),
                    kernel_size=3, stride=1, padding=1, config=config),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 168, 126),
                    kernel_size=3, stride=1, padding=1, config=config),
        ]
    ]

    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(
        config['DEVICE'])
    forecaster = Forecaster(
        convlstm_forecaster_params[0], convlstm_forecaster_params[1], config=config).to(config['DEVICE'])
    model = EF(encoder, forecaster).to(config['DEVICE'])
    model.load_state_dict(
        torch.load(model_path, map_location='cuda'))

    files = sorted([file for file in glob.glob(global_config['TEST_PATH'])])
    idx = 0
    if start_pred_fn != '':
        try:
            idx = next(i for i,f in enumerate(files) if os.path.basename(f) == start_pred_fn)
        except:
            idx = -1
            print('not found')
    scale_div=4
    data = np.zeros((config['IN_LEN'] + global_config['OUT_TARGET_LEN'], int(global_config['DATA_HEIGHT']/scale_div), int((global_config['DATA_WIDTH'] - 40)/scale_div)), dtype=np.float32)
    for i, file in enumerate(files[idx - config['IN_LEN']:idx + global_config['OUT_TARGET_LEN']]):
        fd = np.fromfile(file, dtype=np.float32).reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))[:,20:-20]
        fd = cv2.resize(fd, (int((global_config['DATA_WIDTH'] - 40)/scale_div), int(global_config['DATA_HEIGHT']/scale_div)), interpolation = cv2.INTER_AREA)
        data[i, :] = fd
    
    data = mm_dbz(data)
    input = data[:in_len]
    label = data[in_len:]
    preds = []
    with torch.no_grad():
        input = torch.from_numpy(input[:, None, None]).to(config['DEVICE'])
        for i in range(18):
            pred = model(input)
            pred[pred<0.2 + 0.035*i] -= 0.04*i
            pred = torch.clamp(pred, min=mm_dbz(0), max=mm_dbz(60))
            preds.append(pred.detach().cpu().numpy()[0, 0, 0])
            input = torch.cat((input[1:], pred), 0)
    pred = np.array(preds)
    print(pred.shape, label.shape)
    pred = np.maximum(dbz_mm(pred), 0)
    label = dbz_mm(label)
    csi = fp_fn_image_csi(pred, label)
    # print('CSI: ', csi)
    csi_multi = fp_fn_image_csi_muti_reg(pred, label)
    # print('CSI Multi: ', csi_multi)
    rmse, rmse_rain, rmse_non_rain = cal_rmse_all(pred, label)
    # print('rmse_all', rmse)
    # print('rmse_rain', rmse_rain)
    # print('rmse_non_rain', rmse_non_rain)

    if not os.path.exists('./imgs_conv'):
        os.makedirs('./imgs_conv')
    path = './imgs_conv/conv_{}_{}_{}_{}/'.format(test_case, in_len, out_len, multitask)
    try:
        os.makedirs(path)
    except:
        pass
    try:
        os.makedirs(path+'imgs/')
    except:
        pass

    #Save erros gif
    errs = np.sqrt(np.square(pred - label))
    make_gif(errs / errs.max() * 255, path + 'errs.gif')
    #Save pred gif
    make_gif(pred / 60 * 255, path + 'pred.gif')
    #Save colored pred gif
    make_gif_color(pred, path + 'pred_colored.gif')
    #Save gt gif
    make_gif(label / 60 * 255, path + 'gt.gif')
    #Save colored gt gif
    make_gif_color(label, path + 'gt_colored.gif')
    #Save imgs
    for i in range(pred.shape[0]):
        cv2.imwrite(path+'imgs/'+str(i)+'.png', 
                    cv2.cvtColor(np.array(pred[i]/80*255, dtype=np.uint8), cv2.COLOR_GRAY2BGR))
    
    # for i in range(pred.shape[0]):
    #     cv2.imwrite(path+str(i)+'.png',
    #                 cv2.cvtColor(np.array(pred[i]/60*255, dtype=np.uint8), cv2.COLOR_GRAY2BGR))

    # cv2.imwrite('conv_pred_1.png', rainfall_shade(pred[-1]))
    # cv2.imwrite('conv_label_1.png', rainfall_shade(data[-1]))

    return [rmse, rmse_rain, rmse_non_rain], csi, csi_multi
