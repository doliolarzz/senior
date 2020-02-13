import torch
import os, glob
import numpy as np
from config import config
from utils.units import dbz_mm, get_crop_boundary_idx
import cv2

def predict(input, model):
    with torch.no_grad():
        input = torch.from_numpy(input[:, None, None]).to(config['DEVICE'])
        output = model(input)
    # return np.maximum(dbz_mm(output.cpu().numpy()[:,0,0]), 0)
    return output.cpu().numpy()[:,0,0]

# def get_each_predictions(input, model):

#     t, height, width = input.shape
#     n_h = int((height - config['IMG_SIZE'])/config['STRIDE']) + 1
#     n_w = int((width - config['IMG_SIZE'])/config['STRIDE']) + 1
# bug hereeeeeeeeeeeeeeeeeeeeeeee v
#     data = np.zeros((config['OUT_LEN'], n_h+1, n_w+1, config['IMG_SIZE'], config['IMG_SIZE']), dtype=np.float32)

#     for h in range(n_h):
#         for w in range(n_w):
#             h_start = h*config['STRIDE']
#             h_end = h*config['STRIDE'] + config['IMG_SIZE']
#             w_start = w*config['STRIDE']
#             w_end = w*config['STRIDE'] + config['IMG_SIZE']
#             data[:, h, w] += predict(input[:, h_start:h_end, w_start:w_end], model)

#     for h in range(n_h):
#         h_start = h*config['STRIDE']
#         h_end = h*config['STRIDE'] + config['IMG_SIZE']
#         data[:, h, n_w] += predict(input[:, h_start:h_end, -config['IMG_SIZE']:], model)

#     for w in range(n_w):
#         w_start = w*config['STRIDE']
#         w_end = w*config['STRIDE'] + config['IMG_SIZE']
#         data[:, n_h, w] += predict(input[:, -config['IMG_SIZE']:, w_start:w_end], model)
    
#     data[:, n_h, n_w] += predict(input[:, -config['IMG_SIZE']:, -config['IMG_SIZE']:], model)

#     return data

def get_data(start_fn, crop=None):

    h1, h2, w1, w2 = 0, config['DATA_HEIGHT'], 0, config['DATA_WIDTH']
    if crop is not None:
        h1, h2, w1, w2 = get_crop_boundary_idx(crop)

    files = sorted([file for file in glob.glob(config['TEST_PATH'])])
    try:
        idx = next(i for i,f in enumerate(files) if os.path.basename(f) == start_fn)
    except:
        idx = -1
        print('not found')

    data = np.zeros((config['IN_LEN'] + config['OUT_TARGET_LEN'], h2 - h1 + 1, w2 - w1 + 1), dtype=np.float32)
    for i, file in enumerate(files[idx:idx+config['IN_LEN']+config['OUT_TARGET_LEN']]):
        data[i, :] = np.fromfile(file, dtype=np.float32).reshape((config['DATA_HEIGHT'], config['DATA_WIDTH']))[h1 : h2 + 1, w1 : w2 + 1]
    
    return data

def prepare_testing(data, model, weight=1, stride=120):

    t, height, width = data.shape
    assert t == config['IN_LEN'] + config['OUT_TARGET_LEN']
    input = data[:config['IN_LEN']]
    label = data[-config['OUT_TARGET_LEN']:]

    assert config['OUT_TARGET_LEN'] % config['OUT_LEN'] == 0

    n_h = int((height - config['IMG_SIZE'])/config['STRIDE']) + 1
    n_w = int((width - config['IMG_SIZE'])/config['STRIDE']) + 1

    pred = np.zeros((config['OUT_TARGET_LEN'], height, width), dtype=np.float32)
    vals = np.zeros((config['OUT_LEN'], height, width), dtype=np.float32)
    counts = np.zeros((config['OUT_LEN'], height, width), dtype=np.float32)
    for i in range(int(config['OUT_TARGET_LEN']/config['OUT_LEN'])):
        vals[:] = 0
        counts[:] = 0
        for h in range(n_h):
            for w in range(n_w):
                h_start = h*config['STRIDE']
                h_end = h*config['STRIDE'] + config['IMG_SIZE']
                w_start = w*config['STRIDE']
                w_end = w*config['STRIDE'] + config['IMG_SIZE']
                vals[:, h_start:h_end, w_start:w_end] += predict(input[:, h_start:h_end, w_start:w_end], model) * weight
                counts[:, h_start:h_end, w_start:w_end] += weight

        for h in range(n_h):
            h_start = h*config['STRIDE']
            h_end = h*config['STRIDE'] + config['IMG_SIZE']
            vals[:, h_start:h_end, -config['IMG_SIZE']:] += predict(input[:, h_start:h_end, -config['IMG_SIZE']:], model) * weight
            counts[:, h_start:h_end, -config['IMG_SIZE']:] += weight

        for w in range(n_w):
            w_start = w*config['STRIDE']
            w_end = w*config['STRIDE'] + config['IMG_SIZE']
            vals[:, -config['IMG_SIZE']:, w_start:w_end] += predict(input[:, -config['IMG_SIZE']:, w_start:w_end], model) * weight
            counts[:, -config['IMG_SIZE']:, w_start:w_end] += weight
        
        vals[:, -config['IMG_SIZE']:, -config['IMG_SIZE']:] += predict(input[:, -config['IMG_SIZE']:, -config['IMG_SIZE']:], model) * weight
        counts[:, -config['IMG_SIZE']:, -config['IMG_SIZE']:] += weight

        pred_i = vals / counts
        pred[i*config['OUT_LEN']:(i+1)*config['OUT_LEN']] = pred_i
        # if i == 16:
        #     cv2.imshow('test', cv2.cvtColor(np.array(pred_i[0]/pred_i.max()*255,dtype=np.uint8), cv2.COLOR_GRAY2BGR))
        #     cv2.waitKey(0)
        input = np.concatenate([input[1:], pred_i], axis=0)

    return pred, label