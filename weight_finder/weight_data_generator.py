import numpy as np
import torch
from utils import get_weight_train_data
from config import config

def predict(input, model):
    assert input.shape[0] == config['IN_LEN']
    assert input.shape[2] == config['IMG_SIZE']
    assert input.shape[3] == config['IMG_SIZE']

    with torch.no_grad():
        input = torch.from_numpy(input[:, :, None]).to(config['DEVICE'])
        output = model(input)
    
    assert output.shape[0] == config['OUT_LEN']
    assert output.shape[3] == config['IMG_SIZE']
    assert output.shape[4] == config['IMG_SIZE']

    return output.cpu().numpy()[:, :, 0]

def weight_data_generator(model, size):

    data = get_weight_train_data(size)

    n, t, height, width = data.shape
    assert t == config['IN_LEN'] + config['OUT_LEN']
    input = data[:, :config['IN_LEN']].swapaxes(0, 1)
    label = data[:, config['IN_LEN']:config['IN_LEN']+config['OUT_LEN']].swapaxes(0, 1)

    n_h = int((height - config['IMG_SIZE'])/config['STRIDE']) + 1
    n_w = int((width - config['IMG_SIZE'])/config['STRIDE']) + 1
    pred = np.zeros((config['OUT_LEN'], config['BATCH_SIZE'], n_h+1, n_w+1, config['IMG_SIZE'], config['IMG_SIZE']), dtype=np.float32)
    
    for i in range(int(np.ceil(n / config['BATCH_SIZE']))):

        pred[:] = 0
        i_start = i*config['BATCH_SIZE']
        i_end = min((i+1)*config['BATCH_SIZE'], n)

        for h in range(n_h):
            for w in range(n_w):
                h_start = h*config['STRIDE']
                h_end = h*config['STRIDE'] + config['IMG_SIZE']
                w_start = w*config['STRIDE']
                w_end = w*config['STRIDE'] + config['IMG_SIZE']
                pred[:, :, h, w] += predict(input[:, i_start:i_end, h_start:h_end, w_start:w_end], model)

        for h in range(n_h):
            h_start = h*config['STRIDE']
            h_end = h*config['STRIDE'] + config['IMG_SIZE']
            pred[:, :, h, n_w] += predict(input[:, i_start:i_end, h_start:h_end, -config['IMG_SIZE']:], model)

        for w in range(n_w):
            w_start = w*config['STRIDE']
            w_end = w*config['STRIDE'] + config['IMG_SIZE']
            pred[:, :, n_h, w] += predict(input[:, i_start:i_end, -config['IMG_SIZE']:, w_start:w_end], model)
        
        pred[:, : n_h, n_w] += predict(input[:, i_start:i_end, -config['IMG_SIZE']:, -config['IMG_SIZE']:], model)
        
        yield pred, label