import numpy as np
import torch
import os, glob
# from utils.predictor import get_weight_train_data
from utils.units import dbz_mm, get_crop_boundary_idx
from global_config import global_config
from tqdm import tqdm

def predict(input, model, config=None):
    assert input.shape[0] == config['IN_LEN']
    assert input.shape[2] == global_config['IMG_SIZE']
    assert input.shape[3] == global_config['IMG_SIZE']

    with torch.no_grad():
        input = torch.from_numpy(input[:, :, None]).to(config['DEVICE'])
        output = model(input)
    
    assert output.shape[0] == config['OUT_LEN']
    assert output.shape[3] == global_config['IMG_SIZE']
    assert output.shape[4] == global_config['IMG_SIZE']

    return output.cpu().numpy()[:, :, 0]

def train_weight_model(model, size, crop=None, epochs=1, learning_rate=1e-5, config=None):

    h1, h2, w1, w2 = 0, global_config['DATA_HEIGHT'] - 1, 0, global_config['DATA_WIDTH'] - 1
    if crop is not None:
        h1, h2, w1, w2 = get_crop_boundary_idx(crop)

    height = h2 - h1 + 1
    width = w2 - w1 + 1

    files = sorted([file for file in glob.glob(global_config['DATA_PATH'])])
    window_size = config['IN_LEN'] + config['OUT_LEN']

    n_h = int((height - global_config['IMG_SIZE'])/global_config['STRIDE']) + 1
    n_w = int((width - global_config['IMG_SIZE'])/global_config['STRIDE']) + 1
    weight = torch.randn(480, 480, device=config['DEVICE'], dtype=torch.float, requires_grad=True)
    sigmoid = torch.nn.Sigmoid()
    
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([weight], lr=learning_rate)
    
    all_itera = 0
    pbar = tqdm(total=epochs*int(np.ceil(size / config['BATCH_SIZE'])))
    for e in range(epochs):
        picked_files = None
        while True:
            picked_files = np.random.choice(len(files) - window_size + 1, size)
            picked_files = np.setdiff1d(picked_files, global_config['MISSINGS'])
            if picked_files.shape[0] == size:
                break
        np.random.shuffle(picked_files) 
        for i in range(int(np.ceil(size / config['BATCH_SIZE']))):

            data = np.zeros((window_size, config['BATCH_SIZE'], h2 - h1 + 1, w2 - w1 + 1), dtype=np.float32)
            for b in range(config['BATCH_SIZE']):
                s_idx = picked_files[i*config['BATCH_SIZE'] + b]
                for f, file in enumerate(files[s_idx:s_idx+window_size]):
                    data[f, b, :] = np.fromfile(file, dtype=np.float32).reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))[h1 : h2 + 1, w1 : w2 + 1]
            input = data[:config['IN_LEN']]
            label = data[config['IN_LEN']:]

            pred = np.zeros((config['OUT_LEN'], config['BATCH_SIZE'], n_h+1, n_w+1, global_config['IMG_SIZE'], global_config['IMG_SIZE']), dtype=np.float32)
            # i_start = i*config['BATCH_SIZE']
            # i_end = min((i+1)*config['BATCH_SIZE'], size)

            for h in range(n_h):
                for w in range(n_w):
                    h_start = h*global_config['STRIDE']
                    h_end = h*global_config['STRIDE'] + global_config['IMG_SIZE']
                    w_start = w*global_config['STRIDE']
                    w_end = w*global_config['STRIDE'] + global_config['IMG_SIZE']
                    pred[:, :, h, w] += predict(input[:, :, h_start:h_end, w_start:w_end], model, config=config)

            for h in range(n_h):
                h_start = h*global_config['STRIDE']
                h_end = h*global_config['STRIDE'] + global_config['IMG_SIZE']
                pred[:, :, h, n_w] += predict(input[:, :, h_start:h_end, -global_config['IMG_SIZE']:], model, config=config)

            for w in range(n_w):
                w_start = w*global_config['STRIDE']
                w_end = w*global_config['STRIDE'] + global_config['IMG_SIZE']
                pred[:, :, n_h, w] += predict(input[:, :, -global_config['IMG_SIZE']:, w_start:w_end], model, config=config)

            pred[:, :, n_h, n_w] += predict(input[:, :, -global_config['IMG_SIZE']:, -global_config['IMG_SIZE']:], model, config=config)

            x = torch.from_numpy(pred.reshape(-1, n_h+1, n_w+1, global_config['IMG_SIZE'], global_config['IMG_SIZE'])).float().to(config['DEVICE'])
            y = torch.from_numpy(label.reshape(-1, height, width)).float().to(config['DEVICE'])
            
            vals = torch.zeros(config['BATCH_SIZE'], height, width, device=config['DEVICE'], dtype=torch.float, requires_grad=False)
            counts = torch.zeros(config['BATCH_SIZE'], height, width, device=config['DEVICE'], dtype=torch.float, requires_grad=False)
            
            for hh in range(n_h):
                for ww in range(n_w):
                    h_start = hh*global_config['STRIDE']
                    h_end = hh*global_config['STRIDE'] + global_config['IMG_SIZE']
                    w_start = ww*global_config['STRIDE']
                    w_end = ww*global_config['STRIDE'] + global_config['IMG_SIZE']
                    vals[:, h_start:h_end, w_start:w_end] += x[:, hh, ww] * sigmoid(weight[None,:])
                    counts[:, h_start:h_end, w_start:w_end] += sigmoid(weight[None,:])

            for hh in range(n_h):
                h_start = hh*global_config['STRIDE']
                h_end = hh*global_config['STRIDE'] + global_config['IMG_SIZE']
                vals[:, h_start:h_end, -global_config['IMG_SIZE']:] += x[:, hh, n_w] * sigmoid(weight[None,:])
                counts[:, h_start:h_end, -global_config['IMG_SIZE']:] += sigmoid(weight[None,:])

            for ww in range(n_w):
                w_start = ww*global_config['STRIDE']
                w_end = ww*global_config['STRIDE'] + global_config['IMG_SIZE']
                vals[:, -global_config['IMG_SIZE']:, w_start:w_end] += x[:, n_h, ww] * sigmoid(weight[None,:])
                counts[:, -global_config['IMG_SIZE']:, w_start:w_end] += sigmoid(weight[None,:])

            vals[:, -global_config['IMG_SIZE']:, -global_config['IMG_SIZE']:] += x[:, n_h, n_w] * sigmoid(weight[None,:])
            counts[:, -global_config['IMG_SIZE']:, -global_config['IMG_SIZE']:] += sigmoid(weight[None,:])

            y_pred = vals / counts

            loss = mse(y_pred, y) #.pow(2).sum()
#             if all_itera % 1 == 0:
#                 print(all_itera, loss.item())
            pbar.set_description('Current Loss %.2f' % loss.item())
            all_itera += 1
            pbar.update(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             with torch.no_grad():
#                 weight -= learning_rate * weight.grad
#                 weight.grad.zero_()
    pbar.close()
    
    return sigmoid(weight).detach().cpu().numpy()