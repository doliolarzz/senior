import numpy as np
import torch
from utils.predictor import get_weight_train_data
from global_config import global_config

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

    data = get_weight_train_data(size, crop=crop, config=config)

    n, t, height, width = data.shape
    assert t == config['IN_LEN'] + config['OUT_LEN']
    input = data[:, :config['IN_LEN']].swapaxes(0, 1)
    label = data[:, config['IN_LEN']:config['IN_LEN']+config['OUT_LEN']].swapaxes(0, 1)

    n_h = int((height - global_config['IMG_SIZE'])/global_config['STRIDE']) + 1
    n_w = int((width - global_config['IMG_SIZE'])/global_config['STRIDE']) + 1
    pred = np.zeros((config['OUT_LEN'], config['BATCH_SIZE'], n_h+1, n_w+1, global_config['IMG_SIZE'], global_config['IMG_SIZE']), dtype=np.float32)
    
#     weight = torch.randn(480, 480, device=config['DEVICE'], dtype=torch.float, requires_grad=True)
    weight = torch.ones(480, 480, device=config['DEVICE'], dtype=torch.float, requires_grad=True)
    
    all_itera = 0
    for e in range(epochs):
    
        for i in range(int(np.ceil(n / config['BATCH_SIZE']))):
            all_itera += 1

            pred[:] = 0
            i_start = i*config['BATCH_SIZE']
            i_end = min((i+1)*config['BATCH_SIZE'], n)

            for h in range(n_h):
                for w in range(n_w):
                    h_start = h*global_config['STRIDE']
                    h_end = h*global_config['STRIDE'] + global_config['IMG_SIZE']
                    w_start = w*global_config['STRIDE']
                    w_end = w*global_config['STRIDE'] + global_config['IMG_SIZE']
                    pred[:, :, h, w] += predict(input[:, i_start:i_end, h_start:h_end, w_start:w_end], model, config=config)

            for h in range(n_h):
                h_start = h*global_config['STRIDE']
                h_end = h*global_config['STRIDE'] + global_config['IMG_SIZE']
                pred[:, :, h, n_w] += predict(input[:, i_start:i_end, h_start:h_end, -global_config['IMG_SIZE']:], model, config=config)

            for w in range(n_w):
                w_start = w*global_config['STRIDE']
                w_end = w*global_config['STRIDE'] + global_config['IMG_SIZE']
                pred[:, :, n_h, w] += predict(input[:, i_start:i_end, -global_config['IMG_SIZE']:, w_start:w_end], model, config=config)

            pred[:, :, n_h, n_w] += predict(input[:, i_start:i_end, -global_config['IMG_SIZE']:, -global_config['IMG_SIZE']:], model, config=config)

            x = torch.from_numpy(pred.reshape(-1, n_h+1, n_w+1, global_config['IMG_SIZE'], global_config['IMG_SIZE'])).float().to(config['DEVICE'])
            y = torch.from_numpy(label[:, i_start:i_end].reshape(-1, height, width)).float().to(config['DEVICE'])
            
            vals = torch.zeros(config['BATCH_SIZE'], height, width, device=config['DEVICE'], dtype=torch.float, requires_grad=False)
            counts = torch.zeros(config['BATCH_SIZE'], height, width, device=config['DEVICE'], dtype=torch.float, requires_grad=False)
            
            for hh in range(n_h):
                for ww in range(n_w):
                    h_start = hh*global_config['STRIDE']
                    h_end = hh*global_config['STRIDE'] + global_config['IMG_SIZE']
                    w_start = ww*global_config['STRIDE']
                    w_end = ww*global_config['STRIDE'] + global_config['IMG_SIZE']
                    vals[:, h_start:h_end, w_start:w_end] += x[:, hh, ww] * weight[None,:]
                    counts[:, h_start:h_end, w_start:w_end] += weight[None,:]

            for hh in range(n_h):
                h_start = hh*global_config['STRIDE']
                h_end = hh*global_config['STRIDE'] + global_config['IMG_SIZE']
                vals[:, h_start:h_end, -global_config['IMG_SIZE']:] += x[:, hh, n_w] * weight[None,:]
                counts[:, h_start:h_end, -global_config['IMG_SIZE']:] += weight[None,:]

            for ww in range(n_w):
                w_start = ww*global_config['STRIDE']
                w_end = ww*global_config['STRIDE'] + global_config['IMG_SIZE']
                vals[:, -global_config['IMG_SIZE']:, w_start:w_end] += x[:, n_h, ww] * weight[None,:]
                counts[:, -global_config['IMG_SIZE']:, w_start:w_end] += weight[None,:]

            vals[:, -global_config['IMG_SIZE']:, -global_config['IMG_SIZE']:] += x[:, n_h, n_w] * weight[None,:]
            counts[:, -global_config['IMG_SIZE']:, -global_config['IMG_SIZE']:] += weight[None,:]

            y_pred = vals / counts

            loss = (y_pred - y).pow(2).sum().mean().sqrt()
            if all_itera % 1 == 0:
                print(all_itera, loss.item())
            loss.backward()

            with torch.no_grad():
                weight -= learning_rate * weight.grad
                weight.grad.zero_()
    
    return weight.detach().cpu().numpy()