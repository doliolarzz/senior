import torch
import numpy as np
from config import config
from utils.units import dbz_mm

def predict(imgs, model):
    with torch.no_grad():
        imgs = torch.from_numpy(imgs[:,None,None]).to(config['DEVICE'])
        output = model(imgs)
    return dbz_mm(output.cpu().numpy()[:,0,0])

def test(imgs, model):
    stride = 100
    t, height, width = imgs.shape
    assert t == config['IN_LEN']
    counts = np.zeros((config['OUT_LEN'], height, width))
    values = np.zeros((config['OUT_LEN'], height, width))
    
    for h in range(0, height - config['IMG_SIZE'], stride):
        for w in range(0, width - config['IMG_SIZE'], config['IMG_SIZE']):
            values[:,h:h+config['IMG_SIZE'], w:w+config['IMG_SIZE']] += \
                predict(imgs[:,h:h+config['IMG_SIZE'], w:w+config['IMG_SIZE']], model)
            counts[:,h:h+config['IMG_SIZE'], w:w+config['IMG_SIZE']] += 1
    
    for h in range(0, height - config['IMG_SIZE'], stride):
        values[:,h:h+config['IMG_SIZE'], -config['IMG_SIZE']:] += \
            predict(imgs[:,h:h+config['IMG_SIZE'], -config['IMG_SIZE']:], model)
        counts[:,h:h+config['IMG_SIZE'], -config['IMG_SIZE']:] += 1

    for w in range(0, width - config['IMG_SIZE'], stride):
        values[:,-config['IMG_SIZE']:, w:w+config['IMG_SIZE']] += \
            predict(imgs[:,-config['IMG_SIZE']:, w:w+config['IMG_SIZE']], model)
        counts[:,-config['IMG_SIZE']:, w:w+config['IMG_SIZE']] += 1

    values[:,-config['IMG_SIZE']:, -config['IMG_SIZE']:] += \
        predict(imgs[:,-config['IMG_SIZE']:, -config['IMG_SIZE']:], model)
    counts[:,-config['IMG_SIZE']:, -config['IMG_SIZE']:] += 1

    return values / counts
    