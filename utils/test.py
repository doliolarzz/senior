import numpy as np
from config import config
from utils.units import dbz_mm

def predict(img, model):
    with torch.no_grad():
        output = model(img)
    return dbz_mm(output)

def test(img, model):
    height, width = img.shape
    counts = np.zeros((height, width))
    values = np.zeros((height, width))
    
    for h in range(0, height - config['IMG_SIZE'], config['IMG_SIZE']):
        for w in range(0, width - config['IMG_SIZE'], config['IMG_SIZE']):
            values[h:h+config['IMG_SIZE'], w:w+config['IMG_SIZE']] = \
                predict(img[h:h+config['IMG_SIZE'], w:w+config['IMG_SIZE']], model)
            counts[h:h+config['IMG_SIZE'], w:w+config['IMG_SIZE']] += 1
    
    for h in range(0, height - config['IMG_SIZE'], config['IMG_SIZE']):
        values[h:h+config['IMG_SIZE'], -config['IMG_SIZE']:] = \
            predict(img[h:h+config['IMG_SIZE'], -config['IMG_SIZE']:], model)
        counts[h:h+config['IMG_SIZE'], -config['IMG_SIZE']:] += 1

    for w in range(0, width - config['IMG_SIZE'], config['IMG_SIZE']):
        values[-config['IMG_SIZE']:, w:w+config['IMG_SIZE']] = \
            predict(img[-config['IMG_SIZE']:, w:w+config['IMG_SIZE']], model)
        counts[-config['IMG_SIZE']:, w:w+config['IMG_SIZE']] += 1

    values[-config['IMG_SIZE']:, -config['IMG_SIZE']:] = \
        predict(img[-config['IMG_SIZE']:, -config['IMG_SIZE']:], model)
    counts[-config['IMG_SIZE']:, -config['IMG_SIZE']:] += 1

    return values / counts
    