import numpy as np
from config import config

height, width = config['DATA_HEIGHT'], config['DATA_WIDTH']
lat_min, lat_max, lon_min, lon_max = config['lAT_MIN'], config['LAT_MAX'], config['LON_MIN'], config['LON_MAX']
def get_crop_boundary_idx(crop):
    crop_lat1, crop_lat2, crop_lon1, crop_lon2 = crop
    if (crop_lat1 < lat_min) | (crop_lat2 > lat_max) | (crop_lon1 < lon_min) | (crop_lon2> lon_max) :
        print("Crop boundary is out of bound.")
        return
    
    lat_min_idx = round((crop_lat1 - lat_min)/(lat_max - lat_min)*height) # min row idx, num rows = height
    lat_max_idx = round((crop_lat2 - lat_min)/(lat_max - lat_min)*height) # max row idx, num rows = height
    lon_min_idx = round((crop_lon1 - lon_min)/(lon_max - lon_min)*width)  # min col idx, num cols = height
    lon_max_idx = round((crop_lon2 - lon_min)/(lon_max - lon_min)*width)  # max col idx, num cols = height    
    return height - lat_max_idx, height - lat_min_idx, lon_min_idx, lon_max_idx


c_back = np.power(200, 5/8)
def dbz_mm(value):
    return np.minimum(np.power(10, value*15/4), 1e6) / c_back - 1

c_f = 16 / (60 * np.log(10))
c_h = 5 / 8 * np.log(200)
def mm_dbz(value):
    return c_f * (c_h + np.log(value + 1)) 

def pixel_to_rainfall(img, a=58.53, b=1.56):
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals

def dBZ_to_rainfall(dBZ, a=58.53, b=1.56):
    return np.power(10, (dBZ - 10 * np.log10(a))/(10*b))

def rainfall_to_dBZ(rainfall, a=58.53, b=1.56):
    return 10*np.log10(a) + 10*b*np.log10(rainfall)

