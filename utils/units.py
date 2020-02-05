import numpy as np

def pixel_to_dBZ(img):
    return img * 70.0 - 10.0

def dBZ_to_pixel(dBZ_img):
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)

c_back = np.power(200, 5/8)
def dbz_mm(value):
    mask = value<=0
    ret_val = np.power(10, value*15/4) / c_back
    ret_val[mask] = 0
    return ret_val

c_f = 16 / (60 * np.log(10))
c_h = 5 / 8 * np.log(200)
def mm_dbz(value):
    mask = value!=0
    ret_val = c_f * (c_h + np.log(value, where=mask, out=value*np.nan))
    ret_val[np.isnan(ret_val)] = -1
    return ret_val

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

