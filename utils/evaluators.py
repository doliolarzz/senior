"""
    File name: evaluate.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2019-02-25
    Edited: Nutnaree Kleawsirikul
    Date Edited: 2019-11-25
    Python Version: 3.6
"""


import numpy as np

def hex2Rgb(tn_rgb):
    return list(int(tn_rgb[i:i+2], 16) for i in (0, 2, 4))

def fp_fn_image(gt, pred, config=None, threshold=125):
    # categorize
    gt, pred = gt.copy(), pred.copy()
    gt[gt < threshold] = 0
    gt[gt > threshold] = 1
    pred[pred < threshold] = 0
    pred[pred > threshold] = 1

    # evaluate
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)
    tp = (gt == 1) & (pred == 1)

    # summarize results
    error = np.zeros_like(gt)
    error[fp] = 1
    error[fn] = 2
    error[tp] = 3

    if config['eval']['plot_color'] is True:
        r, c = error.shape
        rgb_error = np.zeros((r, c, 3), 'uint8')
        rgb_error[..., 0] = error
        rgb_error[..., 1] = error
        rgb_error[..., 2] = error

        tn = np.where((rgb_error[:, :, 0] == 0) & (rgb_error[:, :, 1] == 0) & (rgb_error[:, :, 2] == 0))
        fp = np.where((rgb_error[:, :, 0] == 1) & (rgb_error[:, :, 1] == 1) & (rgb_error[:, :, 2] == 1))
        fn = np.where((rgb_error[:, :, 0] == 2) & (rgb_error[:, :, 1] == 2) & (rgb_error[:, :, 2] == 2))
        tp = np.where((rgb_error[:, :, 0] == 3) & (rgb_error[:, :, 1] == 3) & (rgb_error[:, :, 2] == 3))

        if config==None:
            tn_rgb = '#C6EDE7'
            fp_rgb = '#2832C2'
            fn_rgb = '#FF6464'
            tp_rgb = '#FFFFFF'
        else:
            tn_rgb = config["eval"]["color"]["tn"]
            fp_rgb = config["eval"]["color"]["fp"]
            fn_rgb = config["eval"]["color"]["fn"]
            tp_rgb = config["eval"]["color"]["tp"]

        rgb_error[tn] = hex2Rgb(tn_rgb)
        rgb_error[fp] = hex2Rgb(fp_rgb)
        rgb_error[fn] = hex2Rgb(fn_rgb)
        rgb_error[tp] = hex2Rgb(tp_rgb)

        error = rgb_error

    return error


def fp_fn_image_csi(gt, pred, threshold):
    # categorize
    gt, pred = gt.copy(), pred.copy()
    gt[gt < threshold] = 0
    gt[gt >= threshold] = 1
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1

    # evaluate
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tp = np.sum((gt == 1) & (pred == 1))

    csi = float(tp) / (fp + fn + tp) * 100

    return csi


def fp_fn_image_hit(gt, pred, threshold, mask=None):
    # categorize
    gt, pred = gt.copy(), pred.copy()
    if not (mask is None):
        print("Use mask!!!")
        gt = gt[mask]
        pred = pred[mask]

    gt[gt < threshold] = 0
    gt[gt >= threshold] = 1
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1

    # evaluate
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))

    hit = float(tp + tn) / (fp + fn + tp + tn) * 100

    return hit

def get_summary_table():
    return None