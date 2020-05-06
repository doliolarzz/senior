from summary.case import case
from tqdm import tqdm
from global_config import global_config
from utils.units import dbz_mm, get_crop_boundary_idx
from utils.evaluators import fp_fn_image_csi, cal_rmse_all, fp_fn_image_csi_muti, torch_cal_rmse_all
from utils.visualizers import make_gif_color, rainfall_shade, make_gif, make_gif_color_label
from utils.units import mm_dbz, dbz_mm, denorm, torch_denorm
import torch
import numpy as np
import cv2
import os
import glob
import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sys.path.insert(0, '../')

rs_img = torch.nn.Upsample(size=(
    global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']), mode='bilinear')


def test(model, data_loader, config, save_dir, files, file_name, crop=None):

    h1, h2, w1, w2 = 0, global_config['DATA_HEIGHT'] - \
        1, 0, global_config['DATA_WIDTH'] - 1
    if crop is not None:
        h1, h2, w1, w2 = get_crop_boundary_idx(crop)

    idx = 0
    try:
        idx = next(i for i, f in enumerate(files)
                   if os.path.basename(f) == file_name)
    except:
        print('not found')
        return

    scale = config['SCALE']
    h = int(global_config['DATA_HEIGHT'] * scale)
    w = int(global_config['DATA_WIDTH'] * scale)
    sliced_input = np.zeros((1, config['IN_LEN'], h, w), dtype=np.float32)
    sliced_label = np.zeros(
        (1, global_config['OUT_TARGET_LEN'], global_config['DATA_HEIGHT'] - 12, global_config['DATA_WIDTH'] - 2), dtype=np.float32)

    for i, j in enumerate(range(idx - config['IN_LEN'], idx)):
        f = np.fromfile(files[j], dtype=np.float32) \
            .reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
        sliced_input[0, i] = \
            cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)

    for i, j in enumerate(range(idx, idx+global_config['OUT_TARGET_LEN'])):
        sliced_label[0, i] = np.fromfile(files[j], dtype=np.float32) \
            .reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))[6:-6, 1: -1]

    sliced_input = (mm_dbz(sliced_input) -
                    global_config['NORM_MIN']) / global_config['NORM_DIV']
    sliced_input = torch.from_numpy(sliced_input[:, :, 6:-6, 1: -1].swapaxes(0,1)).to(config['DEVICE'])

    outputs = None
    sliced_input = sliced_input[:, None]
    with torch.no_grad():
        for t in range(int(np.ceil(global_config['OUT_TARGET_LEN']/config['OUT_LEN']))):
            # print('input data', sliced_input.shape)
            output = model(sliced_input)
            # print('output', output.shape)
            if outputs is None:
                outputs = output.detach().cpu().numpy()
            else:
                outputs = np.concatenate(
                    [outputs, output.detach().cpu().numpy()], axis=0)

            sliced_input = output[-config['IN_LEN']:]

    pred = np.array(outputs)
    pred = np.array(outputs)[:global_config['OUT_TARGET_LEN'],0,0][None,:]
    sliced_label = sliced_label[0, None]
    # print('pred label shape', pred.shape, sliced_label.shape)
    pred = denorm(pred)
    pred_resized = np.zeros(
        (pred.shape[0], pred.shape[1], global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred_resized[i, j] = cv2.resize(
                pred[i, j], (global_config['DATA_WIDTH'], global_config['DATA_HEIGHT']), interpolation=cv2.INTER_AREA)

    pred_resized = pred_resized[:, :, h1: h2 + 1, w1: w2 + 1]
    sliced_label = sliced_label[:, :, h1: h2 + 1, w1: w2 + 1]
    # csi = fp_fn_image_csi(pred_resized, sliced_label)
    csis = []
    for c in range(pred.shape[1]):
        csis.append(fp_fn_image_csi(pred_resized[:, c], sliced_label[:, c]))
    csi = np.mean(csis)
    csi_multi, macro_csi = fp_fn_image_csi_muti(pred_resized, sliced_label)
    rmse, rmse_rain, rmse_non_rain = cal_rmse_all(pred_resized, sliced_label)
    result_all = [csi] + list(csi_multi) + [rmse, rmse_rain, rmse_non_rain]

    h_small = int(pred_resized.shape[2] * 0.5)
    w_small = int(pred_resized.shape[3] * 0.5)

    pred_small = np.zeros(
        (sliced_label.shape[0], sliced_label.shape[1], h_small, w_small))
    label_small = np.zeros(
        (sliced_label.shape[0], sliced_label.shape[1], h_small, w_small))
    for i in range(sliced_label.shape[0]):
        for j in range(sliced_label.shape[1]):
            pred_small[i, j] = cv2.resize(
                pred_resized[i, j], (w_small, h_small), interpolation=cv2.INTER_AREA)
            label_small[i, j] = cv2.resize(
                sliced_label[i, j], (w_small, h_small), interpolation=cv2.INTER_AREA)

    path = save_dir + '/imgs'
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(pred_resized.shape[0]):
        # Save pred gif
        # make_gif(pred[i] / 80 * 255, path + '/pred_{}_{}.gif'.format(b, i))
        # Save colored pred gif
        make_gif_color(pred_small[i], path + '/pred_colored.gif')
        # Save gt gif
        # make_gif(label_small[i] / 80 * 255, path + '/gt_{}_{}.gif'.format(b, i))
        # Save colored gt gif
        make_gif_color(label_small[i], path + '/gt_colored.gif')

        labels = [os.path.basename(files[idx+i]) for i in range(global_config['OUT_TARGET_LEN'])]
        make_gif_color_label(label_small[i], pred_small[i], labels, fname=path + '/all.gif')

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.plot(np.arange(len(csis))+1, csis)
    ax.set_xticks(np.arange(global_config['OUT_TARGET_LEN'])+1)
    ax.set_ylabel('Binary - CSI')
    ax.set_xlabel('Time Steps')
    plt.savefig(path + '/csis.png')

    result_all = np.array(result_all)
    result_all = np.around(result_all, decimals=3)
    np.savetxt(save_dir + '/result.txt', result_all, delimiter=',', fmt='%.3f')
    np.savetxt(save_dir + '/csi.txt', np.array(csis), delimiter=',', fmt='%.3f')
