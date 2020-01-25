import cv2
import numpy as np
from PIL import Image

def make_video(image_conv, image_label, h, w):
    out = cv2.VideoWriter('compare_1hr.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, (w*2+3*3,h+3*2))
    mn = min(image_conv.min(), image_label.min())
    mx = max(image_conv.max(), image_label.max())
    for i in range(image_label.shape[1]):
        left = np.ones((h,3)) * mn
        f = np.concatenate([left,image_conv[-1,i,0],left,image_label[-1,i,0],left], axis=1)
        f = cv2.cvtColor(np.array((f - mn) / (mx - mn)*255,dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        top = np.zeros((3,w*2+3*3,3),dtype=np.uint8)
        f = np.concatenate([top,f,top], axis=0)
        out.write(f)
    out.release()

def make_gif(single_seq, fname):
    img_seq = [Image.fromarray(img.astype(np.float32) * 255, 'F').convert("L") for img in single_seq]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:])