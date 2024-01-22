import numpy as np
import os,sys,math
import argparse
from tqdm import tqdm
from einops import rearrange, repeat
from torchvision.utils import save_image
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import scipy.io as sio
from dataset.dataset_motiondeblur import *
import utils
from model import UNet,Uformer
from losses import VisualizeAttnMap
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from utils import is_png_file, load_img_resize, Augment_RGB_torch
from DISTS_pytorch import DISTS
import torchvision.transforms.functional as TF
import cv2
import lpips

import glob

parser = argparse.ArgumentParser(description='Image motion deblurring evaluation on GoPro/HIDE')

parser.add_argument('--output_dir', default='/data1/wangzd/datasets/deblurring/GoPro/test/', type=str, help='Directory of validation images')
parser.add_argument('--trans_dir', default='/data1/wangzd/uformer_cvpr/results_release/deblurring/GoPro/Uformer_B/', type=str, help='Directory for results')

args = parser.parse_args()

output_dir = args.output_dir
trans_dir = args.trans_dir

out_dirs = sorted(glob.glob(output_dir + "/*OUT.png"))
trans_dirs = sorted(glob.glob(trans_dir + "/*.png"))

psnr_val_rgb = []
ssim_val_rgb = []
lpips_val_rgb = []
dists_val_rgb = []
lpips = lpips.LPIPS(net='vgg')
    # loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
D = DISTS()

for output_path, trans_path in zip(out_dirs, trans_dirs):
    gt_i = load_img_resize(trans_path)
    inp_i = load_img_resize(output_path)
    gt_np = np.float32(gt_i)
    inp_np = np.float32(inp_i)
    gt = torch.from_numpy(gt_np).permute(2,0,1).clamp(0,1)
    inp = torch.from_numpy(inp_np).permute(2,0,1).clamp(0,1)

    single_psnr = psnr_loss(gt_np, inp_np)
    single_ssim = ssim_loss(gt_np, inp_np, multichannel=True)
    single_lpips = lpips(gt, inp).item()
    single_dists = D(gt.unsqueeze(dim = 0), inp.unsqueeze(dim = 0)).item()

    psnr_val_rgb.append(single_psnr)
    ssim_val_rgb.append(single_ssim)
    lpips_val_rgb.append(single_lpips)
    dists_val_rgb.append(single_dists)
    
    print(single_psnr)
    # print(single_ssim)
    # print(single_lpips)
    # print(single_dists)

psnr_val_rgb = sum(psnr_val_rgb)/len(out_dirs)
ssim_val_rgb = sum(ssim_val_rgb)/len(out_dirs)
lpips_val_rgb = sum(lpips_val_rgb)/len(out_dirs)
dists_val_rgb = sum(dists_val_rgb)/len(out_dirs)

print(output_dir)
print(psnr_val_rgb)
print(ssim_val_rgb)
print(lpips_val_rgb)
print(dists_val_rgb)