import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as TF
from natsort import natsorted
import glob
augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def ShiftData(image, shift_range, width):
    shift_image = image.clone()

    to_shift = image[ :, :, 0:shift_range]
    rest = image[:,  :, shift_range:width]

    shift_image[:, :, 0:(width-shift_range)] = rest
    shift_image[:, :, (width-shift_range):width] = to_shift

    return shift_image

class PanoramaDataLoader(Dataset):
    def __init__(self, mode, data_dir, opt):
        super(PanoramaDataLoader, self).__init__()
        
        self.mode = mode
        
        gt_dir = 'T' 
        input_dir = 'M'
        # gt_files = sorted(os.listdir(os.path.join(data_dir, gt_dir)))
        # input_files = sorted(os.listdir(os.path.join(data_dir, input_dir)))

        # self.gt_filenames = [os.path.join(data_dir, gt_dir, x) for x in gt_files if is_png_file(x)]
        # self.input_filenames = [os.path.join(data_dir, input_dir, x) for x in input_files if is_png_file(x)]

        self.gt_filenames = glob.glob(f'{data_dir}/{gt_dir}/*.png')
        self.input_filenames = glob.glob(f'{data_dir}/{input_dir}/*.png')
        
        # refer_mask_dir = 'ReferMask'
        # mixed_mask_dir = 'MixedMask'

        # rm_files = sorted(os.listdir(os.path.join(data_dir, refer_mask_dir)))
        # mm_files = sorted(os.listdir(os.path.join(data_dir, mixed_mask_dir)))

        # self.rm_filenames = [os.path.join(data_dir, refer_mask_dir, x) for x in rm_files if is_png_file(x)]
        # self.mm_filenames = [os.path.join(data_dir, mixed_mask_dir, x) for x in mm_files if is_png_file(x)]

        self.opt=opt

        self.tar_size = len(self.gt_filenames)


    def __len__(self):
        return self.tar_size
        # return 10

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1][:-4]
        input_filename = os.path.split(self.input_filenames[tar_index])[-1][:-4]
        # mm_filename = os.path.split(self.mm_filenames[tar_index])[-1][:-4]

        gt_np = np.float32(load_img(self.gt_filenames[tar_index]))
        inp_np = np.float32(load_img(self.input_filenames[tar_index]))
        # mm_np = np.float32(load_img(self.mm_filenames[tar_index]))

        ########################### TRAINING ###########################
        if self.mode == 'train':            
            if self.opt.train_reference_type == 'no':
                rm_np = 1 - np.float32(load_img(self.rm_filenames[tar_index]))
                gt_np = gt_np * rm_np
                inp_np = inp_np * rm_np
            
            elif self.opt.train_reference_type == 'blur':
                rm_np = np.float32(load_img(self.rm_filenames[tar_index]))
                blurred_reference = gaussian_filter(gt_np, sigma=(2,2,0))
                gt_np = np.where(rm_np==np.array([0., 0., 0.]), gt_np, blurred_reference)        
                noisy_np = np.where(rm_np==np.array([0., 0., 0.]), noisy_np, blurred_reference)
            
            # numpy to tensor
            gt = torch.from_numpy(gt_np).permute(2,0,1)
            inp = torch.from_numpy(inp_np).permute(2,0,1)
            # mm = torch.from_numpy(mm_np).permute(2,0,1)

            # Resize to pre-defined size
            gt = TF.resize(gt, (self.opt.img_height, self.opt.img_width))
            inp = TF.resize(inp, (self.opt.img_height, self.opt.img_width))
            # mm = TF.resize(mm, (self.opt.img_height, self.opt.img_width))

            ## DATA AUG
            # Random horizontal flipping
            if random.random() > 0.5:
                gt = TF.hflip(gt)
                inp = TF.hflip(inp)
            
            # Random Shifting
            width = 512
            shift_ratio = round(random.randint(0, 512), 3)
            gt = ShiftData(gt, shift_ratio, width)
            inp = ShiftData(inp, shift_ratio, width)

            # return gt, inp, mm, gt_filename, input_filename
            return gt, inp, gt_filename, input_filename
        
        ########################### VALIDATION ###########################
        elif self.mode == 'val':

            if self.opt.val_reference_type == 'no':
                rm_np = 1 - np.float32(load_img(self.rm_filenames[tar_index]))
                gt_np = gt_np * rm_np
                inp_np = inp_np * rm_np
            
            elif self.opt.val_reference_type == 'blur':
                rm_np = np.float32(load_img(self.rm_filenames[tar_index]))
                blurred_reference = gaussian_filter(gt_np, sigma=(2,2,0))
                gt_np = np.where(rm_np==np.array([0., 0., 0.]), gt_np, blurred_reference)        
                noisy_np = np.where(rm_np==np.array([0., 0., 0.]), noisy_np, blurred_reference)
            
            # numpy to tensor
            gt = torch.from_numpy(gt_np).permute(2,0,1)
            inp = torch.from_numpy(inp_np).permute(2,0,1)
            # mm = torch.from_numpy(mm_np).permute(2,0,1)

            # Resize to pre-defined size
            gt = TF.resize(gt, (self.opt.img_height, self.opt.img_width))
            inp = TF.resize(inp, (self.opt.img_height, self.opt.img_width))
            # mm = TF.resize(mm, (self.opt.img_height, self.opt.img_width))

            # ## DATA AUG
            # # Random horizontal flipping
            # if random.random() > 0.5:
            #     gt = TF.hflip(gt)
            #     inp = TF.hflip(inp)
            
            # # Random Shifting
            # width = 512
            # shift_ratio = round(random.randint(0, 512), 3)
            # gt = ShiftData(gt, shift_ratio, width)
            # inp = ShiftData(inp, shift_ratio, width)
            
            # return gt, inp, mm, gt_filename
            return gt, inp, gt_filename

class PanoramaNRDataLoader(Dataset):
    def __init__(self, data_dir, opt):
        super(PanoramaNRDataLoader, self).__init__()


        input_files = sorted(os.listdir(data_dir))

        self.input_filenames = [os.path.join(data_dir, x) for x in input_files if is_png_file(x)]
        
        # mixed_mask_dir = 'MixedMask'

        # mm_files = sorted(os.listdir(os.path.join(data_dir, mixed_mask_dir)))

        # self.mm_filenames = [os.path.join(data_dir, mixed_mask_dir, x) for x in mm_files if is_png_file(x)]

        self.opt=opt

        self.tar_size = len(self.input_filenames)


    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        input_filename = os.path.split(self.input_filenames[tar_index])[-1][:-4]
        # mm_filename = os.path.split(self.mm_filenames[tar_index])[-1][:-4]

        inp_np = np.float32(load_img(self.input_filenames[tar_index]))
        # mm_np = np.float32(load_img(self.mm_filenames[tar_index]))


        # if self.opt.val_reference_type == 'no':
        #     rm_np = 1 - np.float32(load_img(self.rm_filenames[tar_index]))
        #     gt_np = gt_np * rm_np
        #     inp_np = inp_np * rm_np
        
        # numpy to tensor
        inp = torch.from_numpy(inp_np).permute(2,0,1)
        
        # Resize to pre-defined size
        # gt = TF.resize(gt, (self.opt.img_height, self.opt.img_width))
        inp = TF.resize(inp, (self.opt.img_height, self.opt.img_width))
        # mm = TF.resize(mm, (self.opt.img_height, self.opt.img_width))

        return inp, input_filename

def get_training_data(data_dir, opt):
    assert os.path.exists(data_dir)
    return PanoramaDataLoader('train', data_dir, opt)

def get_validation_data(data_dir, opt):
    assert os.path.exists(data_dir)
    return PanoramaDataLoader('val', data_dir, opt)

def get_test_data(data_dir, img_options=None):
    assert os.path.exists(data_dir)
    return PanoramaDataLoader('test', data_dir, img_options)

def get_test_data_nr(data_dir, opt):
    assert os.path.exists(data_dir)
    return PanoramaNRDataLoader(data_dir, opt)