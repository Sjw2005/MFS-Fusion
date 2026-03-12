import os.path
import random
import numpy as np
import torch
import cv2
import torch.utils.data as data
import utils.utils_image as util


class Dataset(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A = util.get_image_paths(opt['dataroot_A'])
        self.paths_B = util.get_image_paths(opt['dataroot_B'])


    def __getitem__(self, index):
        A_path = self.paths_A[index]
        B_path = self.paths_B[index]


        img_A = cv2.imread(A_path,0)
        img_A = np.expand_dims(img_A, axis=2)
        img_B = cv2.imread(B_path,0)
        img_B = np.expand_dims(img_B, axis=2)
        # After loading grayscale image and channel expand:
        # img_A/img_B: [H, W, 1], dtype=uint8, value range [0,255]

        # print('img.shape',img_A.shape,img_B.shape)

        if self.opt['phase'] == 'train':
            H, W, _ = img_A.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))

            patch_A = img_A[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_B = img_B[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            # Random crop keeps single channel:
            # patch_A/patch_B: [patch_size, patch_size, 1] (default patch_size=256)
            mode = random.randint(0,7)
            patch_A, patch_B = util.augment_img(patch_A, mode=mode), util.augment_img(patch_B, mode=mode)
            # Data augmentation (flip/rotate) keeps shape unchanged:
            # patch_A/patch_B: [patch_size, patch_size, 1]
            img_A = torch.from_numpy(np.ascontiguousarray(patch_A)).permute(2, 0, 1).float().div(255.)
            img_B = torch.from_numpy(np.ascontiguousarray(patch_B)).permute(2, 0, 1).float().div(255.)
            # HWC -> CHW and normalize to [0,1]:
            # img_A/img_B: [1, patch_size, patch_size] (float32)

            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

        else:
            img_A = np.float32(img_A/255.)
            img_B = np.float32(img_B/255.)
            # In test phase, full image resolution is preserved:
            # img_A/img_B: [H, W, 1] (float32)
            img_A = torch.from_numpy(np.ascontiguousarray(img_A)).permute(2, 0, 1).float()
            img_B = torch.from_numpy(np.ascontiguousarray(img_B)).permute(2, 0, 1).float()
            # Convert to model input format:
            # img_A/img_B: [1, H, W]

            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)
