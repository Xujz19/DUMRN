import os
import imageio
import torch
from data import common
import numpy as np
import torch.utils.data as data


class DIV2K(data.Dataset):
    def __init__(self, args, mode='train'):
        super(DIV2K, self).__init__()
        self.repeat = 50
        self.args = args
        self.n_colors = args.n_colors
        self.sigma = args.sigma
        self.mode = mode
        self.root = os.path.join(args.train_data, 'DIV2K_train_HR')

        self.file_list = []

        self._scan()

    def _scan(self):
        for sub, dirs, files in os.walk(self.root):
            if not dirs:
                file_list = [os.path.join(sub, f) for f in files]
                self.file_list += file_list
        return

    def __getitem__(self, idx):
        idx = idx % len(self.file_list)
        if self.n_colors == 3:
            sharp = imageio.imread(self.file_list[idx], pilmode='RGB')
        elif self.n_colors == 1:
            sharp = imageio.imread(self.file_list[idx], pilmode='L')
            sharp = np.expand_dims(sharp, axis=3)

        H, W, C = sharp.shape
        ix = np.random.randint(0, H - self.args.patch_size + 1)
        iy = np.random.randint(0, W - self.args.patch_size + 1)

        sharp_patch = sharp[ix:ix + self.args.patch_size, iy:iy + self.args.patch_size, :]

        aug_mode = np.random.randint(0, 8)
        sharp_patch = common.augment_img(sharp_patch, aug_mode)
        sharp_patch = common.image_to_tensor(sharp_patch)

        noise = torch.randn(sharp_patch.size()).mul_(self.sigma/255.0)
        noisy_patch = sharp_patch.clone()
        noisy_patch.add_(noise)

        return noisy_patch, sharp_patch

    def __len__(self):
        return len(self.file_list) * self.repeat

