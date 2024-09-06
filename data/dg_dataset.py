# -*- coding: utf-8 -*-

import torch
import os
import h5py
import numpy as np
import pickle
from skimage.transform import resize, rescale, rotate
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class Dg_dataset(Dataset):
    '''pytorch dataset for dataloader'''

    def __init__(self, h5path, pklpath, datatype, transform=None):
        super().__init__()
        self.h5data = h5path
        self.transform = transform
        self.pklpath = pklpath
        # Read pkl information
        train_data_id = []
        test_data_id = []
        pklinfo = pickle.load(open(pklpath, 'rb'))
        for pth_id in pklinfo.keys():
            if str(pklinfo[pth_id]['Use']) == '1' and str(pklinfo[pth_id]['Seg']) == '1':
                train_data_id.append(str(pth_id))
            else:
                test_data_id.append(str(pth_id))

        # Select the used.h5 files
        self.chooseslicepath = []
        for root, dirs, files in os.walk(h5path):
            for dir in dirs:
                file_path = os.path.join(root, dir)
                file_temp = file_path
                for slice in os.listdir(file_path):
                    slicename = slice.split('.')[0].split('_')[0] + '_' + slice.split('.')[0].split('_')[1]
                    if datatype == 'train':
                        if slicename in train_data_id:
                            self.chooseslicepath.append(os.path.join(file_temp, slice))
                    elif datatype == 'valid':
                        if slicename in test_data_id:
                            self.chooseslicepath.append(os.path.join(file_temp, slice))
                    else:
                        raise ValueError('The datatype must be set train or test.')

    def __len__(self):
        return len(self.chooseslicepath)

    def __getitem__(self, idx):
        patientpath = self.chooseslicepath[idx]
        # Get data
        image = h5py.File(patientpath, 'r')['image'][:]
        label = h5py.File(patientpath, 'r')['roi'][:]

        # Augmentation
        if self.transform is not None:
            image, label = self.transform((image, label))

        image_tensor = torch.tensor(image.astype(np.float32))
        label_tensor = torch.tensor(label.astype(np.float32))
        # Add one channel
        image_tensor = image_tensor.unsqueeze(0)
        label_tensor = label_tensor.unsqueeze(0)
        return image_tensor, label_tensor


class Dg_dataset_all(Dataset):
    '''pytorch all dataset for dataloader'''

    def __init__(self, h5path, pklpath, transform=None):
        super().__init__()
        self.h5data = h5path
        self.transform = transform
        self.pklpath = pklpath
        # Read pkl information
        alldata = []
        pklinfo = pickle.load(open(pklpath, 'rb'))

        for pth_id in pklinfo.keys():
            if str(pklinfo[pth_id]['Use']) == '1':
                alldata.append(str(pth_id))

        # Select the used.h5 files
        slicelist = os.listdir(h5path)
        self.chooseslicepath = []
        for root, dirs, files in os.walk(h5path):
            for dir in dirs:
                file_path = os.path.join(root, dir)
                file_temp = file_path
                for slice in os.listdir(file_path):
                    slicename = slice.split('.')[0].split('_')[0] + '_' + slice.split('.')[0].split('_')[1]
                    if slicename in alldata:
                        self.chooseslicepath.append(os.path.join(file_temp, slice))
                    else:
                        continue

        print("the length of alldata {}".format(len(alldata)))

    def __len__(self):
        return len(self.chooseslicepath)

    def __getitem__(self, idx):
        patientpath = self.chooseslicepath[idx]
        # Get data
        image = h5py.File(patientpath, 'r')['image'][:]
        label = h5py.File(patientpath, 'r')['roi'][:]
        # Augmentation
        if self.transform is not None:
            image = self.transform(image)
        # Tensor
        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.float32))
        # Add one channel
        image_tensor = image_tensor.unsqueeze(0)
        label_tensor = label_tensor.unsqueeze(0)
        return image_tensor, label_tensor,


def transforms(scale=None, angle=None, flip_prob=None):
    '''Three kinds of data augmentation'''
    transform_list = []
    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return Compose(transform_list)


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            (scale, scale),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (scale, scale),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask


class HorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask
