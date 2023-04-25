from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torch.autograd import Variable
# from .utils import download_url, check_integrity

npysize = 32 * 2
# npysize = 48 * 2

# npypath = '/media/data1/wentao/tianchi/luna16/cls/crop_v3/'
class lunanod(data.Dataset):

    def __init__(self, npypath, fnamelst, labellst, train=True,val=False,
                 transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        # now load the picked numpy arrays
        self.val = val
        if self.train:
            self.train_data = []
            self.train_labels = []
            for label, fentry in zip(labellst, fnamelst):
                file = os.path.join(npypath, fentry)
                self.train_data.append(np.load(file))
                self.train_labels.append(label)
            self.train_data = np.concatenate(self.train_data)
            self.train_data = np.array(self.train_data)
            self.train_labels = np.array(self.train_labels)
            self.train_data = self.train_data.reshape((len(fnamelst), npysize, npysize, npysize)) # 32 32 32
            self.train_len = len(fnamelst)
            print(f'Training data: {len(fnamelst)}')
            print(self.train_data.shape)

        elif self.val:
            self.val_data = []
            self.val_labels = []
            for label, fentry in zip(labellst, fnamelst):
                file = os.path.join(npypath, fentry)
                self.val_data.append(np.load(file))
                self.val_labels.append(label)
            # self.train_data = np.concatenate(self.train_data)  # 若此处执行concatenate操作，后面需要reshape回来
            self.val_data = np.array(self.val_data)
            self.val_labels = np.array(self.val_labels)
            # self.train_data = np.transpose(self.train_data, (0, 2, 3, 1))  # 32 32 32  # transform操作需要C * H * W
            self.val_len = len(fnamelst)
            print(f'Valid data: {len(fnamelst)}')
            print(self.val_data.shape)

        else:  # 对训练集进行同样操作
            self.test_data = []
            self.test_labels = []
            # self.test_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                if not isinstance(fentry,str):  # 用来判断object的类型是不是我们指定的类型
                    self.test_data.append(fentry)
                    self.test_labels.append(label)
                    # print('1')
                else:
                    file = os.path.join(npypath, fentry)
                    self.test_data.append(np.load(file))
                    self.test_labels.append(label)
            self.test_data = np.concatenate(self.test_data)
            self.test_data = np.array(self.test_data)
            self.test_labels = np.array(self.test_labels)
            self.test_data = self.test_data.reshape((len(fnamelst), npysize, npysize, npysize))  # 32 32 32
            self.test_len = len(fnamelst)
            print(f'Testing data: {len(self.test_labels)}')
            print(self.test_data.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target= self.train_data[index], self.train_labels[index]
        elif self.val:
            img, target= self.val_data[index], self.val_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        # img = torch.from_numpy(img)
        # img = img.cuda(async = True)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print('1', img.shape, type(img))
        # img = Image.fromarray(img)
        # print('2', img.size)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img.shape, target.shape, feat.shape)
        # print(target)

        return img, target

    def __len__(self):
        if self.train:
            return self.train_len
        elif self.val:
            return self.val_len
        else:
            return self.test_len