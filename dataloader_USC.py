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

npysize = 32*2

class lunanod(data.Dataset):

    def __init__(self, npypath, fnamelst, labellst,
                 npypath_jinshan, fnamelst_jinshan, labellst_jinshan,
                 fnamelst_another, labellst_another,
                 train=True, another=False, jinshan=False,
                 transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.another = another
        self.jinshan = jinshan
        # now load the picked numpy arrays
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
            print(f'Labeled data: {len(fnamelst)}')
            print(self.train_data.shape)
            if self.another:
                self.another_data = []
                self.another_labels = []
                for label, fentry in zip(labellst_another, fnamelst_another):
                    file = os.path.join(npypath, fentry)
                    self.another_data.append(np.load(file))
                    self.another_labels.append(label)
                self.another_data = np.concatenate(self.another_data)
                self.another_data = np.array(self.another_data)
                self.another_labels = np.array(self.another_labels)
                self.another_data = self.another_data.reshape((len(fnamelst_another), npysize, npysize, npysize))  # 32 32 32
                self.another_len = len(fnamelst_another)
                print(f'Another data: {len(fnamelst_another)}')
                print(self.another_data.shape)
                self.train_data = np.concatenate((self.train_data, self.another_data), axis=0)
                self.train_labels = np.concatenate((self.train_labels, self.another_labels), axis=0)

            if self.jinshan:
                self.jinshan_data = []
                self.jinshan_labels = []
                for label, fentry in zip(labellst_jinshan, fnamelst_jinshan):
                    file = os.path.join(npypath_jinshan, fentry)
                    self.jinshan_data.append(np.load(file))
                    self.jinshan_labels.append(label)
                self.jinshan_data = np.concatenate(self.jinshan_data)
                self.jinshan_data = np.array(self.jinshan_data)
                self.jinshan_labels = np.array(self.jinshan_labels)
                self.jinshan_data = self.jinshan_data.reshape((len(fnamelst_jinshan), npysize, npysize, npysize))  # 32 32 32
                self.jinshan_len = len(fnamelst_jinshan)
                print(f'jinshan data: {len(fnamelst_jinshan)}')
                print(self.jinshan_data.shape)
                self.train_data = np.concatenate((self.train_data, self.jinshan_data), axis=0)
                self.train_labels = np.concatenate((self.train_labels, self.jinshan_labels), axis=0)
                self.unlbl_data = np.concatenate((self.another_data, self.jinshan_data), axis=0)
                print(f'unlbl data: {len(self.unlbl_data)}')
                print('all training data:')
                print(self.train_data.shape)
        else:  # 对测试集进行同样操作
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
            # print(self.test_data.shape)
            self.test_data = self.test_data.reshape((len(fnamelst), npysize, npysize, npysize))  # 32 32 32
            # self.test_labels = np.asarray(self.test_labels)
            # self.test_data = self.test_data.transpose((0, 2, 3, 4, 1))  # convert to HWZC
            self.test_len = len(fnamelst)
            print(f'Testing data: {len(self.test_labels)}')
            print(self.test_data.shape)
            if self.train:
                self.data = self.train_data
            else:
                self.data = self.test_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target= self.train_data[index], self.train_labels[index]
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
        return len(self.data)

        # if self.train:
        #     return self.train_len
        # else:
        #     return self.test_len