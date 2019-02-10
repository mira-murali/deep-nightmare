import sys
import os
import numpy as np
from skimage import io
from PIL import Image
import hyperparameters as hyp
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data


class nightmareDataset(data.Dataset):
    def __init__(self, txtfile='train.txt', isTrain=True, isTest=False):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.toTensor = transforms.ToTensor()

        self.hRotation = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(p=1), \
        self.toTensor, self.normalize])

        self.vRotation = transforms.Compose([transforms.Resize(224), transforms.RandomVerticalFlip(p=1),\
        self.toTensor, self.normalize])

        self.randomCrop = transforms.Compose([transforms.RandomCrop(224), self.toTensor, self.normalize])

        self.lighten = transforms.Compose([transforms.Resize(224), transforms.ColorJitter(brightness=2.0, contrast=2.5, hue=0.0),\
         self.toTensor, self.normalize])

        self.norm = transforms.Compose([transforms.Resize(224), self.toTensor, self.normalize])
        self.transform = [self.hRotation, self.vRotation, self.randomCrop, self.lighten, self.norm]

        self.isTest = isTest
        self.isTrain = isTrain
        lines = open(txtfile).readlines()
        if not self.isTest:
            self.txtfile = [line.strip('\n').split(',') for line in lines]
        else:
            self.txtfile = [line.strip('\n') for line in lines]


    def __len__(self):
        return len(self.txtfile)

    def __getitem__(self, idx):
        if not self.isTest:
            im =Image.open(self.txtfile[idx][0]).convert('RGB')
        else:
            im =Image.open(self.txtfile[idx]).convert('RGB')
        im = np.asarray(im, np.uint8)
        if len(im.shape) > 2 and im.shape[-1] > 3:
            im = im[:, :, :3]
        if im.shape[0] < 224:
            diff = 224 - im.shape[0]
            if diff%2 == 0:
                im = np.pad(im, ((diff//2, diff//2), (0, 0), (0, 0)), mode='constant')
            else:
                im = np.pad(im, ((diff//2, diff//2+1), (0, 0), (0, 0)), mode='constant')
        if im.shape[1] < 224:
            diff = 224 - im.shape[1]
            if diff%2 == 0:
                im = np.pad(im, ((0, 0), (diff//2, diff//2), (0, 0)), mode='constant')
            else:
                im = np.pad(im, ((0, 0), (diff//2, diff//2+1), (0, 0)), mode='constant')

        im = Image.fromarray(im)
        index = np.random.randint(len(self.transform))
        if self.isTrain:
            im = self.transform[index](im)
        else:
            im = self.norm(im)
        if im.shape[1] >= im.shape[2]:
            im = im[:, :im.shape[2], :]
        elif im.shape[2] > im.shape[1]:
            im = im[:, :, :im.shape[1]]
        if not self.isTest:
            label = int(self.txtfile[idx][1])
        else:
            label = 0
        return im, label

def get_loader(loader):
    if loader == 'train':
        dataset = nightmareDataset(txtfile='train.txt', isTrain=True)
        dataloader = data.DataLoader(dataset, shuffle=True, batch_size=48//(hyp.DEPTH//50), pin_memory=True)
    elif loader == 'val':
        dataset = nightmareDataset(txtfile='val.txt', isTrain=False)
        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=48//(hyp.DEPTH//50), pin_memory=True)
    elif loader == 'test':
        dataset = nightmareDataset(txtfile='test.txt', isTrain=False, isTest=True)
        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=48//(hyp.DEPTH//50), pin_memory=True)

    return dataloader
