import sys
import os
import numpy as np
from skimage import io
from PIL import Image

import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data


class nightmareDataset(data.Dataset):
    def __init__(self, txtfile='train.txt', isTrain=True):
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

        lines = open(txtfile).readlines()
        self.txtfile = [line.strip('\n').split(', ') for line in lines]
        self.isTrain = isTrain

    def __len__(self):
        return len(self.txtfile)

    def __getitem__(self, idx):
        im =Image.open(self.txtfile[idx][0]).convert('RGB')
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
        label = int(self.txtfile[idx][1])
        return im, label

if __name__ == "__main__":
    dataset = nightmareDataset(txtfile='val.txt', isTrain=False)
    dataloader = data.DataLoader(dataset, shuffle=True, batch_size=32)
    for i, (data, target) in enumerate(dataloader):
        print(data.shape)
