import sys
import os
import numpy as np
from skimage import io
from utils import merge_files, shuffle_lines
from PIL import Image
import hyperparameters as hyp
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data


class nightmareDataset(data.Dataset):
    def __init__(self, file_path = 'data_files', grades=['A'], jitter = False, isTrain=True, isTest=False, animals=True):
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
        if jitter:

            self.transform = [self.hRotation, self.vRotation, self.randomCrop, self.lighten, self.norm]
        else:
            self.transform = [self.hRotation, self.vRotation, self.randomCrop, self.norm]

        self.isTest = isTest
        self.isTrain = isTrain

        self.grades = grades
        self.animals = animals

        if self.isTrain:
            string='train'
        else:
            string='val'
        if not self.isTest:
            if not self.animals:
                file_list = []
                for grade in self.grades:
                    file_list.append(os.path.join(file_path, string+grade+'.txt'))
                merge_files(file_list, os.path.join(file_path, string+'.txt'))
                shuffle_lines(os.path.join(file_path, string+'.txt'))
                lines = open(os.path.join(file_path, string+'.txt'))
                self.txtfile = [line.strip('\n').split(',') for line in lines]
            else:
                lines = open(os.path.join(file_path, string+'_animals.txt'))
                self.txtfile = [line.strip('\n').split(',') for line in lines]
        else:
            lines = open(os.path.join(file_path, 'test.txt'))
            self.txtfile = [line.strip('\n') for line in lines]


    def __len__(self):
        return len(self.txtfile)

    def __getitem__(self, idx):
        if not self.isTest:
            im =Image.open(self.txtfile[idx][0]).convert('RGB')
        else:
            im =Image.open(self.txtfile[idx]).convert('RGB')
        if self.animals:
            w,h = im.size
            im = im.crop((0,0,w,h-20))
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

def get_loader(loader, grades=None, jitter=False, animals=True):
    if loader == 'train':
        dataset = nightmareDataset(grades=grades, jitter=False, isTrain=True, animals=animals)
        dataloader = data.DataLoader(dataset, shuffle=True, batch_size=int(48//(hyp.DEPTH//20)*2), pin_memory=True)
    elif loader == 'val':
        dataset = nightmareDataset(grades=grades, isTrain=False, animals=animals)
        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=int(48//(hyp.DEPTH//20)*2), pin_memory=True)
    elif loader == 'test':
        dataset = nightmareDataset(isTrain=False, isTest=True)
        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=int(48//(hyp.DEPTH//20)*2), pin_memory=True)

    return dataloader
