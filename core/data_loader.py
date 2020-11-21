"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import math

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets,self.target2 = self._make_dataset(root)
        self.transform = transform
    #改数据集就在这里改
    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        whole=[]
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
            whole+=[cls_fnames]            
        triple1,triple2,triple3=[],[],[]
        label_org,label_trg=[],[]
        for i in range(len(domains)):            
            names1=whole[i]
            for j in range(len(domains)):
                if i==j:
                    continue
                names2=whole[j]
                #trans=[]
                for m in range(len(whole[i])):
                    base=names1[m]
                    trans=[]
                    for n in range(len(whole[j])):                        
                        duibi=names2[n]
                        if base.parts[3].split('_')[0]==duibi.parts[3].split('_')[0]:
                            #triple1+=names1
                            triple1.append(base)                          
                            triple2.append(duibi)
                            trans.append(duibi)
                    triple3+=random.sample(trans,len(trans))
                    label_org+=[i]*len(trans)
                    label_trg+=[j]*len(trans)        
        return list(zip(triple1, triple2,triple3)), label_org,label_trg

    def __getitem__(self, index):
        fname, fname2,fname3 = self.samples[index]
        label_org = self.targets[index]
        label_trg = self.target2[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        img3 = Image.open(fname3).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img, img2,img3, label_org,label_trg

    def __len__(self):
        return len(self.targets)

class ProduceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
    #改数据集就在这里改
    def _make_dataset(self, root):
        domains = os.listdir(root)        
        whole=[]
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)           
            whole+=[cls_fnames]            
        double1,double2=[],[]
        label_pro=[]        
        for i in range(len(domains)):            
            names1=whole[i]
            for j in range(len(domains)):
                if i==j:
                    continue
                names2=whole[j]                                
                for m in range(len(whole[i])):
                    base=names1[m]
                    temp_save=[]                    
                    for n in range(len(whole[j])):
                        duibi=names2[n]
                        if base.parts[3].split('_')[0]==duibi.parts[3].split('_')[0]:                                             
                            temp_save.append(duibi)
                    double1.append(base)
                    if len(temp_save)>0:                                                     
                        k=random.randrange(0,len(temp_save))                        
                        double2.append(temp_save[k])
                    else:
                        k=random.randrange(0,len(whole[j]))                        
                        double2.append(names2[k])
                label_pro+=[j]*len(whole[i])                
        return list(zip(double1,double2)), label_pro
    def __getitem__(self, index):
        fname, fname2= self.samples[index]
        label = self.targets[index]        
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')        
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)            
        return img, img2,label
    def __len__(self):
        return len(self.targets)



def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)
   
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform =transforms.Compose([
        RandomSizedRectCrop(img_size[0], img_size[1]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform,
    ])

    if which == 'source':
        dataset = ImageFolder(root, transform)        
        '''i=1
        for a,b in dataset:            
            unloader = transforms.ToPILImage()
            img=unloader(a)
            filename='%s.jpg' % i
            img.save(filename)
            i=i+1'''
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform) 
    elif which == 'produce':
        dataset = ProduceDataset(root, transform)
        return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)

class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)
class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)

def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size[0], img_size[1]
        #mean = [0.5, 0.5, 0.5]
        #std = [0.5, 0.5, 0.5]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize([img_size[0], img_size[1]]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    '''transform = transforms.Compose([
        transforms.Resize([img_size[0], img_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])'''
    transform = transforms.Compose([
        transforms.Resize([img_size[0], img_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])#实验证明这种标准化比0.5好用
    dataset = ImageFolder(root, transform)
    i=1
    '''for a,b in dataset:            
        unloader = transforms.ToPILImage()
        img=unloader(a)
        filename='%s.jpg' % i
        img.save(filename)
        i=i+1'''
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def _fetch_train(self):
        try:
            x, x2,x3, y1,y2 = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter= iter(self.loader)
            x, x2,x3, y1,y2 = next(self.iter)
        return x, x2,x3, y1,y2

    def _fetch_pro(self):
        try:
            x, x2, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, x2, y = next(self.iter)
        return x, x2, y
    def __next__(self):      
        if self.mode == 'train':
            x,x_ref, x_ref2, y_org, y_ref = self._fetch_train()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y_org, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            x, y = self._fetch_inputs()
            inputs = Munch(x=x, y=y)
        elif self.mode == 'eval':
            x, x_ref,y_ref = self._fetch_pro()
            inputs = Munch(x_src=x,  x_ref=x_ref, y_ref=y_ref)   
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})