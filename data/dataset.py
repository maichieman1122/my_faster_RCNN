#!/usr/bin/env python
# coding: utf-8

# In[23]:


from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from our_utils.config import opt

def pytorch_normalze(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=1024, max_size=2048):
    C, H, W=img.shape
    scale1 =min_size / min(H, W)
    scale2 =max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img/255.
    img=sktsf.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)
    
    normalize=pytorch_normalze
    return normalize(img)

class Transform(object):
    def __init__(self, min_size=1024, max_size=2048):
        self.min_size=min_size
        self.max_size=max_size
    def __call__(self, in_data):
        img, bbox, label=in_data
        _,H,W=img.shape
        img=preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W=img.shape
        scale=o_H / H
        bbox=util.resize_bbox(bbox,(H,W),(o_H, o_W))
        
        #horizontally flip
        img, params=util.random_flip(img, x_random=True, return_param=True)
        bbox=util.flip_bbox(bbox,(o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale
class Dataset:
    def __init__(self, opt):
        self.opt=opt
        self.db=VOCBboxDataset(opt.voc_data_dir)
        self.tsf=Transform(opt.min_size, opt.max_size)
    
    def __getitem__(self, idx):
        ori_img, bbox, label, difficult= self.db[idx]
        img, bbox, label, scale=self.tsf((ori_img, bbox, label))
        return img.copy(), bbox.copy(), label.copy(), scale
    
    def __len__(self):
        return int(self.db.__len__())
    
class TestDataset:
    def __init__(self, opt, split='test',use_difficult=True):
        self.opt=opt
        self.db=VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)
    
    def __getitem__(self, idx):
        ori_img, bbox, label, difficult=self.db[idx]
        img=preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult
    def __len__(self):
        return int(self.db.__len__())

