#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import xml.etree.ElementTree as ET
from .util import read_image
import numpy as np


class VOCBboxDataset:
    def __init__(self,data_dir, split='trainval', use_difficult=False, return_difficult=False,):
        # data_dir: F:/tot_nghiep/dataset/data
        img_files=os.listdir(os.path.join(data_dir, split, 'image_data'))
        ids=[]
        for i in img_files:
            ids.append(i.replace('.JPG',''))
        self.ids=ids
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.split=split
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        id_=self.ids[i]
        anno=ET.parse(os.path.join(self.data_dir, self.split, 'boxes_data', id_+'.xml'))
        bbox=list()
        label=list()
        difficult=list()
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno=obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name=obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox=np.stack(bbox).astype(np.float32)
        label=np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
        
        #load a image
        img_file = os.path.join(self.data_dir,self.split,'image_data', id_+'.JPG')
        img=read_image(img_file, color=True)
        
        return img, bbox, label, difficult
    
VOC_BBOX_LABEL_NAMES=('ong_noi', 'tru_su', 'khoa_neo')

