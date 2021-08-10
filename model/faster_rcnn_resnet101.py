#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import resnet101
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from our_utils import array_tool as at
from our_utils.config import opt

def decom_resnet101():
    model=resnet101(pretrained=True)
    feature_map=t.nn.Sequential(*list(resnet101(pretrained=True).children())[:-2])
    classifier=model.fc
    return feature_map, classifier

class FasterRCNNResnet101(FasterRCNN):
    feat_stride=16
    def __init__(self,n_fg_class=3, 
                ratios=[0.5,1,2],
                anchor_scales=[8,16,32,64]):
        extractor, classifier=decom_resnet101()
        
        rpn=RegionProposalNetwork(2048, 2048,
                                 ratios=ratios,
                                 anchor_scales=anchor_scales,
                                 feat_stride=self.feat_stride,)
        head= Resnet101RoIHead(n_class=n_fg_class+1,
                              roi_size=7,
                              spatial_scale=(1./self.feat_stride),
                              classifier=classifier)
        
        super(FasterRCNNResnet101, self).__init__(
            extractor,
            rpn,
            head,
        )
class Resnet101RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16
    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(Resnet101RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.
        We assume that there are :math:`N` batches.
        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
