import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from dataset.pascal_voc_img import normalize_image, resize_image
from util.bbox import loc2bbox

from .backbone import get_extractor_classifier
from .head import Head
from .rpn import RPN


class FasterRCNN(nn.Module):
    """Base Faster R-CNN model class.

    This is a base class for Faster R-CNN. The following three stages constitute Faster R-CNN.

    * **Feature extraction**: Image feature maps are obtained by backbone.
    * **Region Proposal Network**: Take feature maps and provide a set of anchors as RoIs around 
        objects for the next stage.
    * **Localization and Classification Head**: Using feature maps and provided RoIs classify 
        the categories of objects and revise the localizations. 
    
    Each stage is carried out by one callable, which are :class:'nn.Module' objects 
    :obj:'feature', :obj:'rpn' and :obj:'head'.

    There are two functions :method:'predict' and :method:'__call__'  for  object detection. 
    :method:'predict' is for a scenario, and :method:'__call__' is for training and debugging.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. 
    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image tensor and returns feature maps.
        rpn (nn.Module): Refer to the documentation of :class:'model.rpn.RPN.
        head (nn.Module): A module that takes a BCHW feature map, RoI and batch indices for RoIs. 
            This returns dependent localization parameters and class scores.
        loc_mean (tuple of four floats): Mean values of localization estimates.
        loc_std (tuple of four floats): Standard deviations of localization estimates.
    """
    def __init__(self, n_class=21, backbone='vgg16', loc_mean=(0.,0.,0.,0.), loc_std=(.1,.1,.2,.2)):
        super().__init__()
        self.n_class = n_class
        self.rpn = RPN()
        self.extractor, classifier = get_extractor_classifier(backbone)
        self.head = Head(classifier, n_class)
        self.loc_mean = torch.FloatTensor(loc_mean)
        self.loc_std  = torch.FloatTensor(loc_std)
        self.use_preset('evaluate')

    def use_preset(self, preset, score_threshold=0.05):
        """Use the given preset during prediction.

        This method sets the values of :obj:`self.nms_threshold` and :obj:`self.score_threshold`. 
        These values are for non maximum suppression and discarding low confidence proposals in 
        :method:`predict`, respectively.

        Args:
            preset ({'visualize', 'evaluate'}): Preset mode.
        """
        if preset == 'visualize':
            self.nms_threshold   = 0.3
            self.score_threshold = 0.7
        elif preset == 'evaluate':
            self.nms_threshold   = 0.3
            self.score_threshold = score_threshold
        else:
            raise ValueError('preset must be visualize or evaluate')

    def forward(self, img, scale=1.):
        """Forward Faster R-CNN.

        :obj:'scale' is the scaling parameter used after reading images.

        Notations:

        * :math:'N' is the batch size.
        * :math:'R' is the total number of proposed anchors.
        * :math:'R'' is the total number of RoIs produced across batches.
        * :math:'L' is the number of classes excluding background.

        Args:
            img (torch.Tensor): BCHW image tensor.
            scale (float): scaling parameter during image preprocessing.
        
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)

            * **rpn_loc**: Offsets and scales for anchors. Its shape is :math:'(R, 4)'.
            * **rpn_score**: Class predictions for the anchors. Its shape is :math:'(R, 2)'.
            * **roi** Coordinates for rois. Its shape is :math:'(R', (L + 1) 4)'.
            * **roi_loc** Offsets and scales for rois. Its shape is :math:'(R', (L + 1) 4)'.
            * **roi_score**: Class predictions for the rois. Its shape is :math:'(R', (L + 1))'.
        """
        img_size = img.shape[2:]
        h = self.extractor(img)
        anchor, rpn_loc, rpn_score, roi = self.rpn(h, img_size, scale)
        roi_loc, roi_score = self.head(h, roi)

        return anchor, rpn_loc, rpn_score, roi, roi_loc, roi_score

    def _suppress(self, cls_bbox, cls_prob):
        bbox  = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        cls_bbox = cls_bbox.reshape((-1, self.n_class, 4))
        for l in range(1, self.n_class):
            cls_bbox_ = cls_bbox[:, l, :]
            cls_prob_ = cls_prob[:, l]
            mask = torch.where(cls_prob_ >= self.score_threshold)[0]
            if mask.numel() == 0:
                continue
            cls_bbox_ = cls_bbox_[mask]
            cls_prob_ = cls_prob_[mask]
            keep_index = nms(cls_bbox_, cls_prob_, self.nms_threshold)
            bbox.append(cls_bbox_[keep_index])
            label.append(l * torch.ones_like(keep_index).to(cls_prob_.device))
            score.append(cls_prob_[keep_index])
        if len(bbox) == 0:
            bbox = np.empty((0, 4)).astype(np.float32)
            label = np.empty((0, )).astype(np.int32)
            score = np.empty((0, 21)).astype(np.float32)
        else:
            bbox  = torch.cat(bbox, dim=0).detach().cpu().numpy()
            label = torch.cat(label, dim=0).detach().cpu().numpy()
            score = torch.cat(score, dim=0).detach().cpu().numpy()
        return bbox, label, score

    def predict(self, img, scale, visualize=False):
        """Detect objects from images.

        This method detects objects in each image when testing.

        Args: 
            imgs (numpy.array): Arrays of images in the CHW format and RGB mode. The range of 
            values are :math:'[0, 255]'.
        
        Returns:
            tuple of lists of numpy.array: :obj:'(bboxes, labels, scores)'.

            * **bboxes**: A list of :obj:'numpy.float32' of bounding boxes. A :obj:'numpy.array' 
                has a shape of :math:'(R, 4)'. :math:'R' is the number of bounding boxes.
            * **labels**: A list of :obj:'numpy.int32' of bounding boxes. A :obj:'numpy.array' 
                has a shape of :math:'(R,)'. :math:'R' is the number of bounding boxes and its 
                value is :math:'[0, self.n_class-2]'
            * **scores**: A list of :obj:'numpy.float32' of bounding boxes. A :obj:'numpy.array' 
                has a shape of :math:'(R,)'. :math:'R' is the number of bounding boxes.
        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            img /= 255.
            img = img.transpose((2, 0, 1))
            size = img.shape[1:]
            img = resize_image(img)
            img = normalize_image(img)
            o_size = img.shape[1:]
            scale = o_size[1] / size[1]
            img = torch.from_numpy(img).cuda()
        else:
            self.use_preset('evaluate')

        # using gpu default
        mean = torch.FloatTensor(self.loc_mean).cuda()
        std  = torch.FloatTensor(self.loc_std).cuda()
        _, _, _, roi, roi_loc, roi_score = self(img, scale)

        # Convert prediction to the coordinates of bounding boxes
        roi_loc = roi_loc.view(-1, self.n_class, 4)
        roi_loc = roi_loc * std + mean
        roi = roi.view(-1, 1, 4).expand_as(roi_loc)
        roi = roi.reshape(-1, 4)
        roi_loc = roi_loc.reshape(-1, 4)
        bbox = loc2bbox(roi, roi_loc)
            
        bbox[:, 0::2] = torch.clamp(bbox[:, 0::2], 0, img.shape[2])
        bbox[:, 1::2] = torch.clamp(bbox[:, 1::2], 0, img.shape[3])
        prob = F.softmax(roi_score, dim=1)
        bbox, label, score = self._suppress(bbox, prob)

        self.use_preset('evaluate')
        self.train()

        return bbox, label, score
