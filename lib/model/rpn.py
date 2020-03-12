import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, zeros_

from creator import RoICreator
from util.anchor import enumerate_anchor_base, generate_anchor_base


class RPN(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose 
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. 
    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of the anchors.
        anchor_scales (list of numbers): This is areas of anchors. Those areas will be the 
            product  of the square of an element in :obj:`anchor_scales` and the original area 
            of the reference window.
        feat_stride (int): Stride size after extracting features from an image.
        initialW (callable): Initial weight value. If :obj:`None` then this function uses Gaussian 
            distribution scaled by 0.1 to initialize weight. May also be a callable that takes an 
            array and edits its values.
        proposal_creator_params (dict): Key valued paramters for :class:`model.creator.RoICreator`.

    .. seealso::
        :class:'lib.creator.creator.RoICreator`
    """
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], 
        anchor_scales=[8, 16, 32], feat_stride=16, roi_creator_params=dict()):
        super(RPN, self).__init__()

        self.feat_stride = feat_stride
        self.roi_creator = RoICreator(self, **roi_creator_params)
        self.anchor_base = generate_anchor_base(feat_stride, ratios, anchor_scales)

        n_anchor        = self.anchor_base.shape[0]
        self.conv       = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.loc_conv   = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        self.score_conv = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        normal_(self.conv.weight.data, 0, 0.01)
        normal_(self.loc_conv.weight.data, 0, 0.01)
        normal_(self.score_conv.weight.data, 0, 0.01)
        zeros_(self.conv.bias.data)
        zeros_(self.loc_conv.bias.data)
        zeros_(self.score_conv.bias.data)

    def forward(self, x, img_size, scale):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (torch.Tensor): The Features extracted from images. Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`, which is the scaled image size.
            scale (float): scaling factor of input images after reading them from files.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for anchors. 
                Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for anchors. 
                Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of proposal boxes. 
                This is a concatenation of bounding box arrays from multiple images in the batch. 
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted bounding boxes from the 
                :math:`i` th image, :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to which RoIs correspond to. 
                Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated anchors. Its shape is :math:`(H W A, 4)`.
        """

        # NOTE: only batch size of 1 is currently supported.
        _, _, height, width = x.shape

        # Convolutional layer to calculate loc and score of anchors.
        anchor    = enumerate_anchor_base(self.anchor_base, self.feat_stride, height, width)
        anchor    = torch.from_numpy(anchor).to(x.device)
        h         = F.relu(self.conv(x))
        rpn_loc   = self.loc_conv(h)
        rpn_loc   = rpn_loc.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        rpn_score = self.score_conv(h)
        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        
        # Revise anchor using loc, propose rois using nms.
        rpn_softmax_score = F.softmax(rpn_score, dim=-1)
        rpn_fg_score      = rpn_softmax_score[:, 1]
        roi = self.roi_creator(anchor, rpn_loc, rpn_fg_score, img_size, scale)

        return anchor, rpn_loc, rpn_score, roi
