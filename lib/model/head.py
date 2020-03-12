import torch
from torch import nn
from torch.nn.init import normal_, zeros_
from torchvision.ops import roi_pool


class Head(nn.Module):
    """This class outputs class-wise localizations and classifications based on feature maps in rois.

    Args: 
        n_class (int): The number of foreground object classed.
        roi_size (int): Height and width of features after RoI-pooling.
        spatial_scale (float): The scaling downsample of extractor.
        classifier (nn.Module): Two fully-connected layers from vgg16.
    """
    def __init__(self, classifier, n_class, roi_size=(7, 7), spatial_scale=1 / 16.):
        super().__init__()
        self.classifier = classifier
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        self.loc_fc   = nn.Linear(4096, n_class * 4)
        self.score_fc = nn.Linear(4096, n_class)
        normal_(self.loc_fc.weight, 0, 0.01)
        normal_(self.score_fc.weight, 0, 0.01)
        zeros_(self.loc_fc.bias)
        zeros_(self.score_fc.bias)
    
    def forward(self, x, roi):
        """Forward FasterRCNNVGG16 head.

        Args:
            x (torch.Tensor): Feature maps in BCHW format.
            roi (torch.Tensor): A bounding box tensor with a shape of :math:'(R, 4)'. :math:'R' 
            is the number of bounding boxes. The coordinates are 
            :math:'p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}'.
        """
        roi_ = torch.zeros((roi.shape[0], 5)).float().to(x.device)
        # (ymin, xmin, ymax, xmax) -> (id, xmin, ymin, xmax, ymax)
        roi_[:, 1] = roi[:, 1]
        roi_[:, 2] = roi[:, 0]
        roi_[:, 3] = roi[:, 3]
        roi_[:, 4] = roi[:, 2]
        pool = roi_pool(x, roi_, self.roi_size, self.spatial_scale)
        pool = pool.view(pool.shape[0], -1)
        fc7 = self.classifier(pool)
        roi_cls_loc = self.loc_fc(fc7)
        roi_cls_score = self.score_fc(fc7)

        return roi_cls_loc, roi_cls_score
