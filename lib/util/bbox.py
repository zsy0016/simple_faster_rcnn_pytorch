import torch
from torchvision.ops import box_iou as bbox_iou

def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by :meth:`bbox2loc`, this function decodes 
    the representation to coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding box whose center is 
    :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`, the decoded bounding box's center 
    :math:`\\hat{g}_y`, :math:`\\hat{g}_x` and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are 
    calculated by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. 
    Rich feature hierarchies for accurate object detection and semantic segmentation. CVPR 2014.

    Args:
        src_bbox (Tensor): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are 
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (Tensor): A tensor with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.

    Returns:
        tesnor:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. The second axis contains 
        four values :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin}, \\hat{g}_{ymax}, \\hat{g}_{xmax}`.
    """
    if src_bbox.shape[0] == 0:
        return torch.zeros(0, 4).float().to(src_bbox.device)
    
    src_ctr_y  = (src_bbox[:, 0] + src_bbox[:, 2]) / 2.0
    src_ctr_x  = (src_bbox[:, 1] + src_bbox[:, 3]) / 2.0
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width  = src_bbox[:, 3] - src_bbox[:, 1]

    dst_ctr_y  = src_ctr_y + src_height * loc[:, 0]
    dst_ctr_x  = src_ctr_x + src_width * loc[:, 1]
    dst_height = src_height * torch.exp(loc[:, 2])
    dst_width  = src_width * torch.exp(loc[:, 3])

    dst_bbox = torch.zeros_like(src_bbox).float().to(src_bbox.device)
    dst_bbox[:, 0] = dst_ctr_y - 0.5 * dst_height
    dst_bbox[:, 1] = dst_ctr_x - 0.5 * dst_width
    dst_bbox[:, 2] = dst_ctr_y + 0.5 * dst_height
    dst_bbox[:, 3] = dst_ctr_x + 0.5 * dst_width

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales to match the source bounding 
    boxes to the target bounding boxes. Mathematcially, given a bounding box whose center is 
    :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales :math:`t_y, t_x, t_h, t_w` can 
    be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. 
    Rich feature hierarchies for accurate object detection and semantic segmentation. CVPR 2014.

    Args:
        src_bbox (Tensor): An image coordinate tensor whose shape is :math:`(R, 4)`. :math:`R` is the 
            number of bounding boxes. These coordinates are 
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (Tensor): An image coordinate tensor whose shape is :math:`(R, 4)`. These coordinates 
            are :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

    Returns:
        tensor:
        Bounding box offsets and scales from :obj:`src_bbox` to :obj:`dst_bbox`. This has shape 
        :math:`(R, 4)`. The second axis contains four values :math:`t_y, t_x, t_h, t_w`.
    """
    src_ctr_y  = (src_bbox[:, 0] + src_bbox[:, 2]) / 2.0
    src_ctr_x  = (src_bbox[:, 1] + src_bbox[:, 3]) / 2.0
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width  = src_bbox[:, 3] - src_bbox[:, 1]

    dst_ctr_y  = (dst_bbox[:, 0] + dst_bbox[:, 2]) / 2.0
    dst_ctr_x  = (dst_bbox[:, 1] + dst_bbox[:, 3]) / 2.0
    dst_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_width  = dst_bbox[:, 3] - dst_bbox[:, 1]

    eps = 1e-3 * torch.ones(1).to(src_bbox.device)
    src_height = torch.max(src_height, eps)
    src_width  = torch.max(src_width, eps)

    loc = torch.zeros_like(src_bbox).to(src_bbox.device)
    loc[:, 0] = (dst_ctr_y - src_ctr_y) / src_height
    loc[:, 1] = (dst_ctr_x - src_ctr_x) / src_width
    loc[:, 2] = torch.log(dst_height / src_height)
    loc[:, 3] = torch.log(dst_width / src_width)

    return loc


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales to match the source bounding 
    boxes to the target bounding boxes. Mathematcially, given a bounding box whose center is 
    :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales :math:`t_y, t_x, t_h, t_w` can 
    be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. 
    Rich feature hierarchies for accurate object detection and semantic segmentation. CVPR 2014.

    Args:
        src_bbox (array): An image coordinate array whose shape is :math:`(R, 4)`. :math:`R` is the 
            number of bounding boxes. These coordinates are 
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is :math:`(R, 4)`. These coordinates 
            are :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` to :obj:`dst_bbox`. This has shape 
        :math:`(R, 4)`. The second axis contains four values :math:`t_y, t_x, t_h, t_w`.
    """
    src_ctr_y  = (src_bbox[:, 0] + src_bbox[:, 2]) / 2.0
    src_ctr_x  = (src_bbox[:, 1] + src_bbox[:, 3]) / 2.0
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width  = src_bbox[:, 3] - src_bbox[:, 1]

    dst_ctr_y  = (dst_bbox[:, 0] + dst_bbox[:, 2]) / 2.0
    dst_ctr_x  = (dst_bbox[:, 1] + dst_bbox[:, 3]) / 2.0
    dst_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_width  = dst_bbox[:, 3] - dst_bbox[:, 1]

    eps = 1e-3 * torch.ones(1).to(src_bbox.device)
    src_height = torch.max(src_height, eps)
    src_width  = torch.max(src_width, eps)

    ty = (dst_ctr_y - src_ctr_y) / src_height
    tx = (dst_ctr_x - src_ctr_x) / src_width
    th = torch.log(dst_height / src_height)
    tw = torch.log(dst_width / src_width)

    loc = torch.zeros_like(src_bbox).to(src_bbox.device)
    loc[:, 0] = ty
    loc[:, 1] = tx
    loc[:, 2] = th
    loc[:, 3] = tw

    return loc
