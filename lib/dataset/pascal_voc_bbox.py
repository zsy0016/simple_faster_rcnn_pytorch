import numpy as np


def resize_bbox(bbox, in_size, out_size):
    """Resize bboxes according to image in and out size.

    The bounding boxes are in a two-dimensional tensor. The tensor shape is :math:'(R, 4)'. 
    :math:'R' in the number of bounding boxes, and the four attributes are in the format of 
    :math:'(y_{min}, x_{min}, y_{max}, x_{max})'.

    Args:
        bbox (numpy.array): An array with a shape of :math:'(R, 4)'. :math:'R' is the number 
            of bounding boxes. The coordinates are :math:'p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}'.
        in_size (tuple): A tuple of length 2. The height and width of the image before resized.
        out_size (tuple): A tuple of length 2. The height and width of the image after resized.
    
    Returns:
        numpy.array: Resized bboxes with a shape of :math:'(R, 4)'.
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / float(in_size[0])
    x_scale = float(out_size[1]) / float(in_size[1])
    bbox[:, 0] *= y_scale
    bbox[:, 1] *= x_scale
    bbox[:, 2] *= y_scale
    bbox[:, 3] *= x_scale

    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bboxes accordingly.

    The bounding boxes are in a two-dimensional tensor. The tensor shape is :math:'(R, 4)'. 
    :math:'R' in the number of bounding boxes, and the four attributes are in the format of 
    :math:'(y_{min}, x_{min}, y_{max}, x_{max})'.

    Args:
        bbox (numpy.array):  An array with a shape of :math:'(R, 4)'. :math:'R' is the number 
            of bounding boxes. The coordinates are :math:'p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}'.
        size (tuple): A tuple of length 2. The height and width of the image.
        x_flip (bool): Flip bounding boxes horizontally.
        y_flip (bool): Flip bounding boxes vetically.

    Returns:
        numpy.array: Flipped bboxes with a shape of :math:'(R, 4)'.
    """
    bbox = bbox.copy()
    H, W = size
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max

    return bbox
