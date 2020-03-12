import numpy as np
from torch import from_numpy
from skimage.io import imread
from skimage.util import img_as_float32
from skimage.transform import resize


def read_image(path):
    """Read an image from a path.

    Returns:
        numpy.array: An CHW image with a range of :math:'[0., 1.]'.
    """
    img = imread(path)
    img = img_as_float32(img)
    img = img.transpose((2, 0, 1))

    return img


def flip_image(img, y_random=False, x_random=True, return_param=True):
    """Randomly flip an image vertically and horizontally.

    Args:
        img (numpy.array): An image array in CHW format.
        y_random (bool): If :obj:'True', randomly flip vertically.
        x_random (bool): If :obj:'True', randomly flip horizontally.
        return_param: If :obj:'True', a dict recording flip along y and x axes is also returned.

    Returns:
        np.array or (numpy.array, dict)

        if :obj:'return_param=False', the img is returned.

        If :obj:'return_param=True', a tuple of img and dict is returned. The dict records the 
        flips along y and x axes.
    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = np.random.choice([True, False])
    if x_random:
        x_flip = np.random.choice([True, False])
    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]
    
    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def resize_image(img, min_size=600, max_size=1000):
    """Resize an image with length restrictions.

    The length of the shorter edge is scaled smaller than :obj:'min_size'.
    The length of the longer edge is scaled smaller than :obj:'max_size'.

    Args:
        img (numpy.array): An image array in CHW format.
        min_size (int): The upper limit of the shorter edge.
        max_size (int): The upper limit of the longer edge.
    
    Returns:
        np.array: An image array in CHW format.
    """
    _, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    size = (int(scale * H), int(scale * W))
    img = img.transpose((1, 2, 0))
    img = resize(img, size)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    return img


def normalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize the image.

    Normalize image using z-score normalization.

    Args:
        img (numpy.array): An CHW image with a range of :math:'[0., 1.]'.
        mean (list of floats): Z-score mean values.
        std (list of floats): Z-score std values.
    
    Returns:
        numpy.array: An normalized CHW image.
    """
    mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
    img = (img - mean) / std

    return img
