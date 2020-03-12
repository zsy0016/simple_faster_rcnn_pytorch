import json
import os
from xml.etree import ElementTree

from torch.utils.data import Dataset

from .pascal_voc_img import *
from .pascal_voc_bbox import *


VOC_BBOX_LABEL_NAMES = (
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


class VOCDatasetBase(Dataset):
    """Dataset base for PASCAL `VOC`_. Image is as the original.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`use_difficult == False`,this dataset returns a 
    corresponding :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour. If :obj:`use_difficult == True`, this dataset returns 
    corresponding :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are in a two-dimensional array. The array shape is :math:'(R, 4)'. 
    :math:'R' in the number of bounding boxes, and the four attributes are in the format of 
    :math:'(y_{min}, x_{min}, y_{max}, x_{max})'.

    The labels are packed in a one-dimensional array. The array shape :math:`(R,)`. :math:`R` is 
    the number of bounding boxes.The class name of the label :math:`l` is :math:`l` th element 
    of :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one-dimensional boolean array of shape :math:`(R,)`. 
    :math:`R` is the number of bounding boxes in the image. If :obj:`use_difficult` is :obj:`False`, 
    this array is a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
    """
    def __init__(self, data_dir, split='trainval', use_difficult=False):
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        split_path = os.path.join(data_dir, f'ImageSets/Main/{split}.txt')
        self.ids = [id_.strip() for id_ in open(split_path)]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):
        """Returns the i-th sample.

        Returns a CHW image, its corresponding bounding boxes and its label.
        if :obj:'self.use_difficult', difficult is also returned.

        Args:
            i (int): The index of the sample.

        Returns:
            tuple of a CHW image array, a bounding box array and a label array.
            If :obj:'self.use_difficult' is :obj:'True', an array indicating 
            difficult bounding boxes is returned.
        """

        id_ = self.ids[i]
        ann_path = os.path.join(self.data_dir, f'Annotations/{id_}.xml')
        ann = ElementTree.parse(ann_path)

        bbox = list()
        label = list()
        difficult = list()
        for obj in ann.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_ann = obj.find('bndbox')
            bbox.append([int(bndbox_ann.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))
        bbox = np.array(bbox).astype(np.float32)
        label = np.array(label).astype(np.int32)
        difficult = np.array(difficult).astype(np.bool)

        img_path = os.path.join(self.data_dir, f'JPEGImages/{id_}.jpg')
        img = read_image(img_path)

        if self.use_difficult:
            return img, bbox, label, difficult
        else:
            return img, bbox, label


class VOCDatasetConfig:
    """config for VOCDataset

    This class defines the configurations of random flip, resize and normalization.

    Notations:
    
        * y_flip (bool): If :obj:'True', random flip along the y axis.
        * x_flip (bool): If :obj:'True', random flip along the x axis.
        * min_size (int): The maximum length of the shorter side length after resize.
        * max_size (int): The maximum length of the longer side length after resize.
        * normalize (bool): If :obj:'True', image is Z-score normalized.
    """
    def __init__(self, path):
        with open(path, 'r') as f:
            self.config = json.load(f)
    
    def __getitem__(self, key):
        return self.config[key]

    
class VOCDataset(VOCDatasetBase):
    """VOC dataset for Faster R-CNN. 

    Images are flipped, resized and normalized.

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        cfg (VOCDatasetConfig): Config for VOCDataset.
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
    """

    def __init__(self, data_dir, cfg, split='trainval', use_difficult=False):
        super().__init__(data_dir, split, use_difficult)
        self.cfg = cfg

    def __getitem__(self, i):
        """Returns the i-th sample.

        Returns a CHW image, its corresponding bounding boxes, label, scale, and difficult. 

        Args:
            i (int): The index of the sample.

        Returns:
            tuple of a CHW image array, a bounding box array, a label array and its scale.
            If :obj:'self.use_difficult' is :obj:'True', an array indicating 
            difficult bounding boxes is returned.
        """

        if self.use_difficult:
            img, bbox, label, difficult = super().__getitem__(i)
        else:
            img, bbox, label = super().__getitem__(i)

        in_size = img.shape[1:]
        img, flip_param = flip_image(img, self.cfg['y_random'], self.cfg['x_random'], True)
        bbox = flip_bbox(bbox, in_size, flip_param['y_flip'], flip_param['x_flip'])

        img = resize_image(img, self.cfg['min_size'], self.cfg['max_size'])
        out_size = img.shape[1:]
        bbox = resize_bbox(bbox, in_size, out_size)
        scale = out_size[0] / in_size[0]

        if self.cfg['normalize']:
            img = normalize_image(img)

        if self.use_difficult:
            return img, bbox, label, scale, difficult
        else:
            return img, bbox, label, scale
