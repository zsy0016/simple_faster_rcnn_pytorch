import argparse
import os
import sys
import time
from collections import namedtuple
sys.path.append('lib')

import torch
import torch.nn.functional as F
from catalyst.contrib.optimizers import RAdam
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

from creator import AnchorTargetCreator, RoICreator, RoITargetCreator
from dataset import VOCDataset, VOCDatasetConfig
from eval import eval_detection_voc
from model.faster_rcnn import FasterRCNN


def main():
    parser = argparse.ArgumentParser('Parser for Faster R-CNN.')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--checkpoint_file', type=str)
    parser.add_argument('--pascal_voc_test_cfg', type=str, 
        default='./lib/dataset/pascal_voc_cfg/pascal_voc_test.json')
    args = parser.parse_args()

    test_cfg = VOCDatasetConfig(args.pascal_voc_test_cfg)
    test_dataset = VOCDataset(args.data_dir, test_cfg, split='test', use_difficult=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FasterRCNN()
    trained_net = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_file))
    model.load_state_dict(trained_net)
    model.cuda()

    t_loader = tqdm(test_dataloader, ncols=150)
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for img, gt_bbox, gt_label, scale, gt_difficult in t_loader:
        img, gt_bbox, gt_label, scale, gt_difficult = \
            img.cuda(), gt_bbox[0], gt_label[0], scale.item(), gt_difficult[0]
        pred_bbox, pred_label, pred_score = model.predict(img, scale)
        # t_loader.set_postfix_str('%d' % len(pred_bbox))
        pred_bboxes.append(pred_bbox)
        pred_labels.append(pred_label)
        pred_scores.append(pred_score)
        gt_bboxes.append(gt_bbox.numpy())
        gt_labels.append(gt_label.numpy())
        gt_difficults.append(gt_difficult.numpy())
    eval_result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults)
    sys.stdout.flush()
    print('\nmap: %.4f' % eval_result['map'])
    print(eval_result)
    sys.stdout.flush()

if __name__ == '__main__':
    main()