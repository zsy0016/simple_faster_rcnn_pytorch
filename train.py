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


LossTuple = namedtuple('LossTuple', ['rpn_loc', 'rpn_cls', 'roi_loc', 'roi_cls', 'total'])


def main():
    parser = argparse.ArgumentParser('Parser for Faster R-CNN.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--decay_epoch', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--pascal_voc_train_cfg', type=str, 
        default='./lib/dataset/pascal_voc_cfg/pascal_voc_train.json')
    parser.add_argument('--pascal_voc_test_cfg', type=str, 
        default='./lib/dataset/pascal_voc_cfg/pascal_voc_test.json')
    args = parser.parse_args()

    train_cfg = VOCDatasetConfig(args.pascal_voc_train_cfg)
    test_cfg = VOCDatasetConfig(args.pascal_voc_test_cfg)
    train_dataset = VOCDataset(args.data_dir, train_cfg, split='trainval', use_difficult=False)
    test_dataset = VOCDataset(args.data_dir, test_cfg, split='test', use_difficult=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FasterRCNN()
    trained_net = torch.load('./checkpoint/vgg_03111932_0.65.pth')
    model.load_state_dict(trained_net)
    model.cuda()

    optimizer = RAdam(model.parameters(), lr=args.lr)
    anchor_target_creator = AnchorTargetCreator()
    roi_target_creator = RoITargetCreator()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    best_map = 0
    for i in range(args.epoch):
        losstuple = LossTuple._make([AverageValueMeter() for i in range(len(LossTuple._fields))])
        if i in args.decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2
        t_loader = tqdm(train_dataloader, ncols=150)
        n_acc = 0
        n_acc_pr = 0
        n_acc_gt = 0
        for img, bbox, label, scale in t_loader:
            img, bbox, label, scale = img.cuda(), bbox[0].cuda(), label[0].cuda(), scale.item()
            img_size = img.shape[2:]
            h = model.extractor(img)
            anchor, rpn_loc, rpn_score, roi = model.rpn(h, img_size, scale)
            _, rpn_gt_loc, rpn_gt_cls = anchor_target_creator(anchor, bbox, img_size)
            roi, roi_gt_loc, roi_gt_cls = roi_target_creator(roi, bbox, label)            
            roi_loc, roi_score = model.head(h, roi)
            n_acc_pr += ((roi_score.argmax(dim=1) > 0).sum().item() + 1e-7)
            n_acc_gt += roi_gt_cls[roi_gt_cls>0].numel()
            n_acc += (roi_score[roi_gt_cls>0].argmax(dim=1) == roi_gt_cls[roi_gt_cls>0]).sum().item()
            roi_loc = roi_loc.contiguous().view(roi_score.shape[0], -1, 4)
            roi_loc = roi_loc[torch.arange(roi_loc.shape[0]), roi_gt_cls]
            rpn_loc_loss = F.smooth_l1_loss(rpn_loc[rpn_gt_cls>0], rpn_gt_loc[rpn_gt_cls>0])
            rpn_cls_loss = F.cross_entropy(rpn_score[rpn_gt_cls>=0], rpn_gt_cls[rpn_gt_cls>=0])
            roi_loc_loss = F.smooth_l1_loss(roi_loc[roi_gt_cls>0], roi_gt_loc[roi_gt_cls>0])
            roi_cls_loss = F.cross_entropy(roi_score, roi_gt_cls)
            total_loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losstuple[0].add(rpn_loc_loss.detach().cpu())
            losstuple[1].add(rpn_cls_loss.detach().cpu())
            losstuple[2].add(roi_loc_loss.detach().cpu())
            losstuple[3].add(roi_cls_loss.detach().cpu())
            losstuple[4].add(total_loss.detach().cpu())
            post_str = ''
            for k, v in losstuple._asdict().items():
                post_str += '%s: %.2f ' % (k, v.value()[0])
            t_loader.set_postfix_str(post_str + '%.2f %.2f' % (n_acc / n_acc_pr, n_acc / n_acc_gt))

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
        sys.stdout.flush()
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            chechpoint_path = os.path.join(args.checkpoint_dir, 
                'vgg_%s_%.2f.pth' % (time.strftime('%m%d%H%M'), eval_result['map']))
            torch.save(model.state_dict(), chechpoint_path)

if __name__ == '__main__':
    main()