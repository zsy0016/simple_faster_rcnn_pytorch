## A Simple Implementation of Faster R-CNN using pytorch

This implementation use torchvision.ops to complete the nms operation.  
The batch size is only supported to be 1 currently.  
Dataset format is Pascal VOC 2007/2010.  
The backbone is vgg16 currently.
  
* data preparation
```
ln -s 'your pascal voc data path' ./data
```
* train
```
python train.py --lr lr --epoch 'total epoch' --decay epoch 'decay epochs'
```
* test
```
python test.py --checkpoint_file 'checkpoint name'
```