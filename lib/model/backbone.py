import torchvision
from torch import nn

def _get_extractor_classifier_vgg16(dropout=True):
    model = torchvision.models.vgg16(pretrained=True)
    extractor = list(model.features)[:30]
    classifier = list(model.classifier)[:-1]

    for relu_layer_i in [11, 13, 15, 18, 20, 22, 25, 27, 29]:
        extractor[relu_layer_i] = nn.LeakyReLU(inplace=True)
    # for layer in extractor[:10]:
    #     for p in layer.parameters():
    #         p.requires_grad = False
    extractor = nn.Sequential(*extractor)

    if not dropout:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    return extractor, classifier


_extractor_classifier_dict = {
    'vgg16': _get_extractor_classifier_vgg16
}


def get_extractor_classifier(backbone='vgg16'):
    if backbone not in _extractor_classifier_dict.keys():
        raise ValueError
    return _extractor_classifier_dict[backbone]()