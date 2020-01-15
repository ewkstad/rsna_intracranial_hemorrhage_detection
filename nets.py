
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from global_params import num_classes, is_vanilla


#
def get_dcnn_base(arch, pretrained=True):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        if is_vanilla:
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            model = nn.Sequential(*list(model.features)[:-1])
    else:
        print('No {}!'.format(arch))
        raise ValueError
    return model


#
class DenseNet121Localization(nn.Module):

    def __init__(self, pretrained=True):
        super(DenseNet121Localization, self).__init__()
        self.features = get_dcnn_base('densenet121', pretrained)
        self.classifier = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.BatchNorm2d(1024),
                                        nn.ReLU(),
                                        nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=(1, 1)),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                        nn.Sigmoid()
                                        )



    def forward(self, x) -> 'PxPxK tensor prediction':
        x = self.features(x)
        x = self.classifier(x)
        return x


class DenseNet121Vanilla(nn.Module):

    def __init__(self, pretrained=True):
        super(DenseNet121Vanilla, self).__init__()
        self.features = get_dcnn_base('densenet121', pretrained)
        self.bn0 = nn.BatchNorm2d(1024)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, kernel_size=16)
        x = self.bn0(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.sigmoid(x)
        return x

