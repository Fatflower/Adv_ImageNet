# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : VGG_refine.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 09/10/2021
#  Description: 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.09.10, first created by Zhang wentao
#
# %Header File End--------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class VGG_refine(nn.Module):
    def __init__(
        self, 
        vgg: str = 'vgg16',
        pretrained: bool = True,
        classes_num: int = 1000,
    ) -> None:
        super(VGG_refine, self).__init__()
        if vgg == 'vgg11':
            bone = models.vgg11(pretrained=pretrained)
        elif vgg == 'vgg16':
            bone = models.vgg16(pretrained=pretrained)
        elif vgg == 'vgg19':
            bone = models.vgg19(pretrained=pretrained)


        self.feature = bone.features
        self.avgpool = bone.avgpool
        self.classifier = nn.Sequential(*list(bone.classifier.children())[-7:-1])
        self.last_layer = nn.Linear(4096, classes_num)

    def forward(self, x):

        out = self.feature(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.last_layer(out)
        return out 

def test_VGG_refine():
    net = VGG_refine('vgg19', False, 10)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y)
    summary(net, input_size=(3, 224, 224), device='cpu')
    print(net)


# test model
if __name__ == '__main__':
    test_VGG_refine()

