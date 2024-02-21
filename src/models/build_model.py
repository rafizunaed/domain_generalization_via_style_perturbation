# MIT License
#
# Copyright (c) 2024 Mohammad Zunaed, mHealth Lab, BUET
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from src.models.densenet_ibn import DenseNet121_IBN
from src.models.resnet import ResNet50
from src.models.convnext import ConvNeXt_Tiny
from src.models.efficientnet_V2 import EfficientNet_V2_Small
from src.models.swin_transformer_V2 import Swin_Transformer_V2_Tiny

def create_model(backbone_architecture: str='densenet121_ibn', num_classes: int=1000, init_srm_fl: bool=False, randomization_stage=None):
    assert backbone_architecture in ['densenet121_ibn', 'resnet50', 'convnext_tiny', 'efficientnet_V2_small', 'swin_transformer_V2_tiny']
    
    if backbone_architecture == 'densenet121_ibn':
        return DenseNet121_IBN(num_classes, init_srm_fl=init_srm_fl, randomization_stage=randomization_stage)
    elif backbone_architecture == 'resnet50':
        return ResNet50(num_classes, init_srm_fl=init_srm_fl, randomization_stage=randomization_stage)
    elif backbone_architecture == 'convnext_tiny':
        return ConvNeXt_Tiny(num_classes, init_srm_fl=init_srm_fl, randomization_stage=randomization_stage)
    elif backbone_architecture == 'efficientnet_V2_small':
        return EfficientNet_V2_Small(num_classes, init_srm_fl=init_srm_fl, randomization_stage=randomization_stage)
    elif backbone_architecture == 'swin_transformer_V2_tiny':
        return Swin_Transformer_V2_Tiny(num_classes, init_srm_fl=init_srm_fl, randomization_stage=randomization_stage)
    
    