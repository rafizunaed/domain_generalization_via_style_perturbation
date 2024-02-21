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

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from src.models.perturbation_blocks import SRM_FL
from src.models.losses import calculate_content_style_loss

class ConvNeXt_Tiny(nn.Module):
    def __init__(self, num_classes: int, init_srm_fl=False, randomization_stage=None):
        super().__init__()
        self._init_backbone()
        self._init_head(num_classes)
        self.init_srm_fl = init_srm_fl
        self.randomization_stage = randomization_stage
        
        if init_srm_fl == True:
            assert randomization_stage in ['S1', 'S2', 'S3']
        if randomization_stage != None:
            assert init_srm_fl == True
        
        if randomization_stage == 'S1':
            self.srm_fl = SRM_FL(96)
        elif randomization_stage == 'S2':
            self.srm_fl = SRM_FL(192)
        elif randomization_stage == 'S3':
            self.srm_fl = SRM_FL(384)
        
    def _init_backbone(self):
        # get convnext_tiny architecture
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)                                    
        self.layer1 = nn.Sequential(
                            model.features[0],            
                            model.features[1],
                            )
        self.layer2 = nn.Sequential(
                            model.features[2],            
                            model.features[3],
                            )
        self.layer3 = nn.Sequential(
                            model.features[4],            
                            model.features[5],
                            )
        self.layer4 = nn.Sequential(
                            model.features[6],            
                            model.features[7],
                            )
        
    def _init_head(self, num_classes: int):
        # final fc
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.classifier = model.classifier 
        self.classifier[2] = nn.Linear(768, num_classes)
        self.classifier[2].weight.data.normal_(0, 0.01)
        self.classifier[2].bias.data.zero_()
        
    def grad_turn_on_srm_fl_only(self):
        for x in [self.layer1, self.layer2, self.layer3, self.layer4, self.classifier]:
            for _, p in enumerate(x.parameters()):
                p.requires_grad_(False)

        for _, p in enumerate(self.srm_fl.parameters()):
            p.requires_grad_(True)

    def grad_turn_off_srm_fl_only(self):
        for x in [self.layer1, self.layer2, self.layer3, self.layer4, self.classifier]:
            for _, p in enumerate(x.parameters()):
                p.requires_grad_(True)

        for _, p in enumerate(self.srm_fl.parameters()):
            p.requires_grad_(False)
            
    def _forward_layer1(self, x: torch.Tensor):
        x = self.layer1(x)
        return x
    
    def _forward_layer2(self, x:torch.Tensor):
        x = self.layer2(x)
        return x

    def _forward_layer3(self, x:torch.Tensor):
        x = self.layer3(x)
        return x
    
    def _forward_layer4(self, x:torch.Tensor):
        x = self.layer4(x)
        return x

    def _forward_features(self, x:torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _forward_head(self, x:torch.Tensor):
        return self.classifier(x)    

    def _perturb_features(self, x:torch.Tensor, train_srm_fl: bool):
        N = x.shape[0]
        idx_swap = torch.arange(N).flip(0)
        cont_x = x.clone()
        style_x = x.clone()[idx_swap].detach()
        x_srm = self.srm_fl(cont_x, style_x)
        if train_srm_fl:
            content_loss, style_loss = calculate_content_style_loss(x_srm, cont_x, style_x)
            return {
                'content_loss': content_loss,
                'style_loss': style_loss,
                }
        else:
            return x_srm

    @autocast()
    def forward(self, x:torch.Tensor, use_srm_fl:bool=False, train_srm_fl:bool=False):
        if self.init_srm_fl == False or use_srm_fl == False:
            xg = self._forward_features(x)
            xg_pool = F.adaptive_avg_pool2d(xg, (1,1))
            logits = self._forward_head(xg_pool)
            
        elif self.init_srm_fl == True and use_srm_fl == True:
            x = self._forward_layer1(x)
            if self.randomization_stage == 'S1' and use_srm_fl==True:
                x = self._perturb_features(x, train_srm_fl)
                if train_srm_fl:
                    return x
            
            x = self._forward_layer2(x)
            if self.randomization_stage == 'S2' and use_srm_fl==True:
                x = self._perturb_features(x, train_srm_fl)
                if train_srm_fl:
                    return x
            
            x = self._forward_layer3(x)
            if self.randomization_stage == 'S3' and use_srm_fl==True:
                x = self._perturb_features(x, train_srm_fl)
                if train_srm_fl:
                    return x
                
            xg = self._forward_layer4(x)
            xg_pool = F.adaptive_avg_pool2d(xg, (1,1))
            logits = self.classifier(xg_pool) 
        
        return {
            'logits': logits,
            'gfm': xg,
            'gfm_pool': xg_pool,
            }