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
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from src.models.densenet_ibn_utils import densenet121_ibn_a
from src.models.perturbation_blocks import SRM_FL
from src.models.losses import calculate_content_style_loss

class DenseNet121_IBN(nn.Module):
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
            self.srm_fl = SRM_FL(128)
        elif randomization_stage == 'S2':
            self.srm_fl = SRM_FL(256)
        elif randomization_stage == 'S3':
            self.srm_fl = SRM_FL(512)
        
    def _init_backbone(self):
        # get densenet architecture
        model = densenet121_ibn_a(pretrained=True)
        modules = list(model.children())[0]
        # stem
        self.stem = nn.Sequential(*modules[:4])
        # dense block 1
        self.db1 = nn.Sequential(*modules[4:5])
        self.trn1 = nn.Sequential(*list(modules[5])[:-1])
        # dense block 2
        self.db2 = nn.Sequential(*modules[6:7])
        self.trn2 = nn.Sequential(*list(modules[7])[:-1])
        # dense block 3
        self.db3 = nn.Sequential(*modules[8:9])
        self.trn3 = nn.Sequential(*list(modules[9])[:-1])
        # dense block 4
        self.db4 = nn.Sequential(*modules[10:])
        # avg layer for applying after dense block-1,2,3
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
    def _init_head(self, num_classes: int):
        # final fc
        self.classifier = nn.Linear(1024, num_classes)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()
        
    def grad_turn_on_srm_fl_only(self):
        for x in [self.stem, self.db1, self.trn1, self.db2, self.trn2, self.db3, self.trn3, self.db4, self.classifier]:
            for _, p in enumerate(x.parameters()):
                p.requires_grad_(False)

        for _, p in enumerate(self.srm_fl.parameters()):
            p.requires_grad_(True)

    def grad_turn_off_srm_fl_only(self):
        for x in [self.stem, self.db1, self.trn1, self.db2, self.trn2, self.db3, self.trn3, self.db4, self.classifier]:
            for _, p in enumerate(x.parameters()):
                p.requires_grad_(True)

        for _, p in enumerate(self.srm_fl.parameters()):
            p.requires_grad_(False)
            
    def _forward_db1(self, x: torch.Tensor):
        x = self.db1(x)
        x = self.trn1(x)
        x = self.avg(x)
        return x
    
    def _forward_db2(self, x:torch.Tensor):
        x = self.db2(x)
        x = self.trn2(x)
        x = self.avg(x)
        return x

    def _forward_db3(self, x:torch.Tensor):
        x = self.db3(x)
        x = self.trn3(x)
        x = self.avg(x)
        return x
    
    def _forward_db4(self, x:torch.Tensor):
        x = self.db4(x)
        x = F.relu(x)
        return x

    def _forward_features(self, x:torch.Tensor):
        x = self.stem(x)
        x = self._forward_db1(x)
        x = self._forward_db2(x)
        x = self._forward_db3(x)
        x = self._forward_db4(x)
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
            xg_pool = F.adaptive_avg_pool2d(xg, (1,1)).flatten(1) 
            logits = self._forward_head(xg_pool)
            
        elif self.init_srm_fl == True and use_srm_fl == True:
            x = self.stem(x)
            
            x = self._forward_db1(x)
            if self.randomization_stage == 'S1' and use_srm_fl==True:
                x = self._perturb_features(x, train_srm_fl)
                if train_srm_fl:
                    return x
            
            x = self._forward_db2(x)
            if self.randomization_stage == 'S2' and use_srm_fl==True:
                x = self._perturb_features(x, train_srm_fl)
                if train_srm_fl:
                    return x
            
            x = self._forward_db3(x)
            if self.randomization_stage == 'S3' and use_srm_fl==True:
                x = self._perturb_features(x, train_srm_fl)
                if train_srm_fl:
                    return x
                
            xg = self._forward_db4(x)
            xg_pool = F.adaptive_avg_pool2d(xg, (1,1)).flatten(1)
            logits = self.classifier(xg_pool) 
        
        return {
            'logits': logits,
            'gfm': xg,
            'gfm_pool': xg_pool,
            }