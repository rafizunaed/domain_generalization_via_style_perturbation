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

class SRM_FL(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.eps = 1e-7    
        self.gamma_net = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, 1, 1),                            
                            nn.Conv2d(in_channels, in_channels//2, 3, 1, 1),
                            nn.Conv2d(in_channels//2, in_channels, 3, 1, 1),
                            nn.Conv2d(in_channels, in_channels, 1, 1),
                            )
        self.beta_net = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, 1, 1),
                            nn.Conv2d(in_channels, in_channels//2, 3, 1, 1),
                            nn.Conv2d(in_channels//2, in_channels, 3, 1, 1),
                            nn.Conv2d(in_channels, in_channels, 1, 1),
                            )

    def forward(self, content_feats_orig:torch.Tensor, style_feats:torch.Tensor):  
        gamma = self.gamma_net(style_feats)
        beta = self.beta_net(style_feats)   
        mean_content = torch.mean(content_feats_orig, dim=[2,3], keepdim=True)
        var_content = torch.var(content_feats_orig, dim=[2,3], keepdim=True) 
        content_feats = (content_feats_orig - mean_content) / (var_content + self.eps).sqrt()  
        content_feats = content_feats * gamma + beta
        return content_feats  