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

def generate_gram_matrix(x:torch.Tensor):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features = F.normalize(features, dim=2, eps=1e-7)
    features_t = features.transpose(1, 2)
    gram_matrix = features.bmm(features_t)
    return gram_matrix

def calculate_content_style_loss(transformed:torch.Tensor, content_ref:torch.Tensor, style_ref:torch.Tensor):
    content_loss = F.mse_loss(transformed, content_ref)
    gm_transformed = generate_gram_matrix(transformed)
    gm_style_ref = generate_gram_matrix(style_ref)
    style_loss = F.mse_loss(gm_transformed, gm_style_ref)
    return content_loss, style_loss

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/focal.py
# https://arxiv.org/pdf/1708.02002.pdf        
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        logpt = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-logpt)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = focal_term * logpt
        if self.alpha != 0:
            loss *= self.alpha * target + (1 - self.alpha) * (1 - target)
        return loss.mean()
    
def compute_kld_loss(inp, target):
    kld_loss_criterion = nn.KLDivLoss(reduction="batchmean")
    # input should be a distribution in the log space
    inp = F.log_softmax(inp, dim=1)
    # Sample a batch of distributions. Usually this would come from the dataset
    target = F.softmax(target, dim=1)
    # target = F.sigmoid(target)
    kld_loss = kld_loss_criterion(inp, target)
    return kld_loss

def compute_pdr_loss(logits1, logits2):
    pdr_loss = (compute_kld_loss(logits1, logits2.detach())+compute_kld_loss(logits2, logits1.detach()))/2
    return pdr_loss