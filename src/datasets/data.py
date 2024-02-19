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
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(resize, crop):
    return A.Compose([
            A.Resize(width=resize, height=resize, p=1.0),
            A.RandomCrop(width=crop, height=crop, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0, additional_targets={"image0": "image"},)

def get_valid_transforms(resize, crop):
    return A.Compose([
            A.Resize(width=resize, height=resize, p=1.0),
            A.CenterCrop(width=crop, height=crop, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)

class SRM_IL(object):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        
    def apply_perturbation(self, image):
        image = image.astype(np.float32)
        m2 = np.random.uniform(low=self.min_value, high=self.max_value, size=(1,1,1)).repeat(3, axis=2)
        s2 = np.random.uniform(low=self.min_value, high=self.max_value, size=(1,1,1)).repeat(3, axis=2)
        m1 = np.mean(image, axis=(0,1), keepdims=True)
        v1 = np.var(image, axis=(0,1), keepdims=True)
        s1 = np.sqrt(v1)
        image = (image - m1) / s1
        image = image * s2 + m2
        return image

class ThoracicDataset(Dataset):
    def __init__(self, datasets_root_dir, fpaths, labels, transform, use_SRM_IL, srm_il_min_value=None, srm_il_max_value=None):
        self.fpaths = np.array([datasets_root_dir+x for x in fpaths])
        self.labels = np.array(labels)        
        self.transform = transform
        self.use_SRM_IL = use_SRM_IL
        if self.use_SRM_IL:
            assert srm_il_min_value != None
            assert srm_il_max_value != None
            self.SRM_IL = SRM_IL(srm_il_min_value, srm_il_max_value)
        
    def __len__(self):
        return self.fpaths.shape[0]
    
    def __getitem__(self, index):
        fpath = self.fpaths[index]
        image = Image.open(fpath).convert('RGB')
        image = np.array(image)
        if self.use_SRM_IL:    
            image = self.SRM_IL.apply_perturbation(image.copy())
        transformed = self.transform(image=image)
        transformed_image = transformed['image']
        label = self.labels[index]
        label = torch.tensor(label).float()
        return {
            'image': transformed_image, 
            'target': label,
            }
    
class ThoracicDatasetDual(Dataset):
    def __init__(self, datasets_root_dir, fpaths, labels, transform, srm_il_min_value=None, srm_il_max_value=None):
        self.fpaths = np.array([datasets_root_dir+x for x in fpaths])
        self.labels = np.array(labels)        
        self.transform = transform
        assert srm_il_min_value != None
        assert srm_il_max_value != None
        self.SRM_IL = SRM_IL(srm_il_min_value, srm_il_max_value)
        
    def __len__(self):
        return self.fpaths.shape[0]
    
    def __getitem__(self, index):
        fpath = self.fpaths[index]
        image = Image.open(fpath).convert('RGB')
        image = np.array(image)
        image_srm_il = self.SRM_IL.apply_perturbation(image.copy())
        transformed = self.transform(image=image, image0=image_srm_il)
        transformed_image = transformed['image']
        transformed_image_srm_il = transformed['image0']
        label = self.labels[index]
        label = torch.tensor(label).float()
        return {
            'image': transformed_image, 
            'target': label,
            'image_il_srm': transformed_image_srm_il,
            }
        
class ThoracicDatasetTest(Dataset):
    def __init__(self, datasets_root_dir, fpaths, labels, transform):
        self.fpaths = np.array([datasets_root_dir+x for x in fpaths])
        self.labels = np.array(labels)        
        self.transform = transform
        
    def __len__(self):
        return self.fpaths.shape[0]
    
    def __getitem__(self, index):
        fpath = self.fpaths[index]
        image = Image.open(fpath).convert('RGB')
        image = np.array(image)
        transformed = self.transform(image=image)
        transformed_image = transformed['image']
        label = self.labels[index]
        label = torch.tensor(label).float()
        return {
            'image': transformed_image, 
            'target': label,
            }
        
def collate_fn_img_level_ds(batch):
    x = batch[0]
    keys = x.keys()
    out = {}
    # declare key
    for key in keys:
        out.update({key:[]})
    # append values
    for i in range(len(batch)):
        for key in keys:
            out[key].append(batch[i][key])
    # stack values
    for key in keys:
        out[key] = torch.stack(out[key])
    return out