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

import os
import torch
from torch import nn
from timm.utils.model_ema import ModelEmaV2

from src.configs.base_configs import DEVICE
from src.trainers.trainer_callbacks import MetricStoreBox, ExtraMetricMeter, ProgressBar
from src.models.losses import FocalLoss, compute_pdr_loss

def check_if_best_value(current_value: float, previous_best_value: float, metric_name: str='loss', mode: str='min'):
    if mode == 'min':
        if previous_best_value > current_value:
            print('\033[32;1m' + ' Val {} is improved from {:.4f} to {:.4f}! '.format(metric_name, previous_best_value, current_value) + '\033[0m')
            best_value = current_value
            is_best_value = True
        else:
            print('\033[31;1m' + ' Val {} is not improved from {:.4f}! '.format(metric_name, previous_best_value) + '\033[0m')
            best_value = previous_best_value
            is_best_value = False
    else:
        if previous_best_value < current_value:
            print('\033[32;1m' + ' Val {} is improved from {:.4f} to {:.4f}! '.format(metric_name, previous_best_value, current_value) + '\033[0m')
            best_value = current_value
            is_best_value = True
        else:
            print('\033[31;1m' + ' Val {} is not improved from {:.4f}! '.format(metric_name, previous_best_value) + '\033[0m')
            best_value = previous_best_value
            is_best_value = False
            
    return best_value, is_best_value
      
#%% #################################### Model Trainer Class #################################### 
class ModelTrainer_SRM_IL_CONS():
    def __init__(self, 
                 model: torch.nn.Module, 
                 Loaders: list, 
                 metrics: dict, 
                 lr: float, 
                 epochsTorun: int,
                 checkpoint_saving_path: str,
                 gpu_ids: list,
                 do_grad_accum: False,
                 grad_accum_step: int,
                 fold: int,
                 use_ema: bool,
                 use_focal_loss: bool,
                 focal_loss_alpha: float,
                 focal_loss_gamma: float,
                 ):
        super().__init__()
                   
        self.metrics = metrics
        self.model = model.to(DEVICE)
        self.trainLoader = Loaders[0]
        self.valLoader = Loaders[1]        
        
        self.fold = fold
        if self.fold != None:
            self.checkpoint_saving_path = checkpoint_saving_path + '/fold' + str(self.fold) + '/'
        else:
            self.checkpoint_saving_path = checkpoint_saving_path + '/'    
        os.makedirs(self.checkpoint_saving_path,exist_ok=True)
        
        self.lr = lr
        self.epochsTorun = epochsTorun       
        
        self.best_loss = 9999
        self.best_auc = -9999
        
        if use_focal_loss:
            self.criterion_cls = FocalLoss(gamma=focal_loss_gamma, alpha=focal_loss_alpha)
        else:
            self.criterion_cls = nn.BCEWithLogitsLoss()
            
        self.criterion_mse = nn.MSELoss()
            
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        
        self.do_grad_accum = do_grad_accum
        self.grad_accum_step = grad_accum_step
        
        self.gpu_ids = gpu_ids
        if len(self.gpu_ids) > 1:
            print('using multi-gpu!')
            self.use_data_parallel = True
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        else:
            self.use_data_parallel = False
        
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = ModelEmaV2(self.model, decay=0.997, device=DEVICE)
        
    def get_checkpoint(self, val_logs):
        if self.use_ema and self.use_data_parallel:
            checkpoint_dict = {
                'Epoch': self.current_epoch_no,
                'Model_state_dict': self.model_ema.module.module.state_dict(),
                } 
        elif self.use_ema:
            checkpoint_dict = {
                'Epoch': self.current_epoch_no,
                'Model_state_dict': self.model_ema.module.state_dict(),
                }
        elif self.use_data_parallel:
            checkpoint_dict = {
                'Epoch': self.current_epoch_no,
                'Model_state_dict': self.model.module.state_dict(),
                }
        else:
            checkpoint_dict = {
                'Epoch': self.current_epoch_no,
                'Model_state_dict': self.model.state_dict(),
                }
                            
        for key in val_logs.keys():
            checkpoint_dict.update({key: val_logs[key]})
            
        return checkpoint_dict
    
    def perform_validation(self, use_progbar=True):
        self.model.eval()
        if self.use_ema:
            self.model_ema.eval()
        torch.set_grad_enabled(False)
        val_info_box = MetricStoreBox(self.metrics)
        extra_metric_box = ExtraMetricMeter()
          
        if use_progbar:
            if self.fold == None:
                progbar_description = f'(val) Epoch {self.current_epoch_no}/{self.epochsTorun}'
            else:
                progbar_description = f'(val) Fold {self.fold} Epoch {self.current_epoch_no}/{self.epochsTorun}'
            val_progbar = ProgressBar(len(self.valLoader), progbar_description)
            
        for itera_no, data in enumerate(self.valLoader):                        
            images = data['image'].to(DEVICE) 
            targets = data['target'].to(DEVICE)
            
            with torch.no_grad() and torch.cuda.amp.autocast():
                if self.use_ema:
                    out = self.model_ema.module(images)
                else:
                    out = self.model(images)                                 
                batch_loss = self.criterion_cls(out['logits'], targets)
                        
            # update extra metric      
            y_score = out['logits'].detach().cpu().clone().float().sigmoid().numpy()
            y_true = targets.detach().cpu().data.numpy()
            extra_metric_box.update(y_score, y_true)
            
            # update progress bar, info box
            val_info_box.update({'loss':[batch_loss.detach().item(), targets.shape[0]]})
            logs_to_display=val_info_box.get_value()
            logs_to_display = {f'val_{key}': logs_to_display[key] for key in logs_to_display.keys()}
            if use_progbar:
                val_progbar.update(1, logs_to_display)
        
        # calculate all metrics and close progbar
        logs_to_display=val_info_box.get_value()
        auc = extra_metric_box.feedback()
        logs_to_display.update({'auc': auc})
        logs_to_display = {f'val_{key}': logs_to_display[key] for key in logs_to_display.keys()}
        
        val_logs = logs_to_display
        self.best_loss, is_best_loss = check_if_best_value(val_logs['val_loss'], self.best_loss, 'loss', 'min')
        self.best_auc, is_best_auc = check_if_best_value(val_logs['val_auc'], self.best_auc, 'auc', 'max')
        
        checkpoint_dict = self.get_checkpoint(val_logs)                              
        if is_best_auc:
            if self.fold == None:
                torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_auc.pth')
            else:                                
                torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_auc_fold{}.pth'.format(self.fold))       
        del checkpoint_dict
        
        best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss':self.best_loss}
        logs_to_display.update(best_results_logs)
        if use_progbar:
            val_progbar.update(logs_to_display=logs_to_display)
            val_progbar.close() 
        
    def train_one_epoch(self):
        train_info_box = MetricStoreBox(self.metrics)
        extra_metric_box = ExtraMetricMeter()
        
        if self.fold == None:
            progbar_description = f'(Train) Epoch {self.current_epoch_no}/{self.epochsTorun}'
        else:
            progbar_description = f'(Train) Fold {self.fold} Epoch {self.current_epoch_no}/{self.epochsTorun}'
        train_progbar = ProgressBar(len(self.trainLoader), progbar_description)
        
        self.model.train()
        torch.set_grad_enabled(True) 
        self.optimizer.zero_grad()
        
        if self.use_ema:
            self.model_ema.train()
        
        for itera_no, data in enumerate(self.trainLoader):                                              
            images = data['image'].to(DEVICE)
            images_il_srm = data['image_il_srm'].to(DEVICE)
            targets = data['target'].to(DEVICE)
            
            with torch.cuda.amp.autocast():
                out_1 = self.model(images)  
                out_2 = self.model(images_il_srm)
                
                ccr_loss = self.criterion_mse(out_1['gfm'], out_2['gfm'])
                pdr_loss = compute_pdr_loss(out_2['logits'], out_1['logits'])
                cls_loss = self.criterion_cls(out_2['logits'], targets)
                batch_loss = cls_loss + (ccr_loss + pdr_loss)/2
                
            self.scaler.scale(batch_loss).backward()
            
            if self.do_grad_accum:
                if (itera_no+1)%self.grad_accum_step == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.use_ema:
                        self.model_ema.update(self.model)
            else:
                self.scaler.step(self.optimizer) 
                self.scaler.update()  
                self.optimizer.zero_grad()
                
                if self.use_ema:
                    self.model_ema.update(self.model)
                    
            # update extra metric
            y_score = out_1['logits'].detach().cpu().clone().float().sigmoid().numpy()
            y_true = targets.detach().cpu().data.numpy()
            extra_metric_box.update(y_score, y_true)
            
            # update progress bar, info box
            train_info_box.update({'loss':[batch_loss.detach().item(), targets.shape[0]],
                                   'cls_loss':[cls_loss.detach().item(), targets.shape[0]],
                                   'ccr_loss':[ccr_loss.detach().item(), targets.shape[0]],
                                   'pdr_loss':[pdr_loss.detach().item(), targets.shape[0]],
                                   })
            logs_to_display=train_info_box.get_value()
            logs_to_display = {f'train_{key}': logs_to_display[key] for key in logs_to_display.keys()}
            best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss':self.best_loss}
            logs_to_display.update(best_results_logs)
            train_progbar.update(1, logs_to_display)
            
            # if itera_no == 10:
            #     break
            
        # calculate all metrics and close progbar
        logs_to_display=train_info_box.get_value()
        best_results_logs = {'best_val_auc': self.best_auc, 'best_val_loss':self.best_loss}
        logs_to_display.update(best_results_logs)
        auc = extra_metric_box.feedback()
        logs_to_display.update({'auc': auc})
        logs_to_display = {f'train_{key}': logs_to_display[key] for key in logs_to_display.keys()}
        train_progbar.update(logs_to_display=logs_to_display)
        train_progbar.close()
        
#%% train part starts here
    def fit(self):   
        for epoch in range(self.epochsTorun):
            self.current_epoch_no = epoch+1
            self.train_one_epoch()
            self.perform_validation()