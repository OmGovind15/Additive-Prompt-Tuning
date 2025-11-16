# File: learners/pa_apt_v2.py
"""
Improved PA-APT with:
1. Reduced k (fewer patches)
2. Selective PPF (different alpha for CLS vs patches)
3. Layer-wise patch selection
"""
from __future__ import print_function
import torch
import torch.nn as nn
import copy
from .default import NormalNN, weight_reset, accumulate_acc
from utils.metric import accuracy, AverageMeter, Timer
from utils.schedulers import CosineSchedule
# We still need the zoo file for model creation
import models.zoo_pa_apt_v2 as zoo_pa_apt 

class Prompt_Learner_V2(NormalNN):
    """Base learner structure, copied from Prompt_Learner"""
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']
        super(Prompt_Learner_V2, self).__init__(learner_config)
    
    def update_model(self, inputs, targets):
        # This logic is copied from the original pa_apt.py
        logits = self.model(inputs, train=True)
        
        logits = logits[:,:self.valid_out_dim]
        logits[:,:self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logits, targets.long())       
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits

    def init_optimizer(self):
        # This logic is copied from the original pa_apt.py
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        self.log('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])

    def create_model(self):
        # This is the original pa_apt model loading logic
        cfg = self.config
        model = zoo_pa_apt.__dict__[cfg['model_name']](
            out_dim=self.out_dim, 
            ema_coeff=self.ema_coeff, 
            prompt_flag = 'apt', 
            prompt_param=self.prompt_param, 
            tasks=self.tasks,
            # Pass V2 specific settings to the model's forward pass 
            k_patches=self.k_patches, 
            use_layer_wise=self.use_layer_wise 
        )
        return model

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


class PA_APT_V2_Learner(Prompt_Learner_V2):
    
    def __init__(self, learner_config):
        # Add New Settings [cite: 875-877]
        self.k_patches = learner_config.get('k_patches', 2) 
        self.alpha_cls = learner_config.get('alpha_cls', 0.7) 
        self.alpha_patch = learner_config.get('alpha_patch', 0.5) 
        self.use_layer_wise = learner_config.get('use_layer_wise', True) 
        self.prev_prompts = {} 
        self.cur_task_id = 0 

        super(PA_APT_V2_Learner, self).__init__(learner_config)
    
    # New Method: Layer-wise Patch Selection 
    def select_patches_per_layer(self, layer_idx):
        """Layer-wise patch selection strategy (hardcoded based on ViT layers)"""
        if not self.use_layer_wise:
            return self.k_patches 

        # ViT-B/16 has 12 layers
        if layer_idx < 4:       # Layers 0-3: Early, ignore patches
            return 0 
        elif layer_idx < 9:     # Layers 4-8: Middle, full k patches
            return self.k_patches 
        else:                   # Layers 9-11: Late, fewer patches
            return max(1, self.k_patches // 2) 

    # New Method: Selective PPF Fusion 
    def selective_ppf_fusion(self, old_prompts, new_prompts):
        """Apply different fusion weights for CLS vs Patch prompts"""
        if old_prompts is None:
            return new_prompts 
        
        fused_prompts = {} 
        for name, new_p in new_prompts.items(): 
            old_p = old_prompts.get(name)
            
            if old_p is None: 
                fused_prompts[name] = new_p 
                continue
            
            # Determine fusion weight based on prompt type
            if 'cls' in name.lower() or 'prompt_k' in name or 'prompt_v' in name: 
                # CLS prompts or main key/value prompts [cite: 900]
                alpha = self.alpha_cls 
            else:
                # Patch prompts [cite: 903]
                alpha = self.alpha_patch 
            
            # PPF fusion [cite: 906]
            fused_prompts[name] = alpha * old_p + (1 - alpha) * new_p 
            
        return fused_prompts 

    # Override: learn_batch 
    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        """Override to use selective PPF fusion after training"""

        # Store prompts before training
        if self.cur_task_id > 0: 
            self.prev_prompts = {} 
            for name, param in self.model.named_parameters(): 
                if 'prompt' in name: 
                    self.prev_prompts[name] = param.data.clone() 
        
        # Standard training (call parent's method)
        avg_train_time = super().learn_batch(train_loader, train_dataset, model_save_dir)
        
        # After training: Apply selective PPF fusion
        if self.cur_task_id > 0 and self.prev_prompts is not None: 
            new_prompts = {} 
            for name, param in self.model.named_parameters(): 
                if 'prompt' in name: 
                    new_prompts[name] = param.data.clone() 
            
            # Apply selective fusion [cite: 925]
            fused_prompts = self.selective_ppf_fusion(self.prev_prompts, new_prompts) 
            
            # Update model parameters with fused prompts [cite: 927-929]
            for name, param in self.model.named_parameters(): 
                if name in fused_prompts: 
                    param.data = fused_prompts[name] 
            
            self.log(f"Applied Selective PPF: alpha_cls={self.alpha_cls}, alpha_patch={self.alpha_patch}") 
            
        self.cur_task_id += 1 
        
        return avg_train_time
