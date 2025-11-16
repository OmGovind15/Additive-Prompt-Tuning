# File: learners/pa_apt_v3.py
"""
PHASE 3: Ultimate Combination (V3)
- Patch-Aware Prompts (from Task 1 V2)
- Importance-Based Fusion on CLS token (from Task 2 V2)
- Selective PPF on Patch prompts (from Task 1 V2)
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .default import NormalNN, weight_reset, accumulate_acc
from utils.metric import accuracy, AverageMeter, Timer
from utils.schedulers import CosineSchedule
# We use the new V3 model loader
import models.zoo_pa_apt_v3 as zoo_pa_apt

# --- BASE LEARNER (Contains all config) ---
class Prompt_Learner_V3(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']

        # --- V3: Get all parameters from config ---
        self.k_patches = learner_config.get('k_patches', 2)
        self.use_layer_wise = learner_config.get('use_layer_wise', True)
        self.alpha_cls = learner_config.get('alpha_cls', 0.7)
        self.alpha_patch = learner_config.get('alpha_patch', 0.5)
        self.use_importance_for_cls = learner_config.get('use_importance_for_cls', True)
        self.temperature = learner_config.get('importance_temperature', 2.0)
        self.adjust_range = learner_config.get('adjust_range', 0.15)
        self.min_alpha = learner_config.get('min_alpha', 0.55)
        self.max_alpha = learner_config.get('max_alpha', 0.85)
        # --- End V3 ---

        super(Prompt_Learner_V3, self).__init__(learner_config)

    def update_model(self, inputs, targets):
        logits = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]
        logits[:,:self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logits, targets.long())       
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

    def init_optimizer(self):
        # Find all prompt parameters (CLS and Patch)
        prompt_params = []
        for name, param in self.model.named_parameters():
            if 'prompt' in name and param.requires_grad:
                prompt_params.append(param)
        
        # Add the classifier parameters
        if len(self.config['gpuid']) > 1:
            params_to_opt = prompt_params + list(self.model.module.last.parameters())
        else:
            params_to_opt = prompt_params + list(self.model.last.parameters())

        self.log('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # --- START OF FIX ---
        # create schedules
        if self.schedule_type == 'cosine':
            # Check if schedule is a list or int
            if isinstance(self.schedule, list):
                k_val = self.schedule[-1]
            else:
                k_val = self.schedule
            self.scheduler = CosineSchedule(self.optimizer, K=k_val)
        elif self.schedule_type == 'decay':
            if isinstance(self.schedule, list):
                milestones = self.schedule
            else:
                milestones = [self.schedule] # Wrap int in a list
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
    def create_model(self):
        cfg = self.config
        # Call the new V3 zoo
        model = zoo_pa_apt.__dict__[cfg['model_name']](
            out_dim=self.out_dim, 
            ema_coeff=self.ema_coeff, 
            prompt_flag = 'apt', 
            prompt_param=self.prompt_param, 
            tasks=self.tasks,
            k_patches=self.k_patches, 
            use_layer_wise=self.use_layer_wise,
            patch_selector_fn=self.select_patches_per_layer
        )
        return model

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    # This is the learn_batch from default.py, with the conflict removed
    def learn_batch(self, train_loader, train_dataset, model_save_dir):
        need_train = True
        if not self.overwrite:
            try:
                print("Overwriting ...")
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        if self.reset_optimizer:
            self.log('Optimizer is reset!')
            self.init_optimizer()

        if need_train:
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            if isinstance(self.config['schedule'], list):
                schedule_length = self.config['schedule'][-1]
            else:
                schedule_length = self.config['schedule']

            for epoch in range(schedule_length):
                self.epoch=epoch
                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()

                for i, (x, y, task) in enumerate(train_loader):
                    self.model.train()
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    loss, output= self.update_model(x, y)
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0))
                    batch_timer.tic()

                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=schedule_length))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                losses = AverageMeter()
                acc = AverageMeter()

        self.model.train()
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None

# --- V3 COMBINED LEARNER ---
class PA_APT_V3_Learner(Prompt_Learner_V3):

    def __init__(self, learner_config):
        super(PA_APT_V3_Learner, self).__init__(learner_config)
        self.prev_prompts = {}
        self.fisher_information = {}
        self.cur_task_id = 0

    # Method from PA-APT-V2
    def select_patches_per_layer(self, layer_idx):
        if not self.use_layer_wise:
            return self.k_patches 
        total_layers = 12
        if layer_idx < 4: return 0 
        elif layer_idx < 9: return self.k_patches 
        else: return max(1, self.k_patches // 2) 

    # Method from APT-IMP-V2 (modified for CLS only)
    def compute_fisher_for_cls_prompts(self, model, dataloader):
        model.eval()
        fisher = {}
        # Find CLS-related prompts
        for name, param in model.named_parameters():
            # This logic identifies the CLS prompts (prompt_k/v) vs. patch prompts
            is_cls = 'patch_p_k' not in name and 'patch_p_v' not in name
            if 'prompt' in name and is_cls and param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        self.log("Computing Fisher for CLS prompts...")
        sample_count = 0
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            if batch_idx >= 20: break

            inputs, targets = inputs.cuda(), targets.cuda()

            current_task_mask = (targets >= self.last_valid_out_dim) & (targets < self.valid_out_dim)
            if not current_task_mask.any(): continue

            inputs = inputs[current_task_mask]
            targets = targets[current_task_mask]

            outputs = model(inputs)
            logits = outputs[:, self.last_valid_out_dim:self.valid_out_dim]
            targets_for_task = targets - self.last_valid_out_dim
            loss = F.cross_entropy(logits, targets_for_task.long())

            model.zero_grad()
            loss.backward()

            for name, param in model.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            sample_count += inputs.size(0)

        if sample_count > 0:
            for name in fisher:
                fisher[name] /= sample_count

        self.log(f"CLS Fisher computed over {sample_count} samples")
        return fisher

    # Method from APT-IMP-V2
    def importance_fusion_for_cls(self, old_prompt, new_prompt, importance):
        imp_min = importance.min()
        imp_max = importance.max()
        if imp_max - imp_min > 1e-8:
            imp_norm = (importance - imp_min) / (imp_max - imp_min + 1e-8)
        else:
            imp_norm = torch.ones_like(importance) * 0.5

        imp_smooth = torch.sigmoid((imp_norm - 0.5) * self.temperature)
        # Use alpha_cls as the base
        adaptive_alpha = self.alpha_cls + self.adjust_range * (imp_smooth - 0.5) * 2 
        adaptive_alpha = torch.clamp(adaptive_alpha, self.min_alpha, self.max_alpha)

        return adaptive_alpha * old_prompt + (1 - adaptive_alpha) * new_prompt

    # New Method: Hybrid Fusion
    def hybrid_fusion(self, old_prompts, new_prompts, fisher):
        if old_prompts is None:
            return new_prompts

        fused_prompts = {}
        for key in new_prompts:
            if key not in old_prompts:
                fused_prompts[key] = new_prompts[key]
                continue

            old_p = old_prompts[key]
            new_p = new_prompts[key]

            # Check if it's a CLS prompt (prompt_k/v) or a Patch prompt (patch_p_k/v)
            # This logic finds the main prompts (not patch prompts)
            is_cls = 'patch_p_k' not in key and 'patch_p_v' not in key

            if is_cls and self.use_importance_for_cls and key in fisher:
                # CLS: Use importance-based fusion
                fused_prompts[key] = self.importance_fusion_for_cls(old_p, new_p, fisher[key])
            elif is_cls:
                # CLS: Simple PPF with higher alpha
                fused_prompts[key] = self.alpha_cls * old_p + (1 - self.alpha_cls) * new_p
            else:
                # Patch: Simple PPF with lower alpha
                fused_prompts[key] = self.alpha_patch * old_p + (1 - self.alpha_patch) * new_p

        self.log(f"Hybrid Fusion: CLS (Importance={self.use_importance_for_cls}), Patches (Alpha={self.alpha_patch})")
        return fused_prompts

    # Override: learn_batch
    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        # 1. Store prompts before training
        if self.cur_task_id > 0:
            self.prev_prompts = {}
            for name, param in self.model.named_parameters():
                if 'prompt' in name:
                    self.prev_prompts[name] = param.data.clone()

        # 2. Standard training (call parent)
        avg_train_time = super().learn_batch(train_loader, train_dataset, model_save_dir)

        # 3. After training: Apply hybrid fusion
        if self.cur_task_id > 0 and self.prev_prompts is not None:

            # Compute Fisher for CLS prompts only
            if self.use_importance_for_cls:
                self.fisher_information = self.compute_fisher_for_cls_prompts(
                    self.model, train_loader
                )

            # Get new prompts
            new_prompts = {}
            for name, param in self.model.named_parameters():
                if 'prompt' in name:
                    new_prompts[name] = param.data.clone()

            # Apply hybrid fusion
            fused_prompts = self.hybrid_fusion(
                self.prev_prompts, new_prompts, self.fisher_information
            )

            # Update model with fused prompts
            for name, param in self.model.named_parameters():
                if name in fused_prompts:
                    param.data.copy_(fused_prompts[name]) # Use .copy_

        self.cur_task_id += 1
        return avg_train_time
