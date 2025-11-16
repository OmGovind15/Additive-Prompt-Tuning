# File: learners/apt_imp_v2.py
"""
Improved APT-IMP with:
1. Temperature-scaled importance
2. Balanced fusion (smaller adjustment range)
3. Better Fisher computation
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .default import NormalNN, weight_reset, accumulate_acc
from utils.metric import accuracy, AverageMeter, Timer
from utils.schedulers import CosineSchedule
import models.zoo_imp as zoo_imp # We use the original IMP zoo

# --- BASE LEARNER (Modified from original to be reusable) ---
class Prompt_Learner_V2(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']
        super(Prompt_Learner_V2, self).__init__(learner_config)

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
        cfg = self.config
        model = zoo_imp.__dict__[cfg['model_name']](
            out_dim=self.out_dim, 
            ema_coeff=self.ema_coeff, 
            prompt_flag = 'apt', 
            prompt_param=self.prompt_param, 
            tasks=self.tasks
        )
        return model

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

# --- V2 IMP LEARNER (Implements new fusion logic) ---
class APT_IMP_V2_Learner(Prompt_Learner_V2):

    def __init__(self, learner_config):
        # Improved importance-based fusion parameters [cite: 699-703]
        self.base_alpha = learner_config.get('base_alpha', 0.7)
        self.temperature = learner_config.get('importance_temperature', 2.0)
        self.adjust_range = learner_config.get('adjust_range', 0.15) 
        self.min_alpha = learner_config.get('min_alpha', 0.55) 
        self.max_alpha = learner_config.get('max_alpha', 0.85)
        self.prev_prompts = {}
        self.fisher_information = {}
        self.cur_task_id = 0

        super(APT_IMP_V2_Learner, self).__init__(learner_config)

    # New Method: Improved Fisher Information [cite: 705-729]
    def compute_fisher_information(self, model, dataloader):
        model.eval()
        fisher = {}
        # Initialize Fisher dict for prompt parameters
        for name, param in model.named_parameters():
            if 'prompt' in name and param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        self.log("Computing Fisher Information...")
        sample_count = 0
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            if batch_idx >= 20: # Limit samples for efficiency
                break

            inputs, targets = inputs.cuda(), targets.cuda()

            # Filter for current task data only
            current_task_mask = (targets >= self.last_valid_out_dim) & (targets < self.valid_out_dim)
            if not current_task_mask.any():
                continue

            inputs = inputs[current_task_mask]
            targets = targets[current_task_mask]

            # Forward pass
            outputs = model(inputs)

            # Slice logits and targets for current task
            logits = outputs[:, self.last_valid_out_dim:self.valid_out_dim]
            targets_for_task = targets - self.last_valid_out_dim

            # Use cross-entropy loss
            loss = F.cross_entropy(logits, targets_for_task.long())

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            sample_count += inputs.size(0)

        # Average over samples
        if sample_count > 0:
            for name in fisher:
                fisher[name] /= sample_count

        self.log(f"Fisher computed over {sample_count} samples")
        return fisher

    # New Method: Balanced Importance Fusion [cite: 730-752]
    def balanced_importance_fusion(self, old_prompt, new_prompt, importance):
        # Normalize importance to [0, 1]
        imp_min = importance.min()
        imp_max = importance.max()
        if imp_max - imp_min > 1e-8:
            imp_norm = (importance - imp_min) / (imp_max - imp_min + 1e-8)
        else:
            imp_norm = torch.ones_like(importance) * 0.5

        # Apply temperature scaling to reduce extremes
        imp_smooth = torch.sigmoid((imp_norm - 0.5) * self.temperature)

        # Compute adaptive alpha with bounded adjustment
        adaptive_alpha = self.base_alpha + self.adjust_range * (imp_smooth - 0.5) * 2

        # Clamp to reasonable range
        adaptive_alpha = torch.clamp(adaptive_alpha, self.min_alpha, self.max_alpha)

        # Dimension-wise fusion
        fused_prompt = adaptive_alpha * old_prompt + (1 - adaptive_alpha) * new_prompt
        return fused_prompt

    # New Method: Main merging function
    def merge_prompts_with_importance(self, old_prompts, new_prompts, fisher):
        if old_prompts is None or fisher is None:
            return new_prompts

        fused_prompts = {}
        for key in new_prompts:
            if key not in old_prompts or key not in fisher:
                fused_prompts[key] = new_prompts[key]
                continue

            old_p = old_prompts[key]
            new_p = new_prompts[key]
            imp = fisher[key]

            # Apply balanced fusion
            fused_prompts[key] = self.balanced_importance_fusion(old_p, new_p, imp)

        self.log(f"Importance Fusion Applied: Base alpha={self.base_alpha:.3f}, Temp={self.temperature:.1f}")
        return fused_prompts

    def learn_batch(self, train_loader, train_dataset, model_save_dir):

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                print("Overwriting ...")
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:

            # data weighting
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()

            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()

                for i, (x, y, task) in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, output= self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0))
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

        self.model.train()

        # --- CONFLICTING BLOCK IS REMOVED ---
        # merge_flag = self.model.prompt.merge_flag
        # if merge_flag:
        # ... (code deleted) ...

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None
