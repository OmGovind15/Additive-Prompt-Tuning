from __future__ import print_function
import torch
import models
import models.zoo_imp as zoo_imp
from utils.metric import accuracy, AverageMeter, Timer
from .default import NormalNN, weight_reset, accumulate_acc
from utils.schedulers import CosineSchedule

class Prompt_Learner(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.ema_coeff = learner_config['ema_coeff']
        super(Prompt_Learner, self).__init__(learner_config)

    def update_model(self, inputs, targets):
        # logits
        logits = self.model(inputs, train=True)
        
        logits = logits[:,:self.valid_out_dim]
        logits[:,:self.last_valid_out_dim] = -float('inf')
        total_loss = self.criterion(logits, targets.long())       
        
        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.detach(), logits

    def get_attn_heatmap(self, inputs):
        return 

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

class APT_IMP_Learner(Prompt_Learner):

    def __init__(self, learner_config):
        super(APT_IMP_Learner, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = zoo_imp.__dict__[cfg['model_name']](out_dim=self.out_dim, ema_coeff=self.ema_coeff, prompt_flag = 'apt', prompt_param=self.prompt_param, tasks=self.tasks)
        return model
    # --- START TASK 2 MODIFICATION ---
    def calculate_importance(self, dataloader):
        print("Calculating prompt importance (Fisher Information)...")
        
        # Get the prompt parameters
        try:
            prompt_params = [self.model.module.prompt.prompt_tokens]
        except:
            prompt_params = [self.model.prompt.prompt_tokens]

        # Initialize a buffer to accumulate squared gradients
        fisher_scores = torch.zeros_like(prompt_params[0].data)

        # Set model to evaluation mode
        self.model.eval()
        
        # Limit calculation to save time
        num_batches = 0
        for i, (inputs, targets, _) in enumerate(dataloader):
            if i > 50: 
                break
            num_batches += 1
                
            inputs = inputs.cuda()
            targets = targets.cuda()
            current_task_mask = (targets >= self.last_valid_out_dim) & (targets < self.valid_out_dim)

            # If this batch has no data from the current task, skip it
            if not current_task_mask.any():
                continue

            inputs = inputs[current_task_mask]
            targets = targets[current_task_mask]
            # Zero out gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(inputs, train=True)
            
            # Use only the logits for the current task
            logits = logits[:, self.last_valid_out_dim:self.valid_out_dim]
            
            
            
            loss = self.criterion(logits, targets.long())
            
            # Backward pass to get gradients
            loss.backward()
            
            # Accumulate squared gradients for the prompt
            if prompt_params[0].grad is not None:
                fisher_scores += prompt_params[0].grad.data.pow(2)

        # Average the scores
        if num_batches > 0:
             fisher_scores = fisher_scores / num_batches
             
        # Save the new scores
        try:
            self.model.module.prompt.fisher_importance.copy_(fisher_scores)
            # Before starting a new task, save the just-trained prompt as the "old_prompt"
            self.model.module.prompt.old_prompt.copy_(prompt_params[0].data)
        except:
            self.model.prompt.fisher_importance.copy_(fisher_scores)
            self.model.prompt.old_prompt.copy_(prompt_params[0].data)
            
        print("Importance calculation complete.")
    # --- END TASK 2 MODIFICATION ---
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

        # --- THIS BLOCK IS NOW REMOVED ---
        # merge_flag = self.model.prompt.merge_flag
        # if merge_flag:
        # ... (conflicting code deleted) ...
        # --- END OF REMOVED BLOCK ---

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
