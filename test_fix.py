import torch
import numpy as np
import random
import yaml
import argparse
import dataloaders
from learners import apt_imp  # Import your new learner
from dataloaders.utils import *
from torch.utils.data import DataLoader

print("--- STARTING FIX TEST (v2) ---")

# --- 1. Load Configs (like run.py) ---
try:
    config = yaml.load(open('configs/cifar-100_imp.yaml', 'r'), Loader=yaml.Loader)
    
    # --- FIX: Add missing args that come from the shell script ---
    config['gpuid'] = [0]
    config['seed'] = 1
    config['prompt_param'] = ['0.01']
    config['lr'] = 0.004
    config['ema_coeff'] = 0.7
    config['schedule'] = 30
    config['batch_size'] = 64 # The missing key
    config['workers'] = 4      # Add workers
    config['train_aug'] = True # Add train_aug
    config['rand_split'] = True # Add rand_split
    # --- END FIX ---

    args = argparse.Namespace(**config)

    # --- 2. Setup Learner (like trainer.py) ---
    Dataset = dataloaders.iCIFAR100
    num_classes = 100
    class_order = np.arange(num_classes).tolist()
    
    # Need to shuffle class_order just like the real script
    random.seed(args.seed)
    random.shuffle(class_order)
    
    tasks_logits = [class_order[0:10]] # Just need the first task

    learner_config = {
        'num_classes': num_classes, 'lr': args.lr, 'debug_mode': False,
        'momentum': args.momentum, 'weight_decay': args.weight_decay,
        'schedule': [args.schedule], 'schedule_type': args.schedule_type,
        'model_type': args.model_type, 'model_name': args.model_name,
        'optimizer': args.optimizer, 'gpuid': args.gpuid, 'memory': 0, 'temp': 2.0,
        'out_dim': num_classes, 'overwrite': 0, 'DW': False, 'batch_size': args.batch_size,
        'upper_bound_flag': False, 'tasks': tasks_logits, 'top_k': 1,
        'prompt_param': [10, args.prompt_param], 'ema_coeff': args.ema_coeff
    }

    # Create the learner
    learner = apt_imp.APT_IMP_Learner(learner_config)
    learner.add_valid_output_dim(10) # Set up for Task 1

    # --- 3. Load the Saved Model (The crucial part) ---
    model_path = './checkpoints/cifar-100-imp/seed1/models/repeat-1/task-1/'
    print(f"Loading saved model from: {model_path}")
    learner.load_model(model_path)
    learner.model.eval() # Set to evaluation mode

    # --- 4. Get the Dataloader (Needed for the function) ---
    print("Loading dataloader...")
    train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=True)
    train_dataset = Dataset(args.dataroot, train=True, lab=True, tasks=[class_order[0:10]],
                            download_flag=True, transform=train_transform, 
                            seed=args.seed, rand_split=args.rand_split, validation=False)
    train_dataset.load_dataset(0, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)

    # --- 5. RUN THE BROKEN FUNCTION ---
    print("Attempting to run 'calculate_importance'...")
    learner.calculate_importance(train_loader)
    print("\n--- TEST SUCCESSFUL! ---")
    print("The bug is fixed. You can now delete test_fix.py and run the full experiment.")

except Exception as e:
    print("\n--- TEST FAILED ---")
    print("The code is still broken. Error message:")
    print(e)

