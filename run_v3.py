# run_v3.py
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
from trainer_v3 import Trainer # <-- V3 CHANGE: Imports new trainer
import re

# Set a default seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cifar-100_imp_v2.yaml')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0])
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--overwrite', type=int, default=0) 
    
    # Add the missing arguments
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--ema_coeff', type=float, default=0.5)
    parser.add_argument('--schedule', type=int, default=50)

    args = parser.parse_args(argv)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- V2: Manually parse config sections ---
    learner_config = config.get('learner', {})
    training_config = config.get('training', {})
    dataset_config = config.get('dataset', {})
    model_config = config.get('model', {})

    # Merge CLI args with config file
    cli_args = vars(args)
    
    # Build the final flat 'args' namespace
    final_config = {}
    final_config.update(cli_args)
    final_config.update(learner_config)
    final_config.update(training_config)
    
    # --- FIX for dataset and model keys ---
    if 'name' in dataset_config:
        final_config['dataset'] = dataset_config['name']
    final_config.update(dataset_config) 

    if 'name' in model_config:
        final_config['model_name'] = model_config['name']
    if 'type' in model_config:
        final_config['model_type'] = model_config['type']
    # --- END OF FIX ---
    
    # --- V3 CHANGES ---
    final_config['learner_type'] = 'pa_apt_v3'
    final_config['learner_name'] = 'PA_APT_V3_Learner'
    
    # Set log_dir based on convention
    if args.log_dir is None:
        dataset_name = final_config.get('dataset', 'CUB200').lower()
        final_config['log_dir'] = f'./checkpoints/{dataset_name}-v3/seed{args.seed}'
    # --- END V3 CHANGES ---

    # Add missing args from original run.py for trainer compatibility
    final_config.setdefault('debug_mode', 0)
    final_config.setdefault('oracle_flag', False)
    final_config.setdefault('upper_bound_flag', False)
    final_config.setdefault('memory', 0)
    final_config.setdefault('temp', 2.0)
    final_config.setdefault('DW', False)
    final_config.setdefault('momentum', 0.9)
    final_config.setdefault('weight_decay', 0)
    final_config.setdefault('max_task', -1)

    return argparse.Namespace(**final_config)

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    
    # Set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + f'/output_seed{args.seed}.log' # Unique log file
    sys.stdout = Logger(log_out)
    
    print("Running with V3 Config:")
    print(args)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    
    metric_keys = ['acc','time','general_forgetting','coda_forgetting']
    save_keys = ['global', 'pt']
    global_only = ['time']
    avg_metrics = {}
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []
    
    start_r = 0
    # --- V3: Run only one repeat based on seed ---
    r = 0 # The shell script will control repeats via seeds
    
    print('************************************')
    print(f'* STARTING TRIAL (Seed {args.seed})')
    print('************************************')

    # set up a trainer
    trainer = Trainer(args, seed, r, metric_keys, save_keys)

    # init total run metrics storage
    max_task = trainer.max_task
    if r == 0: 
        for mkey in metric_keys: 
            avg_metrics[mkey]['global'] = np.zeros((max_task,args.repeat))
            if (not (mkey in global_only)):
                avg_metrics[mkey]['pt'] = np.zeros((max_task,max_task,args.repeat))
    
    # train model
    avg_metrics = trainer.train(avg_metrics)  

    # evaluate model
    avg_metrics = trainer.evaluate(avg_metrics)    

    # save results
    for mkey in metric_keys: 
            m_dir = args.log_dir+'/results-'+mkey+'/'
            if not os.path.exists(m_dir): os.makedirs(m_dir)
            for skey in save_keys:
                if (not (mkey in global_only)) or (skey == 'global'):
                    save_file = m_dir+skey+'.yaml'
                    result=avg_metrics[mkey][skey]
                    yaml_results = {}
                    if mkey=='acc':
                        print(skey, mkey, result)
                    if isinstance(result, tuple):
                        yaml_results['mean'] = result[0]
                    elif isinstance(result, list):
                        yaml_results['mean'] = result[0] if len(result)>0 else ""
                    
                    # --- START OF FIX ---
                    elif result.ndim > 2: # Check if it's a 3D array (like 'pt')
                        yaml_results['mean'] = result[:,:,:r+1].mean(axis=2).tolist()
                        if r>1: yaml_results['std'] = result[:,:,:r+1].std(axis=2).tolist()
                        yaml_results['history'] = result[:,:,:r+1].tolist()
                    else: # Handle 2D arrays (like 'global')
                        yaml_results['mean'] = result[:,:r+1].mean(axis=1).tolist()
                        if r>1: yaml_results['std'] = result[:,:r+1].std(axis=1).tolist()
                        yaml_results['history'] = result[:,:r+1].tolist()
                    # --- END OF FIX ---

                    with open(save_file, 'w') as yaml_file:
                        yaml.dump(yaml_results, yaml_file, default_flow_style=False)
    print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
    for mkey in metric_keys: 
        if 'forgetting' not in mkey:
            print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1,:r+1].mean(), 'std:', avg_metrics[mkey]['global'][-1,:r+1].std())
            print(round(avg_metrics[mkey]['global'][-1,:r+1].mean(),2), '\pm', round(avg_metrics[mkey]['global'][-1,:r+1].std(),2))
