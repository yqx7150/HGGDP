import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from .siat import *


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--exe', type=str, default='SIAT_MULTICH', help='execute main code')
    parser.add_argument('--config', type=str, default='siat_config.yml',  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='Path for saving running related data.')
    parser.add_argument('--model', type=str, default='hggdp', help='A string for documentation purpose')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--load_path', type=str, default='./test_image31/test_data_17.mat', help="The directory of image load path")
    parser.add_argument('--save_path', type=str, default='./result_rec', help="The directory of image save path")


    args = parser.parse_args()
    args.log = os.path.join(args.checkpoint, 'logs', args.model)

    # parse config file
    if not args.test:
        with open(os.path.join('hggdp/configs', args.config), 'r') as f:
            config = yaml.load(f)
        new_config = dict2namespace(config)
    else:
        with open(os.path.join(args.log, 'config.yml'), 'r') as f:
            config = yaml.load(f)
        new_config = config

    if not args.test:
        if not args.resume_training:
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            os.makedirs(args.log)

        with open(os.path.join(args.log, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    try:
        print(args.exe)
        runner = eval(args.exe)(args, config)
        if not args.test:
            runner.train()
        else:
            runner.test()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
