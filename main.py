import argparse
import os
import torch
import PIL
import torchvision
import ast
from omegaconf import OmegaConf

from utils import setup_logger, get_current_time
from baseline_trainer import trainer_init
from methods.wdiff import WDiff



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def str2none(v):
    if v is None:
        return v
    elif isinstance(v, int):
        return v
    elif v.lower() in ("none"):
        return None
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_opts(opt):
    """
    Convert string arguments to their appropriate types using ast.literal_eval
    """
    return ast.literal_eval(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implementation of W-Diff')
    parser.add_argument('--cfg', default='./configs/eval_fix/cfg_yearbook.yaml', metavar='FILE', help='path to config file', type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER,
                        )
    args = parser.parse_args()

    configs = OmegaConf.load(args.cfg)

    opts_dict = {args.opts[i]: args.opts[i + 1] for i in range(0, len(args.opts), 2)}
    configs_from_opts = OmegaConf.create()
    for key, value in opts_dict.items():
        keys = key.split('.')
        current_level = configs_from_opts
        for k in keys[:-1]:
            if k not in current_level:
                current_level[k] = OmegaConf.create()
            current_level = current_level[k]
        if keys[-1] in ['data_dir', 'backbone']:
            current_level[keys[-1]] = value
        else:
            if value.lower() in ("none"):
                current_level[keys[-1]] = None
            else:
                current_level[keys[-1]] = ast.literal_eval(value)

    cfg = OmegaConf.merge(configs, configs_from_opts)
    cfg.trainer.dim_bottleneck_f = str2none(cfg.trainer.dim_bottleneck_f)

    if not os.path.isdir(cfg.log.log_dir):
        os.makedirs(cfg.log.log_dir)
    logger = setup_logger("main", cfg.log.log_dir, 0, filename=get_current_time() + "_" + cfg.log.log_name)
    logger.info("PTL.version = {}".format(PIL.__version__))
    logger.info("torch.version = {}".format(torch.__version__))
    logger.info("torchvision.version = {}".format(torchvision.__version__))
    logger.info("Running with config:\n{}".format(cfg))

    dataset, criterion, network, diffusion_model, optimizer, scheduler = trainer_init(cfg)
    if cfg.trainer.method == "wdiff":
        trainer = WDiff(cfg, logger, dataset, network, diffusion_model, criterion, optimizer, scheduler)
    else:
        raise ValueError

    trainer.run()





