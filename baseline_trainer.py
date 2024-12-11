import numpy as np
import torch
import torch.nn as nn
import random
import os
from functools import partial

from networks.mini_models import PredictionModelNN, PredictionModelNN_for_WDiff
from networks.yearbook import YearbookNetwork_for_WDiff
from networks.fmow import FMoWNetwork_for_WDiff
from networks.rmnist import RotatedMNISTNetwork_for_WDiff
from networks.article import ArticleNetwork_for_WDiff
from networks.diffusion.util import instantiate_from_config

print = partial(print, flush=True)


def _Moons_init(cfg):
    from data.moon_onp import Moons
    dataset = Moons(cfg)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=cfg.trainer.reduction).cuda()
    backbone = PredictionModelNN(2, [64, 128], 2, classifier_bias=False, leaky=True)
    diffusion_model = instantiate_from_config(cfg.DM).cuda()
    network = PredictionModelNN_for_WDiff(cfg, backbone).cuda()
    optimizer = torch.optim.Adam(network.get_parameters(cfg.trainer.lr), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
    return dataset, criterion, network, diffusion_model, optimizer, scheduler


def _ONP_init(cfg):
    from data.moon_onp import ONP
    dataset = ONP(cfg)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=cfg.trainer.reduction).cuda()
    backbone = PredictionModelNN(58, [128], 2, classifier_bias=False, leaky=True)
    diffusion_model = instantiate_from_config(cfg.DM).cuda()
    network = PredictionModelNN_for_WDiff(cfg, backbone).cuda()
    optimizer = torch.optim.Adam(network.get_parameters(cfg.trainer.lr), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
    return dataset, criterion, network, diffusion_model, optimizer, scheduler


def _yearbook_init(cfg):
    from data.yearbook import Yearbook
    dataset = Yearbook(cfg)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=cfg.trainer.reduction).cuda()
    diffusion_model = instantiate_from_config(cfg.DM).cuda()
    network = YearbookNetwork_for_WDiff(cfg, num_input_channels=3, num_classes=dataset.num_classes).cuda()
    optimizer = torch.optim.Adam(network.get_parameters(cfg.trainer.lr), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
    return dataset, criterion, network, diffusion_model, optimizer, scheduler


def _rmnist_init(cfg):
    from data.rmnist import RotatedMNIST
    dataset = RotatedMNIST(cfg)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=cfg.trainer.reduction).cuda()
    diffusion_model = instantiate_from_config(cfg.DM).cuda()
    network = RotatedMNISTNetwork_for_WDiff(cfg, num_input_channels=1, num_classes=dataset.num_classes).cuda()
    optimizer = torch.optim.Adam(network.get_parameters(cfg.trainer.lr), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
    return dataset, criterion, network, diffusion_model, optimizer, scheduler


def _fmow_init(cfg):
    from data.fmow import FMoW
    dataset = FMoW(cfg)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=cfg.trainer.reduction).cuda()
    diffusion_model = instantiate_from_config(cfg.DM).cuda()
    network = FMoWNetwork_for_WDiff(cfg).cuda()
    optimizer = torch.optim.Adam((network.get_parameters(cfg.trainer.lr)), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    return dataset, criterion, network, diffusion_model, optimizer, scheduler


def _huffpost_init(cfg):
    from data.huffpost import HuffPost
    dataset = HuffPost(cfg)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=cfg.trainer.reduction).cuda()
    diffusion_model = instantiate_from_config(cfg.DM).cuda()
    network = ArticleNetwork_for_WDiff(cfg, num_classes=dataset.num_classes).cuda()
    optimizer = torch.optim.Adam((network.get_parameters(cfg.trainer.lr)), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    return dataset, criterion, network, diffusion_model, optimizer, scheduler


def _arxiv_init(cfg):
    from data.arxiv import ArXiv
    dataset = ArXiv(cfg)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=cfg.trainer.reduction).cuda()
    diffusion_model = instantiate_from_config(cfg.DM).cuda()
    network = ArticleNetwork_for_WDiff(cfg, num_classes=dataset.num_classes).cuda()
    optimizer = torch.optim.Adam((network.get_parameters(cfg.trainer.lr)), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    return dataset, criterion, network, diffusion_model, optimizer, scheduler



def trainer_init(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    torch.set_num_threads(1)  # limiting the usage of cpu
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
    return globals()[f'_{args.data.dataset}_init'](args)


