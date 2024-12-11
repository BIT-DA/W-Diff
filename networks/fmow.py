from copy import deepcopy
import torch.nn as nn
from torchvision.models import densenet121, densenet161, densenet169, densenet201

from .utils import MyQueue

IMG_HEIGHT = 224



class FMoWNetwork_for_WDiff(nn.Module):
    def __init__(self, args, weights=None):
        super(FMoWNetwork_for_WDiff, self).__init__()
        self.args = args
        self.num_classes = 62

        if args.trainer.dim_bottleneck_f is not None:
            if args.trainer.backbone == 'densenet121':
                self.bottleneck = nn.Sequential(
                    nn.Linear(1024, args.trainer.dim_bottleneck_f),
                    nn.BatchNorm1d(args.trainer.dim_bottleneck_f),
                    nn.ReLU()
                )
                self.enc = nn.Sequential(densenet121(pretrained=True).features, nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), self.bottleneck)
            elif args.trainer.backbone == 'densenet161':
                self.bottleneck = nn.Sequential(
                    nn.Linear(2208, args.trainer.dim_bottleneck_f),
                    nn.BatchNorm1d(args.trainer.dim_bottleneck_f),
                    nn.ReLU()
                )
                self.enc = nn.Sequential(densenet161(pretrained=True).features, nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), self.bottleneck)
            elif args.trainer.backbone == 'densenet169':
                self.bottleneck = nn.Sequential(
                    nn.Linear(1664, args.trainer.dim_bottleneck_f),
                    nn.BatchNorm1d(args.trainer.dim_bottleneck_f),
                    nn.ReLU()
                )
                self.enc = nn.Sequential(densenet169(pretrained=True).features, nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), self.bottleneck)
            elif args.trainer.backbone == 'densenet201':
                self.bottleneck = nn.Sequential(
                    nn.Linear(1920, args.trainer.dim_bottleneck_f),
                    nn.BatchNorm1d(args.trainer.dim_bottleneck_f),
                    nn.ReLU()
                )
                self.enc = nn.Sequential(densenet201(pretrained=True).features, nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), self.bottleneck)
            self.feature_dim = args.trainer.dim_bottleneck_f
        else:
            if args.trainer.backbone == 'densenet121':
                self.enc = nn.Sequential(densenet121(pretrained=True).features, nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
                self.feature_dim = 1024
            elif args.trainer.backbone == 'densenet161':
                self.enc = nn.Sequential(densenet161(pretrained=True).features, nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
                self.feature_dim = 2208
            elif args.trainer.backbone == 'densenet169':
                self.enc = nn.Sequential(densenet169(pretrained=True).features, nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
                self.feature_dim = 1664
            elif args.trainer.backbone == 'densenet201':
                self.enc = nn.Sequential(densenet201(pretrained=True).features, nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
                self.feature_dim = 1920

        if weights is not None:
            self.load_state_dict(deepcopy(weights))

        self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.reference_point_queue = MyQueue(maxsize=args.trainer.L)
        self.anchor_point_and_prototype_queue = MyQueue(maxsize=args.trainer.M)
        self.eps = 1e-6

    def memorize(self, W):
        # W.shape: [C, D]
        self.reference_point_queue.put_item(W)

    def foward_encoder(self, x):
        f = self.enc(x)
        return f

    def foward(self, x):
        f = self.enc(x)
        logits = self.classifier(f)
        return f, logits

    def get_parameters(self, lr):
        params_list = []
        params_list.extend([
            {"params": self.enc.parameters(), 'lr': 1 * lr},
            {"params": self.classifier.parameters(), 'lr': 1 * lr},
        ]
        )
        return params_list
