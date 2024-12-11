import torch
import torch.nn as nn
from .utils import MyQueue


class YearbookNetwork_for_WDiff(nn.Module):
    def __init__(self, args, num_input_channels, num_classes):
        super(YearbookNetwork_for_WDiff, self).__init__()
        self.args = args
        self.enc = nn.Sequential(self.conv_block(num_input_channels, 32), self.conv_block(32, 32),
                                 self.conv_block(32, 32), self.conv_block(32, 32))
        self.feature_dim = 32
        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        self.reference_point_queue = MyQueue(maxsize=args.trainer.L)
        self.anchor_point_and_prototype_queue = MyQueue(maxsize=args.trainer.M)
        self.eps = 1e-6

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def memorize(self, W):
        # W.shape: [C, D]
        self.reference_point_queue.put_item(W)

    def foward_encoder(self, x):
        f = self.enc(x)
        f = torch.mean(f, dim=(2, 3))
        return f

    def foward(self, x):
        f = self.enc(x)
        f = torch.mean(f, dim=(2, 3))
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









