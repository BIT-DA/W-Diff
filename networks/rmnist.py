import torch.nn as nn
from .utils import MyQueue



class RotatedMNISTNetwork_for_WDiff(nn.Module):
    def __init__(self, args, num_input_channels, num_classes):
        super(RotatedMNISTNetwork_for_WDiff, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(num_input_channels, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.enc = nn.Sequential(self.conv1, self.relu, self.bn0, self.conv2, self.relu, self.bn1,
                                 self.conv3, self.relu, self.bn2, self.conv4, self.relu, self.bn3,
                                 self.avgpool)
        self.feature_dim = 128

        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        self.reference_point_queue = MyQueue(maxsize=args.trainer.L)
        self.anchor_point_and_prototype_queue = MyQueue(maxsize=args.trainer.M)
        self.eps = 1e-6

    def memorize(self, W):
        # W.shape: [C, D]
        self.reference_point_queue.put_item(W)

    def foward_encoder(self, x):
        f = self.enc(x)
        f = f.view(len(f), -1)
        return f

    def foward(self, x):
        f = self.enc(x)
        f = f.view(len(f), -1)
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



