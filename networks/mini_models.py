import torch.nn as nn
from typing import Union

from .utils import MyQueue


def init_weights(model):
    if type(model) == nn.Linear:
        nn.init.kaiming_normal_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)


class PredictionModelNN(nn.Module):
    def __init__(self, input_shape, hidden_shapes, output_shape, classifier_bias=True, **kwargs):
        super(PredictionModelNN, self).__init__()

        self.leaky = kwargs['leaky']
        sub_modules = []
        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape

        if len(self.hidden_shapes) == 0:  # single layer NN
            sub_modules.append(nn.Linear(input_shape, output_shape))
            sub_modules.append(nn.LeakyReLU())

        else:
            sub_modules.append(nn.Linear(self.input_shape, self.hidden_shapes[0]))
            if self.leaky:
                sub_modules.append(nn.LeakyReLU())
            else:
                sub_modules.append(nn.ReLU())

            for i in range(len(self.hidden_shapes) - 1):
                sub_modules.append(nn.Linear(self.hidden_shapes[i], self.hidden_shapes[i + 1]))
                if self.leaky:
                    sub_modules.append(nn.LeakyReLU())
                else:
                    sub_modules.append(nn.ReLU())

        self.feature = nn.Sequential(*sub_modules)
        self.fea_dim = self.hidden_shapes[-1]
        if not classifier_bias:
            self.classifier = nn.Linear(self.hidden_shapes[-1], self.output_shape, bias=False)
        else:
            self.classifier = nn.Linear(self.hidden_shapes[-1], self.output_shape)
        self.apply(init_weights)

    def forward(self, X):
        fea = self.feature(X)
        out = self.classifier(fea)
        return out


def get_fea_classifier(network: Union[PredictionModelNN]):
    return network.feature, network.classifier, network.fea_dim




class PredictionModelNN_for_WDiff(nn.Module):
    def __init__(self, args, backbone: Union[PredictionModelNN]):
        super(PredictionModelNN_for_WDiff, self).__init__()
        self.args = args
        self.enc, self.classifier, self.feature_dim = get_fea_classifier(backbone)
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
