import os
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification
from .utils import MyQueue



class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


class ArticleNetwork_for_WDiff(nn.Module):
    def __init__(self, args, num_classes):
        super(ArticleNetwork_for_WDiff, self).__init__()
        self.args = args
        self.featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")
        # self.featurizer = DistilBertFeaturizer.from_pretrained(os.getcwd() + '/data/distilbert-base-uncased')

        if args.trainer.dim_bottleneck_f is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.featurizer.d_out, args.trainer.dim_bottleneck_f),
                nn.BatchNorm1d(args.trainer.dim_bottleneck_f),
                nn.ReLU()
            )
            self.feature_dim = args.trainer.dim_bottleneck_f
            self.enc = nn.Sequential(self.featurizer, self.bottleneck)
        else:
            self.feature_dim = self.featurizer.d_out   # 768
            self.enc = nn.Sequential(self.featurizer)
        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
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

