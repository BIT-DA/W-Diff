import torch
from collections import defaultdict
from collections import deque




def prepare_data(x, y, dataset_name: str):
    if dataset_name in ['arxiv', 'huffpost']:
        x = x.to(dtype=torch.int64).cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
    elif dataset_name in ['fmow', 'yearbook', 'rmnist']:
        if isinstance(x, tuple):
            x = (elt.cuda() for elt in x)
        else:
            x = x.cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
        else:
            y = y.cuda()
    else:
        x = x.cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
        else:
            y = y.cuda()
    return x, y


def forward_pass(x, y, dataset, network, criterion):
    logits = network(x)
    if str(dataset) in ['drug']:
        logits = logits.squeeze().double()
        y = y.squeeze().double()
    elif str(dataset) in ['arxiv', 'fmow', 'huffpost', 'yearbook']:
        if len(y.shape) > 1:
            y = y.squeeze(1)
    loss = criterion(logits, y)

    return loss, logits, y


def split_into_groups(g):
    """
    From https://github.com/p-lambda/wilds/blob/f384c21c67ee58ab527d8868f6197e67c24764d4/wilds/common/utils.py#L40.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


def get_collate_functions(args, train_dataset):
    train_collate_fn = None
    eval_collate_fn = None
    return train_collate_fn, eval_collate_fn



class SmoothedValue(object):
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.8f} ({:.8f})".format(name, meter.series[meter.count - 1], meter.global_avg)
            )
        return self.delimiter.join(loss_str)

