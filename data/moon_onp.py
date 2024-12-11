import numpy as np
import torch
from torch.utils.data import Dataset
import json
from copy import deepcopy

ID_HELD_OUT = 0.1


def load_data(root):
    X = np.load("{}/X.npy".format(root))
    Y = np.load("{}/Y.npy".format(root))
    U = np.load("{}/U.npy".format(root))
    indices = json.load(open("{}/indices.json".format(root)))
    return X, U, Y, indices


class DatasetBase(Dataset):
    def __init__(self, args):
        super().__init__()
        X_data, U_data, Y_data, indices = load_data(self.data_path())

        X_data = np.array([X_data[ids] for ids in indices])
        Y_data = np.array([Y_data[ids] for ids in indices])
        U_data = np.array([U_data[ids] for ids in indices])

        if args.data.dataset == 'Moons':
            ENV = range(10)
        else:
            ENV = range(len(X_data))

        datasets = {}
        for env in ENV:
            mode_dict = {"data": X_data[env], "labels": Y_data[env], "time": U_data[env]}
            datasets[env] = {0: deepcopy(mode_dict), 1: deepcopy(mode_dict), 2: deepcopy(mode_dict)}


        self.datasets = datasets
        self.args = args
        self.ENV = list(sorted(self.datasets.keys()))
        self.current_time = 0

        self.mode = 0
        self.task_idxs = {}
        start_idx = 0
        for i in self.ENV:
            end_idx = start_idx + len(self.datasets[i][self.mode]['labels'])
            self.task_idxs[i] = {}
            self.task_idxs[i][self.mode] = [start_idx, end_idx]
            start_idx = end_idx

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['data'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['data'], self.datasets[time][self.mode]['data']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['labels'], self.datasets[time][self.mode]['labels']), axis=0)
        self.datasets[time][self.mode]['time'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['time'], self.datasets[time][self.mode]['time']), axis=0)
        if data_del:
            del self.datasets[prev_time]

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        data = self.datasets[self.current_time][self.mode]['data'][index]
        label = self.datasets[self.current_time][self.mode]['labels'][index]

        data_tensor = torch.FloatTensor(data)
        label_tensor = torch.LongTensor([label])
        return data_tensor, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])

    def data_path(self):
        raise NotImplementedError



class Moons(DatasetBase):
    def __init__(self, args):
        self.num_classes = 2
        self.args = args
        super().__init__(args)

    def data_path(self):
        return self.args.data.data_dir + "/processed"

    def __str__(self):
        return "Moons"


class ONP(DatasetBase):
    def __init__(self, args):
        self.num_classes = 2
        self.args = args
        super().__init__(args)

    def data_path(self):
        return self.args.data.data_dir + "/processed"

    def __str__(self):
        return "ONP"
