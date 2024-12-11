import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import initialize_distilbert_transform

PREPROCESSED_FILE = 'huffpost.pkl'
MAX_TOKEN_LENGTH = 300
ID_HELD_OUT = 0.1

class HuffPostBase(Dataset):
    def __init__(self, args):
        super().__init__()

        self.data_file = f'{str(self)}.pkl'
        preprocess(args)
        self.datasets = pickle.load(open(os.path.join(args.data.data_dir, self.data_file), 'rb'))

        self.args = args
        self.ENV = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
        self.num_classes = 11
        self.num_tasks = len(self.ENV)
        self.current_time = 0
        self.mini_batch_size = args.data.mini_batch_size
        self.task_indices = {}
        self.transform = initialize_distilbert_transform(max_token_length=MAX_TOKEN_LENGTH)
        self.mode = 0

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        start_idx = 0
        self.task_idxs = {}
        for i, year in enumerate(self.ENV):
            # Store task indices
            end_idx = start_idx + len(self.datasets[year][self.mode]['category'])
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

            # Store class id list
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(np.array(self.datasets[year][self.mode]['category']) == classid)[0]
                self.class_id_list[classid][year] = sel_idx
            print(f'Year {str(year)} loaded')

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['headline'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['headline'], self.datasets[time][self.mode]['headline']), axis=0)
        self.datasets[time][self.mode]['category'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['category'], self.datasets[time][self.mode]['category']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['category'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'huffpost'



class HuffPost(HuffPostBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        headline = self.datasets[self.current_time][self.mode]['headline'][index]
        category = self.datasets[self.current_time][self.mode]['category'][index]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])
        return x, y

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['category'])



"""
News Categories to IDs:
    {'BLACK VOICES': 0, 'BUSINESS': 1, 'COMEDY': 2, 'CRIME': 3, 
    'ENTERTAINMENT': 4, 'IMPACT': 5, 'QUEER VOICES': 6, 'SCIENCE': 7, 
    'SPORTS': 8, 'TECH': 9, 'TRAVEL': 10}
"""

def preprocess(args):
    if not os.path.isfile(os.path.join(args.data.data_dir, 'huffpost.pkl')):
        raise RuntimeError('dataset huffpost.pkl is not yet ready! Please download from   https://drive.google.com/u/0/uc?id=1jKqbfPx69EPK_fjgU9RLuExToUg7rwIY&export=download   and save it as huffpost.pkl')


