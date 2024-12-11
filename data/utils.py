import os
import gdown
import torch
from transformers import DistilBertTokenizer, BertTokenizer


def initialize_distilbert_transform(max_token_length):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # tokenizer = DistilBertTokenizer.from_pretrained(os.getcwd() + '/data/distilbert-base-uncased')
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_token_length,
            return_tensors='pt')
        x = torch.stack((tokens['input_ids'], tokens['attention_mask']), dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform

    
def download_gdrive(url, save_path, is_folder):
    """ Download the preprocessed data from Google Drive. """
    if not is_folder:
        gdown.download(url=url, output=save_path, quiet=False)
    else:
        gdown.download_folder(url=url, output=save_path, quiet=False)

