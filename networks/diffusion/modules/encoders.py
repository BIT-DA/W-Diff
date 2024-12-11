import torch.nn as nn



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch):
        # this is for use in crossattn
        c = batch[:, None]
        c = self.embedding(c)
        return c


