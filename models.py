import torch
import torch.nn as nn 


class TransactionEncoder(nn.Module):
    def __init__(self, embedding_dict, linear_proj=None):
        super().__init__()

        self.features = embedding_dict.keys()
        self.embeddings = nn.ModuleDict({key: nn.Embedding(vocab, dim) for key, (vocab, dim) in embedding_dict.items()})
        self.linear_proj = nn.Identity()
        if linear_proj is not None:
            self.linear_proj = nn.Linear(sum([dim for key, (vocab, dim) in embedding_dict.items()]), linear_proj)

    def forward(self, x):
        embeddings = [self.embeddings[key](x[key]) for key in self.features]
        proj = self.linear_proj(torch.cat(embeddings, dim=2))
        return proj