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
    

class Head(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x):
        return self.head(x)


class TransformerModel(nn.Module):
    def __init__(self, encoder, n_head, dim_feedforward, dropout, num_layers, head_hidden):
        super().__init__()

        self.transaction_encoder = encoder
        self.cat_cols = list(self.transaction_encoder.embeddings.keys())

        self.embedding_dim = self.transaction_encoder.embedding_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            self.embedding_dim, 
            n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.heads = nn.ModuleDict({
            key: Head(
                self.embedding_dim, 
                head_hidden, 
                self.transaction_encoder.embeddings[key].num_embeddings
            ) for key in self.cat_cols
        })

    def forward(self, x):
        N, S = x[self.cat_cols[0]].shape
        embeddings = self.transaction_encoder(x)

        attn_mask = self.generate_square_subsequent_mask(S)
        padding_mask = self.generate_padding_mask(x[self.cat_cols[0]])

        encoded = self.transformer_encoder(embeddings, mask=attn_mask, is_causal=True, src_key_padding_mask=padding_mask)
        logits = {key: self.heads[key](encoded) for key in self.cat_cols}
        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    
    @staticmethod
    def generate_padding_mask(x):
        return torch.where(x == 0, float('-inf'), 0)
