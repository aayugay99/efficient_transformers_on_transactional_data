import torch
import torch.nn as nn 


class TransactionEncoder(nn.Module):
    def __init__(self, feature_embeddings, linear_proj=None):
        super().__init__()
        
        self.feature_embeddings = feature_embeddings
        self.embeddings = nn.ModuleDict({key: nn.Embedding(vocab, dim) for key, (vocab, dim) in feature_embeddings.items()})
        self.linear_proj = nn.Identity()
        if linear_proj is not None:
            self.embedding_dim = linear_proj
            self.linear_proj = nn.Linear(sum([dim for key, (vocab, dim) in feature_embeddings.items()]), linear_proj)
        else:
            self.embedding_dim = sum([dim for key, (vocab, dim) in feature_embeddings.items()])

    def forward(self, x, device="cpu"):
        embeddings = [self.embeddings[key](x[key].to(device)) for key in self.feature_embeddings]
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
    def __init__(
            self, 
            feature_embeddings, 
            linear_proj=None,
            n_head=8, 
            dim_feedforward=128, 
            dropout=0.1, 
            num_layers=6, 
            head_hidden=128,
        ):
        super().__init__()

        self.transaction_encoder = TransactionEncoder(feature_embeddings, linear_proj=linear_proj)
        self.cat_cols = list(feature_embeddings.keys())
        self.num_classes_dict = {key: num_classes for key, (num_classes, _) in feature_embeddings.items()}

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
                num_classes
            ) for key, num_classes in self.num_classes_dict.items()
        })

    def forward(self, x, device="cpu"):
        N, S = x[self.cat_cols[0]].shape
        embeddings = self.transaction_encoder(x, device=device)#.to(device)
        
        attn_mask = self.generate_square_subsequent_mask(S).to(device)
        padding_mask = self.generate_padding_mask(x[self.cat_cols[0]]).to(device)

        encoded = self.transformer_encoder(embeddings, mask=attn_mask, is_causal=True, src_key_padding_mask=padding_mask)
        logits = {key: self.heads[key](encoded) for key in self.cat_cols}
        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.full((sz, sz), True), diagonal=1).bool()
    
    @staticmethod
    def generate_padding_mask(x):
        return torch.where(x == 0, True, 0).bool()
