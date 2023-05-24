import torch
import torch.nn as nn 
from reformer_pytorch import LSHSelfAttention, Autopadder
from typing import Union

import copy


class TransactionEncoder(nn.Module):
    def __init__(self, feature_embeddings: dict[str, tuple[int, int]], linear_proj: int=None):
        super().__init__()
        
        self.feature_embeddings = feature_embeddings
        self.embeddings = nn.ModuleDict({key: nn.Embedding(vocab, dim) for key, (vocab, dim) in feature_embeddings.items()})
        
        if linear_proj is not None:
            self.embedding_dim = linear_proj
            self.linear_proj = nn.Linear(sum([dim for key, (vocab, dim) in feature_embeddings.items()]), linear_proj)
        else:
            self.embedding_dim = sum([dim for key, (vocab, dim) in feature_embeddings.items()])
            self.linear_proj = nn.Identity()

    def forward(self, x: torch.Tensor, device: str="cpu") -> torch.Tensor:
        embeddings = [self.embeddings[key](x[key].to(device)) for key in self.feature_embeddings]
        proj = self.linear_proj(torch.cat(embeddings, dim=2))
        return proj


class Head(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float=0.1, max_len: int=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-9.21034037198 / embedding_dim))
        pe = torch.zeros(1, max_len, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
            self, 
            feature_embeddings: dict[str, tuple[int, int]], 
            linear_proj: int=None,
            n_head: int=8, 
            dim_feedforward: int=128, 
            dropout: float=0.1, 
            num_layers: int=6, 
            head_hidden: int=128,
            max_len: int=1000,
        ):
        super().__init__()

        self.transaction_encoder = TransactionEncoder(feature_embeddings, linear_proj=linear_proj)
        self.embedding_dim = self.transaction_encoder.embedding_dim
        self.cat_cols = list(feature_embeddings.keys())
        self.num_classes_dict = {key: num_classes for key, (num_classes, _) in feature_embeddings.items()}
        
        self.pos_emb = PositionalEncoding(self.embedding_dim, dropout, max_len)

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

    def forward(self, x: torch.Tensor, device: Union[str, torch.device]="cpu") -> torch.Tensor:
        N, S = x[self.cat_cols[0]].shape
        embeddings = self.transaction_encoder(x, device=device)
        embeddings = self.pos_emb(embeddings)
        
        attn_mask = self.generate_square_subsequent_mask(S).to(device)
        padding_mask = self.generate_padding_mask(x[self.cat_cols[0]]).to(device)
        embeddings = self.transformer_encoder(embeddings, mask=attn_mask, is_causal=True, src_key_padding_mask=padding_mask)

        logits = {key: self.heads[key](embeddings) for key in self.cat_cols}
        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), True), diagonal=1).bool()
    
    @staticmethod
    def generate_padding_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x == 0, True, False).bool()


class LinearTransformerModel(nn.Module):
    # TODO: implement model
    pass


class PerformerModel(nn.Module):
    # TODO: implement model
    pass


class ReformerBlock(nn.Module):
    def __init__(
            self, 
            d_model, 
            n_head, 
            bucket_size,
            n_hashes,
            dim_feedforward,
            dropout
        ):
        super().__init__()

        self.self_attn = Autopadder(
            LSHSelfAttention(
                dim=d_model, 
                heads=n_head, 
                bucket_size=bucket_size, 
                n_hashes=n_hashes, 
                causal=True, 
                n_local_attn_heads=4
            )
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, x, input_mask=None):
        x = self.norm1(x + self._sa_block(x, input_mask=input_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, input_mask=None):
        x = self.self_attn(x, input_mask=input_mask)
        return self.dropout1(x)
    
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

class ReformerEncoder(nn.Module):
    def __init__(self, reformer_block, num_layers):
        super().__init__()

        self.blocks = nn.ModuleList([copy.deepcopy(reformer_block) for i in range(num_layers)])

    def forward(self, x, input_mask=None):
        for block in self.blocks:
            x = block(x, input_mask)
        return x

    
class ReformerModel(nn.Module):
    def __init__(
            self, 
            feature_embeddings: dict[str, tuple[int, int]], 
            linear_proj: int=None,
            n_head: int=8, 
            dim_feedforward: int=128, 
            dropout: float=0.1, 
            num_layers: int=6, 
            bucket_size: int=25,
            n_hashes: int=8,
            head_hidden: int=128,
            max_len: int=1000,
        ):
        super().__init__()

        self.transaction_encoder = TransactionEncoder(feature_embeddings, linear_proj=linear_proj)
        self.embedding_dim = self.transaction_encoder.embedding_dim
        self.cat_cols = list(feature_embeddings.keys())
        self.num_classes_dict = {key: num_classes for key, (num_classes, _) in feature_embeddings.items()}
        
        self.pos_emb = PositionalEncoding(self.embedding_dim, dropout, max_len)
        
        self.reformer_block = ReformerBlock(
            d_model=self.embedding_dim,
            n_head=n_head, 
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.reformer_encoder = ReformerEncoder(self.reformer_block, num_layers)

        self.heads = nn.ModuleDict({
            key: Head(
                self.embedding_dim, 
                head_hidden, 
                num_classes
            ) for key, num_classes in self.num_classes_dict.items()
        })

    def forward(self, x: torch.Tensor, device: Union[str, torch.device]="cpu") -> torch.Tensor:
        embeddings = self.transaction_encoder(x, device=device)
        embeddings = self.pos_emb(embeddings)
        
        padding_mask = self.generate_padding_mask(x[self.cat_cols[0]]).to(device)
        embeddings = self.reformer_encoder(embeddings, input_mask=padding_mask)

        logits = {key: self.heads[key](embeddings) for key in self.cat_cols}
        
        return logits
    
    @staticmethod
    def generate_padding_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x == 0, True, False).bool()
