import torch
import torch.nn as nn 

import math

from performer_pytorch import Performer

class TransactionEncoder(nn.Module):
    def __init__(self, feature_embeddings, linear_proj: int=None):
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
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
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
            feature_embeddings, 
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

    def forward(self, x: torch.Tensor, device: str="cpu") -> torch.Tensor:
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
    def __init__(
            self, 
            feature_embeddings, 
            linear_proj: int=None,
            n_head: int=8, 
            dropout: float=0.1, 
            num_layers: int=6, 
            head_hidden: int=128,
            max_len: int=1000,
            dim_feedforward: int=128,
            dim_head: int=32,
            local_attn_heads: int=0,
            local_window_size: int=256,
            ff_mult: int=4,
            feature_redraw_interval: int=1000,
            reversible: bool=False,
            ff_chunks: int=1,
            generalized_attention: bool=False,
            use_scalenorm: bool=False,
            use_rezero: bool=False,
            ff_glu: bool=False,
            cross_attend: bool=False,
            no_projection: bool=False,
            auto_check_redraw: bool=True,
            qkv_bias: bool=True,
            attn_out_bias: bool=True,
            shift_tokens: bool=False
        ):
        super().__init__()

        self.transaction_encoder = TransactionEncoder(feature_embeddings, linear_proj=linear_proj)
        self.embedding_dim = self.transaction_encoder.embedding_dim
        self.cat_cols = list(feature_embeddings.keys())
        self.num_classes_dict = {key: num_classes for key, (num_classes, _) in feature_embeddings.items()}
        
        self.pos_emb = PositionalEncoding(self.embedding_dim, dropout, max_len)

        self.transformer_encoder = Performer(
            dim = self.embedding_dim, 
            depth = num_layers,
            heads = n_head, 
            ff_dropout = dropout,
            causal = True,
            dim_head = dim_head,
            local_attn_heads = local_attn_heads,
            local_window_size = local_window_size,
            ff_mult = ff_mult,
            feature_redraw_interval = feature_redraw_interval,
            reversible = reversible,
            ff_chunks = ff_chunks,
            generalized_attention = generalized_attention,
            use_scalenorm = use_scalenorm,
            use_rezero = use_rezero,
            ff_glu = ff_glu,
            cross_attend = cross_attend,
            no_projection = no_projection,
            auto_check_redraw = auto_check_redraw,
            qkv_bias = qkv_bias,
            attn_out_bias = attn_out_bias,
            shift_tokens = shift_tokens
        )
        
        self.heads = nn.ModuleDict({
            key: Head(
                self.embedding_dim, 
                head_hidden, 
                num_classes
            ) for key, num_classes in self.num_classes_dict.items()
        })

    def forward(self, x: torch.Tensor, device: str="cpu") -> torch.Tensor:
        N, S = x[self.cat_cols[0]].shape
        embeddings = self.transaction_encoder(x, device=device)
        embeddings = self.pos_emb(embeddings)
        
        attn_mask = self.generate_square_subsequent_mask(S).to(device)
        padding_mask = self.generate_padding_mask(x[self.cat_cols[0]]).to(device)
        embeddings = self.transformer_encoder(embeddings, input_mask=padding_mask)

        logits = {key: self.heads[key](embeddings) for key in self.cat_cols}
        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), True), diagonal=1).bool()
    
    @staticmethod
    def generate_padding_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x == 0, True, False).bool()


class ReformerModel(nn.Module):
    # TODO: implement model
    pass
