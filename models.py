import torch
import torch.nn as nn 
from reformer_pytorch import LSHSelfAttention, Autopadder
from performer_pytorch import SelfAttention as PerformerSelfAttention
from linear_attention_transformer import SelfAttention as LinearSelfAttention
from typing import Union

import copy

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
    def __init__(
            self, 
            feature_embeddings, 
            linear_proj: int=None,
            n_head: int=8, 
            dim_feedforward: int=128, 
            dropout: float=0.0, 
            num_layers: int=6, 
            head_hidden: int=128,
            max_len: int=1000,
            dim_head: int=32,
            local_attn_window_size: int=128,
            receives_context: bool=False,
            blindspot_size: int=1,
            n_local_attn_heads: int=0,
            attn_dropout: float=0.0
        ):
        super().__init__()

        self.transaction_encoder = TransactionEncoder(feature_embeddings, linear_proj=linear_proj)
        self.embedding_dim = self.transaction_encoder.embedding_dim
        self.cat_cols = list(feature_embeddings.keys())
        self.num_classes_dict = {key: num_classes for key, (num_classes, _) in feature_embeddings.items()}
        
        self.pos_emb = PositionalEncoding(self.embedding_dim, dropout, max_len)

        sa_module = LinearSelfAttention(
            self.embedding_dim, 
            n_head,
            causal=True, 
            dim_head=dim_head,
            blindspot_size=blindspot_size,
            n_local_attn_heads=n_local_attn_heads,
            local_attn_window_size=local_attn_window_size,
            receives_context=receives_context,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        self.encoder_layer = Block(
            self.embedding_dim, 
            dim_feedforward, 
            dropout, 
            sa_module
        )
        self.transformer_encoder = Encoder(self.encoder_layer, num_layers)
        
        self.heads = nn.ModuleDict({
            key: Head(
                self.embedding_dim, 
                head_hidden, 
                num_classes
            ) for key, num_classes in self.num_classes_dict.items()
        })

    def forward(self, x: torch.Tensor, device: str="cpu") -> torch.Tensor:
        embeddings = self.transaction_encoder(x, device=device)
        embeddings = self.pos_emb(embeddings)
        
        padding_mask = self.generate_padding_mask(x[self.cat_cols[0]]).to(device)
        embeddings = self.transformer_encoder(embeddings, input_mask=padding_mask)

        logits = {key: self.heads[key](embeddings) for key in self.cat_cols}
        return logits
    
    @staticmethod
    def generate_padding_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x == 0, True, False).bool()


class Block(nn.Module):
    def __init__(
            self, 
            d_model, 
            dim_feedforward,
            dropout,
            self_attention
        ):
        super().__init__()

        self.self_attn = self_attention
        
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
    

class Encoder(nn.Module):
    def __init__(self, block, num_layers):
        super().__init__()

        self.blocks = nn.ModuleList([copy.deepcopy(block) for i in range(num_layers)])

    def forward(self, x, input_mask=None):
        for block in self.blocks:
            x = block(x, input_mask)
        return x


class PerformerModel(nn.Module):
    def __init__(
            self, 
            feature_embeddings, 
            linear_proj: int=None,
            n_head: int=8, 
            dropout: float=0.1, 
            num_layers: int=6, 
            dim_feedforward: int=128,
            head_hidden: int=128,
            max_len: int=1000,
            dim_head: int=32,
            local_heads: int=0,
            local_window_size: int=256,
            feature_redraw_interval: int=1000,
            no_projection: bool=False,
            qkv_bias: bool=True,
            attn_out_bias: bool=True
        ):
        super().__init__()

        self.transaction_encoder = TransactionEncoder(feature_embeddings, linear_proj=linear_proj)
        self.embedding_dim = self.transaction_encoder.embedding_dim
        self.cat_cols = list(feature_embeddings.keys())
        self.num_classes_dict = {key: num_classes for key, (num_classes, _) in feature_embeddings.items()}
        
        self.pos_emb = PositionalEncoding(self.embedding_dim, dropout, max_len)

        sa_module = PerformerSelfAttention(
            self.embedding_dim, 
            causal=True, 
            heads=n_head,
            dim_head=dim_head,
            local_heads=local_heads,
            local_window_size=local_window_size,
            feature_redraw_interval=feature_redraw_interval,
            no_projection=no_projection,
            qkv_bias=qkv_bias,
            attn_out_bias=attn_out_bias
        )
        self.encoder_layer = Block(
            self.embedding_dim, 
            dim_feedforward, 
            dropout, 
            sa_module
        )
        self.transformer_encoder = Encoder(self.encoder_layer, num_layers)
        
        self.heads = nn.ModuleDict({
            key: Head(
                self.embedding_dim, 
                head_hidden, 
                num_classes
            ) for key, num_classes in self.num_classes_dict.items()
        })

    def forward(self, x: torch.Tensor, device: str="cpu") -> torch.Tensor:
        embeddings = self.transaction_encoder(x, device=device)
        embeddings = self.pos_emb(embeddings)
        
        padding_mask = self.generate_padding_mask(x[self.cat_cols[0]]).to(device)
        embeddings = self.transformer_encoder(embeddings, input_mask=padding_mask)

        logits = {key: self.heads[key](embeddings) for key in self.cat_cols}
        return logits
    
    @staticmethod
    def generate_padding_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x == 0, True, False).bool()

    
class ReformerModel(nn.Module):
    def __init__(
            self, 
            feature_embeddings, 
            linear_proj: int=None,
            n_head: int=8, 
            dropout: float=0.1, 
            num_layers: int=6,
            dim_feedforward: int=128, 
            head_hidden: int=128,
            max_len: int=1000,
            attn_chunks: int=1,
            n_local_attn_heads: int=4,
            bucket_size: int=25,
            n_hashes: int=8,
            dim_head: int=32,
            random_rotations_per_head: bool = False,
            attend_across_buckets: bool = True,
            allow_duplicate_attention: bool = True,
            num_mem_kv: int = 0,
            one_value_head: bool = False
        ):
        super().__init__()

        self.transaction_encoder = TransactionEncoder(feature_embeddings, linear_proj=linear_proj)
        self.embedding_dim = self.transaction_encoder.embedding_dim
        self.cat_cols = list(feature_embeddings.keys())
        self.num_classes_dict = {key: num_classes for key, (num_classes, _) in feature_embeddings.items()}
        
        self.pos_emb = PositionalEncoding(self.embedding_dim, dropout, max_len)
        
        sa_module = Autopadder(
            LSHSelfAttention(
                dim=self.embedding_dim,
                heads=n_head,
                attn_chunks=attn_chunks,
                bucket_size=bucket_size,
                n_hashes=n_hashes,
                causal=True,
                dim_head=dim_head,
                n_local_attn_heads=n_local_attn_heads,
                random_rotations_per_head=random_rotations_per_head,
                attend_across_buckets=attend_across_buckets,
                allow_duplicate_attention=allow_duplicate_attention,
                num_mem_kv=num_mem_kv,
                one_value_head=one_value_head
            )
        )
        self.encoder_layer = Block(
            d_model=self.embedding_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            self_attention=sa_module
        )
        self.transformer_encoder = Encoder(self.encoder_layer, num_layers)

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
        embeddings = self.transformer_encoder(embeddings, input_mask=padding_mask)

        logits = {key: self.heads[key](embeddings) for key in self.cat_cols}
        
        return logits
    
    @staticmethod
    def generate_padding_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x == 0, True, False).bool()
