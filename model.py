import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Literal

class InputEmbeddings(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        vocab_size: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        max_seq_len: int = 5000, 
        dropout: float = 0.1
    ) -> None: 
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("positional_encoding", self._get_positional_encoding(d_model, max_seq_len))

    def _get_positional_encoding(self, d_model: int, max_seq_len: int) -> torch.Tensor:
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :])
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(d_model)) # bias is a learnable parameter

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: Literal['relu', 'gelu'], dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation not in ["relu", "gelu"]:
            raise ValueError(f"Unknown activation function: {activation}")
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        x = self.gate(x)
        x = self.activation(x)
        x = self.down(x)
        x = self.dropout(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads

        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention_weights = None

    @staticmethod
    def _scaled_dot_product_attention(q, k, v, mask, dropout: nn.Dropout):
        # q: (batch, seq_len, d_k)
        # k: (batch, seq_len, d_k)
        # v: (batch, seq_len, d_v)
        # mask: (batch, seq_len, seq_len)
        
        d_k = k.shape[-1]
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        return (attention_weights @ v), attention_weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_weights = self._scaled_dot_product_attention(
            query, key, value, mask, self.dropout
        )
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        activation: Literal['relu', 'gelu'], 
        dropout: float
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, activation, dropout)
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout), # For self-attention
            ResidualConnection(d_model, dropout)  # For feed-forward
        ])

    def forward(self, x, src_mask):
        # Self-attention sublayer
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        # Feed-forward sublayer
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        activation: Literal['relu', 'gelu'], 
        dropout: float, 
        num_blocks: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, activation, dropout) 
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
class DecoderBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        activation: Literal['relu', 'gelu'], 
        dropout: float
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, activation, dropout)
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout), # For self-attention
            ResidualConnection(d_model, dropout), # For cross-attention
            ResidualConnection(d_model, dropout)  # For feed-forward
        ])

    def forward(self, x, context, src_mask, tgt_mask):
        # Self-attention sublayer
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        # Cross-attention sublayer
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, context, context, src_mask))
        # Feed-forward sublayer
        x = self.residual_connections[2](x, self.feed_forward)
        return x
class Decoder(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        activation: Literal['relu', 'gelu'], 
        dropout: float, 
        num_blocks: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, activation, dropout) 
            for _ in range(num_blocks)
        ])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, context, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, context, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.projection(x)

class Transformer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        activation: Literal['relu', 'gelu'], 
        dropout: float, 
        num_encoder_blocks: int, 
        num_decoder_blocks: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_seq_len: int,
        use_gradient_checkpointing: bool = False,
        tie_weights: bool = True
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Embeddings and Positional Encoding
        self.src_embedding = InputEmbeddings(d_model, src_vocab_size)
        self.src_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Encoder and Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, activation, dropout, num_encoder_blocks)
        self.decoder = Decoder(d_model, num_heads, d_ff, activation, dropout, num_decoder_blocks)
        
        # Projection layer
        self.projection = ProjectionLayer(d_model, tgt_vocab_size)
        
        # Weight tying
        if tie_weights:
            self.projection.projection.weight = self.tgt_embedding.embedding.weight
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, LayerNormalization):
                nn.init.ones_(module.alpha)
                nn.init.zeros_(module.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos_encoding(src)
        
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self.encoder, src, src_mask, use_reentrant=True
            )
        
        return self.encoder(src, src_mask)

    def decode(self, tgt, context, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos_encoding(tgt)
        
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self.decoder, tgt, context, src_mask, tgt_mask, use_reentrant=True
            )
            
        return self.decoder(tgt, context, src_mask, tgt_mask)

    def project(self, x):
        return self.projection(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.project(decoder_output)