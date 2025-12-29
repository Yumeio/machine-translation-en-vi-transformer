import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Literal, Optional, Tuple

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1) -> None: 
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

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (Batch, Heads, Seq, Dim)
    # cos, sin: (Seq, Dim) -> Broadcasts to (1, 1, Seq, Dim) 
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class QKNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_rmsnorm: bool = True):
        super().__init__()
        if use_rmsnorm:
            self.query_norm = RMSNorm(dim, eps=eps)
            self.key_norm = RMSNorm(dim, eps=eps)
        else:
            self.query_norm = nn.LayerNorm(dim, eps=eps)
            self.key_norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, q, k):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        multiple_of: int = 256,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            d_ff = int(2 * d_ff / 3)
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

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
    def __init__(self, d_model: int, dropout: float, use_rmsnorm: bool = False) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        if use_rmsnorm:
            self.layer_norm = RMSNorm(d_model)
        else:
            self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float,
        use_rmsnorm: bool = True,
        use_qknorm: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_qknorm = use_qknorm
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads

        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        if use_qknorm:
            self.qk_norm = QKNorm(self.d_k, use_rmsnorm=use_rmsnorm)
        else:
            self.qk_norm = None
        
        self.attention_weights = None

    @staticmethod
    def _scaled_dot_product_attention(q, k, v, mask, dropout: nn.Dropout):
        # q, k, v shape: (Batch, Heads, Seq, D_k)
        d_k = k.shape[-1]
        
        # Attention scores: (Batch, Heads, Seq, Seq)
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        if dropout is not None:
            attention_weights = dropout(attention_weights)
            
        return (attention_weights @ v), attention_weights

    def forward(self, q, k, v, mask=None, rope=None):
        batch_size = q.shape[0]
        
        # 1. Linear Projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # 2. Reshape to (Batch, Heads, Seq, D_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Apply QKNorm (Applying BEFORE RoPE is standard practice if used)
        if self.qk_norm is not None:
            query, key = self.qk_norm(query, key)

        # 4. Apply RoPE (Rotary Positional Embedding)
        if rope is not None:
            cos, sin = rope
            # apply_rotary_pos_emb handles broadcasting automatically
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # 5. Scaled Dot Product Attention
        x, self.attention_weights = self._scaled_dot_product_attention(
            query, key, value, mask, self.dropout
        )
        
        # 6. Concatenate Heads and Output Projection
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        activation: Literal['relu', 'gelu'], 
        dropout: float,
        use_rmsnorm: bool = False,
        use_qknorm: bool = False,
        use_swiglu: bool = False
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(
            d_model, num_heads, dropout, use_rmsnorm, use_qknorm
            )
        if use_swiglu:
            self.feed_forward = SwiGLU(d_model, d_ff, dropout=dropout)
        else:
            self.feed_forward = FeedForwardBlock(d_model, d_ff, activation, dropout)

        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout, use_rmsnorm), 
            ResidualConnection(d_model, dropout, use_rmsnorm)  
        ])

    def forward(self, x, src_mask, rope=None):
        # Pass rope to self_attention
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask, rope=rope))
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
        num_blocks: int,
        use_rmsnorm: bool = False,
        use_qknorm: bool = False,
        use_swiglu: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model, num_heads, d_ff, activation, dropout,
                use_rmsnorm, use_qknorm, use_swiglu
                ) 
            for _ in range(num_blocks)
        ])
        if use_rmsnorm:
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask, rope=None):
        for layer in self.layers:
            x = layer(x, src_mask, rope=rope)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        activation: Literal['relu', 'gelu'], 
        dropout: float,
        use_rmsnorm: bool = False,
        use_qknorm: bool = False,
        use_swiglu: bool = False
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(
            d_model, num_heads, dropout, use_rmsnorm, use_qknorm
            )
        self.cross_attention = MultiHeadAttentionBlock(
            d_model, num_heads, dropout, use_rmsnorm, use_qknorm
            )
        if use_swiglu:
            self.feed_forward = SwiGLU(d_model, d_ff, dropout=dropout)
        else:
            self.feed_forward = FeedForwardBlock(d_model, d_ff, activation, dropout)

        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout), 
            ResidualConnection(d_model, dropout), 
            ResidualConnection(d_model, dropout)  
        ])

    def forward(self, x, context, src_mask, tgt_mask, rope=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask, rope=rope))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, context, context, src_mask, rope=None))
        
        # Feed-forward
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
        num_blocks: int,
        use_rmsnorm: bool = False,
        use_qknorm: bool = False,
        use_swiglu: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model, num_heads, d_ff, activation, dropout,
                use_rmsnorm, use_qknorm, use_swiglu
            ) 
            for _ in range(num_blocks)
        ])
        if use_rmsnorm:
            self.norm = RMSNorm(d_model)
        else:
            self.norm = LayerNormalization(d_model)

    def forward(self, x, context, src_mask, tgt_mask, rope=None):
        for layer in self.layers:
            x = layer(x, context, src_mask, tgt_mask, rope=rope)
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
        tie_weights: bool = True,

        use_rmsnorm: bool = True,
        use_qknorm: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        rope_base: float = 10000.0
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_rope = use_rope

        features = []
        if use_rmsnorm: features.append("RMSNorm")
        if use_qknorm: features.append("QKNorm")
        if use_rope: features.append("RoPE")
        if use_swiglu: features.append("SwiGLU")
        print(f"Building Transformer with features: {', '.join(features) if features else 'Base (no improvements)'}")


        # Embeddings and Positional Encoding
        self.src_embedding = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)
        
        if use_rope:
            self.rope = RotaryEmbedding(
                dim=d_model // num_heads,
                max_seq_len=max_seq_len,
                base=rope_base
            )
            self.src_pos_encoding = nn.Dropout(dropout) 
            self.tgt_pos_encoding = nn.Dropout(dropout)
        else:
            self.rope = None
            self.src_pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
            self.tgt_pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder = Encoder(
            d_model, num_heads, d_ff, activation, dropout, num_encoder_blocks,
            use_rmsnorm, use_qknorm, use_swiglu
        )
        self.decoder = Decoder(
            d_model, num_heads, d_ff, activation, dropout, num_decoder_blocks,
            use_rmsnorm, use_qknorm, use_swiglu
        )
        
        self.projection = ProjectionLayer(d_model, tgt_vocab_size)
        
        if tie_weights:
            self.projection.projection.weight = self.tgt_embedding.embedding.weight
        
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
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        rope = None
        if self.rope is not None:
            # Calculate RoPE cos/sin for this sequence length
            cos, sin = self.rope(src)
            rope = (cos, sin)
            src = self.src_pos_encoding(src)
        else:
            src = self.src_pos_encoding(src)
        
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self.encoder, src, src_mask, rope, use_reentrant=True
            )
        
        return self.encoder(src, src_mask, rope)

    def decode(self, tgt, context, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        rope = None
        if self.rope is not None:
            # Calculate RoPE cos/sin for this sequence length
            cos, sin = self.rope(tgt)
            rope = (cos, sin)
            tgt = self.tgt_pos_encoding(tgt)
        else:
            tgt = self.tgt_pos_encoding(tgt)
        
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(
                self.decoder, tgt, context, src_mask, tgt_mask, rope, use_reentrant=True
            )
            
        return self.decoder(tgt, context, src_mask, tgt_mask, rope)

    def project(self, x):
        return self.projection(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.project(decoder_output)