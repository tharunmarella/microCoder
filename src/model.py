"""
Model Architecture: GPT-like Transformer with modern optimizations.

Features:
- RMSNorm (Llama-style normalization)
- Rotary Positional Embeddings (RoPE)
- Flash Attention
- Weight Tying
- GPT-style initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for rotary positional embeddings (RoPE).
    
    Args:
        dim: Dimension of the head
        end: Maximum sequence length
        theta: Rotation base frequency
    
    Returns:
        Complex tensor of shape (end, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """
    Apply rotary positional embeddings to query/key tensors.
    
    Args:
        x: Input tensor of shape (B, T, n_heads, head_dim)
        freqs_cis: Precomputed frequencies
    
    Returns:
        Tensor with RoPE applied
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, -1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with RoPE and Flash Attention.
    """
    
    def __init__(self, d_model, n_heads, block_size, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        
        # Combined QKV projection (more efficient)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention (PyTorch's optimized version)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0,
            is_causal=True  # Causal masking for autoregressive generation
        )

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (from Llama).
    More stable and efficient than LayerNorm.
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-normalization (GPT-3 style).
    """
    
    def __init__(self, d_model, n_heads, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = RMSNorm(d_model)
        
        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, freqs_cis):
        # Pre-norm + residual connections
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTLikeModel(nn.Module):
    """
    GPT-like decoder-only transformer model for code generation.
    
    Architecture features:
    - RMSNorm for stable training
    - RoPE for better positional encoding
    - Flash Attention for speed
    - Weight tying between embedding and output
    - GPT-style weight initialization
    """
    
    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_layers=10,
        d_model=768,
        n_heads=12,
        dropout=0.1
    ):
        super().__init__()
        self.block_size = block_size
        
        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output head
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying (share weights between embedding and output)
        self.tok_emb.weight = self.head.weight
        
        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(d_model // n_heads, block_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections (GPT-2/3 style)
        for pn, p in self.named_parameters():
            if pn.endswith('out.weight') or pn.endswith('mlp.2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
    
    def _init_weights(self, module):
        """Initialize weights using GPT-style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        Forward pass through the model.
        
        Args:
            idx: Token indices of shape (B, T)
        
        Returns:
            logits: Predictions of shape (B, T, vocab_size)
        """
        B, T = idx.size()
        
        # Token embeddings
        x = self.tok_emb(idx)
        x = self.drop(x)
        
        # Apply transformer blocks
        freqs_cis = self.freqs_cis[:T]
        for block in self.blocks:
            x = block(x, freqs_cis)
        
        # Final norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

    def count_parameters(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
