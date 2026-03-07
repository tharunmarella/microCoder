import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import math
import random
import json
import os
import shutil

# Optional: Tiktoken for BPE
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

# Optional: HuggingFace datasets for data loading
try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    load_dataset = None

# Optional: TensorBoard for visualization
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# 1) GLOBAL CONFIG (aiming for ~100M params)
# Default vocabulary size: 50257 for tiktoken (GPT-2 BPE), falls back to 256 for byte-level
vocab_size = 50257
block_size = 1024       # context length
n_layers = 10              # number of Transformer blocks
d_model = 2048             # hidden dimension (parameter count scales with d_model^2)
n_heads = 16               # must divide d_model
dropout = 0.1

# 2) MODEL COMPONENTS
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    # x: (B, T, n_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, -1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        # x: (B, T, C)
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # reshape for heads: (B, T, n_heads, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # reshape for attention: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention (scaled_dot_product_attention)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0, 
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

        # idx: (B, T) token indices
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x

class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size=50257, block_size=1024, n_layers=10, d_model=2048, n_heads=16, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying
        self.tok_emb.weight = self.head.weight
        
        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model // n_heads, block_size))
        
        # Initialize weights with GPT-style initialization
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections (GPT-2 style)
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
        B, T = idx.size()
        x = self.tok_emb(idx)
        x = self.drop(x)
        
        freqs_cis = self.freqs_cis[:T]
        for block in self.blocks:
            x = block(x, freqs_cis)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# 3) DATA LOADING (HuggingFace Datasets)
def get_tokenizer():
    if TIKTOKEN_AVAILABLE:
        return tiktoken.get_encoding("gpt2")
    return None

def load_hf_dataset(dataset_name="flytech/python-codes-25k", config=None, split="train", max_bytes=None, vocab_size=50257):
    if not HF_AVAILABLE:
        raise RuntimeError("datasets library is not available. Install 'datasets' to load HF datasets.")
    ds = load_dataset(dataset_name, config, split=split)
    tokenizer = get_tokenizer()
    
    all_tokens = []
    total_bytes = 0
    for item in ds:
        s = item.get('text') or item.get('code') or item.get('content') or item.get('instruction', '')
        if not isinstance(s, str) or len(s) == 0:
            continue
        
        if tokenizer:
            tokens = tokenizer.encode_ordinary(s)
            all_tokens.extend(tokens)
            total_bytes += len(s)
        else:
            # Fallback to byte-level encoding (vocab_size should be 256)
            b = s.encode('utf-8')
            all_tokens.extend([min(x, vocab_size - 1) for x in list(b)])
            total_bytes += len(b)
            
        if max_bytes is not None and total_bytes >= max_bytes:
            break
    
    print(f"Loaded {len(all_tokens)} tokens from HF dataset {dataset_name}/{config}")
    if tokenizer:
        print(f"Using tiktoken (GPT-2 BPE) with vocab_size={vocab_size}")
    else:
        print(f"Using byte-level encoding with vocab_size={vocab_size}")
    return torch.tensor(all_tokens, dtype=torch.long)

def load_custom_text_file(filepath, max_bytes=None, vocab_size=50257):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        data = f.read()
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    
    tokenizer = get_tokenizer()
    if tokenizer:
        tokens = tokenizer.encode_ordinary(data)
        print(f"Loaded {len(tokens)} tokens from {filepath}")
        print(f"Using tiktoken (GPT-2 BPE) with vocab_size={vocab_size}")
    else:
        # Fallback to byte-level encoding
        tokens = [min(x, vocab_size - 1) for x in list(data.encode('utf-8'))]
        print(f"Loaded {len(tokens)} tokens from {filepath}")
        print(f"Using byte-level encoding with vocab_size={vocab_size}")
        
    return torch.tensor(tokens, dtype=torch.long)

# 4) TRAINING UTILITIES
class DataLoader:
    def __init__(self, data_tensor, block_size, batch_size, device='cpu'):
        self.data = data_tensor
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data_len = data_tensor.size(0)
    
    def get_batch(self):
        idx = torch.randint(0, self.data_len - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in idx])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in idx])
        return x.to(self.device), y.to(self.device)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_lr_scheduler_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    Creates a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of steps for linear warmup
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of initial lr (default: 0.1)
    """
    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return step / warmup_steps
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def generate_text(model, start_text, max_new=256, temperature=1.0, top_k=None, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    tokenizer = get_tokenizer()
    
    if tokenizer:
        input_ids = torch.tensor([tokenizer.encode_ordinary(start_text)], dtype=torch.long, device=device)
    else:
        start_bytes = start_text.encode('utf-8')
        input_ids = torch.tensor([list(start_bytes)], dtype=torch.long, device=device)
    
    produced_ids = input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_new):
            # crop input_ids to the maximum context length
            input_ids_cropped = input_ids if input_ids.size(1) <= model.block_size else input_ids[:, -model.block_size:]
            logits = model(input_ids_cropped)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            produced_ids.append(int(next_id.item()))
            
    if tokenizer:
        return tokenizer.decode(produced_ids)
    else:
        out_bytes = bytes([t % 256 for t in produced_ids])
        try:
            return out_bytes.decode('utf-8', errors='replace')
        except Exception:
            return out_bytes.decode('latin1', errors='replace')

def check_memory(args, device):
    if device.type == 'cuda':
        # Estimate model size in GB (assuming float32, 4 bytes per param)
        # 1B params * 4 bytes = 4GB
        # Plus optimizer states (AdamW: 8 bytes per param) = 8GB
        # Plus activations and gradients...
        model_params = (args.n_layers * (12 * args.d_model**2)) # rough estimate for transformer
        # More accurate count
        temp_model = GPTLikeModel(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads
        )
        total_params = count_params(temp_model)
        del temp_model
        
        param_size_gb = (total_params * 4) / (1024**3)
        total_mem_gb = param_size_gb * 3 # model + grad + opt
        
        free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Estimated GPU Memory needed: ~{total_mem_gb:.2f} GB")
        print(f"Available GPU Memory: {free_mem:.2f} GB")
        
        if total_mem_gb > free_mem * 0.9:
            print("WARNING: Model might not fit in GPU memory. Consider reducing d_model or n_layers.")

def train(args):
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize TensorBoard if available and enabled
    writer = None
    if args.tensorboard and TENSORBOARD_AVAILABLE:
        log_dir = f"logs/tensorboard/run_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard logging to: {log_dir}")
        print(f"   View with: tensorboard --logdir logs/tensorboard")
    elif args.tensorboard and not TENSORBOARD_AVAILABLE:
        print("⚠️  TensorBoard requested but not installed. Install with: pip install tensorboard")
    
    check_memory(args, device)
    
    # Build model
    model = GPTLikeModel(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    model.to(device)
    
    total_params = count_params(model)
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    if args.print_model:
        print(model)
        print("\n" + "="*50)
    
    # Load data
    if args.data_source == 'preprocessed':
        # Load preprocessed dataset from disk
        print(f"Loading preprocessed dataset from: {args.data_file}")
        dataset_dict = torch.load(args.data_file)
        data = dataset_dict['data']
        metadata = dataset_dict.get('metadata', {})
        print(f"Loaded {len(data):,} tokens")
        if metadata:
            print(f"Dataset info: {metadata.get('dataset_name', 'Unknown')}")
            print(f"Tokenizer: {metadata.get('tokenizer', 'Unknown')}")
    elif args.data_source == 'hf':
        if not HF_AVAILABLE:
            print("Error: HuggingFace datasets library not installed. Install with: pip install datasets")
            sys.exit(1)
        data = load_hf_dataset(
            dataset_name=args.hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            max_bytes=args.max_data_bytes,
            vocab_size=args.vocab_size
        )
    elif args.data_source == 'file':
        data = load_custom_text_file(
            args.data_file, 
            max_bytes=args.max_data_bytes,
            vocab_size=args.vocab_size
        )
    else:
        # synthetic data for testing
        sample_text = "def hello_world():\n    print('Hello, World!')\n    return True\n" * 1000
        tokenizer = get_tokenizer()
        if tokenizer:
            tokens = tokenizer.encode_ordinary(sample_text)
        else:
            tokens = [min(x, args.vocab_size - 1) for x in list(sample_text.encode('utf-8'))]
        data = torch.tensor(tokens, dtype=torch.long)
        print(f"Using synthetic data with {len(tokens)} tokens")
    
    data = data.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    if args.scheduler:
        warmup_steps = int(args.iterations * args.warmup_ratio)
        scheduler = get_lr_scheduler_with_warmup(
            optimizer, 
            warmup_steps=warmup_steps, 
            total_steps=args.iterations,
            min_lr_ratio=args.min_lr_ratio
        )
        print(f"Using LR scheduler with {warmup_steps} warmup steps and cosine decay to {args.min_lr_ratio * args.lr:.2e}")
    else:
        scheduler = None
    
    # Mixed precision
    if args.amp:
        if device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        elif device.type == 'mps':
            # MPS doesn't support GradScaler yet, but we can use autocast in some versions
            # or just skip scaler for now. For M4, we'll use torch.autocast.
            scaler = None
            print("Note: AMP on MPS uses autocast without GradScaler.")
        else:
            scaler = None
    else:
        scaler = None
    
    # Data loader
    loader = DataLoader(data, args.block_size, args.batch_size, device)
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    # Training loop
    model.train()
    start_time = time.time()
    for it in range(args.iterations):
        x, y = loader.get_batch()
        
        # Determine autocast device type
        autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
        # Note: torch.autocast(device_type='mps') is supported in newer torch versions
        if device.type == 'mps':
            autocast_device = 'mps'

        if args.amp:
            with torch.autocast(device_type=autocast_device):
                logits = model(x)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (it + 1) % args.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        else:
            logits = model(x)
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
            loss.backward()
            if (it + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        
        # Logging
        if (it + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_seen = (it + 1) * args.batch_size * args.block_size
            current_lr = optimizer.param_groups[0]['lr']
            perplexity = math.exp(min(loss.item(), 20))  # Cap to avoid overflow
            
            print(f"Iter {it+1:5d}/{args.iterations} | Loss: {loss.item():.4f} | PPL: {perplexity:.2f} | LR: {current_lr:.2e} | {tokens_seen / elapsed:,.0f} tok/s")
            
            # Track loss for early stopping
            loss_history.append(loss.item())
            
            # TensorBoard logging
            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), it + 1)
                writer.add_scalar('train/perplexity', perplexity, it + 1)
                writer.add_scalar('train/learning_rate', current_lr, it + 1)
                writer.add_scalar('train/tokens_per_sec', tokens_seen / elapsed, it + 1)
                
                # Log gradient norms
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('train/grad_norm', total_norm, it + 1)
        
        # Early stopping check (every checkpoint interval)
        if args.early_stopping and (it + 1) % args.checkpoint_interval == 0 and len(loss_history) >= args.patience:
            # Calculate average loss over last 'patience' intervals
            recent_avg_loss = sum(loss_history[-args.patience:]) / args.patience
            
            # Check if loss improved
            if recent_avg_loss < best_loss - args.min_delta:
                best_loss = recent_avg_loss
                patience_counter = 0
                
                # Save best checkpoint (model only, no optimizer to save disk space)
                if args.save_path:
                    checkpoint_dir = os.path.dirname(args.save_path) or 'models/checkpoints'
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # 1. Save to fast local NVMe storage first
                    tmp_checkpoint_path = f'/tmp/best_model_tmp.pt'
                    # 2. Final destination on network volume
                    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                    
                    print(f"💾 Saving checkpoint to local storage first...")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'iteration': it + 1,
                        'loss': recent_avg_loss,
                        'config': {
                            'vocab_size': args.vocab_size,
                            'block_size': args.block_size,
                            'n_layers': args.n_layers,
                            'd_model': args.d_model,
                            'n_heads': args.n_heads,
                            'dropout': args.dropout,
                        },
                        'total_params': total_params,
                    }, tmp_checkpoint_path)
                    
                    # 3. Move from local NVMe to network volume safely
                    print(f"🚚 Moving checkpoint to network volume...")
                    shutil.move(tmp_checkpoint_path, best_checkpoint_path)
                    
                    print(f"\n✅ Best checkpoint saved: {best_checkpoint_path} (Loss: {recent_avg_loss:.4f})\n")
            else:
                patience_counter += 1
                print(f"📊 No improvement | Patience: {patience_counter}/{args.early_stop_patience} | Best Loss: {best_loss:.4f}")
                
                # Check if we should stop
                if patience_counter >= args.early_stop_patience:
                    print("\n" + "="*70)
                    print("🛑 EARLY STOPPING TRIGGERED")
                    print("="*70)
                    print(f"Training stopped at iteration {it+1}/{args.iterations}")
                    print(f"Best loss: {best_loss:.4f}")
                    print(f"No improvement for {args.early_stop_patience * args.checkpoint_interval} iterations")
                    print("="*70 + "\n")
                    break
        
        # Generate sample
        if (it + 1) % args.sample_interval == 0 and args.sample_interval > 0:
            model.eval()
            sample = generate_text(model, "def ", max_new=64, temperature=0.8, top_k=40, device=device)
            print(f"\n--- Sample after iter {it+1} ---")
            print(sample[:256] + ("..." if len(sample) > 256 else ""))
            print("---\n")
            
            # Log sample to TensorBoard
            if writer is not None:
                writer.add_text('samples/generated', sample[:512], it + 1)
            
            model.train()
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print("📊 TensorBoard logs saved")
    
    # Final generation
    print("\n" + "="*50)
    print("Training finished.")
    model.eval()
    sample = generate_text(model, args.prompt, max_new=args.generate_len, temperature=0.8, top_k=40, device=device)
    print("Generated sample:")
    print(sample)
    
    # Save model if requested
    if args.save_path:
        checkpoint_dir = os.path.dirname(args.save_path) or 'models/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 1. Save to fast local NVMe storage first
        tmp_final_path = f'/tmp/final_model_tmp.pt'
        
        print(f"💾 Saving final model to local storage first...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': args.vocab_size,
                'block_size': args.block_size,
                'n_layers': args.n_layers,
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'dropout': args.dropout,
            },
            'total_params': total_params,
        }, tmp_final_path)
        
        # 2. Move from local NVMe to network volume safely
        print(f"🚚 Moving final model to network volume...")
        shutil.move(tmp_final_path, args.save_path)
        
        print(f"✅ Model saved to {args.save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train a ~100M parameter GPT-like model on text data.')
    
    # Model architecture
    parser.add_argument('--vocab-size', type=int, default=50257, help='Vocabulary size (tiktoken gpt2)')
    parser.add_argument('--block-size', type=int, default=2048, help='Context length')
    parser.add_argument('--n-layers', type=int, default=24, help='Number of transformer layers')
    parser.add_argument('--d-model', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--n-heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Data
    parser.add_argument('--data-source', type=str, default='hf', choices=['hf', 'file', 'synthetic', 'preprocessed'], 
                        help='Data source: hf (HuggingFace), file, synthetic, or preprocessed')
    parser.add_argument('--hf-dataset', type=str, default='flytech/python-codes-25k', help='HF dataset name')
    parser.add_argument('--hf-config', type=str, default=None, help='HF dataset config')
    parser.add_argument('--hf-split', type=str, default='train', help='HF dataset split')
    parser.add_argument('--data-file', type=str, default='', help='Path to text file (if data-source=file) or preprocessed data (if data-source=preprocessed)')
    parser.add_argument('--max-data-bytes', type=int, default=None, help='Max bytes to load from dataset')
    
    # Training
    parser.add_argument('--iterations', type=int, default=5000, help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--scheduler', action='store_true', help='Use LR scheduler with warmup and cosine decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.01, help='Warmup steps as ratio of total iterations (default: 0.01 = 1%)')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1, help='Minimum LR as ratio of initial LR (default: 0.1)')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision (AMP) if available (CUDA/MPS)')
    
    # Logging and generation
    parser.add_argument('--log-interval', type=int, default=100, help='Log loss every N iterations')
    parser.add_argument('--sample-interval', type=int, default=500, help='Generate sample every N iterations')
    parser.add_argument('--prompt', type=str, default="def ", help='Prompt for final generation')
    parser.add_argument('--generate-len', type=int, default=256, help='Length of generated text')
    
    # Checkpointing and early stopping
    parser.add_argument('--checkpoint-interval', type=int, default=500, help='Save checkpoint every N iterations')
    parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on loss plateau')
    parser.add_argument('--patience', type=int, default=10, help='Number of checkpoint intervals to track for early stopping')
    parser.add_argument('--early-stop-patience', type=int, default=5, help='Stop after N consecutive non-improving checkpoints')
    parser.add_argument('--min-delta', type=float, default=0.01, help='Minimum loss improvement to reset patience counter')
    
    # System
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='Device to use')
    parser.add_argument('--print-model', action='store_true', help='Print model architecture')
    parser.add_argument('--save-path', type=str, default='', help='Path to save model checkpoint')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    
    args = parser.parse_args()
    
    # Validate
    if args.data_source == 'file' and not args.data_file:
        print("Error: --data-file must be provided when --data-source=file")
        sys.exit(1)
    if args.data_source == 'preprocessed' and not args.data_file:
        print("Error: --data-file must be provided when --data-source=preprocessed")
        sys.exit(1)
    
    # Run training
    train(args)

if __name__ == "__main__":
    main()
