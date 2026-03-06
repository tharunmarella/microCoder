"""
Utility functions for training and generation.
"""

import torch
import math

# Optional: Tiktoken for text generation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False


def get_tokenizer():
    """Get tiktoken tokenizer or None."""
    if TIKTOKEN_AVAILABLE:
        return tiktoken.get_encoding("gpt2")
    return None


def count_params(model):
    """Count total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def get_lr_scheduler_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    Creates a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of steps for linear warmup
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of initial lr
    
    Returns:
        torch.optim.lr_scheduler.LambdaLR
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
    """
    Generate text from the model.
    
    Args:
        model: The trained model
        start_text: Prompt to start generation
        max_new: Number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (only sample from top k most likely tokens)
        device: Device to run on
    
    Returns:
        str: Generated text
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    tokenizer = get_tokenizer()
    
    # Encode the starting text
    if tokenizer:
        input_ids = torch.tensor(
            [tokenizer.encode_ordinary(start_text)],
            dtype=torch.long,
            device=device
        )
    else:
        start_bytes = start_text.encode('utf-8')
        input_ids = torch.tensor([list(start_bytes)], dtype=torch.long, device=device)
    
    produced_ids = input_ids[0].tolist()
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_new):
            # Crop input to maximum context length
            input_ids_cropped = (
                input_ids if input_ids.size(1) <= model.block_size
                else input_ids[:, -model.block_size:]
            )
            
            # Forward pass
            logits = model(input_ids_cropped)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_id], dim=1)
            produced_ids.append(int(next_id.item()))
    
    # Decode the generated tokens
    if tokenizer:
        return tokenizer.decode(produced_ids)
    else:
        out_bytes = bytes([t % 256 for t in produced_ids])
        try:
            return out_bytes.decode('utf-8', errors='replace')
        except Exception:
            return out_bytes.decode('latin1', errors='replace')


def check_memory(args, device):
    """
    Check if the model will fit in GPU memory (CUDA only).
    
    Args:
        args: Training arguments
        device: torch.device
    """
    from .model import GPTLikeModel
    
    if device.type == 'cuda':
        # Create temporary model to count parameters
        temp_model = GPTLikeModel(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads
        )
        total_params = count_params(temp_model)
        del temp_model
        
        # Estimate memory (params + gradients + optimizer states)
        param_size_gb = (total_params * 4) / (1024**3)
        total_mem_gb = param_size_gb * 3  # model + grad + optimizer
        
        free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Estimated GPU Memory needed: ~{total_mem_gb:.2f} GB")
        print(f"Available GPU Memory: {free_mem:.2f} GB")
        
        if total_mem_gb > free_mem * 0.9:
            print("WARNING: Model might not fit in GPU memory. Consider reducing d_model or n_layers.")
