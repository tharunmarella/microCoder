"""
Data utilities for loading and preprocessing datasets.
"""

import torch

# Optional: Tiktoken for BPE tokenization
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

# Optional: HuggingFace datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


def get_tokenizer():
    """Get tiktoken tokenizer (GPT-2 BPE) or None for byte-level fallback."""
    if TIKTOKEN_AVAILABLE:
        return tiktoken.get_encoding("gpt2")
    return None


def load_hf_dataset(dataset_name="flytech/python-codes-25k", config=None, split="train", max_bytes=None, vocab_size=50257):
    """
    Load and tokenize a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        config: Dataset configuration (optional)
        split: Dataset split (default: 'train')
        max_bytes: Maximum bytes to load (optional)
        vocab_size: Vocabulary size
    
    Returns:
        torch.Tensor: Tokenized data as long tensor
    """
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
            # Fallback to byte-level encoding
            b = s.encode('utf-8')
            all_tokens.extend([min(x, vocab_size - 1) for x in list(b)])
            total_bytes += len(b)
            
        if max_bytes is not None and total_bytes >= max_bytes:
            break
    
    print(f"Loaded {len(all_tokens):,} tokens from HF dataset {dataset_name}/{config}")
    if tokenizer:
        print(f"Using tiktoken (GPT-2 BPE) with vocab_size={vocab_size}")
    else:
        print(f"Using byte-level encoding with vocab_size={vocab_size}")
    
    return torch.tensor(all_tokens, dtype=torch.long)


def load_custom_text_file(filepath, max_bytes=None, vocab_size=50257):
    """
    Load and tokenize a local text file.
    
    Args:
        filepath: Path to text file
        max_bytes: Maximum bytes to load (optional)
        vocab_size: Vocabulary size
    
    Returns:
        torch.Tensor: Tokenized data
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        data = f.read()
    
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    
    tokenizer = get_tokenizer()
    if tokenizer:
        tokens = tokenizer.encode_ordinary(data)
        print(f"Loaded {len(tokens):,} tokens from {filepath}")
        print(f"Using tiktoken (GPT-2 BPE) with vocab_size={vocab_size}")
    else:
        # Fallback to byte-level encoding
        tokens = [min(x, vocab_size - 1) for x in list(data.encode('utf-8'))]
        print(f"Loaded {len(tokens):,} tokens from {filepath}")
        print(f"Using byte-level encoding with vocab_size={vocab_size}")
    
    return torch.tensor(tokens, dtype=torch.long)


class DataLoader:
    """
    Simple dataloader for training.
    Randomly samples sequences from the dataset.
    """
    
    def __init__(self, data_tensor, block_size, batch_size, device='cpu'):
        self.data = data_tensor
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data_len = data_tensor.size(0)
    
    def get_batch(self):
        """
        Get a random batch of sequences.
        
        Returns:
            x: Input sequences of shape (batch_size, block_size)
            y: Target sequences of shape (batch_size, block_size)
        """
        idx = torch.randint(0, self.data_len - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in idx])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in idx])
        return x.to(self.device), y.to(self.device)
