"""
Data Preparation Script for Code Training

This script downloads and preprocesses datasets from HuggingFace,
tokenizes them, and saves them to disk for fast reuse during training.

Usage:
    python prepare_data.py --dataset flytech/python-codes-25k --output data/python_codes.pt
    python prepare_data.py --dataset bigcode/the-stack --config data/python --output data/the_stack.pt
"""

import torch
import argparse
import os
import json
from pathlib import Path

# Optional: Tiktoken for BPE
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
    print("ERROR: HuggingFace datasets library not installed.")
    print("Install with: pip install datasets")
    exit(1)


def get_tokenizer():
    """Get tiktoken tokenizer (GPT-2 BPE) or None for byte-level fallback."""
    if TIKTOKEN_AVAILABLE:
        return tiktoken.get_encoding("gpt2")
    return None


def prepare_hf_dataset(dataset_name, config=None, split="train", max_samples=None, vocab_size=50257):
    """
    Download and tokenize a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'flytech/python-codes-25k')
        config: Dataset configuration (e.g., 'data/python' for The Stack)
        split: Dataset split (default: 'train')
        max_samples: Maximum number of samples to process (None = all)
        vocab_size: Vocabulary size (50257 for tiktoken, 256 for byte-level)
    
    Returns:
        torch.Tensor: Tokenized data as long tensor
        dict: Metadata about the dataset
    """
    print(f"\n{'='*60}")
    print(f"📦 Loading dataset: {dataset_name}")
    if config:
        print(f"   Config: {config}")
    print(f"   Split: {split}")
    print(f"{'='*60}\n")
    
    # Load dataset from HuggingFace
    print("⏳ Downloading from HuggingFace... (this may take a few minutes)")
    if config:
        ds = load_dataset(dataset_name, config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    
    print(f"✅ Dataset loaded: {len(ds)} samples")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    if tokenizer:
        print("🔤 Using tiktoken (GPT-2 BPE) tokenizer")
    else:
        print("🔤 Using byte-level encoding (fallback)")
    
    # Tokenize data
    print("\n⏳ Tokenizing dataset...")
    all_tokens = []
    total_chars = 0
    samples_processed = 0
    
    for i, item in enumerate(ds):
        # Try different field names (different datasets use different keys)
        text = (item.get('text') or 
                item.get('code') or 
                item.get('content') or 
                item.get('instruction', '') or
                item.get('func_code_string') or  # CodeSearchNet
                item.get('whole_func_string', ''))  # CodeSearchNet alternative
        
        if not isinstance(text, str) or len(text) == 0:
            continue
        
        # Tokenize
        if tokenizer:
            tokens = tokenizer.encode_ordinary(text)
        else:
            # Byte-level encoding fallback
            byte_data = text.encode('utf-8')
            tokens = [min(x, vocab_size - 1) for x in list(byte_data)]
        
        all_tokens.extend(tokens)
        total_chars += len(text)
        samples_processed += 1
        
        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i+1:,} samples, {len(all_tokens):,} tokens so far...")
        
        # Stop if we hit max_samples
        if max_samples and samples_processed >= max_samples:
            print(f"\n⚠️  Reached max_samples limit ({max_samples})")
            break
    
    print(f"\n✅ Tokenization complete!")
    print(f"   Samples processed: {samples_processed:,}")
    print(f"   Total tokens: {len(all_tokens):,}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Avg tokens per sample: {len(all_tokens) / samples_processed:.1f}")
    
    # Convert to tensor
    data_tensor = torch.tensor(all_tokens, dtype=torch.long)
    
    # Metadata
    metadata = {
        'dataset_name': dataset_name,
        'config': config,
        'split': split,
        'samples_processed': samples_processed,
        'total_tokens': len(all_tokens),
        'total_chars': total_chars,
        'vocab_size': vocab_size,
        'tokenizer': 'tiktoken_gpt2' if tokenizer else 'byte_level',
    }
    
    return data_tensor, metadata


def prepare_text_file(filepath, vocab_size=50257):
    """
    Load and tokenize a local text file.
    
    Args:
        filepath: Path to text file
        vocab_size: Vocabulary size
    
    Returns:
        torch.Tensor: Tokenized data
        dict: Metadata
    """
    print(f"\n{'='*60}")
    print(f"📄 Loading text file: {filepath}")
    print(f"{'='*60}\n")
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    print(f"✅ File loaded: {len(text):,} characters")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    if tokenizer:
        print("🔤 Using tiktoken (GPT-2 BPE) tokenizer")
        tokens = tokenizer.encode_ordinary(text)
    else:
        print("🔤 Using byte-level encoding (fallback)")
        byte_data = text.encode('utf-8')
        tokens = [min(x, vocab_size - 1) for x in list(byte_data)]
    
    print(f"✅ Tokenization complete: {len(tokens):,} tokens")
    
    data_tensor = torch.tensor(tokens, dtype=torch.long)
    
    metadata = {
        'source_file': filepath,
        'total_tokens': len(tokens),
        'total_chars': len(text),
        'vocab_size': vocab_size,
        'tokenizer': 'tiktoken_gpt2' if tokenizer else 'byte_level',
    }
    
    return data_tensor, metadata


def save_dataset(data_tensor, metadata, output_path):
    """
    Save tokenized dataset to disk.
    
    Args:
        data_tensor: Tokenized data tensor
        metadata: Dataset metadata
        output_path: Path to save the dataset
    """
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving dataset to: {output_path}")
    
    # Save as a dictionary with data and metadata
    save_dict = {
        'data': data_tensor,
        'metadata': metadata
    }
    
    torch.save(save_dict, output_path)
    
    # Also save metadata as readable JSON
    metadata_path = output_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Calculate file sizes
    data_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"✅ Dataset saved successfully!")
    print(f"   Data file: {output_path} ({data_size_mb:.2f} MB)")
    print(f"   Metadata: {metadata_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total tokens: {metadata['total_tokens']:,}")
    print(f"   Vocab size: {metadata['vocab_size']:,}")
    print(f"   Tokenizer: {metadata['tokenizer']}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare and save tokenized datasets for training.'
    )
    
    # Data source
    parser.add_argument(
        '--source',
        type=str,
        default='hf',
        choices=['hf', 'file'],
        help='Data source: hf (HuggingFace) or file (local text file)'
    )
    
    # HuggingFace dataset options
    parser.add_argument(
        '--dataset',
        type=str,
        default='flytech/python-codes-25k',
        help='HuggingFace dataset name (e.g., flytech/python-codes-25k)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Dataset configuration (e.g., data/python for The Stack)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split (default: train)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )
    
    # Local file options
    parser.add_argument(
        '--file',
        type=str,
        default='',
        help='Path to local text file (if source=file)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='data/train_data.pt',
        help='Output path for processed dataset (default: data/train_data.pt)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=50257,
        help='Vocabulary size (default: 50257 for tiktoken)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source == 'file' and not args.file:
        print("ERROR: --file must be provided when --source=file")
        exit(1)
    
    # Prepare dataset
    if args.source == 'hf':
        data_tensor, metadata = prepare_hf_dataset(
            dataset_name=args.dataset,
            config=args.config,
            split=args.split,
            max_samples=args.max_samples,
            vocab_size=args.vocab_size
        )
    else:
        data_tensor, metadata = prepare_text_file(
            filepath=args.file,
            vocab_size=args.vocab_size
        )
    
    # Save dataset
    save_dataset(data_tensor, metadata, args.output)
    
    print(f"\n{'='*60}")
    print(f"✅ DONE! Dataset ready for training.")
    print(f"{'='*60}")
    print(f"\n💡 To use this dataset in training, run:")
    print(f"   python tiny_llm.py --data-source preprocessed --data-file {args.output}")
    print()


if __name__ == "__main__":
    main()
