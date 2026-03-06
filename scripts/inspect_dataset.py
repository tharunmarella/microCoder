"""
Inspect preprocessed dataset - View samples from .pt files

Usage:
    python scripts/inspect_dataset.py data/python_codes_25k.pt --num-samples 5
"""

import torch
import argparse
import json

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except:
    TIKTOKEN_AVAILABLE = False


def get_tokenizer():
    """Get tiktoken tokenizer."""
    if TIKTOKEN_AVAILABLE:
        return tiktoken.get_encoding("gpt2")
    return None


def inspect_dataset(data_path, num_samples=5, sample_length=512):
    """
    Inspect a preprocessed dataset and show sample texts.
    
    Args:
        data_path: Path to .pt file
        num_samples: Number of random samples to show
        sample_length: Length of each sample in tokens
    """
    print(f"\n{'='*60}")
    print(f"📦 INSPECTING DATASET: {data_path}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("⏳ Loading dataset...")
    dataset_dict = torch.load(data_path)
    
    # Check if it's our format or just raw tensor
    if isinstance(dataset_dict, dict):
        data = dataset_dict['data']
        metadata = dataset_dict.get('metadata', {})
    else:
        data = dataset_dict
        metadata = {}
    
    # Show metadata
    if metadata:
        print("📊 Dataset Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        print()
    
    # Show basic stats
    print(f"📈 Dataset Statistics:")
    print(f"   Total tokens: {len(data):,}")
    print(f"   Data type: {data.dtype}")
    print(f"   Shape: {data.shape}")
    print(f"   Min token ID: {data.min().item()}")
    print(f"   Max token ID: {data.max().item()}")
    print(f"   Unique tokens: {len(torch.unique(data)):,}")
    print()
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    if not tokenizer:
        print("⚠️  Tiktoken not available, showing token IDs only")
        print()
    
    # Show random samples
    print(f"{'='*60}")
    print(f"🔍 RANDOM SAMPLES (showing {num_samples} samples)")
    print(f"{'='*60}\n")
    
    max_start = len(data) - sample_length
    for i in range(num_samples):
        # Get random sample
        start_idx = torch.randint(0, max_start, (1,)).item()
        sample_tokens = data[start_idx:start_idx+sample_length].tolist()
        
        print(f"Sample {i+1}:")
        print(f"{'─'*60}")
        print(f"Token range: {start_idx:,} to {start_idx+sample_length:,}")
        print()
        
        if tokenizer:
            # Decode tokens to text
            try:
                text = tokenizer.decode(sample_tokens)
                print(text[:500])  # Show first 500 chars
                if len(text) > 500:
                    print("\n... (truncated)")
            except Exception as e:
                print(f"Error decoding: {e}")
                print(f"Token IDs: {sample_tokens[:50]}...")
        else:
            # Just show token IDs
            print(f"Token IDs: {sample_tokens[:100]}...")
        
        print(f"\n{'─'*60}\n")
    
    # Token frequency analysis
    print(f"{'='*60}")
    print(f"📊 TOKEN FREQUENCY ANALYSIS")
    print(f"{'='*60}\n")
    
    unique_tokens, counts = torch.unique(data, return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    
    print("Top 20 most common tokens:")
    for idx in sorted_indices[:20]:
        token_id = unique_tokens[idx].item()
        count = counts[idx].item()
        percentage = (count / len(data)) * 100
        
        if tokenizer:
            try:
                token_text = tokenizer.decode([token_id])
                # Escape special characters for display
                token_text = repr(token_text)[1:-1]  # Remove outer quotes
            except:
                token_text = f"<token_{token_id}>"
        else:
            token_text = f"<token_{token_id}>"
        
        print(f"   Token {token_id:6d} ({token_text[:20]:20s}): {count:8,} ({percentage:.2f}%)")
    
    print(f"\n{'='*60}")
    print("✅ Inspection complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect preprocessed datasets')
    
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to .pt dataset file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of random samples to show (default: 5)'
    )
    parser.add_argument(
        '--sample-length',
        type=int,
        default=512,
        help='Length of each sample in tokens (default: 512)'
    )
    parser.add_argument(
        '--save-json',
        type=str,
        default=None,
        help='Save samples to JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    inspect_dataset(
        args.data_path,
        num_samples=args.num_samples,
        sample_length=args.sample_length
    )
    
    # Optional: save samples to JSON
    if args.save_json:
        print(f"💾 Saving samples to {args.save_json}...")
        dataset_dict = torch.load(args.data_path)
        if isinstance(dataset_dict, dict):
            data = dataset_dict['data']
        else:
            data = dataset_dict
        
        tokenizer = get_tokenizer()
        samples = []
        
        for i in range(args.num_samples):
            start_idx = torch.randint(0, len(data) - args.sample_length, (1,)).item()
            tokens = data[start_idx:start_idx+args.sample_length].tolist()
            
            sample = {
                'token_ids': tokens[:100],  # First 100 tokens
                'start_index': start_idx
            }
            
            if tokenizer:
                try:
                    sample['text'] = tokenizer.decode(tokens[:100])
                except:
                    pass
            
            samples.append(sample)
        
        with open(args.save_json, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"✅ Saved {len(samples)} samples to {args.save_json}")


if __name__ == "__main__":
    main()
