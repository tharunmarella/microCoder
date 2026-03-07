#!/usr/bin/env python3
"""
Test the trained model directly on RunPod
Generates code from various prompts
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, '/workspace/src')

# Import model architecture
from train import GPTLikeModel, precompute_freqs_cis

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except:
    TIKTOKEN_AVAILABLE = False

def generate_text(model, prompt, tokenizer=None, max_new=200, temperature=0.8, top_k=40, device='cuda'):
    """Generate text from prompt"""
    model.eval()
    
    # Encode prompt
    if tokenizer:
        tokens = tokenizer.encode_ordinary(prompt)
    else:
        tokens = [min(ord(c), 255) for c in prompt]
    
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new):
            # Crop context if too long
            x_cond = x if x.size(1) <= model.block_size else x[:, -model.block_size:]
            
            # Forward pass
            logits = model(x_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop at newline for some prompts
            if next_token.item() == 198:  # newline token
                break
                
            x = torch.cat([x, next_token], dim=1)
    
    # Decode
    tokens = x[0].tolist()
    if tokenizer:
        text = tokenizer.decode(tokens)
    else:
        text = ''.join([chr(min(t, 255)) for t in tokens])
    
    return text

def main():
    print("="*70)
    print("🧪 microCoder 3B Model - Test Suite")
    print("="*70)
    print()
    
    # Load checkpoint
    checkpoint_path = "/workspace/models/checkpoints/microcoder_3b_early_stop.pt"
    
    print(f"📦 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    config = checkpoint['config']
    print(f"✅ Model loaded!")
    print(f"   Parameters: {checkpoint['total_params'] / 1e6:.1f}M")
    print(f"   Layers: {config['n_layers']}")
    print(f"   Hidden dim: {config['d_model']}")
    print(f"   Context: {config['block_size']} tokens")
    print()
    
    # Initialize model
    print("🔧 Initializing model architecture...")
    model = GPTLikeModel(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layers=config['n_layers'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda')
    model.eval()
    print("✅ Model ready!")
    print()
    
    # Load tokenizer
    if TIKTOKEN_AVAILABLE:
        tokenizer = tiktoken.get_encoding("gpt2")
        print("✅ Using tiktoken (GPT-2 BPE)")
    else:
        tokenizer = None
        print("⚠️  No tiktoken, using byte-level")
    print()
    
    # Test prompts
    test_cases = [
        {
            "name": "Binary Search",
            "prompt": "def binary_search(arr, target):",
            "max_tokens": 200
        },
        {
            "name": "Fibonacci",
            "prompt": "def fibonacci(n):",
            "max_tokens": 150
        },
        {
            "name": "Reverse String",
            "prompt": "def reverse_string(s):",
            "max_tokens": 100
        },
        {
            "name": "Find Duplicates",
            "prompt": "def find_duplicates(arr):",
            "max_tokens": 150
        },
        {
            "name": "Binary Tree Class",
            "prompt": "class BinaryTree:\n    def __init__(self):",
            "max_tokens": 200
        },
    ]
    
    print("="*70)
    print("GENERATION TESTS")
    print("="*70)
    print()
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_cases)}: {test['name']}")
        print(f"{'='*70}")
        print(f"\n📝 Prompt:")
        print(f"```python")
        print(test['prompt'])
        print(f"```")
        print(f"\n🤖 Generated Code:")
        print(f"```python")
        
        try:
            output = generate_text(
                model, 
                test['prompt'], 
                tokenizer=tokenizer,
                max_new=test['max_tokens'],
                temperature=0.7,
                top_k=40,
                device='cuda'
            )
            print(output)
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"```")
        print()
    
    print("="*70)
    print("✅ Testing Complete!")
    print("="*70)
    print()
    print("Your model is ready to use! 🚀")
    print()

if __name__ == "__main__":
    main()
