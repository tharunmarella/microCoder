"""
Generate text from a trained model checkpoint.

Usage:
    python src/generate.py models/checkpoints/model.pt "def hello_world():" --max-new 100
"""

import torch
import sys
import argparse
from model import GPTLikeModel
from utils import generate_text


def generate_from_checkpoint(checkpoint_path, prompt, max_new=100, temperature=0.8, top_k=40):
    """
    Load a model from checkpoint and generate text.
    
    Args:
        checkpoint_path: Path to model checkpoint
        prompt: Text prompt to start generation
        max_new: Number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = GPTLikeModel(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layers=config['n_layers'],
        d_model=config['d_model'],
        n_heads=config['n_heads']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\n📊 Model info:")
    print(f"   Parameters: {model.count_parameters() / 1e6:.1f}M")
    print(f"   Layers: {config['n_layers']}")
    print(f"   Hidden dim: {config['d_model']}")
    print(f"\n🎯 Generating with prompt: {prompt}")
    print(f"   Temperature: {temperature}")
    print(f"   Top-k: {top_k}")
    print(f"\n{'='*60}")
    
    # Generate
    output = generate_text(
        model,
        prompt,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        device=device
    )
    
    print(output)
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate text from a trained model.')
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        'prompt',
        type=str,
        help='Text prompt to start generation'
    )
    parser.add_argument(
        '--max-new',
        type=int,
        default=256,
        help='Number of new tokens to generate (default: 256)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k sampling (default: 40)'
    )
    
    args = parser.parse_args()
    
    generate_from_checkpoint(
        args.checkpoint,
        args.prompt,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
