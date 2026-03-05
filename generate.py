import torch
import sys
from tiny_llm import GPTLikeModel, get_tokenizer, generate_text

def generate_from_checkpoint(checkpoint_path, prompt, gen_len=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = GPTLikeModel(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layers=config['n_layers'],
        d_model=config['d_model'],
        n_heads=config['n_heads']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Generating with prompt: {prompt}")
    output = generate_text(model, prompt, max_new=gen_len, device=device)
    print("\nOutput:")
    print(output)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate.py <checkpoint_path> <prompt> [gen_len]")
        sys.exit(1)
    
    cp_path = sys.argv[1]
    prompt = sys.argv[2]
    gen_len = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    generate_from_checkpoint(cp_path, prompt, gen_len)
