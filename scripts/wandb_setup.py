#!/usr/bin/env python3
"""
Weights & Biases (wandb) Integration for microCoder

W&B provides cloud-based experiment tracking with:
- Beautiful interactive dashboards
- Automatic hyperparameter logging
- Model checkpointing and versioning
- Easy collaboration and sharing
- Free for personal projects!

Setup:
    1. Install: pip install wandb
    2. Login: wandb login
    3. Add to training script (see below)

Usage:
    # In your training script:
    import wandb
    
    # Initialize
    wandb.init(
        project="microCoder",
        name="3b-codesearchnet",
        config={
            "n_layers": 32,
            "d_model": 2560,
            "n_heads": 32,
            "batch_size": 2,
            "learning_rate": 1.5e-4,
            "dataset": "CodeSearchNet Python"
        }
    )
    
    # Log metrics during training
    wandb.log({
        "loss": loss.item(),
        "learning_rate": lr,
        "perplexity": math.exp(loss.item()),
        "iteration": iteration
    })
    
    # Log generated samples
    wandb.log({
        "samples": wandb.Table(
            columns=["prompt", "generated"],
            data=[[prompt, generated_text]]
        )
    })
    
    # Save model checkpoint
    wandb.save("models/checkpoints/model.pt")
    
    # Finish
    wandb.finish()

Benefits over TensorBoard:
- ✅ Cloud-hosted (access from anywhere)
- ✅ Better UI and visualizations
- ✅ Easy sharing with team/community
- ✅ Automatic hyperparameter tracking
- ✅ Model versioning and artifact tracking
- ✅ Compare multiple runs easily
- ✅ Mobile app for monitoring on-the-go

View your runs at: https://wandb.ai/<your-username>/microCoder
"""


def setup_wandb_example():
    """
    Example integration code for training script
    """
    
    example_code = '''
# Add to src/train.py or your training script:

import wandb
import math

def train_with_wandb(args):
    """Training function with W&B integration"""
    
    # 1. Initialize W&B
    wandb.init(
        project="microCoder",
        name=f"{args.n_layers}L-{args.d_model}d",  # e.g., "32L-2560d"
        config={
            # Model architecture
            "n_layers": args.n_layers,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "block_size": args.block_size,
            "vocab_size": args.vocab_size,
            "dropout": args.dropout,
            
            # Training config
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "learning_rate": args.lr,
            "iterations": args.iterations,
            "warmup_ratio": args.warmup_ratio,
            
            # Dataset
            "dataset": args.data_file,
            "device": str(args.device),
        },
        tags=["code-generation", "gpt", "transformer"]
    )
    
    # 2. Watch model (logs gradients and parameters)
    wandb.watch(model, log="all", log_freq=100)
    
    # 3. Training loop
    for iteration in range(args.iterations):
        # ... your training code ...
        
        # Log metrics every step
        wandb.log({
            "train/loss": loss.item(),
            "train/perplexity": math.exp(min(loss.item(), 20)),  # Cap for stability
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/grad_norm": grad_norm,
            "train/iteration": iteration,
        }, step=iteration)
        
        # Log samples periodically
        if iteration % args.sample_interval == 0:
            with torch.no_grad():
                generated = generate_text(model, tokenizer, prompt, max_tokens=100)
            
            wandb.log({
                "samples": wandb.Table(
                    columns=["iteration", "prompt", "generated"],
                    data=[[iteration, prompt, generated]]
                )
            }, step=iteration)
    
    # 4. Save final model as artifact
    artifact = wandb.Artifact(
        name=f"model-{wandb.run.id}",
        type="model",
        description=f"{args.n_layers}L {args.d_model}d GPT model"
    )
    artifact.add_file(args.save_path)
    wandb.log_artifact(artifact)
    
    # 5. Finish
    wandb.finish()
    print(f"🎉 Training complete! View results at: {wandb.run.get_url()}")

# Usage:
# python src/train.py --use-wandb ...
'''
    
    return example_code


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("EXAMPLE INTEGRATION CODE:")
    print("="*70)
    print(setup_wandb_example())
