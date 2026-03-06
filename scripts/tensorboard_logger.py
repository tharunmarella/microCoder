#!/usr/bin/env python3
"""
TensorBoard Integration for microCoder Training

This script adds TensorBoard logging to track:
- Training loss over time
- Learning rate schedule
- Gradient norms
- Model parameters
- Sample generations
- GPU memory usage

Usage:
    # During training, logs are automatically saved to logs/tensorboard/
    
    # View in TensorBoard:
    tensorboard --logdir logs/tensorboard --port 6006
    
    # Then open: http://localhost:6006
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


class TrainingMonitor:
    """Enhanced training monitor with TensorBoard integration"""
    
    def __init__(self, log_dir="logs/tensorboard", experiment_name=None):
        """
        Initialize training monitor with TensorBoard
        
        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name for this training run (auto-generated if None)
        """
        if experiment_name is None:
            experiment_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_path, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_path)
        self.step = 0
        
        print(f"📊 TensorBoard logging to: {self.log_path}")
        print(f"   View with: tensorboard --logdir {log_dir}")
        
    def log_metrics(self, metrics, step=None):
        """
        Log scalar metrics to TensorBoard
        
        Args:
            metrics: Dict of metric_name -> value
            step: Global step (uses internal counter if None)
        """
        if step is None:
            step = self.step
            self.step += 1
        
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(name, value, step)
    
    def log_model_graph(self, model, input_shape):
        """Log model architecture graph"""
        try:
            dummy_input = torch.randint(0, 50257, input_shape)
            self.writer.add_graph(model, dummy_input)
            print("✅ Model graph logged to TensorBoard")
        except Exception as e:
            print(f"⚠️  Could not log model graph: {e}")
    
    def log_text_generation(self, prompt, generated_text, step):
        """Log generated text samples"""
        text = f"**Prompt:** {prompt}\n\n**Generated:**\n```python\n{generated_text}\n```"
        self.writer.add_text("generations", text, step)
    
    def log_histograms(self, model, step):
        """Log weight and gradient histograms"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"weights/{name}", param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, step)
    
    def log_lr(self, lr, step):
        """Log learning rate"""
        self.writer.add_scalar("learning_rate", lr, step)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


def add_tensorboard_to_training():
    """
    Instructions for integrating TensorBoard into training loop
    
    Add this to your training script:
    
    1. Import:
        from scripts.tensorboard_logger import TrainingMonitor
    
    2. Initialize before training:
        monitor = TrainingMonitor(
            log_dir="logs/tensorboard",
            experiment_name="3b_codesearchnet"
        )
    
    3. In training loop:
        # Log loss and metrics
        monitor.log_metrics({
            "train/loss": loss.item(),
            "train/perplexity": math.exp(loss.item()),
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/grad_norm": grad_norm
        }, step=iteration)
        
        # Log generations (every N steps)
        if iteration % sample_interval == 0:
            monitor.log_text_generation(prompt, generated, iteration)
        
        # Log histograms (every N steps, expensive)
        if iteration % 1000 == 0:
            monitor.log_histograms(model, iteration)
    
    4. Close at end:
        monitor.close()
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print(add_tensorboard_to_training.__doc__)
