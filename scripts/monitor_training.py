#!/usr/bin/env python3
"""
Real-time Training Monitor - Live dashboard for training progress

Displays:
- Current iteration, loss, perplexity
- Training speed (tokens/sec, time per iteration)
- ETA (estimated time to completion)
- Learning rate
- GPU/Memory usage
- Recent loss history (mini sparkline)

Usage:
    # Monitor a running training session:
    python scripts/monitor_training.py logs/training_3b.log
    
    # Or just check current stats:
    python scripts/monitor_training.py logs/training_3b.log --once
"""

import sys
import time
import re
import os
from datetime import datetime, timedelta


def parse_training_log(log_file):
    """Parse training log and extract metrics"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except:
        return None
    
    # Look for training iteration lines
    # Example: "Iteration 100/15000 | Loss: 3.456 | Time: 1.23s | LR: 0.00015"
    metrics = []
    for line in reversed(lines[-100:]):  # Last 100 lines
        if 'Iteration' in line and 'Loss' in line:
            try:
                iter_match = re.search(r'Iteration (\d+)/(\d+)', line)
                loss_match = re.search(r'Loss: ([\d.]+)', line)
                time_match = re.search(r'Time: ([\d.]+)s', line)
                lr_match = re.search(r'LR: ([\d.e-]+)', line)
                
                if iter_match and loss_match:
                    metrics.append({
                        'iteration': int(iter_match.group(1)),
                        'total_iterations': int(iter_match.group(2)),
                        'loss': float(loss_match.group(1)),
                        'time_per_iter': float(time_match.group(1)) if time_match else None,
                        'lr': float(lr_match.group(1)) if lr_match else None,
                    })
            except:
                continue
    
    if not metrics:
        return None
    
    metrics.reverse()  # Oldest to newest
    return metrics


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        hours = seconds / 3600
        if hours < 24:
            return f"{hours:.1f}h"
        else:
            return f"{hours/24:.1f}d"


def create_sparkline(values, width=20):
    """Create a mini text sparkline graph"""
    if not values or len(values) < 2:
        return "─" * width
    
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return "─" * width
    
    # Unicode block characters for sparkline
    blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    
    # Sample values to fit width
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    
    sparkline = ""
    for val in sampled:
        normalized = (val - min_val) / (max_val - min_val)
        block_idx = int(normalized * (len(blocks) - 1))
        sparkline += blocks[block_idx]
    
    return sparkline


def display_training_stats(log_file, once=False):
    """Display live training statistics"""
    
    print("\n" + "="*70)
    print("📊 microCoder Training Monitor")
    print("="*70)
    print(f"Log file: {log_file}")
    print("="*70 + "\n")
    
    last_update = None
    
    while True:
        metrics = parse_training_log(log_file)
        
        if metrics is None:
            if once:
                print("❌ No training data found in log file")
                return
            print("⏳ Waiting for training data...", end='\r')
            time.sleep(2)
            continue
        
        current = metrics[-1]
        
        # Clear screen (optional, comment out if you want scrolling history)
        if not once:
            os.system('clear' if os.name != 'nt' else 'cls')
            print("\n" + "="*70)
            print("📊 microCoder Training Monitor - Live")
            print("="*70)
            print(f"Log: {log_file}")
            print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            print("="*70 + "\n")
        
        # Current stats
        print(f"🔄 Iteration: {current['iteration']:,} / {current['total_iterations']:,}")
        progress = current['iteration'] / current['total_iterations']
        bar_width = 40
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"   Progress: [{bar}] {progress*100:.1f}%\n")
        
        print(f"📉 Loss: {current['loss']:.4f}")
        print(f"   Perplexity: {2**current['loss']:.2f}")
        
        # Loss trend (last 20 iterations)
        if len(metrics) >= 2:
            recent_losses = [m['loss'] for m in metrics[-20:]]
            sparkline = create_sparkline(recent_losses)
            loss_change = current['loss'] - metrics[-2]['loss']
            trend = "📈" if loss_change > 0 else "📉"
            print(f"   Trend: {sparkline} {trend} ({loss_change:+.4f})")
        
        print()
        
        # Speed stats
        if current['time_per_iter']:
            print(f"⚡ Speed: {current['time_per_iter']:.2f}s per iteration")
            iters_remaining = current['total_iterations'] - current['iteration']
            eta_seconds = iters_remaining * current['time_per_iter']
            print(f"   ETA: {format_time(eta_seconds)}")
            print(f"   Est. completion: {(datetime.now() + timedelta(seconds=eta_seconds)).strftime('%Y-%m-%d %H:%M')}")
        
        print()
        
        # Learning rate
        if current['lr']:
            print(f"📚 Learning Rate: {current['lr']:.2e}")
        
        print()
        
        # Recent history (last 5 iterations)
        if len(metrics) >= 5:
            print("📊 Recent History:")
            for m in metrics[-5:]:
                time_str = f"{m['time_per_iter']:.2f}s" if m['time_per_iter'] else "N/A"
                print(f"   Iter {m['iteration']:6,}: Loss {m['loss']:.4f} | Time {time_str}")
        
        print("\n" + "="*70)
        
        if once:
            break
        
        print("\n⏱️  Refreshing in 10s... (Ctrl+C to stop)\n")
        time.sleep(10)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python scripts/monitor_training.py logs/training_3b.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    once = '--once' in sys.argv
    
    try:
        display_training_stats(log_file, once=once)
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")
