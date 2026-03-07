#!/usr/bin/env python3
"""
Real-time GPU Training Visualization
Monitor your RunPod training in real-time
"""

import subprocess
import time
import os
from datetime import datetime, timedelta

POD_SSH = "ssh -p 15717 root@64.247.201.48"

def get_gpu_stats():
    """Get GPU stats from pod"""
    cmd = f"{POD_SSH} 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits'"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu_util': float(parts[0]),
                'mem_used': float(parts[1]),
                'mem_total': float(parts[2]),
                'temp': float(parts[3]),
                'power': float(parts[4])
            }
    except:
        pass
    return None

def get_training_status():
    """Check if training is running"""
    cmd = f"{POD_SSH} 'ps aux | grep \"python src/train.py\" | grep -v grep | wc -l'"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return int(result.stdout.strip()) > 0
    except:
        return False

def create_bar(value, max_value, width=40):
    """Create a text progress bar"""
    filled = int((value / max_value) * width)
    bar = '█' * filled + '░' * (width - filled)
    return bar

def main():
    print("\n" + "="*70)
    print("🎨 microCoder Training - Real-Time GPU Visualization")
    print("="*70)
    print()
    print("Pod: w7q36vtw8s573q")
    print("GPU: NVIDIA H100 80GB HBM3")
    print()
    print("Press Ctrl+C to stop monitoring")
    print("="*70)
    
    start_time = datetime.now()
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("\n" + "="*70)
            print("🎨 microCoder Training - Real-Time GPU Visualization")
            print("="*70)
            print()
            
            # Runtime
            runtime = datetime.now() - start_time
            print(f"⏱️  Runtime: {str(runtime).split('.')[0]}")
            print()
            
            # Training status
            is_running = get_training_status()
            status_emoji = "✅" if is_running else "❌"
            status_text = "TRAINING" if is_running else "STOPPED"
            print(f"{status_emoji} Status: {status_text}")
            print()
            
            # GPU stats
            stats = get_gpu_stats()
            if stats:
                # GPU Utilization
                gpu_bar = create_bar(stats['gpu_util'], 100)
                print(f"🎮 GPU Utilization:")
                print(f"   [{gpu_bar}] {stats['gpu_util']:.0f}%")
                print()
                
                # Memory
                mem_pct = (stats['mem_used'] / stats['mem_total']) * 100
                mem_bar = create_bar(stats['mem_used'], stats['mem_total'])
                print(f"💾 GPU Memory:")
                print(f"   [{mem_bar}] {stats['mem_used']:.0f} / {stats['mem_total']:.0f} MB ({mem_pct:.1f}%)")
                print()
                
                # Temperature
                temp_bar = create_bar(stats['temp'], 100)
                temp_status = "🟢" if stats['temp'] < 70 else "🟡" if stats['temp'] < 85 else "🔴"
                print(f"🌡️  Temperature:")
                print(f"   [{temp_bar}] {stats['temp']:.0f}°C {temp_status}")
                print()
                
                # Power
                power_bar = create_bar(stats['power'], 700)
                print(f"⚡ Power Draw:")
                print(f"   [{power_bar}] {stats['power']:.0f} / 700 W")
                print()
            else:
                print("⚠️  Could not fetch GPU stats")
                print()
            
            # Cost estimate
            hours = runtime.total_seconds() / 3600
            cost = hours * 2.69
            estimated_total_hours = 7  # estimate
            estimated_total_cost = estimated_total_hours * 2.69
            remaining_hours = max(0, estimated_total_hours - hours)
            
            print("="*70)
            print("💰 Cost Tracking:")
            print(f"   Current: ${cost:.2f} ({hours:.2f} hours)")
            print(f"   Estimated Total: ${estimated_total_cost:.2f} ({estimated_total_hours} hours)")
            print(f"   Remaining: ~{remaining_hours:.1f} hours")
            print()
            
            print("="*70)
            print(f"📊 Last updated: {datetime.now().strftime('%H:%M:%S')}")
            print("   Refreshing in 5 seconds...")
            print()
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")
        print()

if __name__ == "__main__":
    main()
