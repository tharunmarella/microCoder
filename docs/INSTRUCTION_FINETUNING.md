# Fine-Tuning an Instruct Model - Complete Guide

Transform your base code generation model into an instruction-following AI assistant!

---

## 🎯 What is Instruction Fine-Tuning?

**Base Model** (what you have now):
```python
Input:  "def sort_list"
Output: "(arr):\n    return sorted(arr)"  # Just completes
```

**Instruct Model** (after fine-tuning):
```python
Input:  "Write a function to sort a list"
Output: "Here's a function to sort a list:

def sort_list(arr):
    '''Sort a list in ascending order'''
    return sorted(arr)

This function takes a list and returns a sorted copy."
```

---

## 📋 Overview

Instruction fine-tuning teaches your model to:
- ✅ Follow natural language commands
- ✅ Answer questions about code
- ✅ Explain and document code
- ✅ Debug and fix errors
- ✅ Refactor and optimize
- ✅ Generate complete solutions from descriptions

---

## 🔧 Two-Stage Process

### **Stage 1: Supervised Fine-Tuning (SFT)** ⭐ Recommended
Train on instruction-response pairs to teach following commands

**Time:** 2-4 hours  
**Cost:** ~$5-10 (H100 80GB)  
**Difficulty:** Easy  

### **Stage 2: RLHF** (Optional - Advanced)
Reinforcement Learning from Human Feedback for alignment

**Time:** 4-6 hours  
**Cost:** ~$10-15  
**Difficulty:** Advanced  

---

## 📊 Cost & Quality Comparison

| Approach | Time | Cost | Quality | Use Case |
|----------|------|------|---------|----------|
| **Base only** | 6-8h | $16-22 | ⭐⭐⭐⭐ | Autocomplete |
| **Base + SFT** ⭐ | 8-12h | $21-32 | ⭐⭐⭐⭐⭐ | AI assistant |
| **Base + SFT + RLHF** | 16-24h | $41-62 | ⭐⭐⭐⭐⭐⭐ | Production |

**Recommended:** Base + SFT

---

## 🚀 Quick Start (After Base Training)

### Prerequisites:
- ✅ Base model trained (`codesearchnet_3b.pt`)
- ✅ RunPod pod with H100 80GB
- ✅ 2-4 hours available
- ✅ ~$5-10 budget

### Step 1: Prepare Instruction Dataset
```bash
python scripts/prepare_instruct_data.py \
  --dataset code_alpaca \
  --output data/code_alpaca_instruct.pt
```

### Step 2: Fine-tune on Instructions
```bash
python scripts/finetune_instruct.py \
  --base-model models/checkpoints/codesearchnet_3b.pt \
  --data-file data/code_alpaca_instruct.pt \
  --epochs 3 \
  --lr 2e-5 \
  --batch-size 4 \
  --output models/checkpoints/microcoder_3b_instruct.pt \
  --tensorboard
```

### Step 3: Test Your Instruct Model
```bash
python scripts/test_instruct_model.py \
  models/checkpoints/microcoder_3b_instruct.pt
```

---

## 📚 Instruction Datasets

### **Code Alpaca** ⭐ Recommended
- **Size:** 20,000 instruction-response pairs
- **Focus:** Code generation tasks
- **Quality:** High
- **Best for:** First instruct fine-tune

**Example:**
```json
{
  "instruction": "Write a function to check if a number is prime",
  "input": "",
  "output": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
}
```

### **Other Options:**

| Dataset | Size | Focus | Quality | Use When |
|---------|------|-------|---------|----------|
| **WizardCoder** | 78K | Complex tasks | ⭐⭐⭐⭐⭐ | Advanced |
| **Evol-Instruct** | 110K | Multi-turn | ⭐⭐⭐⭐⭐ | Production |
| **ShareGPT-Code** | 50K | Conversations | ⭐⭐⭐⭐ | Chat |

---

## 🔧 Training Configuration

### Recommended Settings for 3B Model:

```bash
# Model architecture (same as base)
--block-size 2048
--n-layers 32
--d-model 2560
--n-heads 32

# Training (different from base!)
--epochs 3                # Fewer epochs than base
--lr 2e-5                 # 10x LOWER than base!
--batch-size 4            # Smaller batches
--grad-accum-steps 4      # To simulate batch size 16
--warmup-ratio 0.05       # Short warmup
--min-lr-ratio 0.1

# Regularization
--weight-decay 0.01
--dropout 0.1
--max-grad-norm 1.0

# Optimization
--amp                     # Mixed precision
--scheduler               # Cosine decay
--tensorboard             # Visualization
```

### ⚠️ Critical: Lower Learning Rate!

**Why 2e-5 instead of 1.5e-4?**
- Base model already knows language patterns
- Fine-tuning just adapts behavior
- Too high LR → "catastrophic forgetting"
- 2e-5 = 10x lower = safe adaptation

---

## 📊 Training Process

### What Happens During Fine-Tuning:

**1. Load Base Model** (your pre-trained 3B)
```
✓ Already knows Python syntax
✓ Already understands code patterns
✓ Can generate coherent code
```

**2. Format Data** (instruction → response)
```
Input:  ### Instruction: Write factorial function
        ### Response:
Target: def factorial(n):
            if n <= 1: return 1
            return n * factorial(n-1)
```

**3. Masked Training**
```
✗ Don't learn to predict instruction
✓ Only learn to predict response
```

**4. Validation**
```
Test: Does it follow new instructions?
Test: Did it forget base knowledge?
```

---

## 📈 Monitoring Fine-Tuning

### Key Metrics in TensorBoard:

**1. Loss Curve**
- Start: ~2.5-3.0
- Target: ~1.5-2.0
- Should decrease smoothly
- Converges faster than base training

**2. Instruction Following**
- Test on validation set
- Target: >90% appropriate responses
- Examples shown in TEXT tab

**3. Base Capability Retention**
- Test autocomplete still works
- Loss shouldn't increase dramatically
- Code quality maintains

**4. Learning Rate Schedule**
- Brief warmup (5% of training)
- Cosine decay to min LR
- Stays low to avoid forgetting

---

## 🎯 Expected Results

### Before SFT (Base Model):
```
>>> "Write a factorial function"
def factorial function(n):
    return n * factorial(n-1)  # Incomplete, no base case
```

### After SFT (Instruct Model):
```
>>> "Write a factorial function"
Here's a factorial function:

def factorial(n):
    '''Calculate factorial of n recursively'''
    if n <= 1:
        return 1
    return n * factorial(n - 1)

This function calculates the factorial using recursion with a base case.
```

---

## 🧪 Testing Framework

### Test Your Instruct Model:

**1. Code Generation**
```python
test_cases = [
    "Write a function to reverse a string",
    "Create a binary search implementation",
    "Make a function to find prime numbers"
]
```

**2. Code Explanation**
```python
test_cases = [
    "Explain this code: [snippet]",
    "What does this function do?",
    "How does binary search work?"
]
```

**3. Debugging**
```python
test_cases = [
    "Fix this bug: [buggy code]",
    "Why isn't this working: [code]",
    "Debug this error: [error message]"
]
```

**4. Refactoring**
```python
test_cases = [
    "Optimize this function: [code]",
    "Make this code more readable",
    "Refactor using list comprehension"
]
```

---

## 🚀 RunPod Deployment

### One-Command Fine-Tuning:

```bash
# On RunPod pod (after base training completes)
./scripts/finetune_instruct_runpod.sh
```

This script will:
1. ✅ Download Code Alpaca dataset
2. ✅ Prepare instruction format
3. ✅ Load your base model
4. ✅ Fine-tune for 3 epochs
5. ✅ Save instruct model
6. ✅ Run test cases

**Time:** 2-4 hours  
**Cost:** $5-10 (H100 80GB Spot)

---

## 📊 Quality Benchmarks

### Your Instruct Model Should Achieve:

| Task | Target | Comparable To |
|------|--------|---------------|
| **Code generation** | >80% correct | Early Copilot |
| **Explanation** | >75% clear | Good docs |
| **Bug fixing** | >60% correct | Junior dev |
| **Autocomplete** | >85% (maintained) | Base model |
| **Instruction following** | >90% appropriate | ChatGPT-like |

---

## 🎯 After Fine-Tuning: Usage Examples

### Interactive Chat:
```bash
$ python scripts/chat_instruct.py

You: Write a function to find duplicates in a list