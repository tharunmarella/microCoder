# Tiny LLM

A minimal, high-performance implementation of a GPT-like Large Language Model (LLM) in PyTorch. This project is designed for educational purposes, providing a clear, step-by-step look at modern transformer architectures (similar to Llama and GPT-4).

## 🚀 Features

- **Modern Architecture**: Implements `RMSNorm`, `Rotary Positional Embeddings (RoPE)`, and `Multi-Head Self-Attention`.
- **Flash Attention**: Uses PyTorch's optimized attention kernels for faster training.
- **Flexible Data Sources**: Support for HuggingFace datasets, local text files, or synthetic data.
- **Mixed Precision (AMP)**: Support for faster training on NVIDIA GPUs.
- **Tokenization**: Uses `tiktoken` (GPT-2/GPT-4 tokenizer) with a fallback to byte-level encoding.
- **Memory Efficient**: Includes gradient accumulation and memory monitoring.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tiny-llm
   ```

2. **Install dependencies**:
   ```bash
   pip install torch tiktoken datasets
   ```

## 📖 Usage

### Training the Model

To start training with default settings (approx. 100M parameters):

```bash
python tiny_llm.py --data-source hf --hf-dataset wikitext --hf-config wikitext-2-raw-v1
```

### Customizing the Architecture

You can easily adjust the model size using command-line arguments:

```bash
python tiny_llm.py \
    --n-layers 12 \
    --d-model 768 \
    --n-heads 12 \
    --block-size 512 \
    --batch-size 32
```

### Training on a Local File

```bash
python tiny_llm.py --data-source file --data-file path/to/your/text.txt
```

## 🧠 Architecture Overview

The model follows the standard decoder-only transformer architecture:

```text
       [ Input Tokens ]         (e.g., "The cat sat")
              |
      [ Token Embedding ]       (Converts words to vectors)
              |
    +---------+---------+
    |  Rotary Positional |      (Adds position info so model
    |  Embeddings (RoPE) |       knows word order)
    +---------+---------+
              |
      +-------v-------+
      |  Transformer  | x N Layers (The "Brain" of the model)
      |     Block     |
      | +-----------+ |
      | |  RMSNorm  | |  (Normalizes data for stability)
      | +-----------+ |
      | | Attention | |  (Model looks at related words)
      | +-----------+ |
      | |  RMSNorm  | |  (Normalizes again)
      | +-----------+ |
      | |    MLP    | |  (Processes information)
      | +-----------+ |
      +-------+-------+
              |
      [  Output Head  ]         (Maps vectors back to vocabulary)
              |
      [ Next Token Prob ]       (Predicts the next word)

      --------------------------------------------------
      Input:  List of numbers representing words
      Output: Probability for every possible next word
```
```

1.  **Tokenization**: Converts text into integers.
2.  **Embedding Layer**: Maps integers to high-dimensional vectors.
3.  **Rotary Positional Embeddings (RoPE)**: Injects sequence order information.
4.  **Transformer Blocks**:
    - **RMSNorm**: For stable training.
    - **Multi-Head Attention**: Allows the model to focus on different parts of the input.
    - **Feed-Forward Network (MLP)**: Processes the attention output using GELU activation.
5.  **Output Head**: Predicts the probability of the next token.

## 📊 Training Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--iterations` | Number of training steps | 5000 |
| `--lr` | Learning rate | 3e-4 |
| `--amp` | Enable Mixed Precision | False |
| `--grad-accum-steps` | Steps for gradient accumulation | 1 |
| `--save-path` | Where to save the model | `tiny_llm.pt` |

## 📝 License

MIT License. Feel free to use and modify for your own projects!
