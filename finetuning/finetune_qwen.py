import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    # 1. Configuration
    model_name = "Qwen/Qwen2.5-Coder-7B" # The base model
    dataset_name = "sahil2801/CodeAlpaca-20k" # Instruction dataset for coding
    output_dir = "./finetuned_qwen_coder"
    
    print(f"🚀 Starting Fine-Tuning Process for {model_name}")
    print(f"📊 Loading dataset: {dataset_name}")
    
    # 2. Load Dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # 3. Configure Quantization (QLoRA) to save VRAM
    # This allows us to load a 7B model in 4-bit precision
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 4. Load Model and Tokenizer
    print("🧠 Loading model in 4-bit precision...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 5. Prepare Model for LoRA Fine-Tuning
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, # Rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention blocks
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print("✨ Model prepared for LoRA fine-tuning:")
    model.print_trainable_parameters()
    
    # 6. Formatting Function for the Dataset
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction'][i]}\n\n### Response:\n{example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True, # Use fp16 or bf16 depending on GPU
        max_grad_norm=0.3,
        max_steps=200, # Short run for testing; increase for full fine-tuning
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )
    
    # 8. Setup Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # 9. Start Training
    print("🔥 Starting training...")
    trainer.train()
    
    # 10. Save the LoRA adapters
    trainer.model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))
    print(f"🎉 Fine-tuning complete. Adapters saved to {output_dir}/final_adapter")

if __name__ == "__main__":
    main()
