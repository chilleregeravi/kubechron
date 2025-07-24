#!/usr/bin/env python3
"""
Simple LoRA Training Test for Kubernetes Dataset
"""

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

print("🚀 Simple LoRA Training Test")
print("=" * 40)

# System info
print(f"🖥️ System: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"📦 PyTorch: {torch.__version__}")

# Test LoRA availability
try:
    print("✅ PEFT (LoRA) is available!")
    
    # Load small dataset sample
    print("\n📂 Loading dataset sample...")
    dataset_path = "src/output/k8s_dataset.jsonl"
    
    data = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:  # Just 50 examples for quick test
                break
            try:
                record = json.loads(line.strip())
                instruction = record.get("instruction", "")
                output = record.get("output", "")
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}<|endoftext|>"
                data.append({"text": text})
            except:
                continue
    
    print(f"✅ Loaded {len(data)} examples")
    dataset = Dataset.from_list(data)
    
    # Setup model with LoRA
    print("\n🤖 Setting up LoRA model...")
    model_name = "distilgpt2"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📊 Original parameters: {model.num_parameters():,}")
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,  # Very low rank for testing
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["c_attn"],
    )
    
    model = get_peft_model(model, lora_config)
    print("✅ LoRA adapters added!")
    print("📈 Trainable parameters:")
    model.print_trainable_parameters()
    
    print("\n🎉 LoRA setup successful!")
    print("Your system is ready for LoRA training!")
    
    print("\n📋 Recommended next steps:")
    print("   1. Run full training with: python train_k8s_lora.py")
    print("   2. Choose 'quick' for 5-10 minute test")
    print("   3. Choose 'small' for 15-30 minute training")
    print("   4. Use trained model for inference")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
