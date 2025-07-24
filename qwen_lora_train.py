#!/usr/bin/env python3
"""
Qwen Small Language Model LoRA Training
Train Qwen models efficiently on Kubernetes dataset using LoRA
"""

import json
import os
import time
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Qwen model configurations
QWEN_MODELS = {
    "Qwen2-0.5B": {
        "model_name": "Qwen/Qwen2-0.5B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "description": "0.5B parameters - very fast training"
    },
    "Qwen2-1.5B": {
        "model_name": "Qwen/Qwen2-1.5B-Instruct", 
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "description": "1.5B parameters - good balance"
    },
    "Qwen2-7B": {
        "model_name": "Qwen/Qwen2-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "description": "7B parameters - requires more memory"
    }
}

def load_k8s_dataset(max_examples=1000):
    """Load and format Kubernetes dataset for Qwen."""
    print("📂 Loading Kubernetes dataset...")
    
    data = []
    with open("src/output/k8s_dataset.jsonl", 'r') as f:
        for i, line in enumerate(f):
            if len(data) >= max_examples:
                break
            try:
                record = json.loads(line.strip())
                instruction = record.get("instruction", "")
                output = record.get("output", "")
                
                # Format for Qwen chat template
                text = f"<|im_start|>system\nYou are a Kubernetes expert assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                data.append({"text": text})
            except:
                continue
    
    print(f"✅ Loaded {len(data)} examples")
    return Dataset.from_list(data)

def setup_qwen_lora_model(model_choice="Qwen2-0.5B", lora_rank=8):
    """Setup Qwen model with LoRA adapters."""
    if model_choice not in QWEN_MODELS:
        raise ValueError(f"Model choice must be one of: {list(QWEN_MODELS.keys())}")
    
    model_config = QWEN_MODELS[model_choice]
    model_name = model_config["model_name"]
    target_modules = model_config["target_modules"]
    
    print(f"🤖 Loading {model_choice} ({model_name}) with LoRA...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📊 Original parameters: {model.num_parameters():,}")
    
    # Configure LoRA for Qwen
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    print("✅ LoRA adapters added!")
    model.print_trainable_parameters()
    
    return model, tokenizer, model_choice

def train_qwen_lora_model(dataset, model, tokenizer, model_choice, epochs=2, output_dir=None):
    """Train the Qwen model with LoRA."""
    if output_dir is None:
        output_dir = f"qwen_k8s_lora_{model_choice.lower().replace('-', '_')}"
    
    print(f"\n🏋️ Starting Qwen LoRA training...")
    
    # Tokenize dataset
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False,
            max_length=1024,  # Qwen can handle longer sequences
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(f"✅ Tokenized {len(tokenized_dataset)} examples")
    
    # Training arguments optimized for Qwen
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
        gradient_accumulation_steps=8 if torch.cuda.is_available() else 16,
        learning_rate=1e-4,  # Slightly lower for Qwen
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use fp16 only on GPU
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        # Memory optimization
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training info
    batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps = len(tokenized_dataset) // batch_size * epochs
    est_time = steps * 2.0 / 60  # ~2 sec per step for Qwen
    print(f"📊 Training will take ~{est_time:.0f} minutes")
    
    # Train
    start_time = time.time()
    trainer.train()
    training_time = (time.time() - start_time) / 60
    
    print(f"✅ Training completed in {training_time:.1f} minutes!")
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"💾 Model saved to {output_dir}")
    
    return trainer, output_dir

def test_trained_qwen_model(output_dir, model_choice):
    """Test the trained Qwen model."""
    print(f"\n🧪 Testing trained Qwen model...")
    
    try:
        # Load base model
        model_config = QWEN_MODELS[model_choice]
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Load LoRA model
        model = PeftModel.from_pretrained(base_model, output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        
        # Test questions
        questions = [
            "What is a Kubernetes Pod?",
            "How do you create a Service in Kubernetes?",
            "Explain Kubernetes Deployments",
            "How to troubleshoot a CrashLoopBackOff error?"
        ]
        
        print("🤖 Qwen model responses:\n")
        for question in questions:
            prompt = f"<|im_start|>system\nYou are a Kubernetes expert assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "<|im_start|>assistant\n" in response:
                response = response.split("<|im_start|>assistant\n")[-1].strip()
            
            print(f"❓ {question}")
            print(f"🤖 {response}\n")
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def main():
    """Main training function."""
    print("🚀 Qwen Kubernetes LoRA Training")
    print("=" * 50)
    print(f"🖥️ System: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Show available models
    print(f"\n📋 Available Qwen Models:")
    for name, config in QWEN_MODELS.items():
        print(f"   {name}: {config['description']}")
    
    # Training configurations
    configs = {
        "quick": {"examples": 100, "epochs": 1, "rank": 4, "time": "10-20 min"},
        "small": {"examples": 500, "epochs": 2, "rank": 8, "time": "30-60 min"},
        "medium": {"examples": 1000, "epochs": 3, "rank": 8, "time": "60-120 min"},
    }
    
    print(f"\n📋 Training Options:")
    for name, config in configs.items():
        print(f"   {name}: {config['examples']} examples, {config['epochs']} epochs, rank {config['rank']} (~{config['time']})")
    
    # Get model choice
    model_choice = input(f"\n❓ Choose Qwen model (Qwen2-0.5B/Qwen2-1.5B/Qwen2-7B) [Qwen2-0.5B]: ").strip() or "Qwen2-0.5B"
    if model_choice not in QWEN_MODELS:
        print("Invalid choice, using Qwen2-0.5B")
        model_choice = "Qwen2-0.5B"
    
    # Get training config
    choice = input(f"\n❓ Choose training size (quick/small/medium) [small]: ").strip().lower() or "small"
    if choice not in configs:
        print("Invalid choice, using 'small'")
        choice = "small"
    
    config = configs[choice]
    print(f"✅ Using {model_choice} with '{choice}' configuration")
    
    # Confirm
    if input(f"\n❓ Start {choice} training with {model_choice} (~{config['time']})? (y/n): ").lower() != 'y':
        print("👋 Training cancelled")
        return
    
    try:
        # Load dataset
        dataset = load_k8s_dataset(max_examples=config["examples"])
        
        # Setup model
        model, tokenizer, model_choice = setup_qwen_lora_model(model_choice=model_choice, lora_rank=config["rank"])
        
        # Train
        trainer, output_dir = train_qwen_lora_model(dataset, model, tokenizer, model_choice, epochs=config["epochs"])
        
        print(f"\n🎉 Qwen training completed successfully!")
        
        # Test
        if input(f"\n❓ Test the trained Qwen model? (y/n): ").lower() == 'y':
            test_trained_qwen_model(output_dir, model_choice)
        
        print(f"\n🚀 Your Qwen LoRA model is ready!")
        print(f"   📁 Saved in: {output_dir}/")
        print(f"   🎯 Model: {model_choice}")
        print(f"   ⚡ Uses only ~0.1% of parameters with LoRA")
        print(f"   🧠 Specialized for Kubernetes tasks")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 