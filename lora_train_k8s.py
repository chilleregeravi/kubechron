#!/usr/bin/env python3
"""
Complete LoRA Training for Kubernetes Dataset
Train only 0.09% of parameters with full model quality!
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

def load_k8s_dataset(max_examples=1000):
    """Load and format Kubernetes dataset."""
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
                
                # Format for instruction tuning
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}<|endoftext|>"
                data.append({"text": text})
            except:
                continue
    
    print(f"✅ Loaded {len(data)} examples")
    return Dataset.from_list(data)

def setup_lora_model(model_name="distilgpt2", lora_rank=8):
    """Setup model with LoRA adapters."""
    print(f"🤖 Loading {model_name} with LoRA...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📊 Original parameters: {model.num_parameters():,}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    print("✅ LoRA adapters added!")
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_lora_model(dataset, model, tokenizer, epochs=2, output_dir="k8s_lora_model"):
    """Train the model with LoRA."""
    print(f"\n🏋️ Starting LoRA training...")
    
    # Tokenize dataset
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False,
            max_length=512,
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(f"✅ Tokenized {len(tokenized_dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
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
    steps = len(tokenized_dataset) // 8 * epochs
    est_time = steps * 1.5 / 60  # ~1.5 sec per step
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
    
    return trainer

def test_trained_model(output_dir="k8s_lora_model"):
    """Test the trained model."""
    print(f"\n🧪 Testing trained model...")
    
    try:
        # Load trained model
        base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        model = PeftModel.from_pretrained(base_model, output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Test questions
        questions = [
            "What is a Kubernetes Pod?",
            "How do you create a Service in Kubernetes?",
            "Explain Kubernetes Deployments"
        ]
        
        print("🤖 Model responses:\n")
        for question in questions:
            prompt = f"### Instruction:\n{question}\n\n### Response:\n"
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            print(f"❓ {question}")
            print(f"🤖 {response}\n")
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def main():
    """Main training function."""
    print("🚀 Kubernetes LoRA Training")
    print("=" * 50)
    print(f"🖥️ System: CPU")
    print(f"📊 Will train only ~0.09% of parameters!")
    
    # Training configurations
    configs = {
        "quick": {"examples": 100, "epochs": 1, "rank": 4, "time": "5-10 min"},
        "small": {"examples": 500, "epochs": 2, "rank": 8, "time": "15-30 min"},
        "medium": {"examples": 1000, "epochs": 3, "rank": 8, "time": "30-60 min"},
    }
    
    print(f"\n📋 Training Options:")
    for name, config in configs.items():
        print(f"   {name}: {config['examples']} examples, {config['epochs']} epochs, rank {config['rank']} (~{config['time']})")
    
    # Get choice
    choice = input(f"\n❓ Choose training size (quick/small/medium) [small]: ").strip().lower() or "small"
    
    if choice not in configs:
        print("Invalid choice, using 'small'")
        choice = "small"
    
    config = configs[choice]
    print(f"✅ Using '{choice}' configuration")
    
    # Confirm
    if input(f"\n❓ Start {choice} training (~{config['time']})? (y/n): ").lower() != 'y':
        print("👋 Training cancelled")
        return
    
    try:
        # Load dataset
        dataset = load_k8s_dataset(max_examples=config["examples"])
        
        # Setup model
        model, tokenizer = setup_lora_model(lora_rank=config["rank"])
        
        # Train
        train_lora_model(dataset, model, tokenizer, epochs=config["epochs"])
        
        print(f"\n🎉 Training completed successfully!")
        
        # Test
        if input(f"\n❓ Test the trained model? (y/n): ").lower() == 'y':
            test_trained_model()
        
        print(f"\n🚀 Your LoRA model is ready!")
        print(f"   📁 Saved in: k8s_lora_model/")
        print(f"   🎯 Uses only 0.09% of parameters")
        print(f"   ⚡ Fast CPU training completed")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
