#!/usr/bin/env python3
"""
Test Qwen Model Setup
Quick verification that Qwen models can be loaded and LoRA can be applied
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

print("🧪 Testing Qwen Model Setup")
print("=" * 40)

# System info
print(f"🖥️ System: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"📦 PyTorch: {torch.__version__}")

# Test model loading
model_name = "Qwen/Qwen2-0.5B-Instruct"  # Start with smallest model
print(f"\n🤖 Testing {model_name}...")

try:
    # Load model and tokenizer
    print("   Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Parameters: {model.num_parameters():,}")
    
    # Test LoRA setup
    print("\n🔧 Testing LoRA setup...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,  # Small rank for testing
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    print("✅ LoRA adapters added successfully!")
    print("📈 Trainable parameters:")
    model.print_trainable_parameters()
    
    # Test inference
    print("\n🧠 Testing inference...")
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("✅ Inference test successful!")
    print(f"📝 Sample response: {response[-50:]}")  # Show last 50 chars
    
    print(f"\n🎉 All tests passed!")
    print(f"✅ Your system is ready for Qwen training!")
    print(f"\n📋 Next steps:")
    print(f"   1. Run: python qwen_lora_train.py")
    print(f"   2. Choose your preferred model size")
    print(f"   3. Start with 'quick' training for testing")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print(f"\n🔧 Possible solutions:")
    print(f"   1. Install required packages: pip install transformers peft datasets")
    print(f"   2. Make sure you have internet connection for model download")
    print(f"   3. For GPU issues, install proper CUDA/PyTorch version")
    
    import traceback
    traceback.print_exc() 