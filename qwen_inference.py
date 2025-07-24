#!/usr/bin/env python3
"""
Qwen Model Inference Script
Use your trained Qwen LoRA models for Kubernetes Q&A
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Available model configurations (same as training script)
QWEN_MODELS = {
    "Qwen2-0.5B": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2-1.5B": "Qwen/Qwen2-1.5B-Instruct", 
    "Qwen2-7B": "Qwen/Qwen2-7B-Instruct"
}

class QwenInference:
    """Qwen model inference handler."""
    
    def __init__(self, lora_model_path):
        """Initialize the inference model."""
        self.lora_model_path = lora_model_path
        self.model = None
        self.tokenizer = None
        self.base_model_name = None
        
        self._detect_base_model()
        self._load_model()
    
    def _detect_base_model(self):
        """Detect which base model was used for training."""
        for model_name, model_path in QWEN_MODELS.items():
            expected_dir = f"qwen_k8s_lora_{model_name.lower().replace('-', '_')}"
            if expected_dir in self.lora_model_path:
                self.base_model_name = model_path
                print(f"🔍 Detected base model: {model_name}")
                return
        
        # Fallback - try to read from model config
        print("⚠️ Could not auto-detect base model, defaulting to Qwen2-0.5B")
        self.base_model_name = QWEN_MODELS["Qwen2-0.5B"]
    
    def _load_model(self):
        """Load the trained LoRA model."""
        print(f"🤖 Loading trained model from {self.lora_model_path}...")
        
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.lora_model_path, trust_remote_code=True)
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def generate_response(self, question, max_length=200, temperature=0.7):
        """Generate response to a question."""
        # Format prompt using Qwen's chat template
        prompt = f"<|im_start|>system\nYou are a Kubernetes expert assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1].strip()
        
        return response

def interactive_chat(model_path):
    """Run interactive chat with the trained model."""
    print("🚀 Starting Qwen Kubernetes Assistant")
    print("=" * 50)
    
    # Load model
    inference = QwenInference(model_path)
    
    print("\n💬 Chat started! Type 'quit' to exit")
    print("📝 Ask any Kubernetes-related questions...")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🤖 Qwen: ", end="", flush=True)
            response = inference.generate_response(question)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_questions(model_path, questions):
    """Answer a batch of predefined questions."""
    inference = QwenInference(model_path)
    
    print(f"\n📋 Answering {len(questions)} questions...")
    print("-" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. ❓ {question}")
        response = inference.generate_response(question)
        print(f"   🤖 {response}")

def find_trained_models():
    """Find available trained models."""
    models = []
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("qwen_k8s_lora_"):
            models.append(item)
    return models

def main():
    """Main function."""
    print("🚀 Qwen Kubernetes Model Inference")
    print("=" * 50)
    
    # Find available models
    available_models = find_trained_models()
    
    if not available_models:
        print("❌ No trained Qwen models found!")
        print("🔧 Train a model first using: python qwen_lora_train.py")
        return
    
    print(f"📁 Found {len(available_models)} trained model(s):")
    for i, model in enumerate(available_models, 1):
        print(f"   {i}. {model}")
    
    # Select model
    if len(available_models) == 1:
        model_path = available_models[0]
        print(f"✅ Using: {model_path}")
    else:
        while True:
            try:
                choice = int(input(f"\n❓ Select model (1-{len(available_models)}): ")) - 1
                if 0 <= choice < len(available_models):
                    model_path = available_models[choice]
                    break
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Please enter a number!")
    
    # Choose mode
    print(f"\n📋 Inference modes:")
    print(f"   1. Interactive chat")
    print(f"   2. Sample questions")
    
    mode = input(f"\n❓ Choose mode (1/2) [1]: ").strip() or "1"
    
    if mode == "2":
        # Sample questions
        sample_questions = [
            "What is a Kubernetes Pod?",
            "How do you create a Service in Kubernetes?",
            "Explain Kubernetes Deployments",
            "How to troubleshoot a CrashLoopBackOff error?",
            "What is the difference between ReplicaSet and Deployment?",
            "How do you expose a pod externally?",
            "What are Kubernetes namespaces?",
            "How do you check pod logs?"
        ]
        batch_questions(model_path, sample_questions)
    else:
        # Interactive chat
        interactive_chat(model_path)

if __name__ == "__main__":
    main() 