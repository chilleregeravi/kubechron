#!/usr/bin/env python3
"""
Kubernetes Copilot Model Inference Script
Uses the fine-tuned Kubernetes model for answering questions and generating YAML configurations.
"""

import os
import sys
import torch
import logging
from typing import Optional, Dict, Any
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KubernetesModelInference:
    """Inference class for the trained Kubernetes model."""
    
    def __init__(self, model_path: str, max_new_tokens: int = 512):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model directory
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing inference engine from: {model_path}")
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        try:
            from unsloth import FastLanguageModel
            logger.info("Loading model with Unsloth...")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            
        except ImportError:
            logger.warning("Unsloth not available, falling back to transformers...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        logger.info("Model loaded successfully!")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """
        Format the prompt for the model.
        
        Args:
            instruction: The instruction/question
            input_text: Optional input context
            
        Returns:
            Formatted prompt string
        """
        if input_text.strip():
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response to the given instruction.
        
        Args:
            instruction: The instruction/question
            input_text: Optional input context
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response
        """
        # Format the prompt
        prompt = self.format_prompt(instruction, input_text)
        
        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part (after "### Response:")
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        return response
    
    def ask_kubernetes_question(self, question: str) -> str:
        """
        Ask a Kubernetes-related question.
        
        Args:
            question: The question to ask
            
        Returns:
            Generated answer
        """
        instruction = "Explain the following Kubernetes concept or answer the question:"
        return self.generate_response(instruction, question)
    
    def explain_yaml(self, yaml_content: str) -> str:
        """
        Explain a Kubernetes YAML configuration.
        
        Args:
            yaml_content: The YAML content to explain
            
        Returns:
            Explanation of the YAML
        """
        instruction = "Explain the following Kubernetes YAML configuration:"
        return self.generate_response(instruction, yaml_content)
    
    def generate_yaml(self, description: str) -> str:
        """
        Generate YAML configuration based on description.
        
        Args:
            description: Description of what YAML to generate
            
        Returns:
            Generated YAML configuration
        """
        instruction = f"Generate a Kubernetes YAML configuration for: {description}"
        return self.generate_response(instruction)


def interactive_mode(inference_engine: KubernetesModelInference):
    """Run interactive mode for chatting with the model."""
    print("\n🚀 Kubernetes Copilot - Interactive Mode")
    print("=" * 50)
    print("Ask questions about Kubernetes or paste YAML configurations for explanation.")
    print("Commands:")
    print("  /yaml - Switch to YAML explanation mode")
    print("  /generate - Generate YAML configuration")
    print("  /question - Switch to Q&A mode")
    print("  /quit - Exit")
    print("=" * 50)
    
    mode = "question"
    
    while True:
        try:
            if mode == "question":
                user_input = input("\n❓ Ask a Kubernetes question: ").strip()
            elif mode == "yaml":
                user_input = input("\n📝 Paste YAML to explain: ").strip()
            elif mode == "generate":
                user_input = input("\n🏗️  Describe what YAML to generate: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                if command == '/quit':
                    print("👋 Goodbye!")
                    break
                elif command == '/yaml':
                    mode = "yaml"
                    print("📝 Switched to YAML explanation mode")
                    continue
                elif command == '/generate':
                    mode = "generate"
                    print("🏗️ Switched to YAML generation mode")
                    continue
                elif command == '/question':
                    mode = "question"
                    print("❓ Switched to Q&A mode")
                    continue
                else:
                    print("❌ Unknown command")
                    continue
            
            # Generate response based on mode
            print("\n🤖 Generating response...")
            
            if mode == "question":
                response = inference_engine.ask_kubernetes_question(user_input)
            elif mode == "yaml":
                response = inference_engine.explain_yaml(user_input)
            elif mode == "generate":
                response = inference_engine.generate_yaml(user_input)
            
            print(f"\n💡 Response:\n{response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


def main():
    """Main function for the inference script."""
    parser = argparse.ArgumentParser(
        description="Run inference with the trained Kubernetes model"
    )
    parser.add_argument(
        "--model-path",
        default="k8s_model_output",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--question",
        help="Single question to ask (non-interactive mode)"
    )
    parser.add_argument(
        "--yaml-file",
        help="Path to YAML file to explain"
    )
    parser.add_argument(
        "--generate",
        help="Description of YAML to generate"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at: {args.model_path}")
        print("Please train a model first using train_k8s_model.py")
        sys.exit(1)
    
    # Initialize inference engine
    try:
        inference_engine = KubernetesModelInference(
            model_path=args.model_path,
            max_new_tokens=args.max_tokens
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Handle different modes
    if args.question:
        # Single question mode
        print(f"❓ Question: {args.question}")
        response = inference_engine.ask_kubernetes_question(args.question)
        print(f"💡 Answer: {response}")
        
    elif args.yaml_file:
        # YAML explanation mode
        if not os.path.exists(args.yaml_file):
            print(f"❌ YAML file not found: {args.yaml_file}")
            sys.exit(1)
        
        with open(args.yaml_file, 'r') as f:
            yaml_content = f.read()
        
        print(f"📝 Explaining YAML from: {args.yaml_file}")
        response = inference_engine.explain_yaml(yaml_content)
        print(f"💡 Explanation: {response}")
        
    elif args.generate:
        # YAML generation mode
        print(f"🏗️ Generating YAML for: {args.generate}")
        response = inference_engine.generate_yaml(args.generate)
        print(f"📄 Generated YAML:\n{response}")
        
    else:
        # Interactive mode
        interactive_mode(inference_engine)


if __name__ == "__main__":
    main() 