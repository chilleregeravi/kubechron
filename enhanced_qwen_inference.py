#!/usr/bin/env python3
"""
Enhanced Qwen Inference with Verifiers
High-quality Kubernetes Q&A with verification and iterative refinement
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from verifiers import (
    YAMLKubernetesVerifier, 
    FactualKubernetesVerifier, 
    SelfVerificationPrompting,
    IterativeVerificationRefinement,
    VerificationResult
)

class EnhancedQwenInference:
    """Enhanced Qwen inference with multiple verification strategies."""
    
    def __init__(self, lora_model_path: str, verification_mode: str = "full"):
        """
        Initialize enhanced inference.
        
        Args:
            lora_model_path: Path to trained LoRA model
            verification_mode: 'none', 'basic', 'iterative', or 'full'
        """
        self.lora_model_path = lora_model_path
        self.verification_mode = verification_mode
        
        # Load model
        self.model = None
        self.tokenizer = None
        self.base_model_name = None
        self._load_model()
        
        # Initialize verifiers based on mode
        self.verifiers = {}
        if verification_mode != "none":
            self._initialize_verifiers()
    
    def _load_model(self):
        """Load the trained Qwen model."""
        try:
            # Detect base model
            self.base_model_name = self._detect_base_model()
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Enhanced Qwen model loaded: {self.base_model_name}")
            print(f"🔍 Verification mode: {self.verification_mode}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _detect_base_model(self) -> str:
        """Detect base model from adapter config."""
        config_path = Path(self.lora_model_path) / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return config.get("base_model_name_or_path", "Qwen/Qwen2-0.5B-Instruct")
        return "Qwen/Qwen2-0.5B-Instruct"
    
    def _initialize_verifiers(self):
        """Initialize verifiers based on verification mode."""
        try:
            if self.verification_mode in ["basic", "full"]:
                self.verifiers["yaml"] = YAMLKubernetesVerifier()
                self.verifiers["factual"] = FactualKubernetesVerifier()
                print("✅ Basic verifiers initialized")
            
            if self.verification_mode in ["iterative", "full"]:
                self.verifiers["self"] = SelfVerificationPrompting(self.lora_model_path)
                self.verifiers["iterative"] = IterativeVerificationRefinement(
                    self.lora_model_path, max_iterations=3
                )
                print("✅ Advanced verifiers initialized")
                
        except Exception as e:
            print(f"⚠️ Some verifiers failed to load: {e}")
            print("Continuing with basic inference...")
    
    def generate_response(self, question: str, max_length: int = 500, 
                         temperature: float = 0.7, use_verification: bool = True) -> Dict[str, Any]:
        """Generate a response with optional verification."""
        
        # Create the prompt
        prompt = f"""You are a Kubernetes expert. Answer the following question clearly and accurately.

Question: {question}

Answer:"""
        
        start_time = time.time()
        
        try:
            # Generate initial response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            initial_response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            generation_time = time.time() - start_time
            
            # Apply verification if enabled
            if use_verification and self.verification_mode != "none":
                return self._verify_and_enhance_response(
                    question, initial_response, generation_time
                )
            else:
                return {
                    "question": question,
                    "answer": initial_response,
                    "verification_mode": "none",
                    "generation_time": generation_time,
                    "total_time": generation_time,
                    "verification_score": None,
                    "iterations": 1
                }
        
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error generating response: {str(e)}",
                "verification_mode": self.verification_mode,
                "generation_time": time.time() - start_time,
                "total_time": time.time() - start_time,
                "verification_score": 0.0,
                "error": str(e)
            }
    
    def _verify_and_enhance_response(self, question: str, initial_response: str, 
                                   generation_time: float) -> Dict[str, Any]:
        """Apply verification and enhancement to the response."""
        
        verification_start = time.time()
        
        if self.verification_mode == "iterative" and "iterative" in self.verifiers:
            # Use iterative refinement
            refinement_result = self.verifiers["iterative"].refine_response(question, initial_response)
            
            return {
                "question": question,
                "answer": refinement_result["final_answer"],
                "initial_answer": initial_response,
                "verification_mode": "iterative",
                "generation_time": generation_time,
                "verification_time": time.time() - verification_start,
                "total_time": time.time() - verification_start + generation_time,
                "verification_score": refinement_result["final_score"],
                "iterations": len(refinement_result["iterations"]),
                "improvement_achieved": refinement_result["improvement_achieved"],
                "iteration_details": refinement_result["iterations"]
            }
            
        else:
            # Use basic verification
            verifications = {}
            
            # YAML verification
            if "yaml" in self.verifiers and ("apiVersion" in initial_response or "kind:" in initial_response):
                verifications["yaml"] = self.verifiers["yaml"].verify_kubernetes_semantics(initial_response)
            
            # Factual verification
            if "factual" in self.verifiers:
                verifications["factual"] = self.verifiers["factual"].verify_factual_accuracy(initial_response, question)
            
            # Self verification
            if "self" in self.verifiers:
                verifications["self"] = self.verifiers["self"].self_verify(question, initial_response)
            
            # Calculate overall score
            if verifications:
                total_score = sum(v.score for v in verifications.values()) / len(verifications)
                all_valid = all(v.is_valid for v in verifications.values())
            else:
                total_score = 1.0
                all_valid = True
            
            return {
                "question": question,
                "answer": initial_response,
                "verification_mode": "basic",
                "generation_time": generation_time,
                "verification_time": time.time() - verification_start,
                "total_time": time.time() - verification_start + generation_time,
                "verification_score": total_score,
                "all_verifications_passed": all_valid,
                "verification_details": {
                    name: {
                        "valid": result.is_valid,
                        "score": result.score,
                        "errors": result.errors,
                        "suggestions": result.suggestions
                    } for name, result in verifications.items()
                },
                "iterations": 1
            }
    
    def interactive_chat(self):
        """Interactive chat session with verification."""
        print(f"\n🤖 Enhanced Qwen Kubernetes Assistant")
        print(f"📋 Model: {self.base_model_name}")
        print(f"🔍 Verification: {self.verification_mode}")
        print("💬 Type 'quit' to exit, 'mode <mode>' to change verification mode")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower().startswith('mode '):
                    new_mode = question.split(' ', 1)[1]
                    if new_mode in ['none', 'basic', 'iterative', 'full']:
                        self.verification_mode = new_mode
                        self._initialize_verifiers()
                        print(f"✅ Verification mode changed to: {new_mode}")
                    else:
                        print("❌ Invalid mode. Options: none, basic, iterative, full")
                    continue
                
                if not question:
                    continue
                
                print("🔄 Generating response...")
                
                result = self.generate_response(question)
                
                # Display result
                print(f"\n🤖 **Answer:**\n{result['answer']}")
                
                # Display verification info
                if result.get('verification_score') is not None:
                    score = result['verification_score']
                    score_emoji = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                    print(f"\n{score_emoji} **Verification Score:** {score:.2f}")
                    
                    if result.get('iterations', 1) > 1:
                        print(f"🔄 **Iterations:** {result['iterations']}")
                        if result.get('improvement_achieved'):
                            print("✨ **Response was improved through verification!**")
                
                # Display timing info
                print(f"⏱️ **Timing:** {result['total_time']:.1f}s total")
                if result.get('verification_time'):
                    print(f"   (Generation: {result['generation_time']:.1f}s, Verification: {result['verification_time']:.1f}s)")
                
                # Display verification details for basic mode
                if self.verification_mode == "basic" and result.get('verification_details'):
                    print("\n🔍 **Verification Results:**")
                    for name, details in result['verification_details'].items():
                        status = "✅" if details['valid'] else "❌"
                        print(f"   {status} {name.title()}: {details['score']:.2f}")
                        
                        if details['errors']:
                            for error in details['errors'][:2]:  # Show first 2 errors
                                print(f"      ⚠️ {error}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def batch_evaluate(self, questions: List[str], output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Evaluate a batch of questions."""
        print(f"📊 Evaluating {len(questions)} questions...")
        print(f"🔍 Verification mode: {self.verification_mode}")
        
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n🔄 Question {i}/{len(questions)}: {question[:50]}...")
            
            result = self.generate_response(question)
            results.append(result)
            
            score = result.get('verification_score', 0.0)
            score_emoji = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
            print(f"   {score_emoji} Score: {score:.2f}, Time: {result['total_time']:.1f}s")
        
        # Calculate summary statistics
        scores = [r.get('verification_score', 0.0) for r in results if r.get('verification_score') is not None]
        times = [r['total_time'] for r in results]
        
        summary = {
            "total_questions": len(questions),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "average_time": sum(times) / len(times),
            "high_quality_responses": sum(1 for s in scores if s > 0.8),
            "verification_mode": self.verification_mode
        }
        
        print(f"\n📊 **Evaluation Summary:**")
        print(f"   🎯 Average Score: {summary['average_score']:.2f}")
        print(f"   ⏱️ Average Time: {summary['average_time']:.1f}s")
        print(f"   🟢 High Quality: {summary['high_quality_responses']}/{len(questions)} ({summary['high_quality_responses']/len(questions)*100:.1f}%)")
        
        # Save results if requested
        if output_file:
            output_data = {
                "summary": summary,
                "results": results
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"💾 Results saved to: {output_file}")
        
        return results

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Qwen Inference with Verifiers")
    parser.add_argument("--model", required=True, help="Path to trained LoRA model")
    parser.add_argument("--mode", choices=["none", "basic", "iterative", "full"], 
                       default="full", help="Verification mode")
    parser.add_argument("--question", help="Single question to answer")
    parser.add_argument("--batch", help="File containing questions (one per line)")
    parser.add_argument("--output", help="Output file for batch results")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    
    args = parser.parse_args()
    
    # Initialize enhanced inference
    enhanced = EnhancedQwenInference(args.model, args.mode)
    
    if args.interactive:
        enhanced.interactive_chat()
    elif args.question:
        result = enhanced.generate_response(args.question)
        print(f"\n🤖 **Answer:**\n{result['answer']}")
        if result.get('verification_score'):
            print(f"\n🔍 **Score:** {result['verification_score']:.2f}")
    elif args.batch:
        with open(args.batch, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        enhanced.batch_evaluate(questions, args.output)
    else:
        enhanced.interactive_chat()

if __name__ == "__main__":
    main() 