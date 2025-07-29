#!/usr/bin/env python3
"""
Verifier-Guided Training Pipeline
Improve Qwen models using verifier feedback through preference learning
"""

import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import numpy as np

from verifiers import (
    YAMLKubernetesVerifier, 
    FactualKubernetesVerifier, 
    SelfVerificationPrompting,
    VerificationResult
)

@dataclass
class ResponsePair:
    """A pair of responses for preference learning."""
    question: str
    response_a: str
    response_b: str
    score_a: float
    score_b: float
    preferred: str  # "A" or "B"
    verification_details: Dict[str, Any]

class VerifierGuidedTrainer:
    """Train models using verifier feedback for continuous improvement."""
    
    def __init__(self, base_model_path: str, verifier_model_path: Optional[str] = None):
        """
        Initialize verifier-guided trainer.
        
        Args:
            base_model_path: Path to base Qwen model or existing LoRA model
            verifier_model_path: Path to existing LoRA model for verification (optional)
        """
        self.base_model_path = base_model_path
        self.verifier_model_path = verifier_model_path
        
        # Initialize verifiers
        self.yaml_verifier = YAMLKubernetesVerifier()
        self.factual_verifier = FactualKubernetesVerifier()
        self.self_verifier = None
        
        # Model components
        self.model = None
        self.tokenizer = None
        
        # Training data
        self.preference_pairs = []
        self.synthetic_data = []
        
        print("🚀 Verifier-Guided Training Pipeline Initialized")
    
    def load_model_for_training(self, model_choice: str = "Qwen2-0.5B") -> Tuple[Any, Any]:
        """Load model for training with LoRA."""
        
        model_configs = {
            "Qwen2-0.5B": "Qwen/Qwen2-0.5B-Instruct",
            "Qwen2-1.5B": "Qwen/Qwen2-1.5B-Instruct",
            "Qwen2-7B": "Qwen/Qwen2-7B-Instruct"
        }
        
        if self.base_model_path in model_configs.values():
            # Loading from HuggingFace
            model_name = self.base_model_path
        elif os.path.exists(self.base_model_path):
            # Loading existing LoRA model
            config_path = Path(self.base_model_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    model_name = config.get("base_model_name_or_path", model_configs[model_choice])
            else:
                model_name = model_configs[model_choice]
        else:
            model_name = model_configs[model_choice]
        
        print(f"📥 Loading model: {model_name}")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply LoRA if loading existing model
        if os.path.exists(self.base_model_path) and (Path(self.base_model_path) / "adapter_config.json").exists():
            print("🔧 Loading existing LoRA adapters...")
            model = PeftModel.from_pretrained(model, self.base_model_path)
        else:
            # Create new LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # Higher rank for better adaptation
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none"
            )
            
            print("🆕 Applying new LoRA configuration...")
            model = get_peft_model(model, lora_config)
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize self-verifier if we have a model
        if self.verifier_model_path:
            try:
                self.self_verifier = SelfVerificationPrompting(self.verifier_model_path)
                print("✅ Self-verifier loaded")
            except Exception as e:
                print(f"⚠️ Failed to load self-verifier: {e}")
        
        return model, tokenizer
    
    def generate_response_candidates(self, questions: List[str], num_candidates: int = 4, 
                                   temperature_range: Tuple[float, float] = (0.5, 1.2)) -> List[Dict[str, Any]]:
        """Generate multiple candidate responses for each question."""
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model_for_training() first.")
        
        candidates = []
        
        for question in questions:
            question_candidates = []
            
            # Generate multiple candidates with different temperatures
            temperatures = np.linspace(temperature_range[0], temperature_range[1], num_candidates)
            
            for i, temp in enumerate(temperatures):
                prompt = f"""You are a Kubernetes expert. Answer the following question clearly and accurately.

Question: {question}

Answer:"""
                
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=400,
                            temperature=temp,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.1
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    question_candidates.append({
                        "candidate_id": i,
                        "question": question,
                        "response": response,
                        "temperature": temp,
                        "generated_at": time.time()
                    })
                    
                except Exception as e:
                    print(f"❌ Failed to generate candidate {i} for question: {str(e)}")
            
            candidates.extend(question_candidates)
            print(f"✅ Generated {len(question_candidates)} candidates for: {question[:50]}...")
        
        return candidates
    
    def evaluate_candidates_with_verifiers(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all candidates using verifiers."""
        
        evaluated_candidates = []
        
        for candidate in candidates:
            question = candidate["question"]
            response = candidate["response"]
            
            # Run verifications
            verifications = {}
            
            # YAML verification (if response contains YAML)
            if "apiVersion" in response or "kind:" in response:
                verifications["yaml"] = self.yaml_verifier.verify_kubernetes_semantics(response)
            
            # Factual verification
            verifications["factual"] = self.factual_verifier.verify_factual_accuracy(response, question)
            
            # Self verification (if available)
            if self.self_verifier:
                verifications["self"] = self.self_verifier.self_verify(question, response)
            
            # Calculate composite score
            scores = [v.score for v in verifications.values()]
            composite_score = np.mean(scores) if scores else 0.0
            
            # Weight by verification type
            weights = {"yaml": 0.4, "factual": 0.4, "self": 0.2}
            weighted_score = sum(
                verifications[vtype].score * weights.get(vtype, 0.3) 
                for vtype in verifications.keys()
            ) / sum(weights.get(vtype, 0.3) for vtype in verifications.keys())
            
            # Add evaluation results
            evaluated_candidate = candidate.copy()
            evaluated_candidate.update({
                "verifications": {
                    name: {
                        "score": result.score,
                        "is_valid": result.is_valid,
                        "errors": result.errors,
                        "suggestions": result.suggestions
                    } for name, result in verifications.items()
                },
                "composite_score": composite_score,
                "weighted_score": weighted_score,
                "quality_tier": self._classify_quality(weighted_score)
            })
            
            evaluated_candidates.append(evaluated_candidate)
        
        return evaluated_candidates
    
    def _classify_quality(self, score: float) -> str:
        """Classify response quality based on score."""
        if score >= 0.85:
            return "excellent"
        elif score >= 0.70:
            return "good"
        elif score >= 0.55:
            return "fair"
        else:
            return "poor"
    
    def create_preference_pairs(self, evaluated_candidates: List[Dict[str, Any]]) -> List[ResponsePair]:
        """Create preference pairs from evaluated candidates."""
        
        # Group candidates by question
        question_groups = {}
        for candidate in evaluated_candidates:
            question = candidate["question"]
            if question not in question_groups:
                question_groups[question] = []
            question_groups[question].append(candidate)
        
        preference_pairs = []
        
        for question, candidates in question_groups.items():
            if len(candidates) < 2:
                continue
            
            # Sort by weighted score
            candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
            
            # Create pairs: best vs others
            best_candidate = candidates[0]
            
            for other_candidate in candidates[1:]:
                # Only create pair if there's a meaningful difference
                score_diff = best_candidate["weighted_score"] - other_candidate["weighted_score"]
                if score_diff >= 0.1:  # Minimum difference threshold
                    
                    pair = ResponsePair(
                        question=question,
                        response_a=best_candidate["response"],
                        response_b=other_candidate["response"],
                        score_a=best_candidate["weighted_score"],
                        score_b=other_candidate["weighted_score"],
                        preferred="A",
                        verification_details={
                            "score_difference": score_diff,
                            "quality_a": best_candidate["quality_tier"],
                            "quality_b": other_candidate["quality_tier"],
                            "verifications_a": best_candidate["verifications"],
                            "verifications_b": other_candidate["verifications"]
                        }
                    )
                    
                    preference_pairs.append(pair)
        
        print(f"✅ Created {len(preference_pairs)} preference pairs")
        return preference_pairs
    
    def create_synthetic_training_data(self, preference_pairs: List[ResponsePair], 
                                     include_explanations: bool = True) -> List[Dict[str, Any]]:
        """Create synthetic training data from preference pairs."""
        
        synthetic_data = []
        
        for pair in preference_pairs:
            # Create training example with preferred response
            if include_explanations:
                # Add reasoning about why this response is better
                reasoning = self._create_quality_reasoning(pair)
                
                training_text = f"""Question: {pair.question}

High-quality answer (Verification Score: {pair.score_a:.2f}):
{pair.response_a}

Quality reasoning: {reasoning}"""
            else:
                training_text = f"""Question: {pair.question}

Answer: {pair.response_a}"""
            
            synthetic_data.append({
                "text": training_text,
                "question": pair.question,
                "response": pair.response_a,
                "score": pair.score_a,
                "quality_tier": self._classify_quality(pair.score_a),
                "data_type": "synthetic_preference"
            })
        
        # Add negative examples (lower quality responses with explanations)
        for pair in preference_pairs[:len(preference_pairs)//3]:  # Subset to avoid overrepresentation
            if pair.score_b < 0.6:  # Only include clearly poor responses
                
                issues = self._identify_response_issues(pair)
                
                training_text = f"""Question: {pair.question}

Lower-quality answer (Verification Score: {pair.score_b:.2f}):
{pair.response_b}

Issues identified: {issues}

Improved answer:
{pair.response_a}"""
                
                synthetic_data.append({
                    "text": training_text,
                    "question": pair.question,
                    "response": pair.response_a,  # Use the better response as target
                    "score": pair.score_a,
                    "quality_tier": "improvement_example",
                    "data_type": "synthetic_improvement"
                })
        
        print(f"✅ Created {len(synthetic_data)} synthetic training examples")
        return synthetic_data
    
    def _create_quality_reasoning(self, pair: ResponsePair) -> str:
        """Create reasoning for why one response is better than another."""
        reasons = []
        
        # Check verification scores
        for vtype in ["yaml", "factual", "self"]:
            if (vtype in pair.verification_details.get("verifications_a", {}) and 
                vtype in pair.verification_details.get("verifications_b", {})):
                
                score_a = pair.verification_details["verifications_a"][vtype]["score"]
                score_b = pair.verification_details["verifications_b"][vtype]["score"]
                
                if score_a > score_b + 0.1:
                    if vtype == "yaml":
                        reasons.append("better YAML syntax and Kubernetes resource structure")
                    elif vtype == "factual":
                        reasons.append("more factually accurate information about Kubernetes")
                    elif vtype == "self":
                        reasons.append("higher confidence and self-assessed accuracy")
        
        # Check errors
        errors_a = sum(len(v.get("errors", [])) for v in pair.verification_details.get("verifications_a", {}).values())
        errors_b = sum(len(v.get("errors", [])) for v in pair.verification_details.get("verifications_b", {}).values())
        
        if errors_b > errors_a:
            reasons.append(f"fewer verification errors ({errors_a} vs {errors_b})")
        
        # Quality tiers
        if pair.verification_details.get("quality_a") != pair.verification_details.get("quality_b"):
            reasons.append(f"higher overall quality tier ({pair.verification_details.get('quality_a')} vs {pair.verification_details.get('quality_b')})")
        
        if not reasons:
            reasons.append("higher overall verification score")
        
        return "This response is preferred because it has " + ", ".join(reasons[:3]) + "."
    
    def _identify_response_issues(self, pair: ResponsePair) -> str:
        """Identify specific issues with the lower-quality response."""
        issues = []
        
        # Check verifications for response B (lower quality)
        verifications_b = pair.verification_details.get("verifications_b", {})
        
        for vtype, details in verifications_b.items():
            if details["errors"]:
                if vtype == "yaml":
                    issues.extend([f"YAML: {error}" for error in details["errors"][:2]])
                elif vtype == "factual":
                    issues.extend([f"Factual: {error}" for error in details["errors"][:2]])
                elif vtype == "self":
                    issues.extend([f"Accuracy: {error}" for error in details["errors"][:2]])
        
        if not issues:
            issues.append(f"Lower verification score ({pair.score_b:.2f} vs {pair.score_a:.2f})")
        
        return "; ".join(issues[:3])
    
    def train_with_verifier_feedback(self, questions: List[str], 
                                   output_dir: str = "qwen_verified_model",
                                   epochs: int = 2,
                                   num_candidates: int = 4) -> str:
        """Complete training pipeline with verifier feedback."""
        
        print("🚀 Starting Verifier-Guided Training Pipeline")
        print("=" * 60)
        
        # Step 1: Load model
        print("📥 Step 1: Loading model...")
        model, tokenizer = self.load_model_for_training()
        
        # Step 2: Generate candidate responses
        print(f"🎲 Step 2: Generating {num_candidates} candidates per question...")
        candidates = self.generate_response_candidates(questions, num_candidates)
        
        # Step 3: Evaluate with verifiers
        print("🔍 Step 3: Evaluating candidates with verifiers...")
        evaluated_candidates = self.evaluate_candidates_with_verifiers(candidates)
        
        # Step 4: Create preference pairs
        print("⚖️ Step 4: Creating preference pairs...")
        preference_pairs = self.create_preference_pairs(evaluated_candidates)
        
        # Step 5: Create synthetic training data
        print("📝 Step 5: Creating synthetic training data...")
        synthetic_data = self.create_synthetic_training_data(preference_pairs)
        
        # Step 6: Train on synthetic data
        print("🏋️ Step 6: Training on verified data...")
        final_model_path = self._train_on_synthetic_data(synthetic_data, output_dir, epochs)
        
        # Step 7: Evaluation
        print("📊 Step 7: Evaluating improved model...")
        self._evaluate_improvements(questions[:5], final_model_path)
        
        print(f"🎉 Training complete! Model saved to: {final_model_path}")
        return final_model_path
    
    def _train_on_synthetic_data(self, synthetic_data: List[Dict[str, Any]], 
                               output_dir: str, epochs: int) -> str:
        """Train the model on synthetic preference data."""
        
        # Create dataset
        dataset = Dataset.from_list(synthetic_data)
        
        # Tokenize
        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=1024,
                return_tensors=None
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,  # Slightly lower for refinement
            weight_decay=0.01,
            warmup_steps=50,
            logging_steps=25,
            save_steps=100,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        def custom_data_collator(features):
            batch = self.tokenizer.pad(
                features,
                padding='max_length',
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
            return batch
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=self.tokenizer,
            data_collator=custom_data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save
        trainer.save_model()
        
        return output_dir
    
    def _evaluate_improvements(self, test_questions: List[str], model_path: str):
        """Evaluate improvements after verifier-guided training."""
        
        print("🧪 Testing improved model...")
        
        # Load improved model for testing
        try:
            from enhanced_qwen_inference import EnhancedQwenInference
            
            enhanced = EnhancedQwenInference(model_path, verification_mode="basic")
            
            total_score = 0
            for i, question in enumerate(test_questions, 1):
                result = enhanced.generate_response(question)
                score = result.get('verification_score', 0.0)
                total_score += score
                
                print(f"   Test {i}: {score:.2f} - {question[:40]}...")
            
            avg_score = total_score / len(test_questions)
            print(f"📊 Average verification score: {avg_score:.2f}")
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")

def main():
    """Example usage of verifier-guided training."""
    
    # Sample questions for training
    sample_questions = [
        "What is a Kubernetes Pod and how does it work?",
        "How do you create a Deployment in Kubernetes?",
        "What's the difference between a Service and an Ingress?",
        "How do you configure resource limits for containers?",
        "What are Kubernetes namespaces used for?",
        "How do you troubleshoot a failing Pod?",
        "What is a ConfigMap and when would you use it?",
        "How do persistent volumes work in Kubernetes?"
    ]
    
    # Initialize trainer
    trainer = VerifierGuidedTrainer(
        base_model_path="qwen_k8s_lora_qwen2_0.5b",  # Use existing trained model
        verifier_model_path="qwen_k8s_lora_qwen2_0.5b"  # Use same model for self-verification
    )
    
    # Run verifier-guided training
    improved_model_path = trainer.train_with_verifier_feedback(
        questions=sample_questions,
        output_dir="qwen_verified_improved",
        epochs=2,
        num_candidates=3
    )
    
    print(f"🎉 Improved model available at: {improved_model_path}")

if __name__ == "__main__":
    main() 