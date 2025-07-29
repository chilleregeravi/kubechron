#!/usr/bin/env python3
"""
Kubernetes Domain Verifiers
Multiple verification approaches to improve Qwen model outputs
"""

import yaml
import json
import re
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

@dataclass
class VerificationResult:
    """Result of a verification check."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str]
    suggestions: List[str]
    details: Dict[str, Any]

class YAMLKubernetesVerifier:
    """Verifies YAML syntax and Kubernetes resource semantics."""
    
    def __init__(self):
        # Common K8s resource types and their required fields
        self.k8s_schemas = {
            "Pod": {
                "required": ["apiVersion", "kind", "metadata", "spec"],
                "spec_required": ["containers"],
                "apiVersion": ["v1"]
            },
            "Deployment": {
                "required": ["apiVersion", "kind", "metadata", "spec"],
                "spec_required": ["selector", "template"],
                "apiVersion": ["apps/v1"]
            },
            "Service": {
                "required": ["apiVersion", "kind", "metadata", "spec"],
                "spec_required": ["selector", "ports"],
                "apiVersion": ["v1"]
            },
            "ConfigMap": {
                "required": ["apiVersion", "kind", "metadata"],
                "spec_required": [],
                "apiVersion": ["v1"]
            },
            "Secret": {
                "required": ["apiVersion", "kind", "metadata"],
                "spec_required": [],
                "apiVersion": ["v1"]
            }
        }
    
    def verify_yaml_syntax(self, yaml_text: str) -> VerificationResult:
        """Verify YAML syntax is valid."""
        errors = []
        try:
            parsed = yaml.safe_load(yaml_text)
            if parsed is None:
                errors.append("YAML is empty or invalid")
                return VerificationResult(False, 0.0, errors, [], {})
            
            return VerificationResult(True, 1.0, [], [], {"parsed": parsed})
        
        except yaml.YAMLError as e:
            errors.append(f"YAML syntax error: {str(e)}")
            return VerificationResult(False, 0.0, errors, 
                                    ["Check YAML indentation and syntax"], {})
    
    def verify_kubernetes_semantics(self, yaml_text: str) -> VerificationResult:
        """Verify Kubernetes resource semantics."""
        # First check YAML syntax
        syntax_result = self.verify_yaml_syntax(yaml_text)
        if not syntax_result.is_valid:
            return syntax_result
        
        parsed = syntax_result.details["parsed"]
        errors = []
        suggestions = []
        score = 1.0
        
        # Handle multiple documents
        if isinstance(parsed, list):
            documents = parsed
        else:
            documents = [parsed] if parsed else []
        
        for doc in documents:
            if not isinstance(doc, dict):
                errors.append("Document must be a dictionary")
                score *= 0.5
                continue
            
            # Check basic K8s structure
            kind = doc.get("kind")
            if not kind:
                errors.append("Missing 'kind' field")
                score *= 0.7
                continue
            
            if kind in self.k8s_schemas:
                schema = self.k8s_schemas[kind]
                
                # Check required top-level fields
                for required_field in schema["required"]:
                    if required_field not in doc:
                        errors.append(f"Missing required field: {required_field}")
                        score *= 0.8
                
                # Check apiVersion
                api_version = doc.get("apiVersion")
                if api_version not in schema["apiVersion"]:
                    errors.append(f"Invalid apiVersion '{api_version}' for {kind}. Expected: {schema['apiVersion']}")
                    suggestions.append(f"Use apiVersion: {schema['apiVersion'][0]}")
                    score *= 0.9
                
                # Check spec requirements
                if "spec" in doc and schema["spec_required"]:
                    spec = doc["spec"]
                    for spec_field in schema["spec_required"]:
                        if spec_field not in spec:
                            errors.append(f"Missing required spec field: {spec_field}")
                            score *= 0.8
                
                # Check metadata.name
                metadata = doc.get("metadata", {})
                if not metadata.get("name"):
                    errors.append("metadata.name is required")
                    suggestions.append("Add metadata.name field")
                    score *= 0.9
        
        is_valid = len(errors) == 0
        return VerificationResult(is_valid, score, errors, suggestions, 
                                {"parsed": parsed, "kind_count": len(documents)})
    
    def verify_with_kubectl(self, yaml_text: str) -> VerificationResult:
        """Verify using kubectl dry-run (requires kubectl installed)."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_text)
                f.flush()
                
                # Use kubectl dry-run to validate
                result = subprocess.run([
                    'kubectl', 'apply', '--dry-run=client', '-f', f.name
                ], capture_output=True, text=True, timeout=10)
                
                Path(f.name).unlink()  # Clean up
                
                if result.returncode == 0:
                    return VerificationResult(True, 1.0, [], 
                                            ["YAML passes kubectl validation"], 
                                            {"kubectl_output": result.stdout})
                else:
                    errors = [f"kubectl validation failed: {result.stderr}"]
                    suggestions = ["Fix the kubectl validation errors"]
                    return VerificationResult(False, 0.3, errors, suggestions, 
                                            {"kubectl_error": result.stderr})
        
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            return VerificationResult(False, 0.0, [f"kubectl check failed: {str(e)}"], 
                                    ["Ensure kubectl is installed and accessible"], {})

class FactualKubernetesVerifier:
    """Verifies factual accuracy of Kubernetes information."""
    
    def __init__(self):
        # Knowledge base of K8s facts
        self.k8s_facts = {
            "pod": {
                "description": "smallest deployable unit in Kubernetes",
                "contains": "one or more containers",
                "networking": "shares network and storage",
                "lifecycle": "ephemeral"
            },
            "deployment": {
                "description": "manages replica sets and pods",
                "purpose": "declarative updates to applications",
                "features": ["rolling updates", "rollbacks", "scaling"]
            },
            "service": {
                "description": "stable network endpoint for pods",
                "types": ["ClusterIP", "NodePort", "LoadBalancer", "ExternalName"],
                "purpose": "service discovery and load balancing"
            },
            "namespace": {
                "description": "virtual cluster for resource isolation",
                "purpose": "multi-tenancy and resource organization",
                "default": "default namespace exists by default"
            }
        }
    
    def verify_factual_accuracy(self, text: str, context: str = "") -> VerificationResult:
        """Verify factual accuracy of Kubernetes information."""
        text_lower = text.lower()
        errors = []
        suggestions = []
        score = 1.0
        
        # Check for common misconceptions
        misconceptions = [
            ("pod is permanent", "Pods are ephemeral and can be terminated at any time"),
            ("deployment creates pods directly", "Deployments create ReplicaSets which create Pods"),
            ("services contain pods", "Services route traffic to Pods but don't contain them"),
            ("kubernetes is docker", "Kubernetes orchestrates containers, Docker is one container runtime"),
        ]
        
        for misconception, correction in misconceptions:
            if misconception in text_lower:
                errors.append(f"Potential misconception detected: {misconception}")
                suggestions.append(f"Correction: {correction}")
                score *= 0.7
        
        # Check for technical accuracy
        technical_checks = [
            (re.compile(r"pod.*permanent|permanent.*pod", re.I), 
             "Pods are ephemeral, not permanent"),
            (re.compile(r"service.*contain.*pod|pod.*inside.*service", re.I),
             "Services route to Pods but don't contain them"),
            (re.compile(r"deployment.*directly.*create.*pod", re.I),
             "Deployments create ReplicaSets which create Pods"),
        ]
        
        for pattern, correction in technical_checks:
            if pattern.search(text):
                errors.append("Technical inaccuracy detected")
                suggestions.append(correction)
                score *= 0.8
        
        # Positive scoring for correct facts
        correct_facts = 0
        total_checkable_facts = 0
        
        for resource, facts in self.k8s_facts.items():
            if resource in text_lower:
                total_checkable_facts += 1
                if any(fact in text_lower for fact in str(facts.values())):
                    correct_facts += 1
        
        if total_checkable_facts > 0:
            fact_accuracy = correct_facts / total_checkable_facts
            score = (score + fact_accuracy) / 2
        
        is_valid = len(errors) == 0 and score > 0.7
        return VerificationResult(is_valid, score, errors, suggestions, 
                                {"fact_accuracy": score, "misconceptions_found": len(errors)})

class SelfVerificationPrompting:
    """Uses the model itself to verify and improve its outputs."""
    
    def __init__(self, model_path: str):
        """Initialize with trained Qwen model."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained Qwen model."""
        try:
            # Detect base model from adapter config
            base_model_name = self._detect_base_model()
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"✅ Loaded model for self-verification: {base_model_name}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _detect_base_model(self) -> str:
        """Detect base model from adapter config."""
        config_path = Path(self.model_path) / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return config.get("base_model_name_or_path", "Qwen/Qwen2-0.5B-Instruct")
        return "Qwen/Qwen2-0.5B-Instruct"
    
    def self_verify(self, question: str, answer: str) -> VerificationResult:
        """Use the model to verify its own answer."""
        verification_prompt = f"""You are a Kubernetes expert verifying an answer for accuracy.

Question: {question}

Answer to verify: {answer}

Please analyze this answer and provide:
1. Is the answer accurate? (Yes/No)
2. Confidence score (0-100%)
3. Any errors or inaccuracies
4. Suggestions for improvement

Verification:"""

        try:
            inputs = self.tokenizer(verification_prompt, return_tensors="pt", truncation=True, max_length=1000)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,  # Lower temperature for verification
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            verification = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse verification result (simplified)
            score = self._extract_confidence(verification)
            is_accurate = "yes" in verification.lower() and "accurate" in verification.lower()
            
            errors = []
            suggestions = []
            
            if "error" in verification.lower() or "incorrect" in verification.lower():
                errors.append("Self-verification detected potential issues")
            
            if "suggest" in verification.lower():
                suggestions.append("Self-verification provided improvement suggestions")
            
            return VerificationResult(is_accurate, score, errors, suggestions, 
                                    {"verification_text": verification})
        
        except Exception as e:
            return VerificationResult(False, 0.0, [f"Self-verification failed: {str(e)}"], [], {})
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from verification text."""
        # Look for percentage patterns
        import re
        patterns = [
            r"(\d+)%",
            r"confidence.*?(\d+)",
            r"score.*?(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1)) / 100.0
        
        # Default scoring based on keywords
        if "excellent" in text.lower() or "perfect" in text.lower():
            return 0.95
        elif "good" in text.lower() or "accurate" in text.lower():
            return 0.8
        elif "fair" in text.lower() or "mostly" in text.lower():
            return 0.6
        elif "poor" in text.lower() or "incorrect" in text.lower():
            return 0.3
        else:
            return 0.5

class IterativeVerificationRefinement:
    """Iteratively improves responses using multiple verifiers."""
    
    def __init__(self, model_path: str, max_iterations: int = 3):
        self.yaml_verifier = YAMLKubernetesVerifier()
        self.factual_verifier = FactualKubernetesVerifier()
        self.self_verifier = SelfVerificationPrompting(model_path)
        self.max_iterations = max_iterations
    
    def refine_response(self, question: str, initial_answer: str) -> Dict[str, Any]:
        """Iteratively refine response using verifiers."""
        current_answer = initial_answer
        iteration_history = []
        
        for iteration in range(self.max_iterations):
            print(f"🔍 Verification iteration {iteration + 1}")
            
            # Run all verifiers
            verifications = {}
            
            # YAML verification (if answer contains YAML)
            if "apiVersion" in current_answer or "kind:" in current_answer:
                verifications["yaml"] = self.yaml_verifier.verify_kubernetes_semantics(current_answer)
            
            # Factual verification  
            verifications["factual"] = self.factual_verifier.verify_factual_accuracy(current_answer, question)
            
            # Self verification
            verifications["self"] = self.self_verifier.self_verify(question, current_answer)
            
            # Calculate overall score
            total_score = sum(v.score for v in verifications.values()) / len(verifications)
            all_valid = all(v.is_valid for v in verifications.values())
            
            iteration_result = {
                "iteration": iteration + 1,
                "answer": current_answer,
                "verifications": verifications,
                "overall_score": total_score,
                "all_valid": all_valid
            }
            iteration_history.append(iteration_result)
            
            # If answer is good enough, stop
            if all_valid and total_score > 0.85:
                print(f"✅ Verification passed (score: {total_score:.2f})")
                break
            
            # Collect improvement suggestions
            all_errors = []
            all_suggestions = []
            
            for verifier_name, result in verifications.items():
                all_errors.extend([f"{verifier_name}: {error}" for error in result.errors])
                all_suggestions.extend([f"{verifier_name}: {suggestion}" for suggestion in result.suggestions])
            
            if iteration < self.max_iterations - 1:
                # Generate improved answer using suggestions
                current_answer = self._generate_improved_answer(question, current_answer, all_errors, all_suggestions)
                print(f"🔄 Generated improved answer (iteration {iteration + 2})")
        
        return {
            "final_answer": current_answer,
            "iterations": iteration_history,
            "final_score": iteration_history[-1]["overall_score"],
            "improvement_achieved": len(iteration_history) > 1
        }
    
    def _generate_improved_answer(self, question: str, current_answer: str, errors: List[str], suggestions: List[str]) -> str:
        """Generate an improved answer based on verification feedback."""
        improvement_prompt = f"""You are a Kubernetes expert. Improve the following answer based on the feedback provided.

Original Question: {question}

Current Answer: {current_answer}

Issues Found:
{chr(10).join(f"- {error}" for error in errors[:5])}

Suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in suggestions[:5])}

Please provide an improved answer that addresses these issues:

Improved Answer:"""

        try:
            inputs = self.self_verifier.tokenizer(improvement_prompt, return_tensors="pt", truncation=True, max_length=1200)
            
            with torch.no_grad():
                outputs = self.self_verifier.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.4,
                    do_sample=True,
                    pad_token_id=self.self_verifier.tokenizer.eos_token_id
                )
            
            improved_answer = self.self_verifier.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return improved_answer if improved_answer else current_answer
            
        except Exception as e:
            print(f"❌ Failed to generate improved answer: {e}")
            return current_answer

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Kubernetes Verifiers")
    print("=" * 50)
    
    # Test YAML verifier
    yaml_verifier = YAMLKubernetesVerifier()
    
    test_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: nginx
    image: nginx:1.21
"""
    
    result = yaml_verifier.verify_kubernetes_semantics(test_yaml)
    print(f"YAML Verification: {'✅' if result.is_valid else '❌'} (Score: {result.score:.2f})")
    
    # Test factual verifier
    factual_verifier = FactualKubernetesVerifier()
    
    test_text = "A Pod is the smallest deployable unit in Kubernetes and contains one or more containers."
    result = factual_verifier.verify_factual_accuracy(test_text)
    print(f"Factual Verification: {'✅' if result.is_valid else '❌'} (Score: {result.score:.2f})")
    
    print("\n🎉 Verifier system ready!")
    print("Use with trained Qwen models to improve response quality!") 