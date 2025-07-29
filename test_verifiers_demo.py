#!/usr/bin/env python3
"""
Demo: Test Verifier System with Trained Qwen Model
Showcase different verification approaches for improving model outputs
"""

import os
from pathlib import Path

def test_basic_verifiers():
    """Test basic YAML and factual verifiers."""
    print("🧪 Testing Basic Verifiers")
    print("=" * 40)
    
    from verifiers import YAMLKubernetesVerifier, FactualKubernetesVerifier
    
    # Test YAML verifier
    yaml_verifier = YAMLKubernetesVerifier()
    
    print("\n📝 Testing YAML Verifier:")
    
    # Good YAML
    good_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    ports:
    - containerPort: 80
"""
    
    result = yaml_verifier.verify_kubernetes_semantics(good_yaml)
    print(f"✅ Good YAML - Valid: {result.is_valid}, Score: {result.score:.2f}")
    
    # Bad YAML
    bad_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  # Missing required containers field
  volumes:
  - name: test
"""
    
    result = yaml_verifier.verify_kubernetes_semantics(bad_yaml)
    print(f"❌ Bad YAML - Valid: {result.is_valid}, Score: {result.score:.2f}")
    if result.errors:
        print(f"   Errors: {result.errors[0]}")
    
    # Test factual verifier
    print("\n🎯 Testing Factual Verifier:")
    
    factual_verifier = FactualKubernetesVerifier()
    
    good_text = "A Pod is the smallest deployable unit in Kubernetes and contains one or more containers that share network and storage."
    result = factual_verifier.verify_factual_accuracy(good_text)
    print(f"✅ Good facts - Valid: {result.is_valid}, Score: {result.score:.2f}")
    
    bad_text = "Pods are permanent resources in Kubernetes and services contain pods inside them."
    result = factual_verifier.verify_factual_accuracy(bad_text)
    print(f"❌ Bad facts - Valid: {result.is_valid}, Score: {result.score:.2f}")
    if result.errors:
        print(f"   Issues: {result.errors[0]}")

def test_enhanced_inference():
    """Test enhanced inference with verifiers."""
    print("\n\n🤖 Testing Enhanced Inference")
    print("=" * 40)
    
    # Check if we have a trained model
    model_path = "qwen_k8s_lora_qwen2_0.5b"
    if not os.path.exists(model_path):
        print("❌ No trained model found. Please train a model first using:")
        print("   python qwen_lora_train.py")
        return
    
    try:
        from enhanced_qwen_inference import EnhancedQwenInference
        
        print(f"📥 Loading enhanced inference with model: {model_path}")
        
        # Test different verification modes
        modes = ["none", "basic", "iterative"]
        test_questions = [
            "What is a Kubernetes Pod?",
            "How do I create a deployment YAML file?",
        ]
        
        for mode in modes:
            if mode == "iterative":
                print(f"\n🔍 Testing verification mode: {mode} (this may take longer...)")
            else:
                print(f"\n🔍 Testing verification mode: {mode}")
            
            try:
                enhanced = EnhancedQwenInference(model_path, verification_mode=mode)
                
                for question in test_questions:
                    print(f"\n❓ Question: {question}")
                    
                    result = enhanced.generate_response(question, max_length=200)
                    
                    print(f"🤖 Answer: {result['answer'][:150]}{'...' if len(result['answer']) > 150 else ''}")
                    
                    if result.get('verification_score') is not None:
                        score = result['verification_score']
                        score_emoji = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                        print(f"{score_emoji} Score: {score:.2f}")
                        
                        if result.get('iterations', 1) > 1:
                            print(f"🔄 Iterations: {result['iterations']}")
                        
                        print(f"⏱️ Time: {result['total_time']:.1f}s")
                    
                    # Only test first question for iterative mode (to save time)
                    if mode == "iterative":
                        break
                        
            except Exception as e:
                print(f"❌ Error testing {mode} mode: {e}")
                continue
    
    except ImportError as e:
        print(f"❌ Failed to import enhanced inference: {e}")
        print("Make sure all verifier dependencies are installed.")

def test_self_verification():
    """Test self-verification with trained model."""
    print("\n\n🔍 Testing Self-Verification")
    print("=" * 40)
    
    model_path = "qwen_k8s_lora_qwen2_0.5b"
    if not os.path.exists(model_path):
        print("❌ No trained model found. Please train a model first.")
        return
    
    try:
        from verifiers import SelfVerificationPrompting
        
        print(f"📥 Loading self-verification with model: {model_path}")
        self_verifier = SelfVerificationPrompting(model_path)
        
        # Test self-verification
        question = "What is a Kubernetes Service?"
        answer = "A Kubernetes Service is a stable network endpoint that provides load balancing and service discovery for Pods."
        
        print(f"\n❓ Question: {question}")
        print(f"🤖 Answer: {answer}")
        print("🔍 Self-verifying...")
        
        result = self_verifier.self_verify(question, answer)
        
        print(f"✅ Self-verification result:")
        print(f"   Valid: {result.is_valid}")
        print(f"   Score: {result.score:.2f}")
        
        if result.details.get('verification_text'):
            verification_text = result.details['verification_text'][:200]
            print(f"   Analysis: {verification_text}...")
    
    except Exception as e:
        print(f"❌ Error in self-verification test: {e}")

def demo_verifier_workflow():
    """Demonstrate a complete verifier workflow."""
    print("\n\n🎯 Complete Verifier Workflow Demo")
    print("=" * 40)
    
    # Sample responses to evaluate
    responses = [
        {
            "question": "How do I create a simple Pod?",
            "answer": """To create a simple Pod in Kubernetes, you need to create a YAML file:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    ports:
    - containerPort: 80
```

Save this as pod.yaml and run: kubectl apply -f pod.yaml"""
        },
        {
            "question": "What are Kubernetes Services?",
            "answer": "Services are permanent containers that store Pods inside them and provide networking. They never change and contain the actual application code."
        }
    ]
    
    from verifiers import YAMLKubernetesVerifier, FactualKubernetesVerifier
    
    yaml_verifier = YAMLKubernetesVerifier()
    factual_verifier = FactualKubernetesVerifier()
    
    for i, response in enumerate(responses, 1):
        print(f"\n📝 Response {i}:")
        print(f"Q: {response['question']}")
        print(f"A: {response['answer'][:100]}...")
        
        # YAML verification
        if "apiVersion" in response['answer']:
            yaml_result = yaml_verifier.verify_kubernetes_semantics(response['answer'])
            yaml_emoji = "✅" if yaml_result.is_valid else "❌"
            print(f"   {yaml_emoji} YAML: {yaml_result.score:.2f}")
        
        # Factual verification
        factual_result = factual_verifier.verify_factual_accuracy(response['answer'])
        factual_emoji = "✅" if factual_result.is_valid else "❌"
        print(f"   {factual_emoji} Facts: {factual_result.score:.2f}")
        
        if factual_result.errors:
            print(f"   ⚠️ Issues: {factual_result.errors[0]}")
        
        # Overall assessment
        overall_score = (yaml_result.score if "apiVersion" in response['answer'] else 0.8 + factual_result.score) / 2
        overall_emoji = "🟢" if overall_score > 0.8 else "🟡" if overall_score > 0.6 else "🔴"
        print(f"   {overall_emoji} Overall: {overall_score:.2f}")

def main():
    """Run all verifier demonstrations."""
    print("🚀 Verifier System Demonstration")
    print("This demo showcases how verifiers can improve model outputs")
    print("=" * 60)
    
    # Test 1: Basic verifiers
    test_basic_verifiers()
    
    # Test 2: Enhanced inference (if model exists)
    test_enhanced_inference()
    
    # Test 3: Self-verification (if model exists)
    test_self_verification()
    
    # Test 4: Complete workflow
    demo_verifier_workflow()
    
    print("\n\n🎉 Verifier System Demo Complete!")
    print("=" * 60)
    print("💡 Key Benefits of Verifiers:")
    print("   ✅ Automatic quality assessment")
    print("   ✅ YAML syntax and semantic validation")
    print("   ✅ Factual accuracy checking")
    print("   ✅ Self-verification and improvement")
    print("   ✅ Iterative refinement for better responses")
    print("\n📚 Next Steps:")
    print("   • Use enhanced_qwen_inference.py for verified responses")
    print("   • Run verifier_guided_training.py to improve your model")
    print("   • Integrate verifiers into your own applications")

if __name__ == "__main__":
    main() 