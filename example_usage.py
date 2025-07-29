#!/usr/bin/env python3
"""
Example usage of the trained Kubernetes Copilot model.
Demonstrates different ways to interact with the model.
"""

import os
import sys
from inference_k8s_model import KubernetesModelInference

def example_questions():
    """Example Kubernetes questions."""
    return [
        "What is a Kubernetes Pod?",
        "How do I create a Service in Kubernetes?",
        "What's the difference between Deployment and StatefulSet?",
        "How do I set up Ingress in Kubernetes?",
        "What are ConfigMaps used for?",
        "How do I troubleshoot a Pod that won't start?",
    ]

def example_yaml_configs():
    """Example YAML configurations to explain."""
    return {
        "nginx_deployment": """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
""",
        "service": """
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: ClusterIP
""",
        "configmap": """
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_host: "db.example.com"
  database_port: "5432"
  app_env: "production"
"""
    }

def example_generation_requests():
    """Example YAML generation requests."""
    return [
        "a simple nginx deployment with 2 replicas",
        "a Redis StatefulSet with persistent storage",
        "a LoadBalancer service for a web application",
        "a ConfigMap for database configuration",
        "a Secret for storing API keys",
        "an Ingress resource for routing HTTP traffic",
    ]

def run_examples():
    """Run all examples."""
    model_path = "k8s_model_output"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        print(f"Expected location: {model_path}")
        print("\nTo train a model:")
        print("1. First run: python src/main.py  # Generate dataset")
        print("2. Then run: python train_k8s_model.py  # Train model")
        return
    
    print("🚀 Kubernetes Copilot - Example Usage")
    print("=" * 50)
    
    # Initialize inference engine
    try:
        print("Loading model...")
        inference_engine = KubernetesModelInference(model_path)
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Example 1: Question Answering
    print("📚 Example 1: Question Answering")
    print("-" * 30)
    
    questions = example_questions()
    for i, question in enumerate(questions[:3], 1):  # Show first 3 questions
        print(f"\n{i}. ❓ Question: {question}")
        try:
            answer = inference_engine.ask_kubernetes_question(question)
            print(f"💡 Answer: {answer[:200]}..." if len(answer) > 200 else f"💡 Answer: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 2: YAML Explanation
    print("📝 Example 2: YAML Explanation")
    print("-" * 30)
    
    yaml_configs = example_yaml_configs()
    for name, yaml_content in list(yaml_configs.items())[:2]:  # Show first 2 configs
        print(f"\n🔍 Explaining {name.replace('_', ' ').title()}:")
        print("YAML:")
        print(yaml_content.strip())
        print("\n💡 Explanation:")
        try:
            explanation = inference_engine.explain_yaml(yaml_content)
            print(explanation[:300] + "..." if len(explanation) > 300 else explanation)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 3: YAML Generation
    print("🏗️ Example 3: YAML Generation")
    print("-" * 30)
    
    generation_requests = example_generation_requests()
    for i, request in enumerate(generation_requests[:2], 1):  # Show first 2 requests
        print(f"\n{i}. 🎯 Generate: {request}")
        try:
            generated_yaml = inference_engine.generate_yaml(request)
            print("📄 Generated YAML:")
            print(generated_yaml)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Examples completed!")
    print("\nFor interactive mode, run:")
    print("python inference_k8s_model.py")

def main():
    """Main function."""
    try:
        run_examples()
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 