#!/usr/bin/env python3
"""
Kubernetes Copilot Model Training Script
Fine-tunes a language model on Kubernetes documentation and YAML configurations using Unsloth.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import torch
from datasets import Dataset
from transformers import TrainingArguments
import gc

# Try to import Unsloth with fallback to transformers
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    UNSLOTH_AVAILABLE = True
    logger_message = "✅ Unsloth available - using optimized training"
except (ImportError, NotImplementedError) as e:
    logger_message = f"⚠️  Unsloth not available: {e}\nFalling back to standard transformers training..."
    UNSLOTH_AVAILABLE = False
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        Trainer,
        DataCollatorForLanguageModeling
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print the import status
print(logger_message)

class KubernetesModelTrainer:
    """Kubernetes-focused language model trainer using Unsloth."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        dataset_path: str = "src/output/k8s_dataset.jsonl",
        output_dir: str = "k8s_model_output",
        max_seq_length: int = 2048
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: Base model to fine-tune
            dataset_path: Path to the JSONL dataset
            output_dir: Directory to save the trained model
            max_seq_length: Maximum sequence length for training
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        
        # Unsloth components (will be initialized later)
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing trainer for model: {model_name}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Output directory: {output_dir}")
        
    def load_and_prepare_dataset(self) -> Dataset:
        """
        Load the JSONL dataset and prepare it for training.
        
        Returns:
            Prepared Hugging Face Dataset
        """
        logger.info("Loading dataset...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # Load JSONL data
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} training examples")
        
        # Convert to instruction format expected by Unsloth
        formatted_data = []
        for item in data:
            # Format as instruction-following conversation
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            
            # Create the full conversation text
            if input_text.strip():
                conversation = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            else:
                conversation = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
            
            formatted_data.append({
                "text": conversation,
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "source_type": item.get("metadata", {}).get("source_type", "unknown")
            })
        
        # Create Hugging Face Dataset
        dataset = Dataset.from_list(formatted_data)
        
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        
        # Log some statistics
        source_types = {}
        for item in formatted_data:
            source_type = item["source_type"]
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        logger.info("Dataset composition:")
        for source_type, count in source_types.items():
            logger.info(f"  {source_type}: {count} examples ({count/len(formatted_data)*100:.1f}%)")
        
        return dataset
    
    def initialize_model(self):
        """Initialize the model and tokenizer (Unsloth or standard transformers)."""
        logger.info(f"Loading model: {self.model_name}")
        
        if UNSLOTH_AVAILABLE:
            # Initialize model with Unsloth
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=True,  # Use 4-bit quantization for efficiency
            )
            
            # Add LoRA adapters for efficient fine-tuning
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=42,
            )
            
            logger.info("Model initialized successfully with Unsloth and LoRA adapters")
        else:
            # Fallback to standard transformers
            logger.info("Using standard transformers (CPU/fallback mode)")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            # Resize embeddings if needed
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info("Model initialized successfully with standard transformers")
        
    def format_dataset_for_training(self, dataset: Dataset) -> Dataset:
        """
        Format the dataset for training with proper tokenization.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Tokenizing dataset...")
        
        if UNSLOTH_AVAILABLE:
            # For Unsloth, we can return the dataset as-is since SFTTrainer handles tokenization
            logger.info("Using raw dataset for SFTTrainer")
            return dataset
        else:
            # For standard transformers, we need to tokenize the dataset
            def tokenize_function(examples):
                # Tokenize the conversation text
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=self.max_seq_length,
                    return_tensors=None,
                )
                
                # For instruction tuning, we want the model to predict the response
                # So we set labels = input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized
            
            # Tokenize in batches for efficiency
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing dataset"
            )
            
            logger.info("Dataset tokenization completed")
            return tokenized_dataset
        
        logger.info(f"Tokenized dataset: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train_model(self, dataset: Dataset, validation_split: float = 0.1):
        """
        Train the model on the Kubernetes dataset.
        
        Args:
            dataset: Prepared dataset
            validation_split: Fraction of data to use for validation
        """
        logger.info("Starting model training...")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Split dataset for validation
        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
            logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} validation")
        else:
            train_dataset = dataset
            eval_dataset = None
            logger.info(f"Training on full dataset: {len(train_dataset)} examples")
        
        # Training arguments optimized for instruction tuning
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            
            # Training schedule
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            
            # Optimization
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=10,
            max_grad_norm=1.0,
            
            # Logging and saving
            logging_steps=10,
            save_steps=100,
            eval_steps=100 if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            
            # Performance
            dataloader_num_workers=4,
            remove_unused_columns=False,
            
            # Mixed precision training
            fp16=True,
            
            # Other settings
            seed=42,
            report_to="none",  # Disable wandb logging
        )
        
        # Choose appropriate trainer based on availability
        if UNSLOTH_AVAILABLE:
            # Use SFTTrainer for Unsloth
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
            )
            logger.info("Using SFTTrainer for Unsloth optimized training")
        else:
            # Use standard Trainer with data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            logger.info("Using standard Transformers Trainer")
        
        # Clear GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("Starting training...")
        
        # Train the model
        try:
            trainer.train()
            logger.info("Training completed successfully!")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # Save the final model
        logger.info("Saving trained model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metrics
        if hasattr(trainer.state, 'log_history'):
            metrics_path = os.path.join(self.output_dir, "training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(trainer.state.log_history, f, indent=2)
            logger.info(f"Training metrics saved to {metrics_path}")
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        try:
            logger.info("=== Starting Kubernetes Model Training Pipeline ===")
            
            # Step 1: Load and prepare dataset
            dataset = self.load_and_prepare_dataset()
            
            # Step 2: Initialize model
            self.initialize_model()
            
            # Step 3: Format dataset for training
            formatted_dataset = self.format_dataset_for_training(dataset)
            
            # Step 4: Train the model
            self.train_model(formatted_dataset)
            
            logger.info("=== Training Pipeline Completed Successfully ===")
            logger.info(f"Trained model available at: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run the training script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a Kubernetes-focused language model using Unsloth"
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/DialoGPT-medium",
        help="Base model to fine-tune (default: microsoft/DialoGPT-medium)"
    )
    parser.add_argument(
        "--dataset-path",
        default="src/output/k8s_dataset.jsonl",
        help="Path to the JSONL dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="k8s_model_output",
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    
    args = parser.parse_args()
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow on CPU.")
        response = input("Continue with CPU training? (y/N): ")
        if response.lower() != 'y':
            logger.info("Training cancelled.")
            return
    else:
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
    
    # Initialize trainer
    trainer = KubernetesModelTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length
    )
    
    # Run training
    try:
        trainer.run_training_pipeline()
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {args.output_dir}")
        print(f"🚀 You can now use your Kubernetes-trained model for inference!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 