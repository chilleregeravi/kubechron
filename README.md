# kubechron
KubeChron is a lightweight, specialized language model trained on Kubernetes, GitOps (Flux &amp; ArgoCD), Helm, and Linux — designed for DevOps engineers who need a smart assistant that fits in their stack, not just the cloud.

## Features

- **Production-Grade Reliability**: Comprehensive error handling, retry mechanisms, and logging
- **Configurable**: Environment-based configuration with sensible defaults
- **Data Quality**: Content validation, deduplication, and quality checks
- **Structured Logging**: JSON-formatted logs with metrics tracking
- **Rate Limiting**: Respectful scraping with configurable rate limits
- **Robust Parsing**: Intelligent text chunking with boundary detection

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure (Optional)**:
   Create a `.env` file with your settings:
   ```
   K8S_COPILOT_GITHUB_TOKEN=your_github_token
   K8S_COPILOT_LOG_LEVEL=INFO
   K8S_COPILOT_OUTPUT_DIR=output
   ```

3. **Run the Pipeline**:
   ```bash
   cd src
   python main.py
   ```

## Configuration Options

All configuration can be set via environment variables with the `K8S_COPILOT_` prefix:

- `OUTPUT_DIR`: Output directory (default: `output`)
- `DATASET_FILENAME`: Output filename (default: `k8s_dataset.jsonl`)
- `MAX_CHUNK_LENGTH`: Maximum text chunk size (default: `800`)
- `REQUESTS_PER_SECOND`: Rate limiting (default: `2.0`)
- `GITHUB_TOKEN`: GitHub personal access token (optional)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `LOG_FORMAT`: Log format - `json` or `text` (default: `json`)

## Architecture

- **main.py**: Pipeline orchestrator with comprehensive error handling
- **config.py**: Configuration management with validation
- **logger.py**: Structured logging and metrics collection
- **retry_utils.py**: Robust HTTP requests with exponential backoff
- **docs_scraper.py**: Kubernetes documentation scraping
- **github_scraper.py**: GitHub repository YAML extraction
- **utils.py**: Data validation and quality utilities
- **parser.py**: Intelligent text chunking
- **dataset_formatter.py**: Instruction-tuning format conversion

## Output Format

Creates JSONL files with instruction-tuning format:
```json
{
  "instruction": "Explain the following Kubernetes YAML configuration:",
  "input": "",
  "output": "apiVersion: v1\nkind: Pod\n...",
  "metadata": {
    "source_type": "yaml",
    "content_length": 245,
    "repository": "https://github.com/kubernetes/examples"
  }
}
```

## Production Features

- **Error Recovery**: Continues processing even if individual sources fail
- **Progress Tracking**: Detailed logging of scraping progress
- **Data Validation**: Quality checks and duplicate detection
- **Metrics Collection**: Comprehensive statistics on scraping performance
- **Resource Management**: Proper cleanup of temporary files and resources
- **Rate Limiting**: Respectful of source servers with configurable limits

## Model Training

After generating the dataset, you can train a Kubernetes-focused LLM:

### Prerequisites for Training
```bash
# Install PyTorch with CUDA support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth for efficient training
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# Install other ML dependencies
pip install transformers datasets accelerate peft bitsandbytes
```

### Training the Model
```bash
# Basic training with default settings
python train_k8s_model.py

# Custom training options
python train_k8s_model.py \
    --model-name "microsoft/DialoGPT-medium" \
    --dataset-path "src/output/k8s_dataset.jsonl" \
    --output-dir "k8s_model_output" \
    --epochs 3 \
    --batch-size 4
```

### Using the Trained Model

#### Interactive Mode
```bash
python inference_k8s_model.py
```

#### Single Question
```bash
python inference_k8s_model.py --question "What is a Kubernetes Pod?"
```

#### Explain YAML
```bash
python inference_k8s_model.py --yaml-file deployment.yaml
```

#### Generate YAML
```bash
python inference_k8s_model.py --generate "nginx deployment with 3 replicas"
```

## Features

- **Data Scraping**: Production-grade Kubernetes documentation and YAML extraction
- **Model Training**: Efficient fine-tuning using Unsloth with LoRA adapters
- **Interactive Inference**: Chat interface for asking Kubernetes questions
- **YAML Processing**: Explain existing configurations or generate new ones

## Development

The codebase follows production-grade patterns:
- Type hints throughout
- Comprehensive error handling
- Structured logging
- Configuration validation
- Modular architecture
- Legacy compatibility wrappers 