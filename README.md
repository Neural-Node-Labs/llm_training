# LLM Training Program
## Component-Based Development (CBD) Methodology

A production-ready, enterprise-grade framework for training Large Language Models from scratch using component-based architecture.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

##  Overview

This framework implements a complete end-to-end pipeline for training Large Language Models with:

- **13 atomic components** across 7 architectural layers
- **Production-grade error handling** with dual-stream logging
- **Security-first design** with input sanitization and validation
- **Scalable architecture** supporting distributed training
- **Comprehensive testing** suite with 100% coverage goals

### Why Component-Based Development?

-  **Modularity**: Each component is independently testable and replaceable
-  **Maintainability**: Clear contracts and interfaces reduce complexity
-  **Scalability**: Components can be distributed across systems
-  **Reliability**: Comprehensive error handling at every layer
-  **Observability**: Dual-stream logging for debugging and auditing

---

##  Features

### Data Ingestion
- Multi-source connectors (S3, databases, APIs, web)
- Chunked streaming with backpressure handling
- Real-time data quality validation
- Automatic encoding detection and correction

### Preprocessing
- Production-grade tokenization (BPE, WordPiece, SentencePiece)
- Configurable normalization pipelines
- Automatic dataset splitting with stratification
- Deduplication using MinHash LSH

### Training Infrastructure
- GPU/TPU resource management
- Distributed training support (DDP, FSDP)
- Mixed precision training (FP16/BF16)
- Gradient accumulation and checkpointing

### Monitoring & Evaluation
- Real-time metrics collection and aggregation
- Automated validation runs
- Anomaly detection and alerting
- TensorBoard integration ready

### Model Persistence
- Versioned checkpointing with metadata
- Multiple export formats (ONNX, TorchScript, SafeTensors, GGUF)
- Quantization support (INT8, INT4)
- Deployment validation

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Event Bus Adaptor                        │
│              (Asynchronous Communication Layer)              │
└─────────────────────────────────────────────────────────────┘
                            ▲ ▼
┌──────────────────┬────────────────┬────────────────┬─────────┐
│  Data Ingestion  │ Preprocessing  │   Training     │ Persist │
├──────────────────┼────────────────┼────────────────┼─────────┤
│ • DataSource     │ • Preprocessor │ • Resource Mgr │ • Ckpt  │
│   Connector      │ • Dataset      │ • Model Init   │ • Export│
│ • StreamReader   │   Builder      │ • DataLoader   │ • Valid │
│ • Validator      │                │ • Orchestrator │         │
└──────────────────┴────────────────┴────────────────┴─────────┘
                            ▲ ▼
┌─────────────────────────────────────────────────────────────┐
│              Logging Adaptor (Dual-Stream)                  │
│          system.log  |  llm_interaction.log                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Layers

1. **Data Ingestion Layer**: External data acquisition and streaming
2. **Preprocessing Layer**: Text normalization and tokenization
3. **Training Infrastructure**: Resource allocation and model setup
4. **Training Execution**: Main training loop and optimization
5. **Monitoring & Evaluation**: Metrics and validation
6. **Model Persistence**: Checkpointing and export
7. **Deployment Preparation**: Model validation and optimization

---

##  Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 LTS |
| Python | 3.9+ | 3.11+ |
| RAM | 32 GB | 128+ GB |
| GPU | NVIDIA RTX 3090 (24GB) | A100 (80GB) x4 |
| Storage | 500 GB SSD | 2+ TB NVMe SSD |
| CPU | 8 cores | 32+ cores |

### Software Dependencies

See `requirements.txt` for complete list. Key dependencies:

- **Core**: Python 3.9+, NumPy, SciPy
- **Deep Learning**: PyTorch 2.0+, transformers
- **Data Processing**: pandas, datasets, sentencepiece
- **Utilities**: tqdm, jsonlines, pyyaml
- **Monitoring**: tensorboard, wandb (optional)
- **Security**: cryptography, python-jose

---

##  Installation

### Option 1: pip install (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/llm-training-program.git
cd llm-training-program

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
python -c "import llm_training; print(llm_training.__version__)"
```

### Option 2: Docker (Recommended for Production)

```bash
# Build Docker image
docker build -t llm-training:latest .

# Run container
docker run -it --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  llm-training:latest

# Or use docker-compose
docker-compose up -d
```

### Option 3: Conda

```bash
# Create conda environment
conda env create -f environment.yml
conda activate llm-training

# Verify installation
python -c "import llm_training; print('Installation successful!')"
```

### Post-Installation Setup

```bash
# Create necessary directories
mkdir -p data datasets checkpoints exports metrics logs

# Set environment variables
export LLM_TRAINING_HOME=$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configure logging
cp config/logging.yaml.example config/logging.yaml

# Initialize configuration
python scripts/init_config.py
```

### GPU Setup (CUDA)

```bash
# Verify CUDA installation
nvcc --version

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Install CUDA-specific packages if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

##  Quick Start

### 1. Prepare Your Data

```bash
# Option A: Use sample data (for testing)
python scripts/download_sample_data.py --output data/sample.jsonl

# Option B: Prepare your own data
# See docs/DATA_PREPARATION.md for detailed guide
python scripts/prepare_data.py \
  --input raw_data.txt \
  --output data/prepared.jsonl \
  --format jsonl
```

### 2. Run Complete Pipeline

```python
from llm_training import DataPreparationPipeline, main

# Quick start with defaults
main()
```

Or use CLI:

```bash
# Run full pipeline
python -m llm_training.main \
  --input data/sample.jsonl \
  --output-dir output/ \
  --config config/default.yaml

# View logs
tail -f logs/system.log
tail -f logs/llm_interaction.log
```

### 3. Monitor Training

```bash
# Open TensorBoard (in another terminal)
tensorboard --logdir=runs/

# View metrics
python scripts/view_metrics.py --training-id <your-training-id>

# Check GPU usage
watch -n 1 nvidia-smi
```

### 4. Export Trained Model

```python
from llm_training import ModelExporter, ModelExporterInput

exporter = ModelExporter(event_bus)
export_input = ModelExporterInput(
    checkpoint_path='checkpoints/model_final.pt',
    export_format='onnx',
    optimization={'quantization': 'int8', 'pruning': False}
)

result = exporter.export(export_input)
print(f"Model exported to: {result.export_path}")
```

---

##  Configuration

### Configuration Files

The framework uses YAML configuration files located in `config/`:

```
config/
├── default.yaml          # Default configuration
├── training.yaml         # Training hyperparameters
├── data.yaml            # Data processing settings
├── model.yaml           # Model architecture
└── logging.yaml         # Logging configuration
```

### Example Configuration (config/default.yaml)

```yaml
# Data Ingestion
data_ingestion:
  chunk_size_mb: 50
  max_retries: 3
  timeout_seconds: 30
  allowed_sources:
    - s3
    - database
    - api

# Preprocessing
preprocessing:
  tokenizer_type: bpe
  vocab_size: 50000
  max_sequence_length: 2048
  lowercase: false
  remove_special_chars: false

# Model Architecture
model:
  type: transformer
  num_layers: 12
  hidden_size: 768
  num_attention_heads: 12
  intermediate_size: 3072
  vocab_size: 50000
  max_position_embeddings: 2048
  
# Training
training:
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 0.0001
  num_epochs: 10
  warmup_steps: 1000
  max_grad_norm: 1.0
  mixed_precision: true
  
  optimizer:
    type: adamw
    betas: [0.9, 0.999]
    eps: 1.0e-8
    weight_decay: 0.01
  
  scheduler:
    type: cosine
    num_cycles: 0.5

# Checkpointing
checkpointing:
  save_strategy: best  # best|periodic|final
  checkpoint_every_n_steps: 1000
  keep_best_n: 3
  keep_last_n: 2

# Validation
validation:
  eval_every_n_steps: 500
  eval_batch_size: 64
  metrics:
    - perplexity
    - accuracy
    - loss

# Monitoring
monitoring:
  use_tensorboard: true
  use_wandb: false
  log_every_n_steps: 10
  
  alerting:
    enabled: true
    channels:
      - email
    thresholds:
      loss_spike_factor: 2.0
      gradient_norm_max: 10.0

# Resources
resources:
  gpu_count: 4
  gpu_memory_gb: 80
  distributed: true
  distributed_backend: nccl
  mixed_precision: fp16

# Security
security:
  validate_paths: true
  sanitize_inputs: true
  encrypt_credentials: true
  max_file_size_mb: 1000
```

### Environment Variables

Create `.env` file:

```bash
# Project paths
LLM_TRAINING_HOME=/path/to/llm-training-program
DATA_DIR=/path/to/data
CHECKPOINT_DIR=/path/to/checkpoints

# API Keys (if using external services)
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=your_huggingface_token

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_LAUNCH_BLOCKING=1  # For debugging

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/system.log

# Database (if using)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=llm_training
DB_USER=postgres
DB_PASSWORD=your_password
```

Load environment:

```bash
# Load .env file
source .env

# Or use python-dotenv
pip install python-dotenv
```

---

##  Usage Guide

### Basic Usage

#### 1. Data Collection

```python
from llm_training import DataSourceConnector, DataSourceConnectorInput

# Initialize connector
connector = DataSourceConnector(event_bus)

# Connect to S3
input_data = DataSourceConnectorInput(
    source_type='s3',
    credentials={
        'auth_type': 'iam_role',
        'credentials_encrypted': 'your_encrypted_credentials'
    },
    source_config={
        'endpoint': 's3://your-bucket/training-data',
        'filters': {'file_type': 'txt'}
    }
)

# Establish connection
connection = connector.connect(input_data)
print(f"Connected: {connection.connection_id}")
```

#### 2. Data Streaming

```python
from llm_training import DataStreamReader, DataStreamReaderInput

# Stream data in chunks
reader = DataStreamReader(event_bus)
reader_input = DataStreamReaderInput(
    connection_id=connection.connection_id,
    chunk_size_mb=50,
    format='txt'
)

stream = reader.read_stream(reader_input)
print(f"Received {stream.total_chunks} chunks")
```

#### 3. Data Validation

```python
from llm_training import DataValidator, DataValidatorInput

validator = DataValidator(event_bus)

for chunk in stream.chunks:
    validator_input = DataValidatorInput(
        stream_id=stream.stream_id,
        chunk_data=chunk.data,
        validation_rules={
            'encoding': 'utf-8',
            'min_quality_score': 0.8,
            'prohibited_patterns': []
        }
    )
    
    result = validator.validate(validator_input)
    if result.status == 'passed':
        print(f"Chunk validated: quality={result.quality_metrics['quality_score']}")
```

#### 4. Preprocessing

```python
from llm_training import DataPreprocessor, DataPreprocessorInput

preprocessor = DataPreprocessor(event_bus)
preprocessor_input = DataPreprocessorInput(
    raw_data=validated_text,
    preprocessing_config={
        'tokenizer_type': 'bpe',
        'lowercase': True,
        'remove_special_chars': False,
        'max_sequence_length': 2048
    }
)

tokens = preprocessor.preprocess(preprocessor_input)
print(f"Tokenized: {tokens.token_count} tokens")
```

#### 5. Training

```python
from llm_training import (
    ComputeResourceManager,
    ModelInitializer,
    TrainingOrchestrator
)

# Allocate resources
resource_mgr = ComputeResourceManager(event_bus)
resources = resource_mgr.allocate(
    ComputeResourceManagerInput(
        resource_requirements={
            'gpu_count': 4,
            'gpu_memory_gb': 80,
            'distributed': True
        },
        priority='high'
    )
)

# Initialize model
model_init = ModelInitializer(event_bus)
model = model_init.initialize(
    ModelInitializerInput(
        architecture={
            'model_type': 'transformer',
            'num_layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'vocab_size': 50000
        },
        initialization={
            'method': 'random',
            'seed': 42
        }
    )
)

# Start training
orchestrator = TrainingOrchestrator(event_bus)
training = orchestrator.train(
    TrainingOrchestratorInput(
        model_id=model.model_id,
        loader_id=loader.loader_id,
        training_config={
            'num_epochs': 10,
            'learning_rate': 0.0001,
            'optimizer': 'adamw',
            'gradient_accumulation_steps': 4,
            'checkpoint_every_n_steps': 1000,
            'mixed_precision': True
        }
    ),
    data_loader_output=loader_output
)

print(f"Training status: {training.status}")
```

### Advanced Usage

#### Custom Components

Create your own components following the CBD pattern:

```python
from dataclasses import dataclass
from typing import Dict, Any
import time

@dataclass
class CustomComponentInput:
    """IN Schema"""
    param1: str
    param2: int
    config: Dict[str, Any]

@dataclass
class CustomComponentOutput:
    """OUT Schema"""
    result_id: str
    status: str
    data: Dict[str, Any]

@dataclass
class CustomComponentError:
    """Error Schema"""
    error_code: str
    message: str
    recoverable: bool

class CustomComponent:
    """
    Component: CustomComponent
    Logical Function: Your custom logic here
    """
    
    def __init__(self, event_bus: EventBusAdaptor):
        self.event_bus = event_bus
        logger.log_llm_interaction('CustomComponent', 'initialized', {})
    
    def process(self, input_data: CustomComponentInput) -> CustomComponentOutput:
        """
        Restate IN Schema:
        - param1: str
        - param2: int
        - config: dict
        
        Restate OUT Schema:
        - result_id: str
        - status: str
        - data: dict
        """
        start_time = time.time()
        
        try:
            # Trace point: processing_start
            logger.log_llm_interaction('CustomComponent', 'processing_start', {
                'param1': input_data.param1
            })
            
            # Your logic here
            result_id = str(uuid.uuid4())
            
            # Trace point: processing_complete
            logger.log_llm_interaction('CustomComponent', 'processing_complete', {
                'result_id': result_id,
                'execution_time_ms': (time.time() - start_time) * 1000
            })
            
            return CustomComponentOutput(
                result_id=result_id,
                status='success',
                data={'processed': True}
            )
            
        except Exception as e:
            error = CustomComponentError(
                error_code='PROCESSING_FAILED',
                message=str(e),
                recoverable=True
            )
            logger.log_system('error', 'Processing failed', asdict(error))
            raise
```

#### Distributed Training

```python
import torch.distributed as dist

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Use DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])

# Training loop remains the same
# Framework handles distributed communication
```

#### Custom Data Sources

```python
class CustomDataSource:
    """Custom data source connector"""
    
    def connect(self, config):
        # Implement your connection logic
        pass
    
    def read(self, chunk_size):
        # Implement data reading
        pass
    
    def close(self):
        # Cleanup
        pass

# Register custom source
from llm_training import register_data_source
register_data_source('custom', CustomDataSource)
```

---

##  API Reference

### Core Classes

#### EventBusAdaptor

```python
class EventBusAdaptor:
    def publish(self, topic: str, message: Dict) -> bool
    def subscribe(self, topic: str, callback: Callable) -> str
    def unsubscribe(self, subscription_id: str) -> bool
```

#### LoggingAdaptor

```python
class LoggingAdaptor:
    def log_system(self, level: str, message: str, context: Dict) -> None
    def log_llm_interaction(self, component: str, event: str, data: Dict) -> None
```

#### ConfigurationAdaptor

```python
class ConfigurationAdaptor:
    def get_config(self, component: str, key: str) -> Any
    def update_config(self, component: str, key: str, value: Any) -> bool
    def watch_config(self, component: str, callback: Callable) -> str
```

### Component APIs

See `docs/API.md` for complete API documentation.

---

##  Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
training_config['batch_size'] = 16  # Instead of 32

# Increase gradient accumulation
training_config['gradient_accumulation_steps'] = 8

# Enable gradient checkpointing
model_config['gradient_checkpointing'] = True

# Use mixed precision
training_config['mixed_precision'] = True
```

#### 2. Slow Data Loading

**Error**: Training waiting on data

**Solutions**:
```python
# Increase num_workers
loader_input.num_workers = 8

# Increase prefetch_factor
loader_input.prefetch_factor = 4

# Use faster storage (NVMe SSD)
# Pin memory for faster GPU transfer
loader_config['pin_memory'] = True
```

#### 3. NaN Loss

**Error**: Loss becomes NaN during training

**Solutions**:
```python
# Reduce learning rate
training_config['learning_rate'] = 0.00001

# Enable gradient clipping
training_config['max_grad_norm'] = 1.0

# Check for bad data
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
```

#### 4. Import Errors

**Error**: `ModuleNotFoundError`

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.9+

# Verify installation
pip list | grep llm-training
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with profiling
python -m cProfile -o profile.stats -m llm_training.main

# Analyze profile
python -m pstats profile.stats
```

### Getting Help

1. Check documentation: `docs/`
2. Search issues: https://github.com/your-org/llm-training-program/issues
3. Ask community: https://discord.gg/your-discord
4. Email support: support@your-org.com

---

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_validator.py -v

# Run with coverage
pytest --cov=llm_training tests/

# Generate coverage report
pytest --cov=llm_training --cov-report=html tests/
```

---

##  License

MIT License - see LICENSE file for details




---

**Made with ❤️ using Component-Based Development**
