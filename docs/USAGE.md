# Usage Guide
## LLM Training Program - Complete Usage Documentation

Comprehensive guide for using the LLM Training Program from basic to advanced scenarios.

---

##  Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Workflows](#basic-workflows)
3. [Advanced Usage](#advanced-usage)
4. [Command Line Interface](#command-line-interface)
5. [Python API](#python-api)
6. [Configuration](#configuration)
7. [Monitoring & Debugging](#monitoring--debugging)
8. [Production Deployment](#production-deployment)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

---

##  Quick Start

### 1. Simplest Possible Usage

```bash
# Activate environment
source venv/bin/activate

# Run with defaults
python -m llm_training.main

# This will:
# - Use sample data
# - Train a small model
# - Save checkpoints to ./checkpoints/
# - Log to ./logs/
```

### 2. Basic Training Run

```python
from llm_training import main

# Run complete pipeline
main()
```

### 3. With Custom Data

```bash
python -m llm_training.main \
  --input-file data/my_data.jsonl \
  --output-dir output/experiment_1 \
  --config config/default.yaml
```

---

##  Basic Workflows

### Workflow 1: Data Preparation Only

Prepare data without training:

```python
from llm_training import DataPreparationPipeline

# Initialize pipeline
pipeline = DataPreparationPipeline(output_dir='prepared_data')

# Run data preparation
pipeline.run(
    input_file='raw_data.jsonl',
    max_seq_length=2048
)

# Output:
# - prepared_data/train.json
# - prepared_data/validation.json
# - prepared_data/vocab.json
```

Or via CLI:

```bash
python scripts/prepare_data.py \
  --input raw_data.jsonl \
  --output prepared_data/ \
  --max-length 2048 \
  --vocab-size 50000
```

### Workflow 2: Training from Prepared Data

Train model using pre-prepared data:

```python
from llm_training import (
    EventBusAdaptor,
    ComputeResourceManager,
    ModelInitializer,
    DataLoader,
    TrainingOrchestrator
)

# Initialize components
event_bus = EventBusAdaptor()

# Load data
loader = DataLoader(event_bus)
loader_output = loader.create_loader(
    DataLoaderInput(
        dataset_path='prepared_data/train.json',
        batch_size=32,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2
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
        initialization={'method': 'random', 'seed': 42}
    )
)

# Start training
orchestrator = TrainingOrchestrator(event_bus)
result = orchestrator.train(
    TrainingOrchestratorInput(
        model_id=model.model_id,
        loader_id=loader_output.loader_id,
        training_config={
            'num_epochs': 10,
            'learning_rate': 0.0001,
            'optimizer': 'adamw'
        }
    ),
    data_loader_output=loader_output
)

print(f"Training completed: {result.status}")
```

### Workflow 3: Resume from Checkpoint

Continue training from saved checkpoint:

```bash
python -m llm_training.main \
  --resume-from checkpoints/model_step_5000.pt \
  --config config/training.yaml
```

Or in Python:

```python
from llm_training import load_checkpoint, TrainingOrchestrator

# Load checkpoint
checkpoint = load_checkpoint('checkpoints/model_step_5000.pt')

# Resume training
orchestrator = TrainingOrchestrator(event_bus)
result = orchestrator.train(
    training_input,
    loader_output,
    checkpoint=checkpoint
)
```

### Workflow 4: Distributed Training

Train across multiple GPUs:

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 \
  -m llm_training.main \
  --config config/distributed.yaml

# Or using accelerate
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  -m llm_training.main \
  --config config/distributed.yaml
```

Configuration for distributed training:

```yaml
# config/distributed.yaml
resources:
  gpu_count: 4
  distributed: true
  distributed_backend: nccl

training:
  batch_size: 8  # Per GPU
  gradient_accumulation_steps: 16
```

### Workflow 5: Model Export

Export trained model for deployment:

```python
from llm_training import ModelExporter, ModelExporterInput

exporter = ModelExporter(event_bus)

# Export to ONNX
onnx_output = exporter.export(
    ModelExporterInput(
        checkpoint_path='checkpoints/model_final.pt',
        export_format='onnx',
        optimization={
            'quantization': 'int8',
            'pruning': False
        }
    )
)

print(f"ONNX model: {onnx_output.export_path}")

# Export to TorchScript
torchscript_output = exporter.export(
    ModelExporterInput(
        checkpoint_path='checkpoints/model_final.pt',
        export_format='torchscript',
        optimization={'quantization': 'none'}
    )
)

print(f"TorchScript model: {torchscript_output.export_path}")
```

---

##  Advanced Usage

### Custom Training Loop

Implement custom training logic:

```python
from llm_training import (
    TrainingOrchestrator,
    LossCalculator,
    MetricsCollector,
    ModelCheckpointer
)

class CustomTrainingLoop:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.loss_calc = LossCalculator(event_bus)
        self.metrics = MetricsCollector(event_bus)
        self.checkpointer = ModelCheckpointer(event_bus)
    
    def train_epoch(self, model, data_loader, optimizer):
        """Custom epoch training logic"""
        total_loss = 0
        
        for batch_idx, batch in enumerate(data_loader):
            # Forward pass
            predictions = model(batch['input_ids'])
            
            # Calculate loss
            loss_output = self.loss_calc.calculate(
                LossCalculatorInput(
                    predictions=predictions,
                    targets=batch['labels'],
                    loss_function='cross_entropy',
                    reduction='mean'
                )
            )
            
            # Backward pass
            loss_output.loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Collect metrics
            self.metrics.collect(
                MetricsCollectorInput(
                    training_id='custom_training',
                    step=batch_idx,
                    metrics={
                        'loss': loss_output.loss_value,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    }
                )
            )
            
            total_loss += loss_output.loss_value
        
        return total_loss / len(data_loader)
```

### Custom Components

Create custom data source:

```python
from llm_training import DataSourceConnector
from dataclasses import dataclass

@dataclass
class CustomSourceInput:
    api_endpoint: str
    api_key: str
    filters: dict

class CustomDataSource(DataSourceConnector):
    """Custom data source for proprietary API"""
    
    def connect(self, input_data: CustomSourceInput):
        # Implement custom connection logic
        import requests
        
        headers = {'Authorization': f'Bearer {input_data.api_key}'}
        response = requests.get(
            input_data.api_endpoint,
            headers=headers,
            params=input_data.filters
        )
        
        if response.status_code == 200:
            return DataSourceConnectorOutput(
                connection_id=str(uuid.uuid4()),
                status='connected',
                metadata={'records': len(response.json())}
            )
        else:
            raise ConnectionError(f"API returned {response.status_code}")

# Use custom source
custom_source = CustomDataSource(event_bus)
connection = custom_source.connect(
    CustomSourceInput(
        api_endpoint='https://api.example.com/data',
        api_key='your_api_key',
        filters={'category': 'science'}
    )
)
```

### Multi-Modal Training

Train on text + images:

```python
from llm_training import MultiModalPreprocessor

class MultiModalPreprocessor:
    """Process text and images together"""
    
    def preprocess(self, text, image_path):
        # Tokenize text
        text_tokens = self.tokenize(text)
        
        # Process image
        from PIL import Image
        import torchvision.transforms as transforms
        
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image_tensor = transform(image)
        
        return {
            'text_tokens': text_tokens,
            'image_tensor': image_tensor
        }
```

### Curriculum Learning

Implement progressive difficulty:

```python
class CurriculumTrainer:
    """Train with increasing difficulty"""
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.orchestrator = TrainingOrchestrator(event_bus)
    
    def train_curriculum(self, stages):
        """
        stages = [
            {'max_length': 128, 'epochs': 2},
            {'max_length': 512, 'epochs': 3},
            {'max_length': 2048, 'epochs': 5}
        ]
        """
        model_id = None
        
        for stage_idx, stage in enumerate(stages):
            print(f"\nStage {stage_idx + 1}: max_length={stage['max_length']}")
            
            # Prepare data for this stage
            data = self.prepare_stage_data(stage['max_length'])
            
            # Train
            result = self.orchestrator.train(
                TrainingOrchestratorInput(
                    model_id=model_id,
                    loader_id=data.loader_id,
                    training_config={
                        'num_epochs': stage['epochs'],
                        'learning_rate': 0.0001 / (stage_idx + 1)  # Decay LR
                    }
                ),
                data_loader_output=data
            )
            
            model_id = result.model_id  # Continue from this checkpoint
```

### Active Learning

Select most informative samples:

```python
class ActiveLearningPipeline:
    """Train on most uncertain samples"""
    
    def select_uncertain_samples(self, model, unlabeled_data, n_samples=1000):
        """Select samples with highest uncertainty"""
        uncertainties = []
        
        for sample in unlabeled_data:
            # Get model predictions
            with torch.no_grad():
                logits = model(sample)
                probs = torch.softmax(logits, dim=-1)
                
                # Calculate entropy (uncertainty)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                uncertainties.append((sample, entropy.item()))
        
        # Sort by uncertainty and select top n
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        selected = [sample for sample, _ in uncertainties[:n_samples]]
        
        return selected
    
    def train_active(self, initial_data, unlabeled_pool, iterations=5):
        """Iterative active learning"""
        current_data = initial_data
        
        for iteration in range(iterations):
            # Train on current data
            model = self.train_model(current_data)
            
            # Select new samples
            new_samples = self.select_uncertain_samples(
                model,
                unlabeled_pool,
                n_samples=1000
            )
            
            # Add to training set (after labeling)
            current_data.extend(new_samples)
            
            print(f"Iteration {iteration + 1}: {len(current_data)} samples")
```

---

##  Command Line Interface

### Main Commands

#### Train Command

```bash
# Basic training
llm-train --config config/default.yaml

# With custom parameters
llm-train \
  --config config/default.yaml \
  --data-dir data/ \
  --output-dir experiments/exp_001 \
  --num-epochs 20 \
  --batch-size 16 \
  --learning-rate 0.0001

# Resume training
llm-train \
  --resume checkpoints/model_step_5000.pt \
  --config config/default.yaml

# Distributed training
llm-train \
  --config config/distributed.yaml \
  --num-gpus 4 \
  --distributed-backend nccl
```

#### Data Preparation Command

```bash
# Prepare data
llm-prepare-data \
  --input raw_data.txt \
  --output prepared_data/ \
  --format jsonl \
  --vocab-size 50000 \
  --max-length 2048 \
  --min-quality 0.7 \
  --deduplicate

# Validate prepared data
llm-validate-data \
  --data-dir prepared_data/ \
  --checks quality,distribution,duplicates
```

#### Export Command

```bash
# Export model
llm-export \
  --checkpoint checkpoints/model_final.pt \
  --format onnx \
  --output exports/model.onnx \
  --quantize int8

# Export multiple formats
llm-export \
  --checkpoint checkpoints/model_final.pt \
  --formats onnx,torchscript,safetensors \
  --output-dir exports/
```

#### Evaluation Command

```bash
# Evaluate model
llm-evaluate \
  --model-path checkpoints/model_final.pt \
  --test-data data/test.json \
  --metrics perplexity,accuracy,f1 \
  --output results/eval_results.json

# Compare models
llm-compare \
  --models checkpoints/model_v1.pt checkpoints/model_v2.pt \
  --test-data data/test.json \
  --output results/comparison.html
```

### Utility Commands

```bash
# Check system requirements
llm-check-system

# View training logs
llm-logs --training-id <id> --follow

# Monitor GPU usage
llm-monitor-gpu --interval 1

# Clean up old checkpoints
llm-cleanup \
  --checkpoint-dir checkpoints/ \
  --keep-best 3 \
  --keep-last 2

# Generate dataset statistics
llm-analyze-data \
  --data-file prepared_data/train.json \
  --output stats/analysis.html
```

---

##  Python API

### High-Level API

```python
from llm_training import Trainer

# Simple trainer
trainer = Trainer(
    config_file='config/default.yaml',
    data_dir='data/',
    output_dir='experiments/exp_001'
)

# Train
trainer.train(
    num_epochs=10,
    batch_size=32,
    learning_rate=0.0001
)

# Evaluate
results = trainer.evaluate(test_data='data/test.json')
print(f"Perplexity: {results['perplexity']}")

# Export
trainer.export(
    format='onnx',
    output_path='exports/model.onnx'
)
```

### Mid-Level API

```python
from llm_training import Pipeline

# Create pipeline
pipeline = Pipeline()

# Add stages
pipeline.add_stage('data_collection', source='s3://bucket/data')
pipeline.add_stage('preprocessing', vocab_size=50000)
pipeline.add_stage('training', num_epochs=10)
pipeline.add_stage('evaluation', metrics=['perplexity', 'accuracy'])
pipeline.add_stage('export', format='onnx')

# Run pipeline
results = pipeline.run()

# Access outputs
train_results = results['training']
eval_results = results['evaluation']
```

### Low-Level API

Direct component usage (as shown in previous sections).

---

##  Configuration

### Configuration Hierarchy

```
1. Default values (hardcoded)
2. config/default.yaml
3. Environment variables
4. Command-line arguments
5. Runtime overrides
```

### Configuration Templates

#### Small Model (Testing)

```yaml
# config/small_model.yaml
model:
  num_layers: 6
  hidden_size: 512
  num_attention_heads: 8
  
training:
  batch_size: 64
  num_epochs: 5
  learning_rate: 0.001
```

#### Medium Model (Development)

```yaml
# config/medium_model.yaml
model:
  num_layers: 12
  hidden_size: 768
  num_attention_heads: 12
  
training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 0.0001
  gradient_accumulation_steps: 4
```

#### Large Model (Production)

```yaml
# config/large_model.yaml
model:
  num_layers: 24
  hidden_size: 1024
  num_attention_heads: 16
  
training:
  batch_size: 8
  num_epochs: 20
  learning_rate: 0.00005
  gradient_accumulation_steps: 16
  mixed_precision: true
  
resources:
  gpu_count: 8
  distributed: true
```

### Dynamic Configuration

```python
from llm_training import ConfigurationAdaptor

config = ConfigurationAdaptor()

# Get value
batch_size = config.get_config('training', 'batch_size')

# Update value
config.update_config('training', 'learning_rate', 0.0001)

# Watch for changes
def on_lr_change(new_value):
    print(f"Learning rate changed to: {new_value}")

config.watch_config('training.learning_rate', on_lr_change)
```

---

##  Monitoring & Debugging

### Real-Time Monitoring

```bash
# Terminal 1: Start training
python -m llm_training.main --config config/default.yaml

# Terminal 2: Monitor logs
tail -f logs/system.log

# Terminal 3: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 4: TensorBoard
tensorboard --logdir=runs/ --port=6006
```

### Metrics Dashboard

```python
from llm_training import MetricsDashboard

# Start dashboard
dashboard = MetricsDashboard(port=8080)
dashboard.start()

# Add custom metrics
dashboard.add_metric('custom_score', value=0.95)

# Generate report
dashboard.generate_report('results/metrics_report.html')
```

### Debugging Mode

```python
# Enable debug mode
import os
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enable anomaly detection
import torch
torch.autograd.set_detect_anomaly(True)

# Profile training
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Run training
    model(batch)

# Print profile
print(prof.key_averages().table())
```

### Error Handling

```python
from llm_training import ErrorRecovery

# Set up error recovery
recovery = ErrorRecovery(
    checkpoint_dir='checkpoints/',
    max_retries=3,
    backup_freq=1000
)

try:
    # Training code
    trainer.train()
except Exception as e:
    # Attempt recovery
    recovery.recover_from_error(e)
```

---

##  Production Deployment

### Model Serving

```python
from llm_training import ModelServer

# Start inference server
server = ModelServer(
    model_path='exports/model.onnx',
    port=8000,
    workers=4
)

server.start()

# API endpoint: http://localhost:8000/predict
```

### Batch Inference

```bash
# Run batch inference
llm-infer \
  --model exports/model.onnx \
  --input data/test_inputs.jsonl \
  --output results/predictions.jsonl \
  --batch-size 128
```

### Docker Deployment

```bash
# Build production image
docker build -t llm-model:prod -f Dockerfile.prod .

# Run container
docker run -d \
  --name llm-inference \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/exports:/models \
  llm-model:prod
```

---

##  Best Practices

1. **Always validate data** before training
2. **Use version control** for configs and code
3. **Monitor GPU memory** during training
4. **Save checkpoints frequently**
5. **Test on small dataset** first
6. **Use mixed precision** for faster training
7. **Log everything** for reproducibility
8. **Document experiments** in logs
9. **Validate exported models** before deployment
10. **Keep backups** of important checkpoints

---

##  Examples

See `examples/` directory for complete examples:

- `examples/01_basic_training.py`
- `examples/02_custom_data.py`
- `examples/03_distributed.py`
- `examples/04_fine_tuning.py`
- `examples/05_export_deploy.py`

---
