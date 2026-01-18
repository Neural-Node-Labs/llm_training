# Setup Guide
## LLM Training Program Installation & Configuration

Complete step-by-step guide to get the LLM Training Program running on your system.

---

##  Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Preparation](#system-preparation)
3. [Installation Methods](#installation-methods)
4. [GPU Setup](#gpu-setup)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Common Issues](#common-issues)
8. [Next Steps](#next-steps)

---

##  Prerequisites

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **OS** | Ubuntu 20.04+ / CentOS 8+ / macOS 12+ / Windows 10+ (WSL2) |
| **Python** | 3.9, 3.10, or 3.11 |
| **RAM** | 32 GB |
| **Storage** | 500 GB SSD |
| **GPU** | NVIDIA GPU with 16GB+ VRAM (optional but recommended) |
| **CUDA** | 11.8 or 12.1 (if using GPU) |

### Recommended Setup

| Component | Specification |
|-----------|---------------|
| **OS** | Ubuntu 22.04 LTS |
| **Python** | 3.11 |
| **RAM** | 128+ GB |
| **Storage** | 2+ TB NVMe SSD |
| **GPU** | 4x NVIDIA A100 (80GB) or H100 |
| **CUDA** | 12.1 |
| **Network** | 10+ Gbps for distributed training |

### Software Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential cmake git curl wget

# Install Python development headers
sudo apt install -y python3.11 python3.11-dev python3.11-venv

# Install additional libraries
sudo apt install -y libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev
```

---

##  System Preparation

### 1. Create Project Directory

```bash
# Create project root
mkdir -p ~/llm-training-program
cd ~/llm-training-program

# Create directory structure
mkdir -p data datasets checkpoints exports metrics logs config
```

### 2. Set Up Python Environment

#### Option A: venv (Recommended for simplicity)

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Option B: Conda (Recommended for complex dependencies)

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create conda environment
conda create -n llm-training python=3.11 -y
conda activate llm-training

# Install pip and setuptools
conda install pip setuptools -y
```

#### Option C: pyenv (For version management)

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to ~/.bashrc
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.11
pyenv install 3.11.6
pyenv global 3.11.6

# Create virtual environment
python -m venv venv
source venv/bin/activate
```

### 3. Clone Repository

```bash
# Clone from GitHub
git clone https://github.com/your-org/llm-training-program.git
cd llm-training-program

# Or download release
wget https://github.com/your-org/llm-training-program/archive/refs/tags/v1.0.0.tar.gz
tar -xzf v1.0.0.tar.gz
cd llm-training-program-1.0.0
```

---

##  Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# Ensure you're in the project directory with activated venv
cd llm-training-program
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Verify installation
python -c "import llm_training; print(f'Version: {llm_training.__version__}')"
```

### Method 2: Docker Installation

#### Step 1: Install Docker

```bash
# Install Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Step 2: Build Docker Image

```bash
# Build image
docker build -t llm-training:latest .

# Or pull pre-built image
docker pull your-registry/llm-training:latest
```

#### Step 3: Run Container

```bash
# Run with GPU support
docker run -it --gpus all \
  --name llm-training-container \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/config:/workspace/config \
  -p 6006:6006 \
  llm-training:latest

# Or use docker-compose
docker-compose up -d
```

#### Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  llm-training:
    image: llm-training:latest
    container_name: llm-training
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    volumes:
      - ./data:/workspace/data
      - ./checkpoints:/workspace/checkpoints
      - ./config:/workspace/config
      - ./logs:/workspace/logs
    ports:
      - "6006:6006"  # TensorBoard
      - "8888:8888"  # Jupyter (optional)
    command: python -m llm_training.main
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Method 3: Development Installation

For contributors and developers:

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Install package in development mode
pip install -e ".[dev]"

# Run tests to verify
pytest tests/ -v
```

### Method 4: Production Installation

Minimal installation for production servers:

```bash
# Install production dependencies only
pip install -r requirements-prod.txt

# Install without development tools
pip install --no-dev .

# Disable debugging features
export PYTHONOPTIMIZE=1
export LOG_LEVEL=WARNING
```

---

##  GPU Setup

### CUDA Installation

#### Ubuntu/Debian

```bash
# Remove old CUDA versions
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
 "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"

# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.1
sudo apt-get -y install cuda-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

#### CentOS/RHEL

```bash
# Add CUDA repository
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install CUDA
sudo yum install -y cuda-12-1

# Configure environment
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### PyTorch with CUDA

```bash
# Install PyTorch with CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
```

### Multi-GPU Configuration

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable peer-to-peer access
python -c "
import torch
for i in range(torch.cuda.device_count()):
    for j in range(torch.cuda.device_count()):
        if i != j:
            can_access = torch.cuda.can_device_access_peer(i, j)
            print(f'GPU {i} -> GPU {j}: {can_access}')
"

# Monitor GPU usage
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

### Apple Silicon (M1/M2/M3)

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Note: Some features may be limited on MPS
```

---

##  Configuration

### 1. Environment Variables

Create `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit with your settings
nano .env
```

Example `.env`:

```bash
# Project Configuration
LLM_TRAINING_HOME=/home/user/llm-training-program
DATA_DIR=/mnt/data/llm-training
CHECKPOINT_DIR=/mnt/checkpoints

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_LAUNCH_BLOCKING=0
NCCL_DEBUG=INFO

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# API Keys (optional)
WANDB_API_KEY=your_wandb_key
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key

# Database (if using)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=llm_training
DB_USER=postgres
DB_PASSWORD=secure_password

# Performance Tuning
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
TOKENIZERS_PARALLELISM=true
```

Load environment:

```bash
# Manual load
source .env

# Or use python-dotenv
pip install python-dotenv
python -c "from dotenv import load_dotenv; load_dotenv()"
```

### 2. Configuration Files

```bash
# Copy example configurations
cp config/default.yaml.example config/default.yaml
cp config/training.yaml.example config/training.yaml
cp config/model.yaml.example config/model.yaml

# Edit configurations
nano config/default.yaml
```

### 3. Initialize Configuration

```bash
# Run initialization script
python scripts/init_config.py

# This will:
# - Validate configuration files
# - Create necessary directories
# - Set up logging
# - Check dependencies
```

### 4. Download Required Data

```bash
# Download sample data for testing
python scripts/download_sample_data.py --output data/sample.jsonl

# Download pre-trained tokenizer (optional)
python scripts/download_tokenizer.py --model gpt2 --output models/tokenizer

# Download checkpoints (if continuing training)
python scripts/download_checkpoint.py --checkpoint-id <id> --output checkpoints/
```

---

##  Verification

### 1. System Check

```bash
# Run system diagnostics
python scripts/check_system.py
```

Expected output:
```
✓ Python version: 3.11.6
✓ CUDA available: True
✓ GPU count: 4
✓ GPU 0: NVIDIA A100-SXM4-80GB (79GB)
✓ GPU 1: NVIDIA A100-SXM4-80GB (79GB)
✓ GPU 2: NVIDIA A100-SXM4-80GB (79GB)
✓ GPU 3: NVIDIA A100-SXM4-80GB (79GB)
✓ Disk space: 1.5TB available
✓ RAM: 128GB
✓ All dependencies installed
```

### 2. Import Test

```python
# Test all imports
python -c "
import llm_training
from llm_training import (
    DataSourceConnector,
    DataStreamReader,
    DataValidator,
    DataPreprocessor,
    DatasetBuilder,
    ComputeResourceManager,
    ModelInitializer,
    DataLoader,
    TrainingOrchestrator,
    EventBusAdaptor,
    LoggingAdaptor,
    ConfigurationAdaptor
)
print('✓ All imports successful')
"
```

### 3. Run Quick Test

```bash
# Run quick integration test
python scripts/quick_test.py

# Or use pytest
pytest tests/test_integration.py -v
```

### 4. Verify Logs

```bash
# Check log files are being created
ls -lh logs/

# Should see:
# - system.log
# - llm_interaction.log

# View logs
tail -f logs/system.log
```

### 5. GPU Stress Test

```bash
# Run GPU benchmark
python scripts/gpu_benchmark.py

# Monitor during test
nvidia-smi dmon -i 0
```

---

##  Common Issues

### Issue 1: CUDA Not Detected

**Symptoms:**
```
torch.cuda.is_available() returns False
```

**Solutions:**

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall CUDA toolkit
sudo apt-get install cuda-toolkit-12-1

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.version.cuda)"
```

### Issue 2: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in config
nano config/training.yaml
# Set batch_size: 16 (instead of 32)

# Enable gradient checkpointing
# Add to model config:
# gradient_checkpointing: true
```

### Issue 3: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'llm_training'
```

**Solutions:**

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall package
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue 4: Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

```bash
# Fix directory permissions
sudo chown -R $USER:$USER ~/llm-training-program
chmod -R 755 ~/llm-training-program

# Fix specific directories
chmod 777 logs/
chmod 777 checkpoints/
chmod 777 data/
```

### Issue 5: Slow Installation

**Symptoms:**
Long install times, hanging on certain packages

**Solutions:**

```bash
# Use faster mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install binary packages only
pip install -r requirements.txt --only-binary=:all:

# Increase timeout
pip install -r requirements.txt --timeout=300

# Use conda for problematic packages
conda install -c conda-forge package-name
```

---

##  Next Steps

After successful setup:

1. **Read Documentation**
   ```bash
   # Open documentation
   firefox docs/index.html
   ```

2. **Run Tutorial**
   ```bash
   # Follow quick start tutorial
   python tutorials/01_quick_start.py
   ```

3. **Prepare Your Data**
   ```bash
   # See data preparation guide
   cat docs/DATA_PREPARATION.md
   ```

4. **Start Training**
   ```bash
   # Run first training job
   python -m llm_training.main --config config/tutorial.yaml
   ```

5. **Join Community**
   - Discord: https://discord.gg/your-community
   - Forums: https://forum.your-org.com
   - GitHub: https://github.com/your-org/llm-training-program

---

##  Additional Resources

- **Documentation**: `docs/`
- **Examples**: `examples/`
- **Tutorials**: `tutorials/`
- **API Reference**: `docs/api/`
- **FAQ**: `docs/FAQ.md`

---

##  Tips

- **Use tmux/screen** for long-running jobs
- **Monitor GPU temperature** with nvidia-smi
- **Set up automatic checkpointing** to recover from crashes
- **Use version control** for configuration files
- **Document your experiments** in logs

---
