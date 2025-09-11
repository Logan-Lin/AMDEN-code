# AMDEN Implementation

This repository contains the implementation code for AMDEN (Amorphous Material DEnoising Network), a diffusion model framework for inverse design of amorphous materials.

## Project Structure

The codebase is organized into three main directories:

- **`src/`** - Source code implementation
  - `main.py` - Main entry point for training and inference
  - `pipeline.py` - Training, testing, and inference pipelines
  - `data.py` - Dataset loading and preprocessing utilities
  - `models/` - Neural network architectures
    - `denoisers/` - Denoising model implementations (EGNN-based)
    - `networks/` - Core network architectures
    - `modules/` - Supporting modules (noise schedules, losses)
  - `utils.py` - Utility functions and logging
  - `neighborlists.py` - Neighbor list computation for molecular systems

- **`runtime/`** - Environment setup options (5 different methods)
  - Docker, Singularity, Poetry, Nix Flake, and pip-based setups

- **`settings/`** - Configuration files organized by material system
  - `a-si/` - Amorphous silicon configurations
  - `a-sio2/` - Amorphous silica configurations  
  - `meg/` - Multi-element glass configurations


Datasets can be found under **`datasets/`**:
- **`flake`** - Nix Flake with a development environment to run the scripts for analyzing the datasets
- **`MEG/`** - Multi-element glass dataset
    - `data/` - Structure and property data
    - `workflow/` - LAMMPS and Python scripts used for generating the dataset
    - `src/` - Script to compute the Young's modulus shown in the paper
- **`Si`** - Amorphous Silicon
    - `data/` - Structure and property data
    - `src/` - Script to compute the sheer modulus and average ring size shown in the paper
- **`SiO2`** - Three variants of amorphous Silica with different cooling schedules (melt, quench, anneal)
    - `data/` - Structure and property data
    - `src/` - Script to compute radial distribution functions, bond angle distributions, structure factors and potential energies

## Setup Options

Choose one of the following setup methods based on your environment:

### 1. Docker

```bash
cd runtime/
docker build -t amden .
docker run --gpus all -it -v $(pwd)/..:/workspace amden
```

### 2. Poetry (Python dependency management)

```bash
cd runtime/poetry/
poetry install
poetry shell
```

### 3. Nix Flake (Reproducible environments)

```bash
cd runtime/flake/
# With CUDA support
nix develop .#withCuda
# Without CUDA  
nix develop .#withoutCuda
```

### 4. Singularity

```bash
cd runtime/
singularity build amden.sif ddm.def
singularity shell --nv amden.sif
```

### 5. Traditional pip setup

```bash
cd runtime/
bash init.sh
source $HOME/venv/bin/activate
```

## Usage

### Command Line Interface

The main entry point accepts the following arguments:

```bash
python src/main.py -s <settings_file> -g <gpu_id> [--compile]
```

- `-s, --setting`: Path to YAML/JSON configuration file (required)
- `-g, --cuda`: GPU device index (default: 0, use -1 for CPU)
- `--compile`: Enable PyTorch model compilation for performance

### Configuration System

Configuration files are organized by material system and experiment type. Each config file contains:

- **`model`**: Architecture and model parameters
- **`scheduler`**: Noise schedule parameters for diffusion process
- **`loss`**: Loss function configuration
- **`train`**: Training parameters and data settings
- **`infer`**: Inference parameters and output settings
- **`load`**: Model checkpoint loading settings

### Example Usage

Training a model:
```bash
python src/main.py -s settings/meg/egnn-E/train.yaml -g 0
```

Running inference:
```bash  
python src/main.py -s settings/meg/egnn-E/infer.yaml -g 0
```

## Data Formats

The implementation supports:
- **ExtXYZ files**: Atomic structure data with extended properties
- **JSON property files**: Material properties for conditioning
- **LAMMPS data files**: Alternative input format (via ASE)

Expected file structure for datasets:
```
datasets/
├── material_name/
│   ├── structures.extxyz
│   └── properties.json
```

