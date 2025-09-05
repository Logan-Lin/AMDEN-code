FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Prevent interactive prompts during package installation
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /venv --system-site-packages
ENV PATH="/venv/bin:$PATH"

# Install PyTorch with CUDA support and other dependencies
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# Install MACE layer
RUN git clone https://github.com/ACEsuit/mace-layer.git && \
    pip install ./mace-layer && \
    rm -rf mace-layer

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"] 