cd /work/DiffDisMatter

sudo apt-get update && sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    wget

python3 -m venv $HOME/venv --system-site-packages && source $HOME/venv/bin/activate

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

git clone https://github.com/ACEsuit/mace-layer.git && \
    pip install ./mace-layer && \
    rm -rf mace-layer

pip install -r requirements.txt
