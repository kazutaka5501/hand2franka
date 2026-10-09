#!/usr/bin/env bash
# Environment for h2f.hand_mano (WiLoR needs NumPy 1.x, so it cannot share the main one).
# MANO_RIGHT.pkl must be downloaded from https://mano.is.tue.mpg.de into assets/mano/ by hand.
set -euo pipefail

uv venv --python 3.10 .venv-hand
py=.venv-hand/bin/python
uv pip install --python $py "torch>=2.7" torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $py "numpy<2" smplx==0.1.28 timm einops ultralytics==8.1.34 opencv-python-headless \
    huggingface_hub scikit-image roma av scipy dill pip setuptools wheel six
$py -m pip install --no-deps --no-build-isolation git+https://github.com/mattloper/chumpy  # its setup.py imports pip
uv pip install --python $py --no-deps git+https://github.com/warmshao/WiLoR-mini
