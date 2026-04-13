#!/bin/bash
# ============================================================
# setup_env.sh
# One-time setup for Lab 7 experiments.
# Run this ONCE before running lab7_experiments.py
#
# Usage:
#   cd /path/to/Lab_Session_7
#   bash setup_env.sh
# ============================================================

set -e  # stop on any error

echo "=============================================="
echo " Lab 7 — Environment Setup"
echo "=============================================="

# ---------- 1. Create virtual environment ----------
echo ""
echo "[1/4] Creating virtual environment: lab7_env"
python3 -m venv lab7_env
echo "      Done."

# ---------- 2. Activate venv ----------
source lab7_env/bin/activate
echo ""
echo "[2/4] Virtual environment activated."
echo "      Python: $(which python3)"

# ---------- 3. Install packages ----------
echo ""
echo "[3/4] Installing packages (this may take a few minutes)..."

pip install --upgrade pip --quiet

pip install \
    scikit-learn \
    pandas \
    numpy \
    openpyxl \
    transformers \
    torch \
    --quiet

echo "      All packages installed."

# ---------- 4. Pre-download BERT & RoBERTa model weights ----------
echo ""
echo "[4/4] Pre-downloading BERT and RoBERTa weights from HuggingFace..."
echo "      (saves time when running experiments)"

python3 - <<'PYEOF'
from transformers import AutoTokenizer, AutoModel

print("  Downloading bert-base-uncased ...")
AutoTokenizer.from_pretrained("bert-base-uncased")
AutoModel.from_pretrained("bert-base-uncased")
print("  bert-base-uncased  ✓")

print("  Downloading roberta-base ...")
AutoTokenizer.from_pretrained("roberta-base")
AutoModel.from_pretrained("roberta-base")
print("  roberta-base       ✓")
PYEOF

# ---------- Done ----------
echo ""
echo "=============================================="
echo " Setup complete!"
echo "=============================================="
echo ""
echo " To run the experiments:"
echo ""
echo "   source lab7_env/bin/activate"
echo "   python lab7_experiments.py"
echo ""
echo " Or run the notebook:"
echo ""
echo "   source lab7_env/bin/activate"
echo "   jupyter notebook lab7.ipynb"
echo ""
