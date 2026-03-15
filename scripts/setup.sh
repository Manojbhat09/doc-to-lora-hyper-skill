#!/usr/bin/env bash
# Setup Doc-to-LoRA on macOS (Apple Silicon).
# Downloads model weights and installs dependencies.
#
# Prerequisites (install these first):
#   - Python 3.10+
#   - uv: https://docs.astral.sh/uv/getting-started/installation/
#   - HF_TOKEN: https://huggingface.co/settings/tokens (needed for Gemma access)
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Doc-to-LoRA Setup (macOS) ==="
echo "Repo root: $REPO_ROOT"

# 1. Check prerequisites
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is not installed."
    echo "Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
echo "[1/4] uv found: $(uv --version)"

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set."
    echo "Gemma 2 2B is a gated model. You need a HuggingFace token with Gemma access."
    echo "Get one at: https://huggingface.co/settings/tokens"
    echo "Then: export HF_TOKEN=hf_..."
fi

# 2. Run the Mac install script (creates venv, installs deps)
if [ ! -d ".venv" ]; then
    echo "[2/4] Installing Python dependencies (Mac-compatible)..."
    bash install_mac.sh
else
    echo "[2/4] .venv already exists, skipping dependency install."
fi

# Activate
source .venv/bin/activate

# 3. Install MLX dependencies for fast Apple Silicon inference
echo "[3/4] Installing MLX dependencies..."
uv pip install mlx mlx-lm safetensors 2>/dev/null || true

# 4. Download pretrained D2L weights from HuggingFace
if [ ! -d "trained_d2l" ]; then
    echo "[4/4] Downloading Doc-to-LoRA weights (~3GB)..."
    uv run huggingface-cli download SakanaAI/doc-to-lora --local-dir trained_d2l
else
    echo "[4/4] trained_d2l/ already present, skipping download."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate:  source .venv/bin/activate"
echo ""
echo "Quick test:"
echo "  python demo_dario.py"
