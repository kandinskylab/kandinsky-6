#!/usr/bin/env bash
# Install one third-party framework into its own environment.
# vLLM-Omni, SGLang, and FastVideo need Python 3.12. They do not share
# the environment created by scripts/setup-gpu.sh.
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

name="${1:-}"
if [ "$#" -ne 1 ]; then
    echo "usage: setup-framework.sh vllm|sglang|fastvideo" >&2
    exit 2
fi

ensure_venv() {
    local venv="$1"
    uv venv --no-project --allow-existing --python 3.12 "$venv"
}

clone_main() {
    local url="$1"
    local dest="$2"
    if [ -d "$dest/.git" ]; then
        echo "reusing $dest"
        return 0
    fi
    if [ -e "$dest" ]; then
        echo "$dest exists and is not a git checkout" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$dest")"
    git clone --depth 1 "$url" "$dest"
}

case "$name" in
    vllm)
        ensure_venv .venvs/vllm-omni
        uv pip install --python .venvs/vllm-omni/bin/python \
            "vllm==0.31.0" --torch-backend=auto \
            --extra-index-url https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6
        clone_main https://github.com/vllm-project/vllm-omni.git .third-party/vllm-omni
        uv pip install --python .venvs/vllm-omni/bin/python -e .third-party/vllm-omni
        echo "vLLM-Omni is installed in .venvs/vllm-omni"
        ;;
    sglang)
        # Kandinsky 6 is not in a released SGLang package. Install current main.
        ensure_venv .venvs/sglang
        clone_main https://github.com/sgl-project/sglang.git .third-party/sglang
        uv pip install --python .venvs/sglang/bin/python --prerelease=allow \
            --directory .third-party/sglang -e "python[diffusion]"
        echo "SGLang is installed in .venvs/sglang"
        ;;
    fastvideo)
        ensure_venv .venvs/fastvideo
        backend="${UV_TORCH_BACKEND:-cu130}"
        UV_TORCH_BACKEND="$backend" uv pip install --python .venvs/fastvideo/bin/python fastvideo
        echo "FastVideo is installed in .venvs/fastvideo (UV_TORCH_BACKEND=$backend)"
        ;;
    *)
        echo "unknown framework: $name" >&2
        exit 2
        ;;
esac
