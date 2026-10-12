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

gpu_cap() {
    if ! command -v nvidia-smi >/dev/null; then
        echo "nvidia-smi is not on PATH" >&2
        exit 1
    fi
    local -a caps
    mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/ //g' | sort -u)
    if [ "${#caps[@]}" -ne 1 ]; then
        echo "GPUs report more than one compute capability: ${caps[*]}" >&2
        exit 1
    fi
    echo "${caps[0]}"
}

# FastVideo's FLASH_ATTN selector imports the flash_attn package, then uses
# flash_attn_interface (FA3) when that package is installed.
install_flash_attn() {
    local python="$1"
    if ! "$python" -c "import flash_attn" >/dev/null 2>&1; then
        echo "compiling FlashAttention-2 into $python"
        uv pip install --python "$python" --no-build-isolation flash-attn
    fi
    if ! "$python" -c "import flash_attn_interface" >/dev/null 2>&1; then
        echo "installing FlashAttention-3"
        uv pip install --python "$python" --no-deps \
            --default-index https://download.pytorch.org/whl/cu130 \
            "flash-attn-3==3.0.0"
    fi
    "$python" -c "import flash_attn, flash_attn_interface"
}

# SageAttention 2.2.0. SM121 (GB10) has no kernel in that tag, so the build targets 12.0.
install_sage() {
    local python="$1"
    local arch="$2"
    local src cuda_home
    if [ "$arch" = "12.1" ]; then
        arch="12.0"
    fi
    if "$python" -c "import sageattention" >/dev/null 2>&1; then
        echo "SageAttention is already installed"
        return 0
    fi
    cuda_home="${CUDA_HOME:-/usr/local/cuda}"
    if [ ! -x "$cuda_home/bin/nvcc" ]; then
        echo "nvcc not found at $cuda_home. Set CUDA_HOME to the CUDA toolkit." >&2
        exit 1
    fi
    src="$(mktemp -d)"
    git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git "$src/SageAttention"
    TORCH_CUDA_ARCH_LIST="$arch" CUDA_HOME="$cuda_home" \
        uv pip install --python "$python" --no-build-isolation --no-deps "$src/SageAttention"
    rm -rf "$src"
}

install_fastvideo_attention() {
    local python="$1"
    local cap
    cap=$(gpu_cap)
    case "$cap" in
        9.0)
            echo "compute capability $cap -> FlashAttention-3"
            install_flash_attn "$python"
            ;;
        8.0 | 8.6 | 8.9 | 12.0 | 12.1)
            echo "compute capability $cap -> SageAttention 2.2.0"
            install_sage "$python" "$cap"
            ;;
        *)
            echo "No FastVideo attention profile for compute capability $cap" >&2
            exit 1
            ;;
    esac
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
        uv pip install --python "$PWD/.venvs/sglang/bin/python" --prerelease=allow \
            --directory .third-party/sglang -e "python[diffusion]"
        echo "SGLang is installed in .venvs/sglang"
        ;;
    fastvideo)
        ensure_venv .venvs/fastvideo
        backend="${UV_TORCH_BACKEND:-cu130}"
        UV_TORCH_BACKEND="$backend" uv pip install --python .venvs/fastvideo/bin/python "fastvideo @ git+https://github.com/hao-ai-lab/FastVideo.git"
        install_fastvideo_attention .venvs/fastvideo/bin/python
        echo "FastVideo is installed in .venvs/fastvideo (UV_TORCH_BACKEND=$backend)"
        ;;
    *)
        echo "unknown framework: $name" >&2
        exit 2
        ;;
esac
