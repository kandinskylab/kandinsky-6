# Published commands. Run from the repository root.
#   just generate "a cat on a mat"

# Install runtime dependencies for the GPU in this machine. Skips the dev group.
# Hopper: wheels only. Ampere, Ada, consumer Blackwell: also compiles Sage.
setup *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/setup-gpu.sh" --no-dev {{ARGS}}

# Generate a video: just generate "a cat on a mat"
generate PROMPT *ARGS:
    @uv run kandy generate "{{PROMPT}}" {{ARGS}}

# Super-resolve a clip. Paths and scale come from the SR config or the arguments.
# just generate-sr from-video --input clip.mp4 --output-dir outputs
generate-sr *ARGS:
    @uv run kandy-sr {{ARGS}}

# Download a catalog snapshot into $KANDINSKY_HOME/weights.
# pro, pro-distill, pro-pretrain, lite, lite-distill, lite-pretrain.
# just download pro-distill
# just download pro-distill --cache-dir /path/to/weights
download NAME *ARGS:
    @uv run kandy download {{NAME}} {{ARGS}}

# Export one visual block to an AOTInductor .pt2. Weights stay in safetensors.
# just export --config kandinsky/configs/devices/h100.yaml --out export/dit.pt2 --device cuda:0
export *ARGS:
    @uv run kandy export {{ARGS}}

# vLLM 0.31.0 and vLLM-Omni main, in .venvs/vllm-omni. Separate from `just setup`.
setup-vllm:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/setup-framework.sh" vllm

# SGLang main with the diffusion extra, in .venvs/sglang. Separate from `just setup`.
setup-sglang:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/setup-framework.sh" sglang

# FastVideo in .venvs/fastvideo. CUDA 13.0, or UV_TORCH_BACKEND=cu126 just setup-fastvideo.
setup-fastvideo:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/setup-framework.sh" fastvideo
