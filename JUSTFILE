# Published commands. Run from the repository root.
#   just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'

# Install runtime dependencies for the GPU in this machine. Skips the dev group.
# Hopper: wheels only. Ampere, Ada, consumer Blackwell: also compiles Sage.
setup *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/setup-gpu.sh" --no-dev {{ARGS}}

# Generate a video: just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
generate PROMPT *ARGS:
    @uv run kandy generate {{quote(PROMPT)}} {{ARGS}}

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

# just generate-vllm 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' [--image assets/i2va_input.jpg]
generate-vllm PROMPT *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/generate-framework.sh" vllm {{quote(PROMPT)}} {{ARGS}}

# just generate-sglang 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' [--image assets/i2va_input.jpg]
generate-sglang PROMPT *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/generate-framework.sh" sglang {{quote(PROMPT)}} {{ARGS}}

# just serve-vllm [--model kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers] [--port 8091]
serve-vllm *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/generate-framework.sh" serve-vllm {{ARGS}}

# just serve-sglang [--performance-mode speed] [--port 30000]
serve-sglang *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/generate-framework.sh" serve-sglang {{ARGS}}

# just generate-fastvideo 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' [--image assets/i2va_input.jpg]
generate-fastvideo PROMPT *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/generate-framework.sh" fastvideo {{quote(PROMPT)}} {{ARGS}}
