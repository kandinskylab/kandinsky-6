# Published commands. Run from the repository root.
#   just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'

# Install runtime dependencies for the GPU in this machine. Skips the dev group.
# Hopper: wheels only. Ampere, Ada, consumer Blackwell: also compiles Sage.
setup *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/setup-gpu.sh" --no-dev {{ARGS}}

# Generate a video: just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
# Compose presets: just generate 'A cat ...' --configs h100 spd       (device + accelerations/*)
# Bench or notebook YAMLs: just generate 'A cat ...' --config /path/to/resolved.yaml
# First and last frame: just generate 'The person walks toward the doorway.' --image first.png --last-image last.png
# Audio conditions the video: just generate 'A person speaks.' --audio-file speech.wav
# Video conditions the audio: just generate 'Ocean waves and seabirds.' --video-file silent.mp4
generate PROMPT *ARGS:
    @uv run kandy generate {{quote(PROMPT)}} {{ARGS}}

# Super-resolve a clip. Paths and scale come from the SR config or the arguments.
# just generate-sr from-video --input clip.mp4 --output-dir outputs
generate-sr *ARGS:
    @uv run kandy-sr {{ARGS}}

# Download catalog snapshots into $KANDINSKY_HOME/weights.
# pro, pro-distill, pro-pretrain, lite, lite-distill, lite-pretrain.
# just download pro-distill
# just download lite pro
# just download pro-distill --cache-dir /path/to/weights
download NAME *ARGS:
    @uv run kandy download {{NAME}} {{ARGS}}

# Each checkpoint stays loaded for SD and Full HD. Compile is on.
# Three generations, median time, offload chosen from VRAM.
# Non-distilled checkpoints use MagCache from the checkpoint file.
# Full HD includes super-resolution. Profile JSON files go to
# profiles/ inside the run folder. module-time-report.html is written there too.
# just bench
# Eight t2v timings on one GPU: Lite/Pro, distilled and not, SD/Full HD. just bench [--gpu 0]
bench *ARGS:
    @uv run kandy bench {{ARGS}}

# Leave 16, 24, or 32 GB free on each GPU and generate once per consumer preset.
# cudaMalloc holds the rest outside the PyTorch allocator, so an H100 stands in for that card.
# Jobs run in parallel, one process per GPU.
# just memfit
# just memfit --gpus 0 1
memfit *ARGS:
    @uv run kandy memfit {{ARGS}}

# vLLM 0.31.0 and vLLM-Omni main, in .venvs/vllm-omni. Separate from `just setup`.
setup-vllm:
    #!/usr/bin/env bash
    set -euo pipefail
    exec "{{justfile_directory()}}/scripts/setup-framework.sh" vllm

# SGLang 0.5.22 with the diffusion extra, in .venvs/sglang. Separate from `just setup`.
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
