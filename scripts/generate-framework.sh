#!/usr/bin/env bash
# Run one third-party generate or serve command in the environment from setup-framework.sh.
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

name="${1:-}"
prompt=""
case "$name" in
    serve-vllm|serve-sglang)
        shift
        ;;
    vllm|sglang|fastvideo)
        prompt="${2:-}"
        if [ "$#" -lt 2 ] || [ -z "$prompt" ]; then
            echo "usage: generate-framework.sh vllm|sglang|fastvideo PROMPT [args...]" >&2
            exit 2
        fi
        shift 2
        ;;
    *)
        echo "usage: generate-framework.sh vllm|sglang|fastvideo PROMPT [args...]" >&2
        echo "       generate-framework.sh serve-vllm|serve-sglang [args...]" >&2
        exit 2
        ;;
esac

require_bin() {
    local bin="$1"
    local recipe="$2"
    if [ ! -x "$bin" ]; then
        echo "missing $bin. Run just $recipe first." >&2
        exit 1
    fi
}

# A complete Diffusers snapshot under $KANDINSKY_HOME/weights, or empty.
# Hub ids are not downloaded from here.
local_snapshot() {
    python3 - "$1" "$2" <<'PY'
import sys
from pathlib import Path

repo, home = sys.argv[1:]
path = Path(repo)
if path.is_dir() and (path / "model_index.json").is_file():
    transformer = path / "transformer"
    if transformer.is_dir() and (
        any(transformer.glob("*.safetensors"))
        or (transformer / "diffusion_pytorch_model.safetensors.index.json").is_file()
    ):
        print(path)
    raise SystemExit(0)

root = Path(home) / "weights" / ("models--" + repo.replace("/", "--"))
snapshots = root / "snapshots"

def ready(candidate: Path) -> bool:
    transformer = candidate / "transformer"
    if not (candidate / "model_index.json").is_file() or not transformer.is_dir():
        return False
    if any(transformer.glob("*.safetensors")):
        return True
    return (transformer / "diffusion_pytorch_model.safetensors.index.json").is_file()

if snapshots.is_dir():
    ref = root / "refs" / "main"
    if ref.is_file():
        pointed = snapshots / ref.read_text(encoding="utf-8").strip()
        if ready(pointed):
            print(pointed)
            raise SystemExit(0)
    for candidate in sorted(snapshots.iterdir()):
        if candidate.is_dir() and ready(candidate):
            print(candidate)
            raise SystemExit(0)
PY
}

# A complete local snapshot, or the Hub id so the download lands in $HF_HUB_CACHE.
# A directory that is not a complete snapshot is an error.
resolve_weights() {
    local repo="$1"
    local home="$2"
    local found
    found=$(local_snapshot "$repo" "$home")
    if [ -n "$found" ]; then
        printf '%s\n' "$found"
        return 0
    fi
    if [ -d "$repo" ]; then
        echo "No complete snapshot at $repo." >&2
        return 1
    fi
    printf '%s\n' "$repo"
}

# height width frames steps guidance, from kandinsky/configs/checkpoints.
# The file is chosen by Hub id, directory name, or checkpoint name (pro-distill).
checkpoint_generation() {
    python3 - "$1" <<'PY'
import sys
from pathlib import Path

needle = sys.argv[1]
name = Path(needle).name
root = Path("kandinsky/configs/checkpoints")

def parse(path: Path) -> tuple[str | None, str | None, dict[str, str]]:
    repo = None
    checkpoint = None
    generation: dict[str, str] = {}
    in_generation = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if repo is None and raw.startswith("# ") and "kandinskylab/" in raw:
            repo = raw.split("#", 1)[1].strip().split()[0]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith(" ") and raw.endswith(":"):
            in_generation = raw[:-1] == "generation"
            continue
        if raw.startswith("checkpoint:"):
            checkpoint = raw.split(":", 1)[1].strip()
            continue
        if in_generation and raw.startswith("  ") and ":" in raw:
            key, value = raw.strip().split(":", 1)
            generation[key] = value.strip()
    return repo, checkpoint, generation

folded = needle.replace("/", "--")
parts = set(Path(needle).parts)
match = None
for path in sorted(root.glob("*.yaml")):
    repo, checkpoint, generation = parse(path)
    repo_name = repo.split("/")[-1] if repo else None
    repo_cache = repo.replace("/", "--") if repo else None
    if (
        needle in {repo, repo_name, checkpoint}
        or name in {repo_name, checkpoint}
        or (repo_name is not None and repo_name in parts)
        or (repo_cache is not None and repo_cache in folded)
    ):
        match = (checkpoint, generation)
        break
if match is None:
    raise SystemExit(f"No checkpoint config in {root} matches {needle}")
checkpoint, generation = match
needed = ("height", "width", "sample_frames", "num_steps", "guidance_weight")
missing = [key for key in needed if key not in generation]
if missing:
    raise SystemExit(f"Checkpoint config for {needle} is missing {', '.join(missing)}")
print(
    checkpoint,
    generation["height"],
    generation["width"],
    generation["sample_frames"],
    generation["num_steps"],
    generation["guidance_weight"],
)
PY
}

# JSON cache config for vLLM, or the word none when the checkpoint has no curve.
checkpoint_magcache() {
    python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

needle = sys.argv[1]
name = Path(needle).name
root = Path("kandinsky/configs/checkpoints")

def parse(path: Path) -> tuple[str | None, str | None, dict[str, str]]:
    repo = None
    checkpoint = None
    mag: dict[str, str] = {}
    section = None
    in_mag = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if repo is None and raw.startswith("# ") and "kandinskylab/" in raw:
            repo = raw.split("#", 1)[1].strip().split()[0]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith(" ") and raw.endswith(":"):
            section = raw[:-1]
            in_mag = False
            continue
        if raw.startswith("checkpoint:"):
            checkpoint = raw.split(":", 1)[1].strip()
            continue
        if section == "cache" and raw.startswith("  ") and not raw.startswith("    ") and stripped.endswith(":"):
            in_mag = stripped[:-1] == "magcache"
            continue
        if in_mag and raw.startswith("    ") and ":" in raw:
            key, value = raw.strip().split(":", 1)
            mag[key] = value.strip()
    return repo, checkpoint, mag

folded = needle.replace("/", "--")
parts = set(Path(needle).parts)
found = None
for path in sorted(root.glob("*.yaml")):
    repo, checkpoint, mag = parse(path)
    repo_name = repo.split("/")[-1] if repo else None
    repo_cache = repo.replace("/", "--") if repo else None
    if (
        needle in {repo, repo_name, checkpoint}
        or name in {repo_name, checkpoint}
        or (repo_name is not None and repo_name in parts)
        or (repo_cache is not None and repo_cache in folded)
    ):
        found = mag
        break
if not found:
    print("none")
    raise SystemExit(0)
ratios = json.loads(found["mag_ratios"])
print(
    json.dumps(
        {
            "mag_threshold": float(found["thresh"]),
            "mag_max_skip_steps": int(found["K"]),
            "mag_retention_ratio": float(found["retention_ratio"]),
            "mag_ratios": ratios,
        },
        separators=(",", ":"),
    )
)
PY
}

# Same split as scripts/setup-gpu.sh. Hopper uses FlashAttention (FA3 when
# flash_attn_interface is installed). Ampere, Ada, and consumer Blackwell use SageAttention.
attention_for_gpu() {
    if ! command -v nvidia-smi >/dev/null; then
        echo "nvidia-smi is not on PATH; FastVideo will pick its own attention backend" >&2
        return 0
    fi
    local -a caps
    mapfile -t caps < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/ //g' | sort -u)
    if [ "${#caps[@]}" -ne 1 ]; then
        echo "GPUs report more than one compute capability: ${caps[*]}" >&2
        return 1
    fi
    case "${caps[0]}" in
        9.0) echo FLASH_ATTN ;;
        8.0 | 8.6 | 8.9 | 12.0 | 12.1) echo SAGE_ATTN ;;
        *)
            echo "No FastVideo attention profile for compute capability ${caps[0]}" >&2
            return 1
            ;;
    esac
}

case "$name" in
    vllm)
        require_bin .venvs/vllm-omni/bin/python setup-vllm
        script=".third-party/vllm-omni/examples/offline_inference/text_to_video/text_to_video.py"
        repo="kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
        filtered=()
        skip_next=""
        for arg in "$@"; do
            if [ -n "$skip_next" ]; then
                case "$skip_next" in
                    model) repo="$arg" ;;
                esac
                skip_next=""
                continue
            fi
            case "$arg" in
                --image|--image=*)
                    script=".third-party/vllm-omni/examples/offline_inference/image_to_video/image_to_video.py"
                    filtered+=("$arg")
                    ;;
                --model)
                    skip_next=model
                    ;;
                --model=*)
                    repo="${arg#--model=}"
                    ;;
                --fps)
                    skip_next=drop
                    ;;
                --fps=*)
                    ;;
                *)
                    filtered+=("$arg")
                    ;;
            esac
        done
        if [ ! -f "$script" ]; then
            echo "missing $script. Run just setup-vllm first." >&2
            exit 1
        fi
        home="${KANDINSKY_HOME:-$HOME/.cache/kandinsky}"
        home=$(python3 -c 'import os, sys; print(os.path.expanduser(os.path.expandvars(sys.argv[1])))' "$home")
        export HF_HUB_CACHE="$home/weights"
        unset HF_HUB_OFFLINE
        model=$(resolve_weights "$repo" "$home")
        stamp=$(date +%Y-%m-%dT%H-%M-%S)
        run="$home/outputs/vllm_$stamp"
        if [ -e "$run" ]; then
            run="$home/outputs/vllm_$(date +%Y-%m-%dT%H-%M-%S-%N)"
        fi
        mkdir -p "$run/generations" "$run/logs"
        python3 - "$run/launch.json" "$prompt" "$model" <<'PY'
import json
import sys

path, prompt, model = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {"name": "vllm", "prompt": prompt, "model": model},
        handle,
        indent=2,
    )
    handle.write("\n")
PY
        echo "run: $run"
        read -r gen_checkpoint gen_height gen_width gen_frames gen_steps gen_guidance < <(checkpoint_generation "$repo")
        mag=$(checkpoint_magcache "$repo")
        cache_args=()
        user_sets_cache=0
        for arg in "${filtered[@]}"; do
            case "$arg" in
                --cache-backend|--cache-backend=*) user_sets_cache=1 ;;
            esac
        done
        # A checkpoint without cache.magcache stays uncached. A later --cache-backend replaces this.
        if [ "$user_sets_cache" -eq 0 ] && [ "$mag" != "none" ]; then
            cache_args=(--cache-backend mag_cache --cache-config "$mag")
        fi
        set -o pipefail
        .venvs/vllm-omni/bin/python "$script" \
            --model "$model" \
            --prompt "$prompt" \
            --seed 42 \
            --height "$gen_height" \
            --width "$gen_width" \
            --num-frames "$gen_frames" \
            --num-inference-steps "$gen_steps" \
            --guidance-scale "$gen_guidance" \
            --fps 24 \
            --enable-cpu-offload \
            --output "$run/generations/output.mp4" \
            "${cache_args[@]}" \
            "${filtered[@]}" \
            2>&1 | tee "$run/logs/vllm.log"
        ;;
    sglang)
        require_bin .venvs/sglang/bin/sglang setup-sglang
        repo="kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
        filtered=()
        skip_next=""
        output_file=""
        for arg in "$@"; do
            if [ -n "$skip_next" ]; then
                case "$skip_next" in
                    model) repo="$arg" ;;
                    image) filtered+=(--image-path "$arg") ;;
                    output) output_file="$arg" ;;
                    drop) ;;
                esac
                skip_next=""
                continue
            fi
            case "$arg" in
                --model|--model-path) skip_next=model ;;
                --model=*|--model-path=*) repo="${arg#*=}" ;;
                --image) skip_next=image ;;
                --image=*) filtered+=(--image-path "${arg#--image=}") ;;
                --fps) skip_next=drop ;;
                --fps=*) ;;
                --output) skip_next=output ;;
                --output=*) output_file="${arg#--output=}" ;;
                *) filtered+=("$arg") ;;
            esac
        done
        home="${KANDINSKY_HOME:-$HOME/.cache/kandinsky}"
        home=$(python3 -c 'import os, sys; print(os.path.expanduser(os.path.expandvars(sys.argv[1])))' "$home")
        export HF_HUB_CACHE="$home/weights"
        unset HF_HUB_OFFLINE
        model=$(resolve_weights "$repo" "$home")
        stamp=$(date +%Y-%m-%dT%H-%M-%S)
        run="$home/outputs/sglang_$stamp"
        if [ -e "$run" ]; then
            run="$home/outputs/sglang_$(date +%Y-%m-%dT%H-%M-%S-%N)"
        fi
        mkdir -p "$run/generations" "$run/logs"
        read -r gen_checkpoint gen_height gen_width gen_frames gen_steps gen_guidance < <(checkpoint_generation "$repo")
        hub_name=$(sed -n 's/^# kandinskylab\///p' "kandinsky/configs/checkpoints/${gen_checkpoint}.yaml" | head -n 1 | awk '{print $1}')
        out_dir="$run/generations"
        out_name="output.mp4"
        if [ -n "$output_file" ]; then
            case "$output_file" in
                *.mp4)
                    out_dir=$(dirname "$output_file")
                    out_name=$(basename "$output_file")
                    ;;
                *)
                    out_dir="$output_file"
                    ;;
            esac
            mkdir -p "$out_dir"
        fi
        python3 - "$run/launch.json" "$prompt" "$model" "$gen_checkpoint" <<'PY'
import json
import sys

path, prompt, model, checkpoint = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {"name": "sglang", "prompt": prompt, "model": model, "checkpoint": checkpoint},
        handle,
        indent=2,
    )
    handle.write("\n")
PY
        echo "run: $run"
        set -o pipefail
        .venvs/sglang/bin/sglang generate \
            --model-path "$model" \
            --model-id "$hub_name" \
            --performance-mode memory \
            --attention-backend fa \
            --warmup-mode off \
            --prompt "$prompt" \
            --seed 42 \
            --height "$gen_height" \
            --width "$gen_width" \
            --num-frames "$gen_frames" \
            --num-inference-steps "$gen_steps" \
            --guidance-scale "$gen_guidance" \
            --fps 24 \
            --save-output \
            --output-path "$out_dir" \
            --output-file-name "$out_name" \
            "${filtered[@]}" \
            2>&1 | tee "$run/logs/sglang.log"
        ;;
    serve-vllm)
        require_bin .venvs/vllm-omni/bin/vllm setup-vllm
        repo="kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
        filtered=()
        skip_next=""
        host="127.0.0.1"
        port="8091"
        for arg in "$@"; do
            if [ -n "$skip_next" ]; then
                case "$skip_next" in
                    model) repo="$arg" ;;
                    host) host="$arg" ;;
                    port) port="$arg" ;;
                esac
                skip_next=""
                continue
            fi
            case "$arg" in
                --model) skip_next=model ;;
                --model=*) repo="${arg#--model=}" ;;
                --host) skip_next=host ;;
                --host=*) host="${arg#--host=}" ;;
                --port) skip_next=port ;;
                --port=*) port="${arg#--port=}" ;;
                *) filtered+=("$arg") ;;
            esac
        done
        home="${KANDINSKY_HOME:-$HOME/.cache/kandinsky}"
        home=$(python3 -c 'import os, sys; print(os.path.expanduser(os.path.expandvars(sys.argv[1])))' "$home")
        export HF_HUB_CACHE="$home/weights"
        unset HF_HUB_OFFLINE
        model=$(resolve_weights "$repo" "$home")
        read -r _ _ _ _ gen_steps gen_guidance < <(checkpoint_generation "$repo")
        sampling=$(python3 -c 'import json, sys; print(json.dumps({"0": {"num_inference_steps": int(sys.argv[1]), "guidance_scale": float(sys.argv[2])}}, separators=(",", ":")))' "$gen_steps" "$gen_guidance")
        mag=$(checkpoint_magcache "$repo")
        cache_args=()
        sampling_args=(--default-sampling-params "$sampling")
        user_sets_cache=0
        for arg in "${filtered[@]}"; do
            case "$arg" in
                --cache-backend|--cache-backend=*) user_sets_cache=1 ;;
                --default-sampling-params|--default-sampling-params=*) sampling_args=() ;;
            esac
        done
        if [ "$user_sets_cache" -eq 0 ] && [ "$mag" != "none" ]; then
            cache_args=(--cache-backend mag_cache --cache-config "$mag")
        fi
        echo "listening: http://$host:$port"
        exec .venvs/vllm-omni/bin/vllm serve "$model" --omni \
            --host "$host" --port "$port" \
            --num-gpus 1 \
            --enable-cpu-offload \
            "${sampling_args[@]}" \
            "${cache_args[@]}" \
            "${filtered[@]}"
        ;;
    serve-sglang)
        require_bin .venvs/sglang/bin/sglang setup-sglang
        repo="kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
        filtered=()
        skip_next=""
        host="127.0.0.1"
        port="30000"
        for arg in "$@"; do
            if [ -n "$skip_next" ]; then
                case "$skip_next" in
                    model) repo="$arg" ;;
                    host) host="$arg" ;;
                    port) port="$arg" ;;
                esac
                skip_next=""
                continue
            fi
            case "$arg" in
                --model|--model-path) skip_next=model ;;
                --model=*|--model-path=*) repo="${arg#*=}" ;;
                --host) skip_next=host ;;
                --host=*) host="${arg#--host=}" ;;
                --port) skip_next=port ;;
                --port=*) port="${arg#--port=}" ;;
                *) filtered+=("$arg") ;;
            esac
        done
        home="${KANDINSKY_HOME:-$HOME/.cache/kandinsky}"
        home=$(python3 -c 'import os, sys; print(os.path.expanduser(os.path.expandvars(sys.argv[1])))' "$home")
        export HF_HUB_CACHE="$home/weights"
        unset HF_HUB_OFFLINE
        model=$(resolve_weights "$repo" "$home")
        read -r gen_checkpoint _ < <(checkpoint_generation "$repo")
        hub_name=$(sed -n 's/^# kandinskylab\///p' "kandinsky/configs/checkpoints/${gen_checkpoint}.yaml" | head -n 1 | awk '{print $1}')
        echo "listening: http://$host:$port"
        exec .venvs/sglang/bin/sglang serve \
            --model-path "$model" \
            --model-id "$hub_name" \
            --performance-mode memory \
            --attention-backend fa \
            --host "$host" \
            --port "$port" \
            "${filtered[@]}"
        ;;
    fastvideo)
        require_bin .venvs/fastvideo/bin/python setup-fastvideo
        home="${KANDINSKY_HOME:-$HOME/.cache/kandinsky}"
        home=$(python3 -c 'import os, sys; print(os.path.expanduser(os.path.expandvars(sys.argv[1])))' "$home")
        export HF_HUB_CACHE="$home/weights"
        if [ -z "${FASTVIDEO_ATTENTION_BACKEND:-}" ]; then
            FASTVIDEO_ATTENTION_BACKEND=$(attention_for_gpu)
            export FASTVIDEO_ATTENTION_BACKEND
        fi
        stamp=$(date +%Y-%m-%dT%H-%M-%S)
        run="$home/outputs/fastvideo_$stamp"
        if [ -e "$run" ]; then
            run="$home/outputs/fastvideo_$(date +%Y-%m-%dT%H-%M-%S-%N)"
        fi
        mkdir -p "$run/generations" "$run/logs"
        python3 - "$run/launch.json" "$prompt" "$FASTVIDEO_ATTENTION_BACKEND" <<'PY'
import json
import sys

path, prompt, backend = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {"name": "fastvideo", "prompt": prompt, "attention_backend": backend},
        handle,
        indent=2,
    )
    handle.write("\n")
PY
        echo "run: $run"
        fv_repo="kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
        prev=""
        for arg in "$@"; do
            if [ "$prev" = "--model" ]; then
                fv_repo="$arg"
            fi
            case "$arg" in
                --model=*) fv_repo="${arg#--model=}" ;;
            esac
            prev="$arg"
        done
        read -r gen_checkpoint gen_height gen_width gen_frames gen_steps gen_guidance < <(checkpoint_generation "$fv_repo")
        set -o pipefail
        .venvs/fastvideo/bin/python examples/fastvideo_ti2va.py "$prompt" \
            --height "$gen_height" \
            --width "$gen_width" \
            --num-frames "$gen_frames" \
            --num-inference-steps "$gen_steps" \
            --guidance-scale "$gen_guidance" \
            --output "$run/generations/output.mp4" \
            "$@" \
            2>&1 | tee "$run/logs/fastvideo.log"
        ;;
    *)
        echo "unknown framework: $name" >&2
        exit 2
        ;;
esac
