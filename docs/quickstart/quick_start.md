# Quick start

The Kandinsky pipeline. Framework ports have their own guides: [vLLM-Omni](quick_start_vllm.md), [SGLang](quick_start_sglang.md), [FastVideo](quick_start_fastvideo.md). The recipe index is [just.md](../just.md). Generation modes are [generation modes](../generation_modes.md).

An NVIDIA GPU and Python 3.13 or 3.14. You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Run every recipe from the repository root.

`$KANDINSKY_HOME` is the root for weights, logs, and runs. When it is unset, that root is `~/.cache/kandinsky`.

## Setup

```bash
git clone https://github.com/kandinskylab/kandinsky-6.git
cd kandinsky-6
just setup
```

`just setup` installs the pipeline into `.venv` and reads the compute capability of the GPUs in this machine. Every GPU must report the same capability. It installs one PyTorch build:

| GPUs | Compute capability | Installed |
|---|---|---|
| Hopper (H100, H200) | 9.0 | PyTorch for CUDA 13.0, FlashAttention 3 |
| Ampere, Ada, consumer Blackwell | 8.0, 8.6, 8.9, 12.0, 12.1 | PyTorch for CUDA 13.0, then SageAttention 2.2.0 |

Ampere, Ada, and consumer Blackwell compile SageAttention, so `nvcc` has to be on `PATH`. `CUDA_HOME` defaults to `/usr/local/cuda`. Extra arguments are passed to `uv sync`.

## Download

```bash
just download pro-distill
just download lite pro
```

Downloads each named catalog snapshot into `$KANDINSKY_HOME/weights` when it is not already there. Names: `pro`, `pro-distill`, `pro-pretrain`, `lite`, `lite-distill`, `lite-pretrain`.

```bash
just download pro-distill --cache-dir /path/to/weights
```

`just generate` also downloads the checkpoint named in the config if it is not already on disk.

## Generate

```bash
just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
```

The first run downloads [Kandinsky-6.0-Pro-distill-5s](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) into `$KANDINSKY_HOME/weights`. Without `--out`, the clip is `$KANDINSKY_HOME/outputs/generate_<YYYY-MM-DDTHH-MM-SS>/generations/output.mp4`. That folder also holds `launch.json`, `expanded_prompt.txt`, `logs/`, and `profiles/`.

Anything after the prompt is a `kandy generate` flag:

```bash
just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' --configs rtx-5090 --out clip.mp4
```

Presets compose by name. `--configs h100 spd` deep-merges the device preset with `configs/accelerations/spd.yaml`, switches to the Pro checkpoint, and wraps the base denoising loop with the staged multi-resolution [SPEED](https://arxiv.org/abs/2605.18736) strategy. `--config <path>` loads a single YAML (bench runs, resolved configs) and excludes `--configs`.

Image-to-video-and-audio uses the shipped frame. The other modes are in [generation modes](../generation_modes.md).

```bash
just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' --image assets/i2va_input.jpg
```

| Flag | Role |
|---|---|
| `--configs` | Preset tokens resolved across `configs/{devices,checkpoints,accelerations}` and deep-merged left to right. The default is `--configs h100` |
| `--config` | One YAML path. Excludes `--configs` |
| `--out` | Output file. Skips the stamped run directory |
| `--image` | Reference frame for image-to-video-and-audio. The shipped frame is `assets/i2va_input.jpg` |
| `--last-image` | Last reference of one video. Requires `--image` |
| `--audio-file` | Audio file that conditions the video |
| `--video-file` | Video file that conditions the audio |
| `--no-audio` | Video only |
| `--steps`, `--guidance`, `--seed` | Sampler overrides |
| `--height`, `--width`, `--time-length`, `--latent-frames` | Geometry overrides |
| `--device` | Torch device. The default is `cuda` |
| `--attention-engine` | Attention override from the config |
| `--offload` | `none`, `module`, or `block` |
| `--magcache`, `--navicache` | Cache override. The two flags exclude each other. MagCache ratios, threshold, K, and retention come from `cache.magcache` in the checkpoint file. Pro and Lite have a curve |
| `--negative` | Negative prompt |
| `--sr` | Run super-resolution after the clip |
| `--warmup` | One unsaved pass, then the saved run |
| `--no-progress` | Hide the step bar |

Presets live in `kandinsky/configs/{devices,checkpoints,accelerations}/`. Device presets pre-validate one GPU, checkpoint tokens override the model, and acceleration tokens flip a coordinated subset of sampler-level settings.

## Super-resolution

```bash
just generate-sr from-video --input clip.mp4 --output-dir outputs
```

Runs `kandy-sr` in `.venv` on an existing clip. Arguments are the super-resolution CLI: the subcommand, the input clip, and the output directory. Paths and scale also come from the SR config.


## Temporal and Adaptive Block Cache

Use these presets with Pro Distill PiFlow for one 864×480 text-to-video clip
with audio, 31 full-grid video latents, and ten DiT calls. They support eager
execution and regional torch.compile with no offload or module offload.

```bash
just generate 'A person speaks.' --configs h100 pro-distill temporal-fixed
just generate 'A person speaks.' --configs h100 pro-distill temporal-a2
just generate 'A person speaks.' --configs h100 pro-distill adaptive-block-cache
just generate 'A person speaks.' --configs h100 pro-distill temporal-a2 adaptive-block-cache

just generate 'A person speaks.' --configs h100 pro-distill temporal-fixed adaptive-block-cache
```

Fixed Temporal switches to 31 video latents after two calls on 16 latents.
Adaptive Temporal integrates the predicted clean video's high-frequency energy
fraction over the full-grid noise schedule and switches after at most eight
reduced-grid calls. Audio follows the base PiFlow schedule in both modes.

Adaptive Block Cache can run on its own or with either Temporal preset. It
computes the first fused block on every call and reuses the remaining blocks'
paired video and audio residuals when the drift estimates are within their
thresholds. It needs two full calls on the 31-frame grid before it can hit and
always computes the final call in full.

With `profile.mode: time_mem_module`, the profile JSON includes the number of
reduced-grid calls before the transition and the zero-based cache hit indices.
