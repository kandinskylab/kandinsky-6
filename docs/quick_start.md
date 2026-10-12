# Quick start

The Kandinsky pipeline. Framework ports have their own guides: [vLLM-Omni](quick_start_vllm.md), [SGLang](quick_start_sglang.md), [FastVideo](quick_start_fastvideo.md). The recipe index is [just.md](just.md).

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
```

Downloads one catalog snapshot into `$KANDINSKY_HOME/weights` when it is not already there. Names: `pro`, `pro-distill`, `pro-pretrain`, `lite`, `lite-distill`, `lite-pretrain`.

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
just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' --config kandinsky/configs/devices/rtx-5090.yaml --out clip.mp4
```

Image-to-video-and-audio uses the shipped frame:

```bash
just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' --image assets/i2va_input.jpg
```

| Flag | Role |
|---|---|
| `--config` | YAML preset. The default is `kandinsky/configs/devices/h100.yaml` |
| `--out` | Output file. Skips the stamped run directory |
| `--image` | Reference frame for image-to-video-and-audio. The shipped frame is `assets/i2va_input.jpg` |
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

Presets live in `kandinsky/configs/devices/`.

## Super-resolution

```bash
just generate-sr from-video --input clip.mp4 --output-dir outputs
```

Runs `kandy-sr` in `.venv` on an existing clip. Arguments are the super-resolution CLI: the subcommand, the input clip, and the output directory. Paths and scale also come from the SR config.
