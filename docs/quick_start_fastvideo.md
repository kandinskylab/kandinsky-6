# Quick start: FastVideo

[FastVideo](https://haoailab.com/FastVideo/inference/kandinsky6/) for Kandinsky 6. This environment is separate from [the pipeline](quick_start.md). The recipe index is [just.md](just.md).

You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Run every recipe from the repository root.

`$KANDINSKY_HOME` is the root for weights and runs. When it is unset, that root is `~/.cache/kandinsky`.

## Setup

```bash
just setup-fastvideo
```

Installs FastVideo into `.venvs/fastvideo` (Python 3.12). There is no checkout directory; the package is installed from git. The PyTorch build is CUDA 13.0. On CUDA 12.6:

```bash
UV_TORCH_BACKEND=cu126 just setup-fastvideo
```

The same command installs the attention package for this GPU:

| GPUs | Compute capability | Attention |
|---|---|---|
| Hopper | 9.0 | FlashAttention 3 (`FLASH_ATTN`; the `flash_attn` package is installed so FastVideo does not fall back to SDPA) |
| Ampere, Ada, consumer Blackwell | 8.0, 8.6, 8.9, 12.0, 12.1 | SageAttention 2.2.0 (`SAGE_ATTN`) |

SageAttention needs `nvcc`. `CUDA_HOME` defaults to `/usr/local/cuda`. This does not replace the `.venv` from `just setup`.

`just generate-fastvideo` stops and names this recipe when the environment is missing.

## Generate

```bash
just generate-fastvideo 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
```

Uses `.venvs/fastvideo` and `examples/fastvideo_ti2va.py`. The default checkpoint is Pro-distill. A checkpoint already stored in `$KANDINSKY_HOME/weights` is reused. Hugging Face downloads for this recipe go to that same weights directory.

Frame size, frame count, step count, and guidance come from the matching file in `kandinsky/configs/checkpoints/`. The default model reads `pro-distill.yaml`: 864×480, 121 frames, 10 steps, guidance 1. `--model kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers` reads `pro.yaml` instead (50 steps, guidance 5). A later flag replaces the value from that file. The clip is 24 fps. The model is trained at that rate.

The recipe sets `FASTVIDEO_ATTENTION_BACKEND` from the GPU: `FLASH_ATTN` on Hopper, `SAGE_ATTN` on Ampere, Ada, and consumer Blackwell. Set the variable yourself to override it. Other FastVideo names are `TORCH_SDPA` and `SAGE_ATTN_THREE`.

The run is `$KANDINSKY_HOME/outputs/fastvideo_<YYYY-MM-DDTHH-MM-SS>/`. The clip is `generations/output.mp4`, the log is `logs/fastvideo.log`, and `launch.json` sits beside them.

The first argument is the prompt. Anything after it is a flag:

```bash
just generate-fastvideo 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' --image assets/i2va_input.jpg
```

| Argument | Role |
|---|---|
| prompt | Required. The text FastVideo encodes. Spoken words go in `<S>...<E>`. The audio description goes in `<AUDCAP>...<ENDAUDCAP>`. |
| `--image` | Reference frame for image-to-video-and-audio. Omit it for text only. The shipped frame is `assets/i2va_input.jpg`. |
| `--model` | Checkpoint. The default is `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`. A local directory is used as given. A Hub id already present under `$KANDINSKY_HOME/weights` is linked under the repository name, so a distilled snapshot keeps the distilled preset. |
| `--height` | Frame height in pixels. From the checkpoint config (`generation.height`). |
| `--width` | Frame width in pixels. From the checkpoint config (`generation.width`). |
| `--num-frames` | Number of frames. From the checkpoint config (`generation.sample_frames`). |
| `--num-inference-steps` | Sampling steps. From the checkpoint config (`generation.num_steps`). |
| `--guidance-scale` | Classifier-free guidance. From the checkpoint config (`generation.guidance_weight`). |
| `--num-gpus` | GPUs used to load and run the pipeline. The default is 1. |
| `--output` | Clip path. The recipe writes `$KANDINSKY_HOME/outputs/fastvideo_<YYYY-MM-DDTHH-MM-SS>/generations/output.mp4`. A later `--output` replaces that path. A path ending in `.mp4` is the filename. A directory makes FastVideo name the file from the prompt. |

## Super-resolution

`just generate-fastvideo` does not run super-resolution. SR is a separate checkpoint and takes an existing clip. The prompt is ignored, and the source audio is kept. These commands use ×2. The other scales are 4 and 2.25 (a 1.125 pixel pre-upscale, then ×2 tiles). Omitting `sr_resolution_scale` leaves FastVideo's own default, which is 2.25.

[Kandinsky-6.0-VSR-5s](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers) uses 4 Euler steps per tile. [Kandinsky-6.0-VSR-distilled2steps-5s](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers) uses 2 PiFlow steps. A Hub id is cached under `$KANDINSKY_HOME/weights`.

```bash
HF_HUB_CACHE="${KANDINSKY_HOME:-$HOME/.cache/kandinsky}/weights" \
.venvs/fastvideo/bin/python - <<'PY'
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers",
    num_gpus=1,
)
generator.generate_video(
    video_path="clip.mp4",
    output_path="output.mp4",
    save_video=True,
    sr_resolution_scale=2,
)
PY
```

`sr_tiles_batch_size` defaults to 1 and `sr_tile_min_overlap` to 0.2. `sr_target_resolution` (`hd`, `fullhd`, `2k`, or `WxH`) only downscales the result. `sr_target_resize_mode` is `fit` or `exact`.
