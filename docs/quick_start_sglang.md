# Quick start: SGLang

Kandinsky 6 is not in a released SGLang package. This environment is separate from [the pipeline](quick_start.md). The recipe index is [just.md](just.md). The full upstream recipe is the [SGLang Kandinsky 6 page](https://docs.sglang.io/cookbook/diffusion/Kandinsky/Kandinsky6).

You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Run every recipe from the repository root.

`$KANDINSKY_HOME` is the root for weights and runs. When it is unset, that root is `~/.cache/kandinsky`.

## Setup

```bash
just setup-sglang
```

Checks out [SGLang](https://github.com/sgl-project/sglang) `main` into `.third-party/sglang` and installs the `diffusion` extra into `.venvs/sglang` (Python 3.12). A second run reuses that checkout. Delete `.third-party/sglang` first to pick up a newer `main`. This does not replace the `.venv` from `just setup`.

`just generate-sglang` stops and names this recipe when the environment is missing.

## Generate

```bash
just generate-sglang 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
```

Uses `.venvs/sglang`. The default checkpoint is [`kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers). SGLang loads PiFlow from that checkpoint. A snapshot already stored in `$KANDINSKY_HOME/weights` is reused. Otherwise the recipe downloads it there. The clip is 24 fps. The model is trained at that rate, so the recipe does not take another value.

Frame size, frame count, step count, and guidance come from the matching file in `kandinsky/configs/checkpoints/`. The default model reads `pro-distill.yaml`: 864×480, 121 frames, 10 steps, guidance 1. `--model kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers` reads `pro.yaml` instead (50 steps, guidance 5). A later flag replaces the value from that file.

The run is `$KANDINSKY_HOME/outputs/sglang_<YYYY-MM-DDTHH-MM-SS>/`. The clip is `generations/output.mp4`, the log is `logs/sglang.log`, and `launch.json` sits beside them.

Attention is FlashAttention (`fa`), performance mode is `memory`, and warmup is off. On RTX PRO 6000, pass `--attention-backend torch_sdpa`. SGLang has no MagCache. Cache-DiT stays off.

`memory` moves the text encoder and VAEs off the GPU and brings the whole DiT on only for the step. `--performance-mode speed` keeps every component on the GPU. Pro is about 56 GiB for the DiT and 15 GiB for the Qwen text encoder, so `speed` runs out of memory on an 80 GB GPU at the first denoising step. If `memory` still runs out of memory, add `--dit-layerwise-offload`. That streams DiT layers instead of keeping the whole DiT resident.

The first argument is the prompt. Anything after it is a flag:

```bash
just generate-sglang 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' --image assets/i2va_input.jpg
```

| Argument | Role |
|---|---|
| prompt | Required. The text SGLang encodes. Spoken words go in `<S>...<E>`. The audio description goes in `<AUDCAP>...<ENDAUDCAP>`. |
| `--image` | Reference frame for image-to-video-and-audio. Omit it for text only. The shipped frame is `assets/i2va_input.jpg`. Passed to SGLang as `--image-path`. |
| `--model` | Checkpoint Hub id or a local directory. `--model-path` is the same override. The default is `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`. A Hub id is used only when `$KANDINSKY_HOME/weights` already holds a complete snapshot. |
| `--height` | Frame height in pixels. From the checkpoint config (`generation.height`). |
| `--width` | Frame width in pixels. From the checkpoint config (`generation.width`). |
| `--num-frames` | Number of frames. From the checkpoint config (`generation.sample_frames`). |
| `--num-inference-steps` | Sampling steps. From the checkpoint config (`generation.num_steps`). |
| `--guidance-scale` | Classifier-free guidance. From the checkpoint config (`generation.guidance_weight`). |
| `--attention-backend` | Attention kernel. The recipe passes `fa`. A later value replaces it. |
| `--performance-mode` | `memory` by default: the text encoder and VAEs leave the GPU, and the DiT is resident only during the step. `speed` keeps every component on the GPU. |
| `--seed` | Random seed. The default is 42. |
| `--output` | Clip path. The recipe writes `generations/output.mp4` inside the run directory. A later `--output` replaces that path. A path ending in `.mp4` is the filename. |

## Serve

```bash
just serve-sglang
```

Loads the same default checkpoint as `just generate-sglang` and stays in the foreground. A snapshot already under `$KANDINSKY_HOME/weights` is reused. Otherwise the recipe downloads it there. Performance mode is `memory`: the text encoder and VAEs leave the GPU, and the DiT is resident only during a step. Attention is FlashAttention (`fa`). The server listens on `127.0.0.1:30000`.

Step count and guidance follow the checkpoint. The recipe passes the Hub name as `--model-id`, so a distilled snapshot uses 10 steps and guidance 1, and the others use 50 steps and guidance 5. A request that omits `num_inference_steps` and `guidance_scale` keeps that pair. A distilled model rejects a guidance other than 1. Frame size and frame count stay on the request.

Anything after the recipe name is an `sglang serve` flag. A later flag replaces a recipe flag. `--model` selects another catalog checkpoint. On RTX PRO 6000, pass `--attention-backend torch_sdpa`. If `memory` still runs out of memory, add `--dit-layerwise-offload`.

```bash
just serve-sglang --port 30001
```

```bash
VID=$(curl -s -X POST http://127.0.0.1:30000/v1/videos \
  -F 'prompt=A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' \
  -F size=864x480 -F num_frames=121 -F seed=42 \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
curl -s http://127.0.0.1:30000/v1/videos/$VID
curl -s -o kandinsky6.mp4 http://127.0.0.1:30000/v1/videos/$VID/content
```

An image-to-video request sends the reference frame as `input_reference`. `size` and `num_frames` match the checkpoint configs: 864×480 and 121 frames. `--model kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers` switches the omitted step count and guidance to 50 and 5.

## Super-resolution

`just generate-sglang` does not run super-resolution. SR is a separate checkpoint and takes an existing clip. The prompt is ignored, and the source audio is kept. These commands use ×2. The other scales are 4 and 2.25 (a 1.125 pixel pre-upscale, then ×2 tiles). Omitting `--sr-resolution-scale` leaves SGLang's own default, which is 2.25.

[Kandinsky-6.0-VSR-5s](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers) uses 4 Euler steps per tile. [Kandinsky-6.0-VSR-distilled2steps-5s](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers) uses 2 PiFlow steps and ignores a different step count. A Hub id is cached under `$KANDINSKY_HOME/weights`.

```bash
HF_HUB_CACHE="${KANDINSKY_HOME:-$HOME/.cache/kandinsky}/weights" \
.venvs/sglang/bin/sglang generate \
  --model-path kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers \
  --video-path clip.mp4 \
  --sr-resolution-scale 2 \
  --save-output \
  --output-file-path output.mp4 \
  --seed 42
```

`--sr-tiles-batch-size` defaults to 1 and `--sr-tile-min-overlap` to 0.2. `--sr-target-resolution` (`hd`, `fullhd`, `2k`, or `WxH`) only downscales the result. `--sr-target-resize-mode` is `fit` or `exact`.
