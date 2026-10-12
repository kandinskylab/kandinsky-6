# Quick start: vLLM-Omni

Kandinsky 6 on [vLLM-Omni](https://github.com/vllm-project/vllm-omni) `main`. This environment is separate from [the pipeline](quick_start.md). The recipe index is [just.md](just.md).

You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Run every recipe from the repository root.

`$KANDINSKY_HOME` is the root for weights and runs. When it is unset, that root is `~/.cache/kandinsky`.

## Setup

```bash
just setup-vllm
```

Installs vLLM 0.31.0 and current vLLM-Omni `main` into `.venvs/vllm-omni` (Python 3.12). The source is cloned to `.third-party/vllm-omni`. A second run reuses that checkout. This does not replace the `.venv` from `just setup`.

`just generate-vllm` stops and names this recipe when the environment is missing.

## Generate

```bash
just generate-vllm 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
```

Uses `.venvs/vllm-omni` and the vLLM-Omni examples. The default checkpoint is [`kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers`](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers). vLLM-Omni samples with Euler and has no PiFlow sampler, so the distilled checkpoints are not the default. A snapshot already stored in `$KANDINSKY_HOME/weights` is reused. Otherwise the recipe downloads it there. The clip is 24 fps. The model is trained at that rate, so the recipe does not take another value.

Frame size, frame count, step count, and guidance come from the matching file in `kandinsky/configs/checkpoints/`. The default model reads `pro.yaml`: 864×480, 121 frames, 50 steps, guidance 5. A later flag replaces the value from that file.

The recipe does not set an attention backend. On Hopper the automatic choice is FlashAttention 3 when `fa3-fwd` is installed. On Blackwell it is cuDNN when cuDNN is 9.5 or newer.

MagCache turns on when the checkpoint file has `cache.magcache`. Pro uses the i2va SFT curve (threshold 0.02, K 4, retention 0.1). Lite uses the t2va SFT curve (threshold 0.01, K 2, retention 0.2). Pretrain and the distilled checkpoints have no curve, so the cache stays off. `--cache-backend none` turns it off.

The run is `$KANDINSKY_HOME/outputs/vllm_<YYYY-MM-DDTHH-MM-SS>/`. The clip is `generations/output.mp4`, the log is `logs/vllm.log`, and `launch.json` sits beside them.

The first argument is the prompt. Anything after it is a flag:

```bash
just generate-vllm 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' --image assets/i2va_input.jpg
```

| Argument | Role |
|---|---|
| prompt | Required. The text vLLM-Omni encodes. Spoken words go in `<S>...<E>`. The audio description goes in `<AUDCAP>...<ENDAUDCAP>`. |
| `--image` | Reference frame for image-to-video-and-audio. Omit it for text only. The shipped frame is `assets/i2va_input.jpg`. |
| `--model` | Checkpoint Hub id or a local directory. The default is `kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers`. A Hub id is used only when `$KANDINSKY_HOME/weights` already holds a complete snapshot. |
| `--height` | Frame height in pixels. From the checkpoint config (`generation.height`). |
| `--width` | Frame width in pixels. From the checkpoint config (`generation.width`). |
| `--num-frames` | Number of frames. From the checkpoint config (`generation.sample_frames`). |
| `--num-inference-steps` | Sampling steps. From the checkpoint config (`generation.num_steps`). |
| `--guidance-scale` | Classifier-free guidance. From the checkpoint config (`generation.guidance_weight`). |
| `--cache-backend` | Step cache. `mag_cache` when the checkpoint file has `cache.magcache`. `none` turns it off. |
| `--seed` | Random seed. The default is 42. |
| `--output` | Clip path. The recipe writes `generations/output.mp4` inside the run directory. A later `--output` replaces that path. |

CPU offload is on, which an 80 GB GPU needs. Audio is on.

## Serve

```bash
just serve-vllm
```

Loads the same default checkpoint as `just generate-vllm` and stays in the foreground. A snapshot already under `$KANDINSKY_HOME/weights` is reused. Otherwise the recipe downloads it there. CPU offload is on. MagCache turns on when the checkpoint file has `cache.magcache`, the same way as `just generate-vllm`. The server listens on `127.0.0.1:8091`.

Step count and guidance come from the checkpoint file and are the server defaults. Pro, Lite, and the pretrain checkpoints use 50 steps and guidance 5. The distilled checkpoints use 10 steps and guidance 1. A request that omits `num_inference_steps` and `guidance_scale` keeps that pair. Frame size and frame count stay on the request.

Anything after the recipe name is a `vllm serve` flag. A later flag replaces a recipe flag. `--model` selects another catalog checkpoint. `--cache-backend none` turns MagCache off.

```bash
just serve-vllm --port 8092
```

```bash
VID=$(curl -s -X POST http://127.0.0.1:8091/v1/videos \
  -F 'prompt=A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>' \
  -F size=864x480 -F num_frames=121 -F seed=42 \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
curl -s http://127.0.0.1:8091/v1/videos/$VID
curl -s -o kandinsky6.mp4 http://127.0.0.1:8091/v1/videos/$VID/content
```

The API reference is the [Kandinsky 6 module page](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/kandinsky6/).
