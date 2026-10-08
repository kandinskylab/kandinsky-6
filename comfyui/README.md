# Kandinsky 6

Text-to-video+audio and image-to-video+audio with Kandinsky 6.
Supports **Pro, Pro Distill, Lite and Lite Distill**.
Two ready-to-run workflows default to **Pro Distill PiFlow (10 steps, CFG=1)**,
with an I2VA reference portrait. MagCache is supported only for non-distilled Pro
and automatically bypassed for distilled models.
Both include native **Qwen3.5-9B** prompt beautification and use the separate
**Kandinsky6 SR** extension for super resolution.

Requires ComfyUI **0.38.0+** and Python 3.10+. Model loading and offloading are managed
by ComfyUI; the Python inference pipeline and Diffusers library are not required.
No ComfyUI core patches are needed. For older NVIDIA drivers, see **Compatibility** below.

## Install

Once the Registry versions are available:

1. Open **ComfyUI Manager** and its custom-node list.
2. Search for `kandinsky6` and `kandinsky6-sr` (publisher `kandinskylab`) and click **Install** for both.
3. Restart ComfyUI, then follow **Models** and **Run** below.

Alternatively, with [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)
and ComfyUI Manager installed:

```bash
comfy --workspace /path/to/ComfyUI node install kandinsky6 kandinsky6-sr
```

Replace `/path/to/ComfyUI` with your ComfyUI folder and restart after installation.
The base generation nodes work without SR, but the bundled workflows need both extensions.

For manual installation, clone `kandinskylab/kandinsky-6` outside `custom_nodes/`,
copy the contents of `comfyui/` into `ComfyUI/custom_nodes/kandinsky6/`, then run with
**ComfyUI's Python** and restart:

```bash
python -m pip install -r ComfyUI/custom_nodes/kandinsky6/requirements.txt
```

## Models

Open a bundled workflow and click **Download models** in its setup note,
or choose **Kandinsky 6 → Kandinsky 6 — Download models** from the top menu.

The button downloads Pro distill 5s, Qwen3.5-9B (~19.3 GB), Qwen2.5, CLIP, the video/audio VAEs,
BigVGAN and VSR, including all required JSON configs. Files are placed in
ComfyUI's model folders automatically; existing files, including configured
extra model paths, are reused. Nothing is downloaded during extension install
or startup. Allow enough disk space for large HF weights.

For gated/private models, obtain access and run `hf auth login` on the ComfyUI
server first.

### Manual downloads

Download the files separately if you prefer not to use the button. Destinations
below are relative to your **ComfyUI folder**; create missing directories.

| Download | Destination |
| --- | --- |
| [Pro distill 5s — default transformer](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers/resolve/main/transformer/diffusion_pytorch_model.safetensors) | `models/diffusion_models/Kandinsky-6.0-Pro-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors` |
| [Lite distill 5s — optional, smaller transformer](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers/resolve/main/transformer/diffusion_pytorch_model.safetensors) | `models/diffusion_models/Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors` |
| [Qwen3.5-9B — beautifier](https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_9b_bf16.safetensors) | `models/text_encoders/qwen3.5_9b_bf16.safetensors` |
| [Qwen2.5 — text encoder](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors) | `models/text_encoders/qwen_2.5_vl_7b.safetensors` |
| [CLIP-L — text encoder](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors) | `models/text_encoders/clip_l.safetensors` |
| [HunyuanVideo VAE](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors) | `models/vae/hunyuan_video_vae_bf16.safetensors` |
| [v1-44.pth — audio VAE](https://huggingface.co/hkchengrex/MMAudio/resolve/main/ext_weights/v1-44.pth) | `models/audio_vae/v1-44.pth` |
| [bigvgan_generator.pt — vocoder](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/bigvgan_generator.pt) | `models/audio_vae/bigvgan_vocoder/bigvgan_generator.pt` |
| [config.json — vocoder config](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/config.json) | `models/audio_vae/bigvgan_vocoder/config.json` |

Both bundled workflows also need the files in the
[SR manual-download table](https://github.com/kandinskylab/kandinsky-6-sr/blob/main/comfyui/README.md#manual-downloads).
Keep all listed JSON configs in their specified folders.

For non-distilled Pro, use [this transformer instead](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers/resolve/main/transformer/diffusion_pytorch_model.safetensors),
saved as `models/diffusion_models/Kandinsky-6.0-Pro-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors`;
see **Run** below for its sampling settings. HF access may be required for this
non-distilled checkpoint. You do not need both transformers.

For **Lite distilled**, download its transformer from the table and select it in
**Load Diffusion Model** in either template. All other weights and settings stay
the same: **10 steps, CFG=1, denoise=1, audio scaling=0.417**. Detection is automatic;
MagCache is bypassed. The download button still fetches the default Pro distilled
bundle, not the optional Lite transformer.

Models on another drive can use ComfyUI's configured extra model paths.
For audio, the simplest option is a symbolic link (directory junction on Windows)
from `models/audio_vae` to your audio-model folder on that drive, keeping the
same layout above; BigVGAN is loaded from that folder. Restart ComfyUI after
placing the files and select them in the loader nodes. The download button is
not required for manual installation.

## Run

For faster inference, try SageAttention: launch ComfyUI with `--use-sage-attention` instead of `--use-flash-attention` (requires `sageattention` in ComfyUI's Python environment and a supported NVIDIA GPU).

Open **Workflow → Browse Templates → kandinsky6** and choose:

- **Kandinsky 6.0 Text to Video+Audio**
- **Kandinsky 6.0 Image to Video+Audio**

Select the downloaded models and run the workflow to save an MP4 with audio.
Edit the video/audio captions in **Beautify Prompt**; it expands them automatically.
For I2VA it also sees the original **Load Image** reference. The result appears in
**Preview as Text**. Set `enabled=false` to use your captions directly.
Put exact spoken lines in the video caption as `<S>Look there!<E>`; describe the
voice and other sounds in the audio caption. Thinking and MTP are off; no vLLM,
Transformers model or separate LLM server is needed. Qwen3.5 is managed by ComfyUI
and is separate from the Qwen2.5 text encoder required by Kandinsky.
Before your first generation, use **Download models** or the manual tables above,
including the companion JSON configs and audio files.
Use `weight_dtype=default` in **Load Diffusion Model** for the first run.
I2VA includes a portrait; replace it in **Load Image** to use your own image.

**Kandinsky 6 Sampler** selects PiFlow automatically for distilled Pro or Lite; use
**10 steps, CFG=1, denoise=1** and audio VAE **scaling=0.417**. Its sampler/scheduler
selectors apply only to non-distilled models. **MagCache automatically bypasses
distilled models**, even if its node is connected.
For [non-distilled Pro](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers),
select its transformer and use **50 steps, CFG=5, audio scaling=0.5302**.
Set MagCache's `steps` to the sampler's step count; it is calibrated for
**non-distilled Pro only**, not Lite or VSR. For
[non-distilled Lite](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers),
use its transformer with the same 50-step settings and remove/bypass the MagCache
node. Both templates include VSR.
The default VSR scale is **2.25x**. For **4x**, select `4x` in both the VSR node
and the latent-upscaler loader; **2x/2.25x** use the loader's `2x` entry.
VSR's `use_nabla` defaults to **off**, using ComfyUI's selected attention backend
without NABLA warmup. Enable it to compare sparse NABLA attention; its first run
compiles kernels. The models themselves are not compiled.

For standalone video upscaling, use the
[SR extension and template](https://github.com/kandinskylab/kandinsky-6-sr/tree/main/comfyui).

## Compatibility

If native Qwen fails because `comfy-kitchen` requires a newer NVIDIA driver,
update the driver or use its official Python-only wheel. For ComfyUI **0.38.0**
with `comfy-kitchen==0.2.36`, this was tested on H100 with driver 570 and Torch 2.10.
Run with **ComfyUI's Python**, then restart:

```bash
python -m pip install --force-reinstall --no-deps \
  https://files.pythonhosted.org/packages/38/23/a6787aac01d7c28ae3cb07579ba839297a35e6fad66baac096916246cc7f/comfy_kitchen-0.2.36-py3-none-any.whl
```

Inference stays on GPU. For another ComfyUI version, match its own
`comfy-kitchen` requirement rather than forcing this pin.
