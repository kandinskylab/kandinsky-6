<div align="center">
  <img src="assets/readme/promo.webp" width="100%">
</div>

<br>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/Kandinsky_LOGO_6_Horizontal_white.png">
  <img src="assets/readme/Kandinsky_LOGO_6_Horizontal_black.png" width="60%">
</picture>
</div>

<br>

<div align="center">

<a href="https://kandinskylab.ai/"><img alt="KandinskyLab" src="https://img.shields.io/badge/KandinskyLab-76E0B7?style=for-the-badge"></a>
<a href="https://arxiv.org/abs/2610.05608"><img alt="Report" src="https://img.shields.io/badge/Report-9C2731?style=for-the-badge"></a>
<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky6"><img alt="Diffusers" src="https://img.shields.io/badge/Diffusers-F8D44E?style=for-the-badge"></a>
<a href="https://huggingface.co/spaces/kandinskylab/Kandinsky-6.0-Pro-distill-5s"><img alt="HF Demo" src="https://img.shields.io/badge/HF%20Demo-D9622B?style=for-the-badge"></a>
<a href="https://registry.comfy.org/nodes/kandinsky6"><img alt="ComfyUI" src="https://img.shields.io/badge/ComfyUI-EBF764?style=for-the-badge"></a>
<a href="https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/kandinsky6/"><img alt="vLL-Omni" src="https://img.shields.io/badge/vLLM Omni-54A0F8?style=for-the-badge"></a>
<!-- <a href="#"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-3171B2?style=for-the-badge"></a> -->
<!-- <a href="#"><img alt="SGLang" src="https://img.shields.io/badge/SGLang-C6602D?style=for-the-badge"></a>
<a href="#"><img alt="FastVideo" src="https://img.shields.io/badge/FastVideo-436BF6?style=for-the-badge"></a> -->


</div>

<h1>Kandinsky 6.0: A family of diffusion models for Video + Audio generation</h1>

We present Kandinsky 6.0 Video, a family of foundation diffusion models for synchronized text-to-audio-video generation, comprising Kandinsky 6.0 Video Lite (3B parameters)
and Kandinsky 6.0 Video Pro (29B parameters). Both models generate 5-second video clips
with synchronized 44 kHz audio, including lip-sync, in text-to-audio-video (T2AV) and image-to-audio-video (TI2AV) modes; a plugin-in super-resolution model raises the output resolution to
Full-HD (1920×1080).


## Project Updates

- ```2026/10/06```: We added [vLLM-omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/kandinsky6/) support
- ```2026/10/06```: We added Hugging Face Space for [Kandinsky 6.0 Pro Distill](https://huggingface.co/spaces/kandinskylab/Kandinsky-6.0-Pro-distill-5s)
- ```2026/10/06```: We have open-sourced `Kandinsky 6.0`


## Quick start

An NVIDIA GPU and Python 3.13 or 3.14.

The first run downloads [Kandinsky-6.0-Pro-distill-5s](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers) into `$KANDINSKY_HOME/weights`. When `$KANDINSKY_HOME` is unset, that directory is `~/.cache/kandinsky`. The clip is written to `$KANDINSKY_HOME/outputs/generate_<YYYY-MM-DDTHH-MM-SS>/generations/output.mp4`, with the expanded prompt in `expanded_prompt.txt` next to `launch.json`, `logs/`, and `profiles/`. The generate command downloads the checkpoint named in the config if it is not already on disk. Other catalog names are `pro`, `pro-pretrain`, `lite`, `lite-distill`, and `lite-pretrain`.

### Clone the repository

You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
git clone https://github.com/kandinskylab/kandinsky-6.git
cd k6_video
just setup
```

`just setup` reads the GPU and installs one PyTorch build:

| GPUs | Compute capability | Installed |
|---|---|---|
| Hopper (H100, H200) | 9.0 | PyTorch for CUDA 13.0, FlashAttention 3 |
| Ampere, Ada, consumer Blackwell | 8.0, 8.6, 8.9, 12.0, 12.1 | PyTorch for CUDA 13.0, then SageAttention 2.2.0 |

Ampere, Ada, and consumer Blackwell compile SageAttention, so `nvcc` has to be on `PATH`. `CUDA_HOME` defaults to `/usr/local/cuda`.

```bash
just download pro-distill
just generate "a cat on a mat"
```

Another GPU uses its preset:

```bash
just generate "a cat on a mat" --config kandinsky/configs/devices/rtx-5090.yaml --out clip.mp4
```

Presets live in `kandinsky/configs/devices/`.

## ComfyUI

For ComfyUI, install [kandinsky6](https://registry.comfy.org/nodes/kandinsky6) and [kandinsky6-sr](https://registry.comfy.org/nodes/kandinsky6-sr) through **ComfyUI Manager**, then restart ComfyUI.
The `comfyui/` directory contains extension source code; no manual copying is needed — see the [setup guide](comfyui/README.md) and [manual model downloads](comfyui/README.md#manual-downloads).

## Performance

Working time (s) for a 5-second clip on the non-distilled model, after warmup. Weight loading and MP4 encoding are excluded. 

RTX PRO 6000 (96 GB), A100 80 GB, and H100 80 GB use module offload. The consumer cards use block offload.

RTX 4090, RTX 5060 Ti, RTX 5080, RTX 5090, RTX PRO 6000, and A100 80 GB use SageAttention2++ for visual self-attention. The H100 uses FlashAttention 3. 

| | RTX 4090 | RTX 5060 Ti | RTX 5080 | RTX 5090 | RTX PRO 6000 | A100 80 GB | H100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lite SD | 437 | 1310 | 577 | 309 | 242 | 532 | 239 |
| Lite HD | 422 | 1328 | 579 | 296 | 274 | 472 | 203 |
| Lite Full HD | 578 | 1774 | 770 | 406 | 387 | 664 | 284 |
| Pro SD | 936 | 3080 | 1336 | 754 | 621 | 972 | 356 |
| Pro HD | 1194 | 2716 | 1189 | 638 | 561 | 791 | 292 |
| Pro Full HD | 1247 | 3530 | 1546 | 854 | 765 | 1106 | 402 |
