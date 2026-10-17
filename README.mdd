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

<a href="https://kandinskylab.ai/models/video/"><img alt="KandinskyLab" src="https://img.shields.io/badge/KandinskyLab-76E0B7?style=for-the-badge"></a>
<a href="https://huggingface.co/papers/2610.05608"><img alt="Report" src="https://img.shields.io/badge/Report-9C2731?style=for-the-badge"></a>
<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky6"><img alt="Diffusers" src="https://img.shields.io/badge/Diffusers-F8D44E?style=for-the-badge"></a>
<a href="https://huggingface.co/spaces/kandinskylab/Kandinsky-6.0-Pro-distill-5s"><img alt="HF Demo" src="https://img.shields.io/badge/HF%20Demo-D9622B?style=for-the-badge"></a>
<a href="https://registry.comfy.org/nodes/kandinsky6"><img alt="ComfyUI" src="https://img.shields.io/badge/ComfyUI-EBF764?style=for-the-badge"></a>
<a href="https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/kandinsky6/"><img alt="vLL-Omni" src="https://img.shields.io/badge/vLLM Omni-54A0F8?style=for-the-badge"></a>
<a href="https://docs.sglang.io/cookbook/diffusion/Kandinsky/Kandinsky6"><img alt="SGLang" src="https://img.shields.io/badge/SGLang-C6602D?style=for-the-badge"></a>
<a href="https://haoailab.com/FastVideo/inference/kandinsky6/"><img alt="FastVideo" src="https://img.shields.io/badge/FastVideo-436BF6?style=for-the-badge"></a>
<!-- <a href="#"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-3171B2?style=for-the-badge"></a> -->

</div>

<h1>Kandinsky 6.0: A family of diffusion models for Video + Audio generation</h1>

We present Kandinsky 6.0 Video, a family of foundation diffusion models for synchronized text-to-audio-video generation, comprising Kandinsky 6.0 Video Lite (3B parameters)
and Kandinsky 6.0 Video Pro (29B parameters). Both models generate 5-second video clips with synchronized 44 kHz audio, including lip-sync, in text-to-audio-video (T2AV) and image-to-audio-video (TI2AV) modes. A plug-in [super-resolution model](https://github.com/kandinskylab/kandinsky-6-sr) raises the output resolution to Full HD (1920×1080).


## Project Updates

- 2026/10/06: Kandinsky 6.0 is now available in [Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky6), [ComfyUI](https://registry.comfy.org/nodes/kandinsky6), [vLLM-omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/kandinsky6/), [SGLang](https://docs.sglang.io/cookbook/diffusion/Kandinsky/Kandinsky6) and [FastVideo](https://haoailab.com/FastVideo/inference/kandinsky6)
- 2026/10/06: We added Hugging Face Space for [Kandinsky 6.0 Pro Distill](https://huggingface.co/spaces/kandinskylab/Kandinsky-6.0-Pro-distill-5s)
- 2026/10/06: We have open-sourced **Kandinsky 6.0** and [**Kandinsky 6.0 Video Super-Resolution**](https://github.com/kandinskylab/kandinsky-6-sr)



## Quick start

An NVIDIA GPU. You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Each path has its own environment and generate command:

| Path | Guide |
|---|---|
| Pipeline | [docs/quick_start.md](docs/quick_start.md) |
| vLLM-Omni | [docs/quick_start_vllm.md](docs/quick_start_vllm.md) |
| SGLang | [docs/quick_start_sglang.md](docs/quick_start_sglang.md) |
| FastVideo | [docs/quick_start_fastvideo.md](docs/quick_start_fastvideo.md) |

The recipe index is [docs/just.md](docs/just.md).

```bash
git clone https://github.com/kandinskylab/kandinsky-6.git
cd kandinsky-6
just setup
just download pro-distill
just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
```

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
