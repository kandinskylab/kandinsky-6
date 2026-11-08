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
<a href="https://huggingface.co/docs/diffusers/en/api/pipelines/kandinsky6"><img alt="Diffusers" src="https://img.shields.io/badge/Diffusers-F8D44E?style=for-the-badge"></a>
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

- 2026/11/08: Added [SPEED](https://arxiv.org/abs/2605.18736), a spectral progressive diffusion strategy, which accelerates the Pro model by 14%. See [Quick start](docs/quickstart/quick_start.md).
- 2026/10/25: The native pipeline now generates from a first and last frame, from an audio file, and from a video file. The calls and `just generate` flags are in [Generation modes](docs/generation_modes.md).
- 2026/10/06: Kandinsky 6.0 is now available in [Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/kandinsky6), [ComfyUI](https://registry.comfy.org/nodes/kandinsky6), [vLLM-omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/kandinsky6/), [SGLang](https://docs.sglang.io/cookbook/diffusion/Kandinsky/Kandinsky6) and [FastVideo](https://haoailab.com/FastVideo/inference/kandinsky6)
- 2026/10/06: We added Hugging Face Space for [Kandinsky 6.0 Pro Distill](https://huggingface.co/spaces/kandinskylab/Kandinsky-6.0-Pro-distill-5s)
- 2026/10/06: We have open-sourced **Kandinsky 6.0** and [**Kandinsky 6.0 Video Super-Resolution**](https://github.com/kandinskylab/kandinsky-6-sr)



## Quick start

An NVIDIA GPU. You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Each path has its own environment and generate command:

| Path | Guide |
|---|---|
| Pipeline | [docs/quickstart/quick_start.md](docs/quickstart/quick_start.md) |
| vLLM-Omni | [docs/quickstart/quick_start_vllm.md](docs/quickstart/quick_start_vllm.md) |
| SGLang | [docs/quickstart/quick_start_sglang.md](docs/quickstart/quick_start_sglang.md) |
| FastVideo | [docs/quickstart/quick_start_fastvideo.md](docs/quickstart/quick_start_fastvideo.md) |

The recipe index is [docs/just.md](docs/just.md). The default call is text to video with audio. One image starts from a reference frame, two images set the first and last frames, an audio file drives the video, and a video file drives the audio. The Python calls and `just generate` flags are in [Generation modes](docs/generation_modes.md).

```bash
git clone https://github.com/kandinskylab/kandinsky-6.git
cd kandinsky-6
just setup
just download pro-distill
just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
```

## Performance

Working time (s) for a 5-second text-to-video clip with audio. Weight loading and MP4 encoding are excluded. The base clip is 480×864. Full HD super-resolved at scale 2.25. Compile is torch. Prompt expansion uses Qwen2.5.

Distilled checkpoints use PiFlow, 10 steps. The other checkpoints use 50 steps and MagCache.

The H100 column is `just bench`: FlashAttention 3, Qwen in BF16, module offload. Each cell is median of three generations.

RTX 4090, RTX 5090, and A100 use SageAttention and Qwen in BF16. Lite stays resident on the RTX 5090 and the A100, and uses module offload on the RTX 4090. Pro uses block offload on the RTX 4090 and the RTX 5090, and module offload on the A100. Each cell is one generation after warmup.

The RTX 5060 Ti uses SageAttention and Qwen in NF4. Lite stays resident. Pro uses block offload. Each cell is one generation after warmup.

| | RTX 4090 24 GB | RTX 5060 Ti 16 GB | RTX 5090 32 GB | A100 80 GB | H100 80 GB |
|---|---:|---:|---:|---:|---:|
| Lite Distill SD | 68.0 | 173.7 | 44.2 | 78.6 | 31.2 |
| Lite Distill Full HD | 208.9 | 608.4 | 140.3 | 207.1 | 78.0 |
| Lite SD | 267.9 | 834.9 | 201.3 | 371.6 | 143.2 |
| Lite Full HD | 408.7 | 1286.7 | 296.2 | 508.0 | 192.2 |
| Pro Distill SD | 235.0 | 717.6 | 174.5 | 236.0 | 87.8 |
| Pro Distill Full HD | 381.0 | 1153.0 | 269.5 | 375.4 | 133.2 |
| Pro SD | 955.4 | 3143.6 | 768.0 | 980.4 | 351.5 |
| Pro Full HD | 1096.7 | 3571.4 | 862.8 | 1109.8 | 399.7 |

Module breakdowns:

- [H100 80 GB](docs/performance/h100.md)
- [RTX 4090 24 GB](docs/performance/rtx-4090.md)
- [RTX 5060 Ti 16 GB](docs/performance/rtx-5060-ti.md)
- [RTX 5090 32 GB](docs/performance/rtx-5090.md)
- [A100 80 GB](docs/performance/a100.md)

## Community

- [Kandinsky T8](https://github.com/T8mars/Comfyui-Kandinsky-T8) by [T8mars](https://github.com/T8mars), a ComfyUI adaptation for Lite, Lite Distill, and Pro Distill.
  - [INT8 ConvRot weights](https://huggingface.co/t8star/Kandinsky-Comfy)
- [Native ComfyUI implementation](https://github.com/Comfy-Org/ComfyUI/pull/16825) by [kabachuha](https://github.com/kabachuha).
  - [INT8 ConvRot Pro and Lite weights, including distilled variants](https://huggingface.co/collections/kabachuha/kandinsky-60-quantized-int8-convrot)
