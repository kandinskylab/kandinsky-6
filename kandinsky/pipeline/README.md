# Audio and Video Conditioning

Two modes are selected automatically by the input argument:

- `audio=...`: A2V, generate video conditioned on audio.
- `video=...`: V2A, generate audio conditioned on video.
- Without either argument, existing T2AV/I2AV behavior is unchanged.

## Usage

Minimal `config.yaml` for Lite; use `checkpoint: lite-distill` for PiFlow:

```yaml
checkpoint: lite
cache: {mode: none}
offload: {strategy: module}
compile: {strategy: none}
beautifier: {name: none}
```

```python
from kandinsky import get_pipeline

pipe = get_pipeline('config.yaml')
pipe(text='A person speaks.', audio='speech.wav', save_path='a2v.mp4')
pipe(text='Ocean waves and seabirds.', video='silent.mp4', save_path='v2a.mp4')
```

## Behavior and Limits

- Text remains an input in both modes.
- The supplied modality is VAE-encoded, fixed during denoising, and receives
  `t=0` on every step. The other modality starts from noise. Both towers
  interact through fused attention; this also applies to PiFlow.
- Output reuses source video frames or audio, not their VAE reconstruction.
  MP4 compression can still change pixels or audio.
- Audio becomes mono at the model sample rate. Video is sampled at the
  pipeline FPS and resized to the requested dimensions.
- Length follows `latent_frames`/`time_length`. Longer inputs are trimmed;
  shorter inputs raise `ValueError`. Audio shortages below one video frame
  are zero-padded.
- One media file per call. Prompt batches, both media inputs together,
  image/media combinations, and caching are rejected.
- A multimodal model, audio VAE and vocoder are required. For A2V, keep
  `audio_vae.need_vae_encoder=true`. Do not set `sample_audio=False`.
- GPU-tested on Lite and Lite-Distill/PiFlow; Pro and compilation are untested.

## Code Layout

File decoding and device placement belong to the native pipeline (`media.py`,
`pipeline.py`). Tensor-only VAE encoding is in `core/algo/conditioning.py`,
exported through `core/ports.py` for framework ports. The helpers do not load
weights, move modules, read files, or require PyAV. Existing sampler signatures
are unchanged; the pipeline passes clean latents and the sampling flags.
