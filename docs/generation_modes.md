# Generation modes

The native pipeline generates from a text prompt. An image, a second image, an audio file, or a video file selects the mode. Setup and the shared `just generate` flags are in the [quick start](quickstart/quick_start.md).

```python
from kandinsky import get_pipeline

pipe = get_pipeline("kandinsky/configs/devices/rtx-4090.yaml")
# Equivalent token form:
# pipe = get_pipeline(["rtx-4090"])
# With an acceleration preset layered on top:
# pipe = get_pipeline(["rtx-4090", "spd"])
```

`just generate` reads the same preset. Anything after the prompt is a `kandy generate` flag.

## Text to video and audio

The default. The prompt produces a video and a synchronized audio track.

```python
pipe(
    text="A cat is lying on a mat, turns toward the camera, opens its mouth and says Hello.",
    save_path="t2av.mp4",
)
```

```bash
just generate 'A cat is lying on a mat, turns toward the camera, opens its mouth and says <S>Hello<E>'
```

## Text to video

The same call with the audio track omitted.

```python
pipe(
    text="A cat is lying on a mat and turns toward the camera.",
    sample_audio=False,
    save_path="t2v.mp4",
)
```

```bash
just generate 'A cat is lying on a mat and turns toward the camera.' --no-audio
```

## Image to video and audio

One reference image.

```python
pipe(
    text="The person walks forward.",
    image="first.png",
    visual_cond_scheme="tail_cond_first_frame",
    save_path="i2av.mp4",
)
```

`visual_cond_scheme` is optional. When it is omitted, the preset's scheme is used. `"tail_cond_first_frame"` keeps the reference as a conditioning frame and generates every output frame, including the first.

```bash
just generate 'The person walks forward.' --image assets/i2va_input.jpg
```

`just generate --image` passes one image. It does not set `visual_cond_scheme`, so the preset applies.

## First and last frame

Two reference images for one video: the first and the last. Pass them as one pair. Two images select tail conditioning on their own. The output frames are generated; the references are not pasted over the first and last frames. The last reference is encoded at the same resolution as the first.

```python
pipe(
    text="The person walks toward the doorway.",
    image=[["first.png", "last.png"]],
    save_path="flf2av.mp4",
)
```

A missing image, or any count other than two, raises `ValueError` before encoding.

The prompt rewriter sees only the first image. Set `beautifier.name: none` when the prompt already describes the transition between the two frames.

```bash
just generate 'The person walks toward the doorway.' --image first.png --last-image last.png
```

`--last-image` requires `--image`. The two paths are the first and last references of one video.

## Audio to video

The audio file conditions the video. Text is still required. The file is read as mono at the model sample rate. The saved clip keeps that source audio.

```python
pipe(text="A person speaks.", audio="speech.wav", save_path="a2v.mp4")
```

```bash
just generate 'A person speaks.' --audio-file speech.wav
```

## Video to audio

The video file conditions the audio. Text is still required. Frames are sampled at the pipeline frame rate and resized to the requested size. The saved clip keeps those source frames.

```python
pipe(text="Ocean waves and seabirds.", video="silent.mp4", save_path="v2a.mp4")
```

```bash
just generate 'Ocean waves and seabirds.' --video-file silent.mp4
```

## Audio and video file limits

These two calls accept one media file and one prompt.

- Pass `audio` or `video`, not both, and not together with `image`.
- Do not pass `sample_audio=False`.
- The preset must use `cache.mode: none`. The shipped device presets already do.
- The checkpoint must be a multimodal model with an audio VAE and a vocoder. For audio-to-video, the audio VAE encoder has to be enabled (`audio_vae.need_vae_encoder: true`).
- Length follows `latent_frames` or `time_length`. A longer file is trimmed. A shorter file raises `ValueError`. An audio file that is short by less than one video frame is padded with zeros.
- `--audio-file` and `--video-file` follow the same limits. They exclude each other, `--image`, `--last-image`, and `--no-audio`.

## Shared command flags

```bash
just generate 'A cat turns toward the camera.' --configs rtx-5090 --out clip.mp4
```

| Flag | Role |
|---|---|
| `--configs` | Preset tokens from `configs/{devices,checkpoints,accelerations}`, deep-merged left to right. The default is `--configs h100` |
| `--config` | One YAML path. Excludes `--configs` |
| `--out` | Output file. Skips the stamped run directory |
| `--image` | One reference image |
| `--last-image` | Last reference. Requires `--image` |
| `--audio-file` | Audio file that conditions the video |
| `--video-file` | Video file that conditions the audio |
| `--no-audio` | Video only |
| `--steps`, `--guidance`, `--seed` | Sampler overrides |
| `--height`, `--width`, `--time-length`, `--latent-frames` | Geometry overrides |
| `--device` | Torch device. The default is `cuda` |
| `--attention-engine` | Attention override from the config |
| `--offload` | `none`, `module`, or `block` |
| `--magcache`, `--navicache` | Cache override. The two flags exclude each other |
| `--negative` | Negative prompt |
| `--sr` | Run super-resolution after the clip |
| `--warmup` | One unsaved pass, then the saved run |
| `--no-progress` | Hide the step bar |
