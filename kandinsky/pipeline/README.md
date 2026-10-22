# Image conditioning: I2AV and FLF2AV

The native pipeline accepts one or two reference images per sample. The outer
image list is always the batch; a nested pair specifies the first and last
references of one video.

```python
from kandinsky import get_pipeline

pipe = get_pipeline("kandinsky/configs/devices/h100.yaml")

# One I2AV sample; existing single-image API is preserved.
pipe(
    text="The person walks forward.", image="first.png",
    visual_cond_scheme="tail_cond_first_frame", save_path="i2av.mp4",
)

# One FLF2AV sample; two references select tail conditioning automatically.
pipe(
    text="The person walks toward the doorway.",
    image=[["first.png", "last.png"]], save_path="flf2av.mp4",
)

# Two independent I2AV samples.
pipe(
    text=["A person walks.", "A cat turns around."],
    image=["person.png", "cat.png"],
    visual_cond_scheme="tail_cond_first_frame",
    save_path=["person_i2av.mp4", "cat_i2av.mp4"],
)

# Two FLF2AV samples.
pipe(
    text=["A person walks.", "A cat turns around."],
    image=[["person_first.png", "person_last.png"], ["cat_first.png", "cat_last.png"]],
    save_path=["person_flf2av.mp4", "cat_flf2av.mp4"],
)
```

All samples must have the same number of reference images. Mixed one-/two-frame
batches, empty reference lists, more than two references, and None references
raise ValueError before image encoding. Nested single-image lists are accepted.
For one image, the existing visual_cond_scheme setting is preserved.

## Conditioning layout

For T generated latent frames, FLF2AV uses:

```text
Sequence: [generated 0 ... generated T-1, first reference, last reference]
Role IDs: [0           ... 0,             1,               1]
RoPE t:   [0           ... T-1,           0,               T-1]
Mask:     [0           ... 0,             1,               1]
```

Both references use the existing conditioning embedding. They are kept clean
during denoising and removed before VAE decoding. Video and audio duration
exclude reference tokens. The first and last output frames are generated,
not replaced with the reference images. The last reference is encoded at the
same output resolution as the first.

The two-reference path requires temporal patch_size=1 and at least two role
embeddings. Architecture, weights and training code are unchanged.

## Prompt handling and entry points

Prompt beautification currently sees only the first reference. Set
`beautifier.name: none` in the config when using captions that already describe
the transition between both frames. The nested-image API belongs to the native
Python pipeline; CLI --image still accepts one image. ComfyUI is not extended.

## Code layout

The native pipeline handles image inputs, VAE offload and batch validation.
Shared tensor transforms live in `core/algo/visual_conditioning.py`, exported
through `core/ports.py`: reference packing, role ids, decode mask, boundary RoPE
and clean-reference restoration. Euler and PiFlow use the same helpers.
No new sampler, role embedding, runtime policy or FLF2AV-specific denoise loop
is added. Existing sampler signatures and the single-image API are preserved.
