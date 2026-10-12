"""Generate one Kandinsky 6 clip with FastVideo.

A complete snapshot under ``$KANDINSKY_HOME/weights`` is reused. FastVideo
treats a local directory as the distilled checkpoint only when its name
contains ``distill``, so the snapshot is linked under that repo name.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def kandinsky_home() -> Path:
    raw = os.environ.get("KANDINSKY_HOME")
    if raw:
        return Path(os.path.expandvars(os.path.expanduser(raw)))
    return Path.home() / ".cache" / "kandinsky"


def use_kandinsky_weights() -> Path:
    """Point Hugging Face downloads at ``$KANDINSKY_HOME/weights``."""
    cache = kandinsky_home() / "weights"
    os.environ["HF_HUB_CACHE"] = str(cache)
    return cache


def ready_snapshot(repo_id: str, home: Path | None = None) -> Path | None:
    """Return a local snapshot that already has the DiT weights, if one exists."""
    root = (home or kandinsky_home()) / "weights" / ("models--" + repo_id.replace("/", "--"))
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return None

    def ready(path: Path) -> bool:
        transformer = path / "transformer"
        if not (path / "model_index.json").is_file() or not transformer.is_dir():
            return False
        if any(transformer.glob("*.safetensors")):
            return True
        return (transformer / "diffusion_pytorch_model.safetensors.index.json").is_file()

    ref = root / "refs" / "main"
    if ref.is_file():
        pointed = snapshots / ref.read_text(encoding="utf-8").strip()
        if ready(pointed):
            return pointed
    for path in sorted(snapshots.iterdir()):
        if path.is_dir() and ready(path):
            return path
    return None


def model_for_fastvideo(repo_id: str, link_dir: Path) -> str:
    """Use the on-disk snapshot when it is complete. Otherwise keep the Hub id.

    The link name is the repo id's last component, so a distilled checkpoint
    stays on the distilled preset.
    """
    use_kandinsky_weights()
    if Path(repo_id).is_dir():
        return repo_id
    snapshot = ready_snapshot(repo_id)
    if snapshot is None:
        return repo_id
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / repo_id.split("/")[-1]
    target = snapshot.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            link.unlink()
            link.symlink_to(target)
    elif link.exists():
        return repo_id
    else:
        link.symlink_to(target)
    return str(link)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--image", default=None)
    parser.add_argument(
        "--model",
        default="kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
    )
    parser.add_argument("--output", default="kandinsky6_fastvideo")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output)
    model = model_for_fastvideo(args.model, output.resolve().parent)
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        model,
        num_gpus=args.num_gpus,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    sampling = {}
    if args.num_inference_steps is not None:
        sampling["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        sampling["guidance_scale"] = args.guidance_scale
    generator.generate_video(
        args.prompt,
        image_path=args.image,
        output_path=str(output),
        save_video=True,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        **sampling,
    )


if __name__ == "__main__":
    main()
