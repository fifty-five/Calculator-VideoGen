import argparse
import json
import os
import sys

from utils import load_yaml
from ml.compute_wh import run_ml


MODEL_CONFIGS = {
    # --- UNet models ---
    "AnimateDiff": {"arch": "unet", "params": 0.9},
    "Stable Video Diffusion": {"arch": "unet", "params": 1.5},
    "Pika 1.0": {"arch": "unet", "params": 1.5},
    "ModelScopeT2V": {"arch": "unet", "params": 1.7},
    "Lumiere": {"arch": "unet", "params": 5.0},
    "MagicVideo-V2": {"arch": "unet", "params": 1.5},

    # --- DiT models ---
    "Sora": {"arch": "dit", "params": 10.0},
    "WAN2.1-T2V-1.3B": {"arch": "dit", "params": 1.3},
    "WAN2.1-T2V-14B": {"arch": "dit", "params": 14.0},
    "Mochi 1": {"arch": "dit", "params": 10.0},
    "ContentV": {"arch": "dit", "params": 8.0},
    "VEO": {"arch": "dit", "params": 10.0},
    "Latte-XL": {"arch": "dit", "params": 0.67},

    # --- Hybrid (Transformer + 3D VAE) ---
    "CogVideoX-5B": {"arch": "hybrid", "params": 5.0},
    "CogVideoX-2B": {"arch": "hybrid", "params": 2.0},
}


def get_model_archi(model: str) -> dict:
    """Returns architecture and parameters for a given model."""
    return MODEL_CONFIGS.get(model, {"arch": "error", "params": 0})


def _yaml_get(cfg: dict, key: str):
    """Read a key from a loaded yaml, tolerating the legacy resolution_witdh typo."""
    if cfg is None:
        return None
    if key == "resolution_width":
        return cfg.get("resolution_width", cfg.get("resolution_witdh"))
    return cfg.get(key)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the environmental footprint of a video generation run.",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (see supported models).")
    parser.add_argument("--duration", type=int, default=None, help="Video duration in seconds.")
    parser.add_argument("--resolution-height", type=int, default=None, help="Video height in pixels.")
    parser.add_argument("--resolution-width", type=int, default=None, help="Video width in pixels.")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second.")
    parser.add_argument("--denoising-steps", type=int, default=None, help="Number of denoising steps.")
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["text", "image"],
        default=None,
        help='Input modality: "text" or "image".',
    )
    parser.add_argument("--country", type=str, default=None, help="Country for carbon intensity lookup.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path. CLI flags override file values. "
             "Defaults to input.yaml if it exists in the current directory.",
    )
    return parser


def _resolve_params(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """Merge CLI flags with optional YAML fallback. Errors if anything is missing."""
    cfg = None
    if args.config is not None:
        if not os.path.isfile(args.config):
            parser.error(f"--config file not found: {args.config}")
        cfg = load_yaml(args.config)
    elif os.path.isfile("input.yaml"):
        cfg = load_yaml("input.yaml")

    fields = [
        ("model", args.model, "--model"),
        ("duration", args.duration, "--duration"),
        ("resolution_height", args.resolution_height, "--resolution-height"),
        ("resolution_width", args.resolution_width, "--resolution-width"),
        ("fps", args.fps, "--fps"),
        ("denoising_steps", args.denoising_steps, "--denoising-steps"),
        ("input_type", args.input_type, "--input-type"),
        ("country", args.country, "--country"),
    ]

    resolved = {}
    missing = []
    for key, cli_value, flag in fields:
        value = cli_value if cli_value is not None else _yaml_get(cfg, key)
        if value is None:
            missing.append(flag)
        else:
            resolved[key] = value

    if missing:
        parser.error(f"missing required parameter(s): {', '.join(missing)}")

    return resolved


def run(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    params = _resolve_params(args, parser)

    model_config = get_model_archi(params["model"])
    if model_config["arch"] == "error":
        sys.stderr.write(f"error: unknown model '{params['model']}'\n")
        return 1

    total_frames = params["duration"] * params["fps"]
    predictions = run_ml(
        steps=params["denoising_steps"],
        res=params["resolution_height"] * params["resolution_width"],
        frames=total_frames,
        fps=params["fps"],
        duration=params["duration"],
        params=model_config["params"],
        arch=model_config["arch"],
        input_type=params["input_type"],
        country=params["country"],
    )

    if "error" in predictions:
        sys.stderr.write(f"error: {predictions['error']}\n")
        return 1

    output = {
        "inputs": {
            "model": params["model"],
            "steps": params["denoising_steps"],
            "resolution": f"{params['resolution_height']}x{params['resolution_width']}",
            "frames": total_frames,
        },
        "predictions": {
            "energy": predictions["energy"],
            "run_time": predictions["run_time"],
            "carbon": predictions["carbon"],
            "water_used": predictions["water_used"],
        },
    }
    sys.stdout.write(json.dumps(output, separators=(",", ":"), default=_json_default) + "\n")
    return 0


def _json_default(value):
    """Coerce numpy scalars (returned by the ML layer) to plain Python types."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    sys.exit(run())
