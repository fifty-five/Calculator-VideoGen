from utils import check_dict, load_yaml
from ml.compute_wh import run_ml
import numpy as np


def get_model_archi(model: str) -> str:
    UNet_models = [
        "AnimateDiff",
        "Stable Video Diffusion",
        "Runway (Gen-1, Gen-2)",
        "Pika",
        "ModelScopeT2V",
        "Lumiere",
        "MagicVideoazy"
    ]

    dit_models = [
        "Sora",
        "Veo",
        "Latte",
        "WAN2.1",
        "WAN2.2",
        "ContentV",
        "MAGI-1",
        "CogVideoX-5B",
        "CogVideoX-2B",
        "CogVideoX-1.5",
        "Runway Gen-4",
        "Mochi"
    ]

    if model in UNet_models:
        return "unet"
    elif model in dit_models:
        return "dit"
    return "error"


def run():
    cfg = load_yaml("input.yaml")
    check_dict(cfg, ["model", "duration", "fps", "resolution_height", "resolution_witdh", "denoising_steps", "country"])
    cfg["frame"] = cfg["duration"] * cfg["fps"]
    model_type = get_model_archi(cfg["model"])
    if model_type == "error":
        raise ValueError("Error, model can't be handled or is badly written")
    prepared_data = np.array([cfg["denoising_steps"], cfg["resolution_height"] * cfg["resolution_witdh"], cfg["frame"]])
    predictions = run_ml(prepared_data, model_type, cfg["country"])
    # Afficher les inputs
    print("📥 INPUTS:")
    print(f"  Model: {cfg['model']} ({model_type})")
    print(f"  Steps: {cfg['denoising_steps']}")
    print(f"  Resolution: {cfg['resolution_height']}x{cfg['resolution_witdh']}")
    print(f"  Frames: {cfg['frame']}\n")

    print("\n📊 RESULTS:")
    print(f"  Energy: {predictions[0]:.2f} Wh")
    print(f"  RunTime: {predictions[1]:.2f} s")
    print(f"  Total emissions: {predictions[2]:.2f} gCO2e")

    output = {
        'inputs': {
            'model': cfg['model'],
            'steps': cfg['denoising_steps'],
            'resolution': f"{cfg['resolution_height']}x{cfg['resolution_witdh']}",
            'frames': cfg['frame']
        },
        'results': {
            'energy_wh': round(predictions[0], 2),
            'runtime': round(predictions[1], 2),
            'Total emissions': round(predictions[2], 2),
        }
    }

    return output


if __name__ == "__main__":
    run()
