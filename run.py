from utils import check_dict, load_yaml
from ml.compute_wh import run_ml


def get_model_archi(model: str) -> dict:
    """Returns architecture and parameters for a given model"""

    model_configs = {
        # UNet models
        "AnimateDiff": {"arch": "unet", "params": 0.8},
        "Stable Video Diffusion": {"arch": "unet", "params": 0.5},
        "Runway (Gen-1, Gen-2)": {"arch": "unet", "params": 1.0},
        "Pika": {"arch": "unet", "params": 1.5},
        "ModelScopeT2V": {"arch": "unet", "params": 1.0},
        "Lumiere": {"arch": "unet", "params": 0.6},
        "MagicVideoazy": {"arch": "unet", "params": 0.8},

        # DiT models
        "Sora": {"arch": "dit", "params": 4.0},
        "Veo": {"arch": "dit", "params": 3.0},
        "Latte": {"arch": "dit", "params": 2.0},
        "WAN2.1": {"arch": "dit", "params": 6.0},
        "WAN2.2": {"arch": "dit", "params": 7.0},
        "ContentV": {"arch": "dit", "params": 1.5},
        "MAGI-1": {"arch": "dit", "params": 2.5},
        "CogVideoX-5B": {"arch": "cog", "params": 5.0},
        "CogVideoX-2B": {"arch": "cog", "params": 2.0},
        "CogVideoX-1.5": {"arch": "cog", "params": 1.5},
        "Runway Gen-4": {"arch": "dit", "params": 3.5},
        "Mochi": {"arch": "dit", "params": 2.0},
    }

    if model in model_configs:
        return model_configs[model]
    return {"arch": "error", "params": 0}


def run():
    # Fixed parameters
    DENOISING_STEPS = 50
    FPS = 24
    
    cfg = load_yaml("input.yaml")
    check_dict(cfg, ["model", "duration", "resolution_height", "resolution_witdh", "country"])
    total_frames = cfg["duration"] * FPS
    model_config = get_model_archi(cfg["model"])

    if model_config["arch"] == "error":
        raise ValueError("Error, model can't be handled or is badly written")

    model_type = model_config["arch"]
    params = model_config["params"]

    input_type = cfg.get("input_type", "text")  # Default to text if not specified
    predictions = run_ml(
        steps=DENOISING_STEPS,
        res=cfg["resolution_height"] * cfg["resolution_witdh"],
        frames=total_frames,
        params=params,
        arch=model_type,
        input_type=input_type,
        country=cfg["country"]
    )

    # Check for errors
    if "error" in predictions:
        print(f"❌ Error: {predictions['error']}")
        return predictions

    # Display results
    print("\n📥 INPUTS:")
    print(f"  Model: {cfg['model']} ({model_type}, {params}B params)")
    print(f"  Steps: {DENOISING_STEPS}")
    print(f"  Resolution: {cfg['resolution_height']}x{cfg['resolution_witdh']}")
    print(f"  Frames: {total_frames}\n")

    print("📊 RESULTS:")
    print(f"  Energy: {predictions['energy']['value_wh']:.2f} ± {predictions['energy']['uncertainty_wh']:.2f} Wh")
    print(f"    Model: {predictions['energy']['model']} (R²={predictions['energy']['r2']})")
    print(f"    95% interval: {predictions['energy']['best_case_wh']:.2f} - {predictions['energy']['worst_case_wh']:.2f} Wh")

    print(f"\n  Duration: {predictions['duration']['value_s']:.2f} ± {predictions['duration']['uncertainty_s']:.2f} s ({predictions['duration']['value_min']:.2f} min)")
    print(f"    Model: {predictions['duration']['model']} (R²={predictions['duration']['r2']})")
    print(f"    95% interval: {predictions['duration']['best_case_s']:.2f} - {predictions['duration']['worst_case_s']:.2f} s")

    print(f"\n  Carbon emissions: {predictions['carbon']['value_gco2e']:.2f} gCO2e")
    print(f"    95% interval: {predictions['carbon']['best_case_gco2e']:.2f} - {predictions['carbon']['worst_case_gco2e']:.2f} gCO2e")

    output = {
        'inputs': {
            'model': cfg['model'],
            'steps': DENOISING_STEPS,
            'resolution': f"{cfg['resolution_height']}x{cfg['resolution_witdh']}",
            'frames': total_frames
        },
        'predictions': {
            'energy': predictions['energy'],
            'duration': predictions['duration'],
            'carbon': predictions['carbon']
        }
    }
    return output


if __name__ == "__main__":
    run()
