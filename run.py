from utils import check_dict, load_yaml
from ml.compute_wh import run_ml


def get_model_archi(model: str) -> dict:
    """Returns architecture and parameters for a given model"""

    model_configs = {
        # --- UNet models ---
        "AnimateDiff": {"arch": "unet", "params": 0.9},
        "Stable Video Diffusion": {"arch": "unet", "params": 1.5},
        "Runway Gen-2": {"arch": "unet", "params": 1.5},
        "Pika 1.0": {"arch": "unet", "params": 1.5},
        "ModelScopeT2V": {"arch": "unet", "params": 1.7},
        "Lumiere": {"arch": "unet", "params": 5.0},
        "MagicVideo-V2": {"arch": "unet", "params": 1.5},

        # --- DiT models ---
        "Sora": {"arch": "dit", "params": 10.0},
        "WAN2.1-T2V-1.3B": {"arch": "dit", "params": 1.3},
        "WAN2.1-T2V-14B": {"arch": "dit", "params": 14.0},
        "Mochi 1": {"arch": "dit", "params": 10.0},
        "MAGI-1": {"arch": "dit", "params": 24.0},
        "ContentV": {"arch": "dit", "params": 8.0},
        "VEO": {"arch": "dit", "params": 10.0},
        "Latte-XL": {"arch": "dit", "params": 0.67},

        # --- Hybrid (Transformer + 3D VAE) ---
        "CogVideoX-5B": {"arch": "hybrid", "params": 5.0},
        "CogVideoX-2B": {"arch": "hybrid", "params": 2.0},
    }

    if model in model_configs:
        return model_configs[model]
    return {"arch": "error", "params": 0}


def run():

    cfg = load_yaml("input.yaml")
    check_dict(cfg, ["model", "duration", "fps", "resolution_height", "resolution_witdh", "country", "denoising_steps"])
    model_config = get_model_archi(cfg["model"])

    if model_config["arch"] == "error":
        raise ValueError("Error, model can't be handled or is badly written")

    model_type = model_config["arch"]
    params = model_config["params"]
    fps = cfg["fps"]
    duration = cfg["duration"]
    steps = cfg["denoising_steps"]
    total_frames = duration * fps

    input_type = cfg.get("input_type", "text")  # Default to text if not specified
    predictions = run_ml(
        steps=steps,
        res=cfg["resolution_height"] * cfg["resolution_witdh"],
        frames=total_frames,
        fps=fps,
        duration=duration,
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
    print(f"  Steps: {steps}")
    print(f"  Resolution: {cfg['resolution_height']}x{cfg['resolution_witdh']}")
    print(f"  Frames: {total_frames}\n")

    print("📊 RESULTS:")
    print(f"  Energy: {predictions['energy']['value_wh']:.2f} ± {predictions['energy']['uncertainty_wh']:.2f} Wh")
    print(f"    Model: {predictions['energy']['model']} (R²={predictions['energy']['r2']})")
    print(f"    95% interval: {predictions['energy']['best_case_wh']:.2f} - {predictions['energy']['worst_case_wh']:.2f} Wh")

    print(f"\n  run_time: {predictions['run_time']['value_s']:.2f} ± {predictions['run_time']['uncertainty_s']:.2f} s ({predictions['run_time']['value_min']:.2f} min)")
    print(f"    Model: {predictions['run_time']['model']} (R²={predictions['run_time']['r2']})")
    print(f"    95% interval: {predictions['run_time']['best_case_s']:.2f} - {predictions['run_time']['worst_case_s']:.2f} s")

    print(f"\n  Carbon emissions: {predictions['carbon']['value_gco2e']:.2f} gCO2e")
    print(f"    Embodied: {predictions['carbon']['g_co2_embodied']:.2f} gCO2e")
    print(f"    Electricity: {predictions['carbon']['g_co2_electricity']:.2f} gCO2e")
    print(f"    95% interval: {predictions['carbon']['best_case_gco2e']:.2f} - {predictions['carbon']['worst_case_gco2e']:.2f} gCO2e")

    print(f"\n  Water used: {predictions['water_used']['value_water_used']:.2f} L")
    print(f"    95% interval: {predictions['water_used']['best_case_water_used']:.2f} - {predictions['water_used']['worst_case_water_used']:.2f} L")

    output = {
        'inputs': {
            'model': cfg['model'],
            'steps': steps,
            'resolution': f"{cfg['resolution_height']}x{cfg['resolution_witdh']}",
            'frames': total_frames
        },
        'predictions': {
            'energy': predictions['energy'],
            'run_time': predictions['run_time'],
            'carbon': predictions['carbon'],
            'water_used': predictions['water_used']
        }
    }
    return output


if __name__ == "__main__":
    run()
