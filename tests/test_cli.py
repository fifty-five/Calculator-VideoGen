import json


ALL_FLAGS = [
    "--model", "CogVideoX-5B",
    "--duration", "5",
    "--fps", "24",
    "--resolution-height", "1280",
    "--resolution-width", "720",
    "--denoising-steps", "40",
    "--input-type", "image",
    "--country", "China",
]


def test_all_flags_emits_single_line_json(run_cli, empty_yaml):
    code, stdout, stderr = run_cli(*ALL_FLAGS, config=empty_yaml)
    assert code == 0, stderr
    assert stdout.count("\n") == 1
    payload = json.loads(stdout)
    assert set(payload.keys()) == {"inputs", "predictions"}
    assert payload["inputs"]["model"] == "CogVideoX-5B"
    assert payload["inputs"]["resolution"] == "1280x720"
    assert payload["inputs"]["frames"] == 5 * 24
    assert set(payload["predictions"].keys()) == {"energy", "run_time", "carbon", "water_used"}


def test_yaml_fallback_fills_missing_flags(run_cli, tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "duration: 5\n"
        "fps: 24\n"
        "resolution_height: 1280\n"
        "resolution_width: 720\n"
        "denoising_steps: 40\n"
        "input_type: image\n"
        "country: China\n"
    )
    code, stdout, stderr = run_cli("--model", "CogVideoX-5B", config=yaml_path)
    assert code == 0, stderr
    payload = json.loads(stdout)
    assert payload["inputs"]["model"] == "CogVideoX-5B"
    assert payload["inputs"]["resolution"] == "1280x720"


def test_legacy_typo_key_accepted(run_cli, tmp_path):
    yaml_path = tmp_path / "legacy.yaml"
    yaml_path.write_text(
        "model: CogVideoX-5B\n"
        "duration: 5\n"
        "fps: 24\n"
        "resolution_height: 1280\n"
        "resolution_witdh: 720\n"  # legacy typo
        "denoising_steps: 40\n"
        "input_type: image\n"
        "country: China\n"
    )
    code, stdout, stderr = run_cli(config=yaml_path)
    assert code == 0, stderr
    payload = json.loads(stdout)
    assert payload["inputs"]["resolution"] == "1280x720"


def test_missing_param_errors(run_cli, empty_yaml):
    code, stdout, stderr = run_cli("--model", "CogVideoX-5B", config=empty_yaml)
    assert code != 0
    assert stdout == ""
    assert "--duration" in stderr or "missing" in stderr.lower()


def test_bad_model_errors(run_cli, empty_yaml):
    args = ["--model", "Nonsense"] + ALL_FLAGS[2:]
    code, stdout, stderr = run_cli(*args, config=empty_yaml)
    assert code != 0
    assert stdout == ""
    assert "Nonsense" in stderr or "unknown model" in stderr


def test_stdout_is_clean_json(run_cli, empty_yaml):
    code, stdout, stderr = run_cli(*ALL_FLAGS, config=empty_yaml)
    assert code == 0, stderr
    # Exactly one trailing newline, one line of content.
    assert stdout.endswith("\n")
    assert stdout.count("\n") == 1
    line = stdout.rstrip("\n")
    # Must parse as JSON and have no emoji / ascii art markers that the old prints emitted.
    json.loads(line)
    for marker in ("📥", "📊", "INPUTS:", "RESULTS:", "Energy:", "Carbon"):
        assert marker not in stdout
