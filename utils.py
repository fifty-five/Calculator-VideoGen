import yaml


def load_yaml(file: str) -> dict:
    with open(file, 'r') as file_content:
        yaml_data = yaml.safe_load(file_content)
    return yaml_data


def check_dict(cfg: dict, list_keys: list):
    for key in list_keys:
        try:
            cfg[key]
        except KeyError:
            print("Missing", key, "in the input.yaml")
