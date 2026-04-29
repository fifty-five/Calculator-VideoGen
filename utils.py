import os

import polars as pl
import yaml


def load_yaml(file: str) -> dict:
    with open(file, "r", encoding="utf-8") as file_content:
        yaml_data = yaml.safe_load(file_content)
    return yaml_data


def dump_yaml(data, filename):
    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(data, file)


def dump_csv_polars(data: pl.DataFrame, filename):
    foldername = "/".join(filename.split("/")[:-1])
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    data.write_csv(filename)


def check_dict(cfg: dict, list_keys: list):
    missing = [key for key in list_keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing keys in config: {', '.join(missing)}")
