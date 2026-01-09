import yaml
import os
import polars as pl


def load_yaml(file: str) -> dict:
    with open(file, 'r') as file_content:
        yaml_data = yaml.safe_load(file_content)
    return yaml_data


def dump_yaml(data, filename):
    with open(filename, 'w') as file:
        yaml.dump(data, file)


def dump_csv_polars(data: pl.DataFrame, filename):
    foldername = '/'.join(filename.split('/')[:-1])
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    data.write_csv(filename)


def check_dict(cfg: dict, list_keys: list):
    for key in list_keys:
        try:
            cfg[key]
        except KeyError:
            print("Missing", key, "in the input.yaml")
