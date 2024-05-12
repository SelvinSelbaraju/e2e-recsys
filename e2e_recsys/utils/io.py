import json


def load_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        json_dict = json.load(f)
    return json_dict
