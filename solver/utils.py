import pathlib
from typing import Any

from ruamel.yaml import YAML


def read_yaml(filename: pathlib.Path) -> Any:
    if not filename.exists():
        raise Exception(f"Filename {filename} does not exists")

    with open(filename, "r", encoding="utf-8") as f:
        try:
            yaml = YAML(typ="safe")
            return yaml.load(f)
        except Exception as e:
            raise Exception(
                f"Unable to load YAML from {filename}. Exception {e}"
            ) from e


def save_yaml(data: Any, filename: pathlib.Path):
    def repr_path(representer, data):
        return representer.represent_scalar("tag:yaml.org,2002:str", str(data))

    with open(filename, "w") as f:
        with YAML(typ="rt", output=f) as yaml:
            for p in [pathlib.PosixPath, pathlib.WindowsPath]:
                yaml.representer.add_representer(p, repr_path)
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.explicit_start = True
            yaml.dump(data, f)
