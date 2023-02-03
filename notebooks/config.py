# DATA_DIR = "../data"

# This script has tha purpose to declare global variables related to the dirs in the project
#
#
import pathlib
import sys

from typing import Optional, Dict, Any, NewType

import numpy as np

TMP_DIR = pathlib.Path("/tmp").resolve()
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
PYTHON_DIR = pathlib.Path(sys.executable).parent.parent.resolve()

_dict_type = Dict[str, Any]
dict_type = NewType("dict_type", _dict_type)


def struct_constructor(base_dir: pathlib.Path) -> dict_type:
    if not base_dir.is_dir():
        raise OSError(f"{base_dir} is not a directory.")
    dir_init: dict_type = dict_type({dir.parts[-1]: None for dir in base_dir.iterdir() if dir.is_dir() and not dir.parts[-1].startswith(".")})
    for key in dir_init.keys():
        d_dir = base_dir.joinpath(key)
        sub_dict = struct_constructor(pathlib.Path(d_dir))
        if sub_dict:
            dir_init.update({key: sub_dict})
        else:
            dir_init.update({key: None})
    return dir_init


def declare_consts(struct: Optional[dict_type], pre: str = "") -> None:
    globals_dict: dict = {}
    if struct is not None:
        for key, value in struct.items():
            assert isinstance(key, str)
            name_const = key.upper() + "_DIR"
            new_value = BASE_DIR.joinpath(pre, key)
            globals_dict.update({name_const: new_value})
            if isinstance(value, dict):
                declare_consts(dict_type(value), pre=str(key))
    globals().update(globals_dict)


struct = struct_constructor(BASE_DIR)
declare_consts(dict_type(struct))
print(f"declared {len(struct)} variables") # TODO LOGGING


import matplotlib as mpl

mpl_config = {
    "xtick.bottom": False,
    "xtick.labelbottom": False,
    "ytick.left": False,
    "ytick.labelleft": False,
    "image.cmap": "gray"
}

mpl.rcParams.update(mpl_config)

_ = list(map(
    lambda x: mpl.rcParams.update({x: False}),
    list(
        filter(lambda x: True if "spine" in x else False, mpl.rcParams.keys())
    )
))

np.set_printoptions(suppress=True)


