from collections.abc import Sequence
from typing import Literal

import numpy as np

from .casadi_helper import casadi_vars_to_str
from .ode_model import OdeModel
from .scaling import Scaling


def scale_data(
    scalings: dict[str, Scaling], names: Sequence[str], data: np.ndarray
) -> np.ndarray:
    assert len(names) == data.shape[0]

    data = data.copy()

    for i, name in enumerate(names):
        if name not in scalings.keys():
            print(f"no scaling for signal '{name}' given")
            continue

        if len(data.shape) == 1:
            data[i] = scalings[name].scale(data[i])
        else:
            data[i, :] = scalings[name].scale(data[i, :])

    return data


def unscale_data(
    scalings: dict[str, Scaling], names: Sequence[str], data: np.ndarray
) -> np.ndarray:
    assert len(names) == data.shape[0]

    data = data.copy()

    for i, name in enumerate(names):
        if name not in scalings.keys():
            print(f"no scaling for signal '{name}' given")
            continue

        if len(data.shape) == 1:
            data[i] = scalings[name].unscale(data[i])
        else:
            data[i, :] = scalings[name].unscale(data[i, :])

    return data


def scale_model_signals(
    model: OdeModel,
    kind: Literal["states", "x", "inputs", "u", "outputs", "y"],
    data: np.ndarray,
) -> np.ndarray:
    if kind in ("states", "x"):
        return scale_data(model.scalings, casadi_vars_to_str(model.states), data)
    elif kind in ("inputs", "u"):
        return scale_data(model.scalings, casadi_vars_to_str(model.inputs), data)
    elif kind in ("ouputs", "y"):
        return scale_data(model.scalings, model.y_names, data)

    assert False


def unscale_model_signals(
    model: OdeModel,
    kind: Literal["states", "x", "inputs", "u", "outputs", "y"],
    data: np.ndarray,
) -> np.ndarray:
    if kind in ("states", "x"):
        return unscale_data(model.scalings, casadi_vars_to_str(model.states), data)
    elif kind in ("inputs", "u"):
        return unscale_data(model.scalings, casadi_vars_to_str(model.inputs), data)
    elif kind in ("ouputs", "y"):
        return unscale_data(model.scalings, model.y_names, data)

    assert False
