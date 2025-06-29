import casadi as cs
import numpy as np

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from .casadi_helper import casadi_vars_to_str, create_casadi_vars
from .scaling import Scaling

ModelFct = Callable[[Optional[tuple[dict[str, Any], dict[str, Any], Any]]], Any]


@dataclass
class OdeModel:
    states: cs.MX
    inputs: cs.MX
    dx: cs.MX
    y: cs.MX
    y_names: tuple[str, ...]
    scalings: dict[str, Scaling]
    scaled: bool


def get_model(model: ModelFct, p) -> OdeModel:

    (vars, scalings) = model()

    x = create_casadi_vars(vars["states"])
    u = create_casadi_vars(vars["inputs"])

    states = cs.vertcat(*x.values())
    inputs = cs.vertcat(*u.values())

    (dx, y) = model((x, u, p))

    dx = cs.vertcat(*(dx[state] for state in x.keys()))

    y_names = tuple(y.keys())
    y = cs.vertcat(*y.values())

    return OdeModel(
        states=states,
        inputs=inputs,
        dx=dx,
        y=y,
        y_names=y_names,
        scalings=scalings,
        scaled=False,
    )


def scale_model(model: OdeModel) -> OdeModel:
    if model.scaled:
        print("model allready scaled")
        return

    return _scale(model, "scale")


def unscale_model(model: OdeModel) -> OdeModel:
    if not model.scaled:
        print("model allready unscaled")
        return

    return _scale(model, "unscale")


def _scale(model: OdeModel, direction: Literal["scale", "unscale"]) -> OdeModel:

    if len(model.scalings) == 0:
        raise ValueError("model doesn't provide scalings")

    scalings = model.scalings

    scaled_names = [v for v in scalings.keys()]

    dx = cs.MX(model.dx)
    y = cs.MX(model.y)

    x_names = casadi_vars_to_str(model.states)
    u_names = casadi_vars_to_str(model.inputs)
    y_names = model.y_names

    for i, name in enumerate(x_names):
        if name in scaled_names:
            if direction == "scale":
                dx[i] = scalings[name].scale_derivate(dx[i])
            else:
                dx[i] = scalings[name].unscale_derivate(dx[i])

    for i, name in enumerate(y_names):
        if name in scaled_names:
            if direction == "scale":
                y[i] = scalings[name].scale(y[i])
            else:
                y[i] = scalings[name].unscale(y[i])

    all_names = x_names + u_names
    all_vars = cs.vertcat(model.states, model.inputs)

    no_scaling_vars = set(all_names) - set(scaled_names)

    if len(no_scaling_vars) > 0:
        print(
            f"no scaling for the following signals given:\n    {', '.join(sorted(no_scaling_vars))}"
        )

    for i, name in enumerate(all_names):
        var = all_vars[i]

        scaling = scalings[name]

        if direction == "scale":
            expr = scaling.unscale(var)
        else:
            expr = scaling.scale(var)

        dx = cs.substitute(dx, var, expr)
        y = cs.substitute(y, var, expr)

    dx = cs.simplify(dx)
    y = cs.simplify(y)

    new_states = cs.MX(model.states)
    new_inputs = cs.MX(model.inputs)

    new_vars = cs.vertcat(new_states, new_inputs)

    dx = cs.substitute(dx, all_vars, new_vars)
    y = cs.substitute(y, all_vars, new_vars)

    return OdeModel(
        new_states,
        new_inputs,
        dx,
        y,
        y_names=y_names,
        scalings=model.scalings.copy(),
        scaled=(direction == "scale"),
    )


def get_linearized_matrices(
    model: OdeModel,
    x: np.ndarray | cs.MX | None = None,
    u: np.ndarray | cs.MX | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[cs.MX, cs.MX, cs.MX, cs.MX]
    | tuple[cs.Function, cs.Function, cs.Function, cs.Function]
):

    df_dx = cs.jacobian(model.dx, model.states)
    df_dx = cs.Function(
        "df_dx", [model.states, model.inputs], [df_dx], ["x", "u"], ["df_dx"]
    )

    df_du = cs.jacobian(model.dx, model.inputs)
    df_du = cs.Function(
        "df_dx", [model.states, model.inputs], [df_du], ["x", "u"], ["df_du"]
    )

    dh_dx = cs.jacobian(model.y, model.states)
    dh_dx = cs.Function(
        "dh_dx", [model.states, model.inputs], [dh_dx], ["x", "u"], ["dh_dx"]
    )

    dh_du = cs.jacobian(model.y, model.inputs)
    dh_du = cs.Function(
        "dh_du", [model.states, model.inputs], [dh_du], ["x", "u"], ["dh_du"]
    )

    if x is None:
        return (df_dx, df_du, dh_dx, dh_du)
    elif isinstance(x, cs.MX):
        return (df_dx(x, u), df_du(x, u), dh_dx(x, u), dh_du(x, u))
    else:
        return (
            df_dx(x, u).full(),
            df_du(x, u).full(),
            dh_dx(x, u).full(),
            dh_du(x, u).full(),
        )