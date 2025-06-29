from dataclasses import dataclass
from typing import Literal

import casadi

from .casadi_helper import casadi_vars_to_str
from .ode_model import OdeModel
from .scaling import Scaling


@dataclass
class DiscreteModel:
    states: casadi.MX
    inputs: casadi.MX
    Ts: float
    x_next: casadi.MX
    y: casadi.MX
    y_names: tuple[str, ...]
    scalings: dict[str, Scaling]
    scaled: bool


def discretize_model(
    model: OdeModel, Ts: float, method: Literal["rk1", "rk4"], substeps: int = 1
) -> DiscreteModel:
    # create real copies of system variables
    x0 = casadi.vertcat(*(casadi.MX.sym(s) for s in casadi_vars_to_str(model.states)))
    u = casadi.vertcat(*(casadi.MX.sym(s) for s in casadi_vars_to_str(model.inputs)))

    f = casadi.Function("f", [model.states, model.inputs], [model.dx])

    dt = Ts / substeps
    x = x0

    if method == "rk1":
        for _ in range(substeps):
            x = x + dt * f(x, u)

    elif method == "rk4":
        for _ in range(substeps):
            k1 = f(x, u)
            k2 = f(x + dt / 2 * k1, u)
            k3 = f(x + dt / 2 * k2, u)
            k4 = f(x + dt * k3, u)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    else:
        assert False

    return DiscreteModel(
        states=x0,
        inputs=u,
        Ts=Ts,
        x_next=x,
        y=casadi.substitute(
            model.y,
            casadi.vertcat(model.states, model.inputs),
            casadi.vertcat(x0, u),
        ),
        y_names=model.y_names,
        scalings=model.scalings,
        scaled=False,
    )
