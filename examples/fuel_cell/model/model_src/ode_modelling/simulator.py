import casadi
import numpy as np

from .casadi_helper import casadi_vars_to_str
from .discrete_model import DiscreteModel
from .ode_model import OdeModel
from .simresult import SimResult


class Simulator:
    def __init__(self, model: OdeModel | DiscreteModel, Ts: float):
        discrete = isinstance(model, DiscreteModel)

        self.Ts = Ts

        if discrete:
            assert Ts == model.Ts

            self.f_x = casadi.Function(
                "f", [model.states, model.inputs], [model.x_next]
            )
            self.F = None
        else:
            t0 = 0
            tf = Ts

            dae = {"x": model.states, "p": model.inputs, "ode": model.dx}
            self.f_x = None
            self.F = casadi.integrator("F", "idas", dae, t0, tf)

        self.f_out = casadi.Function("outputs", [model.states, model.inputs], [model.y])

    def sim_step(
        self, x0: np.ndarray, u: np.ndarray, feedthrough_start: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.f_x is not None:
            x = self.f_x(x0, u)
        else:
            r = self.F(x0=x0, p=u)
            x = r["xf"]

        y = self.f_out(x, u).full()
        x = x.full()

        if not feedthrough_start:
            return (x, y)
        else:
            y0_ft = self.f_out(x0, u).full()

            return (x, y, y0_ft)


def sim_nl(model: OdeModel | DiscreteModel, Ts: float, x0: np.ndarray, u: np.ndarray):

    simulator = Simulator(model, Ts)

    N_sim = u.shape[1]

    x = np.zeros((x0.size, N_sim + 1))
    x.fill(np.nan)

    x[:, 0] = x0
    y0 = simulator.f_out(x0, u[:, 0].T)

    y = np.zeros((y0.numel(), N_sim + 1))
    y.fill(np.nan)

    y_ft = y.copy()

    y[:, 0] = y0

    t = np.arange(N_sim + 1) * Ts

    for i in range(N_sim):
        x_i = x[:, i]
        u_i = u[:, i]

        try:
            (x_next, y_next, y_ft_i) = simulator.sim_step(
                x_i, u_i, feedthrough_start=True
            )
        except RuntimeError:
            print(f"Simulation failed at step {i + 1} (t = {i * Ts})")
            break

        x[:, i + 1] = x_next.reshape((-1,))
        y[:, i + 1] = y_next.reshape((-1,))
        y_ft[:, i] = y_ft_i.reshape((-1,))

    y_ft[:, -1] = y[:, -1]

    return SimResult(
        t=t,
        u=np.hstack((u, np.nan * np.zeros((u.shape[0], 1)))),
        x=x,
        y=y,
        y_ft=y_ft,
        u_names=casadi_vars_to_str(model.inputs),
        x_names=casadi_vars_to_str(model.states),
        y_names=model.y_names,
        scalings=model.scalings,
        scaled=model.scaled,
        constraints={},
        discrete_model=isinstance(model, DiscreteModel),
        Ts=Ts,
        desc="",
    )
