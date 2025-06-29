from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np
from matplotlib import pyplot as plt

from .model_helper import scale_data, unscale_data
from .scaling import Scaling
from .signalinfo import SignalInfo


@dataclass
class SimResult:
    t: np.ndarray
    u: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_ft: Optional[np.ndarray]
    u_names: tuple[str, ...]
    x_names: tuple[str, ...]
    y_names: tuple[str, ...]

    scalings: dict[str, Scaling]
    scaled: bool

    constraints: dict[str, tuple[float, float]]

    discrete_model: bool
    Ts: float
    desc: str

    def copy(self):
        return SimResult(
            t=self.t.copy(),
            u=self.u.copy(),
            x=self.x.copy(),
            y=self.y.copy(),
            y_ft=None if self.y_ft is None else self.y_ft.copy(),
            u_names=self.u_names,
            x_names=self.x_names,
            y_names=self.y_names,
            scalings=self.scalings.copy(),
            scaled=self.scaled,
            constraints=self.constraints.copy(),
            discrete_model=self.discrete_model,
            Ts=self.Ts,
            desc=self.desc,
        )

    def scale(self):
        if self.scaled:
            print("data allready scaled")
            return

        self.u = scale_data(self.scalings, self.u_names, self.u)
        self.x = scale_data(self.scalings, self.x_names, self.x)
        self.y = scale_data(self.scalings, self.y_names, self.y)

        if self.y_ft is not None:
            self.y_ft = scale_data(self.scalings, self.y_names, self.y_ft)

        self.scaled = True

    def unscale(self):
        if not self.scaled:
            print("data allready unscaled")
            return

        self.u = unscale_data(self.scalings, self.u_names, self.u)
        self.x = unscale_data(self.scalings, self.x_names, self.x)
        self.y = unscale_data(self.scalings, self.y_names, self.y)

        if self.y_ft is not None:
            self.y_ft = unscale_data(self.scalings, self.y_names, self.y_ft)

        self.scaled = False


_BASECOLORS: tuple[tuple[float, float, float], ...] = [
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
]


def plot_sim_results(
    results: SimResult | Sequence[SimResult],
    plot_signals: Optional[Sequence[str]] = None,
    *,
    reuse_figures: bool = False,
    signal_infos: Optional[dict[str, SignalInfo]] = None,
    auto_unscale: bool = True,
):
    if plot_signals is None:
        plot_signals = ["@states, @outputs, @inputs"]
    elif isinstance(plot_signals, str):
        plot_signals = [plot_signals]

    if isinstance(results, SimResult):
        results = (results,)

    n_results = len(results)

    x_names: list[str] = []
    y_names: list[str] = []
    u_names: list[str] = []

    scaled = results[0].scaled

    wc_results: list[SimResult] = []

    for i in range(len(results)):
        res = results[i]
        for x in res.x_names:
            if x not in x_names:
                x_names.append(x)

        for y in res.y_names:
            if y not in y_names:
                y_names.append(y)

        for u in res.u_names:
            if u not in u_names:
                u_names.append(u)

        if res.scaled:
            if auto_unscale:
                res = res.copy()
                res.unscale()
            elif res.scaled != scaled:
                raise ValueError("the results must be all scaled or all unscaled")

        wc_results.append(res)

    results = wc_results

    if auto_unscale:
        scaled = False

    legendtexts = []

    if n_results > 1 or results[0].desc != "":
        for i, res in enumerate(results):
            if res.desc == "":
                legendtexts.append(f"sim. {i}")
            else:
                legendtexts.append(res.desc)

    ax_link_master: Optional[plt.Axes] = None

    for i_fig in range(len(plot_signals)):
        (title_str, signals) = parse_plot_info(plot_signals[i_fig])

        signals = replace_category_names(signals, x_names, u_names, y_names)

        if title_str == "":
            if reuse_figures:
                plt.clf()
            else:
                plt.figure()
        elif reuse_figures:
            plt.figure(title_str)
            plt.clf()
        else:
            plt.figure(title_str)
            plt.clf()

        n_subplots = len(signals)

        n_cols = 1 if n_subplots < 5 else 2

        n_rows = int(np.ceil(n_subplots / n_cols))

        vh: list[plt.Axes] = []

        for ip, signal in enumerate(signals):

            vh.append(plt.subplot(n_rows, n_cols, ip + 1, sharex=ax_link_master))

            if ax_link_master is None:
                ax_link_master = vh[-1]

            curlegendtexts = []

            for ir, res in enumerate(results):
                if ir >= len(_BASECOLORS):
                    raise ValueError(
                        f"more than {len(_BASECOLORS)} cannot be plotted together"
                    )

                basecolor = _BASECOLORS[ir]

                plotted = plot_(vh[-1], res, signal, signal_infos, basecolor, scaled)

                if plotted and len(legendtexts) > 0:
                    curlegendtexts.append(legendtexts[ir])

            if len(curlegendtexts) > 0:
                plt.legend(curlegendtexts)

            if ip >= n_subplots - n_cols:
                if "time" in signal_infos:
                    info = signal_infos["time"]
                    vh[-1].set_xlabel(f"{info.tex} / {info.disp_unit}")
                else:
                    vh[-1].set_xlabel("time")

            if (signal_infos is not None) and (signal in signal_infos):
                info = signal_infos[signal]

                sig_title = info.tex

                if scaled:
                    sig_ylabel = f"{info.tex} (scaled)"
                elif info.disp_unit == "":
                    sig_ylabel = info.tex
                else:
                    sig_ylabel = f"{info.tex} / {info.disp_unit}"

            else:
                sig_title = signal

                if scaled:
                    sig_ylabel = f"{signal} (scaled)"
                else:
                    sig_ylabel = signal

            vh[-1].set_title(sig_title)
            vh[-1].set_ylabel(sig_ylabel)

            vh[-1].grid(True)

        plt.tight_layout()


def plot_(
    ax: plt.Axes,
    res: SimResult,
    signal: str,
    signal_infos: dict[str, SignalInfo],
    basecolor: tuple[float, float, float],
    scaled: bool,
) -> bool:
    stair_plot = False

    (kind, idx) = find_signal(res, signal)

    if kind == "x":
        values = res.x[idx, :]
        t = res.t
    elif kind == "y":
        values = res.y[idx, :]
        t = res.t

        if res.y_ft is not None:
            values_ft = res.y_ft[idx, :]

            t = np.hstack((t.reshape((-1, 1)), t.reshape((-1, 1)))).reshape((-1,))[1:-1]
            values = np.hstack(
                (values.reshape((-1, 1)), values_ft.reshape((-1, 1)))
            ).reshape((-1,))[1:-1]

    elif kind == "u":
        stair_plot = True
        values = res.u[idx, :]
        t = res.t
    else:
        return False

    if (not scaled) and (signal_infos is not None) and (signal in signal_infos):
        info = signal_infos[signal]
    else:
        info = SignalInfo(signal, "", "", lambda x: x)

    if (signal_infos is not None) and ("time" in signal_infos):
        t_info = signal_infos["time"]
    else:
        t_info = SignalInfo("time", "", "", lambda x: x)

    if not stair_plot:
        ax.plot(t_info.disp_fct(t), info.disp_fct(values), color=basecolor)
    else:
        ax.step(
            t_info.disp_fct(t), info.disp_fct(values), color=basecolor, where="post"
        )

    if signal in res.constraints:
        (min, max) = res.constraints[signal]

        if min > -np.inf:
            ax.axhline(info.disp_fct(min), linestyle=":", color="k")

        if max < np.inf:
            ax.axhline(info.disp_fct(max), linestyle=":", color="k")

    return True


def parse_plot_info(s: str) -> tuple[str, list[str]]:
    title_signals = s.split(":")

    if len(title_signals) == 1:
        title_signals = [""] + title_signals

    title = title_signals[0].strip()
    signals = [sig.strip() for sig in title_signals[1].split(",")]

    return (title, signals)


def replace_category_names(
    names: Sequence[str],
    x_names: Iterable[str],
    u_names: Iterable[str],
    y_names: Iterable[str],
) -> list[str]:
    new_names = []

    for name in names:
        if name == "@states":
            new_names.extend(x_names)
        elif name == "@inputs":
            new_names.extend(u_names)
        elif name == "@outputs":
            new_names.extend(y_names)
        else:
            new_names.append(name)

    return new_names


def find_index(el: Any, it: Iterable[Any]) -> int:
    for i, v in enumerate(it):
        if v == el:
            return i

    else:
        return -1


def find_signal(result: SimResult, name: str) -> tuple[Literal["x", "u", "y", ""], int]:
    idx = find_index(name, result.x_names)
    if idx >= 0:
        return ("x", idx)

    idx = find_index(name, result.u_names)
    if idx >= 0:
        return ("u", idx)

    idx = find_index(name, result.y_names)
    if idx >= 0:
        return ("y", idx)

    return ("", -1)
