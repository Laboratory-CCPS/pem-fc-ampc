from collections import OrderedDict
from typing import Iterable

import casadi


def create_casadi_vars(vars: Iterable[str]) -> OrderedDict[str, casadi.MX]:
    cvars = OrderedDict()

    for v in vars:
        cvars[v] = casadi.MX.sym(v)

    return cvars


def casadi_vars_to_str(vars: casadi.MX) -> tuple[str, ...]:
    return tuple(vars[i].str() for i in range(vars.numel()))
