from dataclasses import dataclass


@dataclass
class SignalInfo:
    tex: str
    unit: str
    disp_unit: str
    disp_fct: str
