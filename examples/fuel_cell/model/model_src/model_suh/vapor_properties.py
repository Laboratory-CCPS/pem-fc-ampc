from .conversions import celsius2kelvin


def saturation_pressure(T):
    if T == celsius2kelvin(25):
        p_sat = 0.0317395e5
    elif T == celsius2kelvin(80):
        p_sat = 0.47373e5
    elif celsius2kelvin(70):
        p_sat = 0.31176e5
    else:
        assert False

    return p_sat
