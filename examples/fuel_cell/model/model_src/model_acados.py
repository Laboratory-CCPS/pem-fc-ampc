import casadi as cs

from acados_template import AcadosModel

from .ode_modelling.ode_model import OdeModel


def get_acados_model(ode_model: OdeModel) -> AcadosModel:
    """
    Creates an AcadosModel

    Parameters
    ----------
    ``P`` : ParamsSuh
        A dataclass object specifying the parameters for the Acados Model.

    Returns
    -------
    ``model`` : AcadosModel
        An AcadosModel 
    """
    acados_model = AcadosModel()

    acados_model.f_expl_expr = ode_model.dx
    acados_model.x = ode_model.states
    acados_model.u = ode_model.inputs
    acados_model.name = 'SuH'

    return acados_model