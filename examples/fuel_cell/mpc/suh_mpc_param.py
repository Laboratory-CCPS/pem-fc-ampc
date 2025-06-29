import numpy as np

from dataclasses import dataclass, field

from neural_horizon_acados.parameters import * 

from model.model_src.model_suh.model_suh import ParamsSuh, params
from model.model_src.model_suh.conversions import rpm2rad
from mpc.utils import get_scaling




@dataclass
class SuH_MPC_Param(AMPC_Param):
    model_param: ParamsSuh = field(default_factory=lambda: params())
    scale_model: bool = True

    # cost weights
    ny: int = 2
    ny_e: int = 1
    W: np.ndarray = field(default_factory=lambda: np.diag([5e-2, 1.]))                   # cost weight on on (I_st, lambda_O2)
    W_e: np.ndarray = field(default_factory=lambda: np.array([[1.]]))                    # cost weight on (lambda_O2)

    # linear constratins
    lbu:   np.ndarray = field(default_factory=lambda: np.array([50., 1.]))                                  # lower bound on (v_cm, I_st)
    ubu:   np.ndarray = field(default_factory=lambda: np.array([250., 500.]))                               # upper bound on (v_cm, I_st)

    lbx_0: np.ndarray = field(default_factory=lambda: np.array([]))
    ubx_0: np.ndarray = field(default_factory=lambda: np.array([]))
    lbx:   np.ndarray = field(default_factory=lambda: np.array([0.05e5, 0.4e5, rpm2rad(20e3), 1e5]))      # lower bound on (p_O2, p_N2, w_cp, p_sm)
    ubx:   np.ndarray = field(default_factory=lambda: np.array([3e5, 3e5, rpm2rad(105e3), 3e5]))          # upper bound on (p_O2, p_N2, w_cp, p_sm)

    # nonlinear constraints
    lbh:   np.ndarray = field(default_factory=lambda: np.array([1.5]))  
    ubh:   np.ndarray = field(default_factory=lambda: np.array([5.]))

    # inits
    # u0:    np.ndarray = field(default_factory=lambda: np.array([110, 100]))
    x0:    np.ndarray = field(default_factory=lambda: np.array([0.1096e5, 0.7502e5, 5.4982e3, 1.4326e5]))     # starting point for the closed-loop trajectory

    # reference
    yref: np.ndarray = field(default_factory=lambda: np.array([130., 2.]))            # reference cost values on (I_st, lambda_O2)
    yref_e: np.ndarray = field(default_factory=lambda: np.array([2.]))               # reference cost values on (lambda_O2)

    ylabel: list[str] = field(default_factory=lambda: ['I_st', 'lambda_O2'])
    yelabel: list[str] = field(default_factory=lambda: ['lambda_O2'])

    x_scale_factor: np.ndarray = field(default_factory=lambda: np.array([]))
    u_scale_factor: np.ndarray = field(default_factory=lambda: np.array([]))
    h_scale_factor: np.ndarray = field(default_factory=lambda: np.array([]))
    y_scale_factor: np.ndarray = field(default_factory=lambda: np.array([]))
    ye_scale_factor: np.ndarray = field(default_factory=lambda: np.array([]))

    x_scale_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    u_scale_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    h_scale_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    y_scale_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    ye_scale_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # MPC horizon and sampling time
    N_MPC: int        = 10                                                                   # length of MPC horizon
    Ts:    float      = 0.025                                                                # sampling rate
    T_sim: float      = 30                                                                   # simulation horizon (seconds)
    N_sim: int        = field(init=False)

    version: int = 0
    name: str = ''
    file: str = ''


    def __post_init__(self):
        super().__post_init__()
        if self.scale_model:
            scalings = get_scaling()
            self.x_scale_factor = scalings['x'].factor
            self.x_scale_offset = scalings['x'].offset
            self.u_scale_factor = scalings['u'].factor
            self.u_scale_offset = scalings['u'].offset
            self.h_scale_factor = scalings['h'].factor
            self.h_scale_offset = scalings['h'].offset
            self.y_scale_factor = scalings['y'].factor
            self.y_scale_offset = scalings['y'].offset
            self.ye_scale_factor = scalings['y_e'].factor
            self.ye_scale_offset = scalings['y_e'].offset



@dataclass
class SuH_NH_MPC_Param(SuH_MPC_Param, NH_AMPC_Param):
    nn_param : NN_Param | None = None
    N_MPC: int = 6
    N : int = 12
    W_NN : np.ndarray = field(default_factory=lambda: np.diag([5e-2, 1.]))
