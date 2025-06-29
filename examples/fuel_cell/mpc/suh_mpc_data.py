import numpy as np

from dataclasses import dataclass, field

from neural_horizon_acados.mpc.mpc_dataclass import AMPC_data


@dataclass
class SuH_MPC_data(AMPC_data):
    lambda_O2 : np.ndarray = field(init=False)
    lambda_O2_traj : np.ndarray = field(init=False)
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self.lambda_O2 = np.full(self.mpc_param.N_sim, np.nan)
        self.lambda_O2_traj = np.full((self.mpc_param.N_sim, self.mpc_param.N_MPC), np.nan)