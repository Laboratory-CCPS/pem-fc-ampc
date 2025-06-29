import numpy as np
import casadi as cs

from typing import Literal
from copy import deepcopy

from neural_horizon_acados.mpc.mpc_classes_acados import Base_AMPC_class
from neural_horizon_acados.mpc.mpc_dataclass import AMPC_data
from neural_horizon_acados.utils.decorators import enforce_call_order

from .suh_mpc_param import SuH_MPC_Param
from .suh_mpc_data import SuH_MPC_data
from model.model_src.model_suh import model_suh
from model.model_src.model_acados import get_acados_model
from model.model_src.ode_modelling.ode_model import get_model, scale_model, OdeModel




class SuH_AMPC(Base_AMPC_class):
    def __init__(
            self,
            unscaled_model: OdeModel,
            mpc_param: SuH_MPC_Param, 
            solver_verbose = False,
            ignore_status_errors: set[Literal[1, 2, 3, 4, 5, 6]] | None = None
        ) -> None:
        self.unscaled_model = unscaled_model
        self.scaled_model = scale_model(self.unscaled_model)
        self.lambda_O2 = self.scaled_model.y if mpc_param.scale_model else self.unscaled_model.y
        
        acados_model = get_acados_model(self.scaled_model if mpc_param.scale_model else self.unscaled_model)
        super().__init__(
            acados_model,
            mpc_param, 
            solver_verbose = solver_verbose, 
            ignore_status_errors = ignore_status_errors,
        )


    @enforce_call_order('set_constraints')
    def set_nonlinear_constraints(self) -> None:  
        self.ocp.model.con_h_expr = self.lambda_O2
        self.ocp.constraints.lh = self._scale_h(self.P.lbh.reshape((-1,)))
        self.ocp.constraints.uh = self._scale_h(self.P.ubh.reshape((-1,)))

    
    @enforce_call_order('set_cost')
    def set_terminal_cost(self) -> None:
        pass


    @enforce_call_order('set_cost')
    def set_stage_cost(self) -> None:
        u = self.ocp.model.u  
        self.ocp.cost.cost_type = "NONLINEAR_LS"
        self.ocp.model.cost_y_expr = cs.vertcat(u[1], self.lambda_O2)
        self.ocp.cost.W = self._scale_W(self.P.W)
        self.ocp.cost.yref = self._scale_y(self.P.yref[:, 0] if len(self.P.yref.shape) > 1 else self.P.yref)

    
    def postprocess_data(self, mpc_results: AMPC_data) -> SuH_MPC_data:
        return self._add_lambda_O2(mpc_results)
    

    def cost_calc(self, mpc_results: SuH_MPC_data) -> float:
        y_opt = np.stack((mpc_results.U[1, :], mpc_results.lambda_O2) , axis=0) 
        y = y_opt - self.P.yref
        return np.sum(y * np.dot(self.P.W, y))

    
    def _add_lambda_O2(self, mpc_results: AMPC_data) -> SuH_MPC_data:
        suh_results = SuH_MPC_data.from_MPC_data(mpc_results)
        lambda_O2_func = cs.Function('lambda_O2_func', [self.ocp.model.x, self.ocp.model.u], [self.unscaled_model.y])

        for k in range(mpc_results.mpc_param.N_sim):
            x_k = mpc_results.X[:, k]
            u_k = mpc_results.U[:, k]
            suh_results.lambda_O2[k] = lambda_O2_func(x_k, u_k).full().flatten()

            for i in range(mpc_results.mpc_param.N_MPC):
                x_ki = mpc_results.X_traj[k, :, i]
                u_ki = mpc_results.U_traj[k, :, i]
                suh_results.lambda_O2_traj[k, i] = lambda_O2_func(x_ki, u_ki).full().flatten()
        return suh_results
