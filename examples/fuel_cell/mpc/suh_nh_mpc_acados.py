import numpy as np
import casadi as cs

from scipy.linalg import block_diag

from neural_horizon_acados.mpc import *
from neural_horizon_acados.utils import *
from neural_horizon_acados.neural_horizon import *

from model.model_src.model_suh import model_suh
from model.model_src.model_acados import get_acados_model
from model.model_src.ode_modelling.ode_model import get_model, scale_model, OdeModel
from .suh_mpc_data import SuH_MPC_data
from .suh_mpc_acados import SuH_AMPC




class SuH_NH_AMPC(Base_NH_AMPC_class, SuH_AMPC):
    def __init__(
            self, 
            unscaled_model: OdeModel,
            NNmodel: NN_for_casadi, 
            solver_verbose: bool = False,
        ) -> None:
        self.unscaled_model = unscaled_model
        self.scaled_model = scale_model(self.unscaled_model)
        self.lambda_O2 = self.scaled_model.y if NNmodel.P.scale_model else self.unscaled_model.y
        
        acados_model = get_acados_model(self.scaled_model if NNmodel.P.scale_model else self.unscaled_model)
        acados_model.p = cs.vertcat(cs.MX.sym('I_st_ref', 1), cs.MX.sym('lambda_O2_ref', 1))
        Base_NH_AMPC_class.__init__(
            self,
            acados_model,
            NNmodel, 
            solver_verbose=solver_verbose, 
        )
        
        
    @enforce_call_order('set_cost')
    def set_nh_terminal_cost(self):
        """
        Sets an nonlinear least sqares terminal cost for the NH-MPC in acados. 
        The ``cost_y_expr_e`` is a vertical stacked x array:
        """
        yref = self._scale_y((self.P.yref_e if len(self.P.yref_e.shape) == 1 else self.P.yref_e[0]).squeeze())
        self.ocp.parameter_values = yref
        # States
        x = self.ocp.model.x
        p = self.ocp.model.p
        x_NN = self.NNmodel.NN_casadi(cs.vertcat(x, p))

        # set terminal cost
        self.ocp.cost.cost_type_e = "NONLINEAR_LS"
        self.ocp.model.cost_y_expr_e = x_NN
        self.ocp.cost.W_e =  block_diag(*[self._scale_W(self.P.W)] * self.P.N_NN)
        self.ocp.cost.yref_e =  np.repeat(yref, x_NN.shape[0]//self.ny)


    def postprocess_data(self, mpc_results: AMPC_data) -> SuH_MPC_data:
        mpc_results = self._add_lambda_O2(mpc_results)
        return self._add_NN_y(mpc_results)
    

    def _add_NN_y(self, mpc_results: SuH_MPC_data) -> SuH_MPC_data:
        last_u_traj = mpc_results.U_traj[:, :, -1:]
        mpc_results.U_traj = np.concatenate(
            (mpc_results.U_traj, np.repeat(last_u_traj, self.P.N_NN, axis=2)), axis=2
        )
        last_lambda_O2 = mpc_results.lambda_O2_traj[:, -1:]
        mpc_results.lambda_O2_traj = np.concatenate(
            (mpc_results.lambda_O2_traj, np.repeat(last_lambda_O2, self.P.N_NN, axis=1)), axis=1
        )
        x_scaled = self._scale_x(mpc_results.X_traj[:, :, self.P.N_MPC])
        y_ref_scaled = self._scale_y(self.P.yref.T)
        input_vals = np.concatenate((x_scaled, y_ref_scaled), axis=1)
        for k in range(self.P.N_sim):
            input_val = input_vals[k, :]
            y_NN = np.reshape(self.NNmodel.NN_casadi(input_val), (2, -1), order='F')
            unscaled_y_NN = self._unscale_y(y_NN)
            mpc_results.lambda_O2_traj[k, self.P.N_MPC:] = unscaled_y_NN[1, :].squeeze()
            mpc_results.U_traj[k, 1, self.P.N_MPC:] = unscaled_y_NN[0, :].squeeze()

        return mpc_results