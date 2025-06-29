# %%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from typing import Literal
from dataclasses import dataclass, field
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

from model.model_src.model_suh import model_suh
from model.model_src.ode_modelling.ode_model import get_model, scale_model, OdeModel
from model.model_src.ode_modelling.model_helper import scale_data, unscale_data
from model.model_src.ode_modelling.scaling import Scaling
from model.model_src.model_suh.conversions import rpm2rad


@dataclass
class MPC_params:
    T_sim: float = 30.
    ts: float = 0.025
    N: int = 10

    nx: int = 4
    nu: int = 2
    nh: int = 1

    xlabel: list[str] = field(default_factory=lambda: ['p_O2', 'p_N2','w_cp','p_sm'])
    ulabel: list[str] = field(default_factory=lambda: ['v_cm', 'I_st'])
    hlabel: list[str] = field(default_factory=lambda: ['lambda_O2'])
    
    lbu: np.ndarray = field(default_factory=lambda: np.array([50., 1.]))
    ubu: np.ndarray = field(default_factory=lambda: np.array([250., 500.]))

    lbx: np.ndarray = field(default_factory=lambda: np.array([0.05e5, 0.4e5, rpm2rad(20e3), 1e5]))
    ubx: np.ndarray = field(default_factory=lambda: np.array([3e5, 3e5, rpm2rad(105e3), 3e5]))

    x0: np.ndarray = field(default_factory=lambda: np.array([0.1113e5, 0.8350e5, 6049.9, 1.5356e5]))

    lh: np.ndarray = field(default_factory=lambda: np.array([1.5]))
    uh: np.ndarray = field(default_factory=lambda: np.array([5.]))

    yref: np.ndarray = field(default_factory=lambda: np.array([130., 2.]))
    yref_e: np.ndarray = field(default_factory=lambda: np.array([2.]))

    W: np.ndarray = field(default_factory=lambda: np.diag([5e-2, 1.]))
    W_e: np.ndarray = field(default_factory=lambda: np.array([[1.]]))

    def scale(
            self, 
            scalings: dict[Literal['p_O2', 'p_N2','w_cp','p_sm', 'v_cm', 'I_st', 'lambda_O2'], Scaling]
    ) -> None:
        self.lbu = scale_data(scalings, self.ulabel, self.lbu)
        self.ubu = scale_data(scalings, self.ulabel, self.ubu)

        self.lbx = scale_data(scalings, self.xlabel, self.lbx)
        self.ubx = scale_data(scalings, self.xlabel, self.ubx)
        self.x0 = scale_data(scalings, self.xlabel, self.x0)

        self.lh = scale_data(scalings, self.hlabel, self.lh)
        self.uh = scale_data(scalings, self.hlabel, self.uh)

        self.yref = scale_data(scalings, [self.ulabel[1], self.hlabel[0]], self.yref)
        self.yref_e = scale_data(scalings, self.hlabel, self.yref_e)

        self.W = self.scale_weights(scalings, [self.ulabel[1], self.hlabel[0]], self.W)
        self.W_e = self.scale_weights(scalings, self.hlabel, self.W_e)


    @staticmethod
    def scale_weights(
            scalings: dict[Literal['p_O2', 'p_N2','w_cp','p_sm', 'v_cm', 'I_st', 'lambda_O2'], Scaling], 
            names: list[str], 
            weights: np.ndarray
    ) -> np.ndarray:
        scale_values = [scalings[name].factor for name in names]
        scaling_matrix = np.diag(scale_values)
        return scaling_matrix @ weights @ scaling_matrix




# %%
def create_ocp_solver(ode_model: OdeModel, mpc_params: MPC_params) -> AcadosOcpSolver:
    ocp = AcadosOcp()

    # Model
    acados_model = AcadosModel()
    acados_model.name = "suh_mpc_model"
    acados_model.x = ode_model.states
    acados_model.u = ode_model.inputs
    acados_model.f_expl_expr = ode_model.dx
    acados_model.p = []
    ocp.model = acados_model

    # Horizon
    ocp.dims.N = 10

    # Constraints
    ocp.constraints.lbu = mpc_params.lbu
    ocp.constraints.ubu = mpc_params.ubu
    ocp.constraints.idxbu = np.arange(mpc_params.nu)

    ocp.constraints.lbx = mpc_params.lbx
    ocp.constraints.ubx = mpc_params.ubx
    ocp.constraints.idxbx = np.arange(mpc_params.nx)

    ocp.constraints.lbx_e = mpc_params.lbx
    ocp.constraints.ubx_e = mpc_params.ubx
    ocp.constraints.idxbx_e = np.arange(mpc_params.nx)

    ocp.constraints.x0 = mpc_params.x0

    # Nonlinear constraints
    ocp.model.con_h_expr = ode_model.y
    ocp.constraints.lh = mpc_params.lh
    ocp.constraints.uh = mpc_params.uh

    # Cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = cs.vertcat(ode_model.inputs[1], ode_model.y)
    ocp.cost.W = mpc_params.W
    ocp.cost.yref =  mpc_params.yref
    # ocp.cost.cost_type_e = "NONLINEAR_LS"
    # ocp.model.cost_y_expr_e = cs.substitute(ode_model.y, ode_model.inputs, cs.vertcat(0, mpc_params.yref[0]))
    # ocp.cost.W_e =  mpc_params.W_e
    # ocp.cost.yref_e = mpc_params.yref_e

    # Define time horizon and discretization
    ocp.solver_options.tf = mpc_params.ts * mpc_params.N # Ts * N
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.regularize_method = 'PROJECT_REDUC_HESS'
    ocp.solver_options.sim_method_num_stages = 2
    # ocp.solver_options.print_level = 4
    ocp.solver_options.sim_method_num_steps = 4
    # ocp.solver_options.qp_solver_warm_start = 1

    # AS RTI settings
    # ocp.solver_options.as_rti_level = 3
    # ocp.solver_options.as_rti_iter = 3

    # Create the OCP solver
    ocp_solver = AcadosOcpSolver(ocp, json_file="suh_mpc.json")
    return ocp_solver


# %%
def simulate_mpc(
        ocp_solver: AcadosOcpSolver, 
        mpc_params: MPC_params, 
        x0: np.ndarray, 
        u_ref: np.ndarray
):
    N_sim = int(mpc_params.T_sim / mpc_params.ts)
    states = np.zeros((N_sim + 1, mpc_params.nx))
    inputs = np.zeros((N_sim, mpc_params.nu))

    x_current = x0
    states[0, :] = x_current
    yref = mpc_params.yref
    I_ref = u_ref[1]

    for i in range(N_sim):
        ocp_solver.set(0, "lbx", x_current)
        ocp_solver.set(0, "ubx", x_current)

        if not np.array_equal(I_ref[i], yref[1]):
            yref[1] = I_ref[i]
            for step in range(mpc_params.N):
                ocp_solver.set(step, "yref", yref)

        status = ocp_solver.solve()
 
        if status != 0:
            print(f"ACADOS solver failed at step {i} with status {status}.")
            get_stats(ocp_solver)
            break

        x_next = ocp_solver.get(1, "x")

        inputs[i, :] = ocp_solver.get(0, "u")
        states[i + 1, :] = x_next
        x_current = x_next
    
    return states.T, inputs.T


def get_stats(ocp_solver: AcadosOcpSolver):
    residuals = ocp_solver.get_residuals()
    formatted_residuals = ", ".join([f"{r:.2f}" for r in residuals])
    print("="*50)
    print(f"{'Metric':<20} | {'Value'}")
    print("-"*50)
    print(f"{'Residuals':<20} | [{formatted_residuals}]")
    print(f"{'Total time':<20} | {(ocp_solver.get_stats('time_tot')*1e3):.3f} ms")
    print(f"{'QP time':<20} | {(ocp_solver.get_stats('time_qp')*1e3):.3f} ms")
    print(f"{'Linearisation time':<20} | {(ocp_solver.get_stats('time_lin')*1e3):.3f} ms")
    print(f"{'Simulation time':<20} | {(ocp_solver.get_stats('time_sim')*1e3):.3f} ms")
    print(f"{'OCP cost':<20} | {ocp_solver.get_cost():.3f}")
    print("="*50)



def plot_results(states: np.ndarray, inputs: np.ndarray, lambda_O2: np.ndarray, inputs_ref: np.ndarray):
    fig, axs = plt.subplots(7, 1, sharex=True, squeeze=True, figsize=(8, 8))

    T_sim = 30
    sim_time = np.linspace(0, T_sim, states.shape[1])

    state_labels = ['$p_{O2} \, [Pa]$', '$p_{N2} \, [Pa]$', '$w_{cp} \, [rad/s]$', '$p_{sm} \, [Pa]$']
    input_labels = ['$v_{cm} \, [V]$', '$I_{st} \, [A]$']
    lambda_O2_label = '$\lambda_{O2}$'

    # plot states
    for i_x, state in enumerate(states):
        ax = axs[i_x]
        ax.plot(sim_time, state)
        ax.set_ylabel(state_labels[i_x])
        ax.grid()

    # plot inputs
    for i_u, (inp, inp_ref) in enumerate(zip(inputs, inputs_ref)):
        ax = axs[4 + i_u]
        ax.plot(sim_time[:-1], inp)
        ax.plot(sim_time[:-1], inp_ref, color='orange')
        ax.set_ylabel(input_labels[i_u])
        ax.grid()

    # plot lambda_O2
    ax = axs[6]
    ax.plot(sim_time[:-1], lambda_O2)
    ax.set_ylabel(lambda_O2_label)
    ax.grid()

    ax.set_xlabel('$t \, [s]$')

    plt.show()


def get_validation_input(Ts: float) -> tuple[np.ndarray, np.ndarray]:
    t0 = 0
    tend = 30
    t = np.arange(t0, tend, Ts).reshape((1, -1))

    def I_load(t: np.ndarray) -> np.ndarray:
        return (
            100.0
            + 80 * (t >= 2)
            + 40 * (t >= 6)
            - 20 * (t >= 10)
            + 60 * (t >= 14)
            + 60 * (t >= 22)
        )
    
    def v_cm(t: np.ndarray) -> np.ndarray:
        return t*0
    
    x0 = np.array([0.1096e5, 0.7502e5, 5.4982e3, 1.4326e5])
    u = np.vstack((v_cm(t), I_load(t)))
    return (u, x0)



def get_lambda_O2(ode_model: OdeModel, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    lambda_O2_func = cs.Function('lambda_O2_func', [ode_model.states, ode_model.inputs], [ode_model.y])
    lambda_O2 = np.full(inputs.shape[1], np.nan)
    for k in range(inputs.shape[1]):
        x_k = states[:, k + 1]
        u_k = inputs[:, k]
        lambda_O2[k] = lambda_O2_func(x_k, u_k).full().flatten()

    return lambda_O2


# %%
def main(scale: bool):
    # get model
    ode_model = get_model(model_suh.model, model_suh.params())
    mpc_params = MPC_params()
    u_ref, x0 = get_validation_input(mpc_params.ts)
    if scale:
        ode_model = scale_model(ode_model)
        mpc_params.scale(ode_model.scalings)
        u_ref = scale_data(ode_model.scalings, mpc_params.ulabel, u_ref)
        x0 = scale_data(ode_model.scalings, mpc_params.xlabel, x0)

    # create ocp solver
    ocp_solver = create_ocp_solver(ode_model, mpc_params)
    
    states, inputs = simulate_mpc(ocp_solver, mpc_params, x0, u_ref)
    lambda_O2 = get_lambda_O2(ode_model, states, inputs)

    if scale:
        states = unscale_data(ode_model.scalings, mpc_params.xlabel, states)
        inputs = unscale_data(ode_model.scalings, mpc_params.ulabel, inputs)
        lambda_O2 = np.squeeze(unscale_data(ode_model.scalings, mpc_params.hlabel, np.expand_dims(lambda_O2, axis=0)))
        u_ref = unscale_data(ode_model.scalings, mpc_params.ulabel, u_ref)

    plot_results(states, inputs, lambda_O2, u_ref)

    # Output results
    print("Simulation completed.")
    print(f"States:\n{states}")
    print(f"Inputs:\n{inputs}")
    print(f"Lambda_O2:\n{lambda_O2}")



if __name__ == '__main__':
    main(True)
# %%
