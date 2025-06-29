import os
import time
import warnings
import torch
import casadi as cs
import numpy as np
import scipy.linalg
import gc

from typing import Optional, Literal, Self, Callable
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
from tqdm.auto import trange
from copy import deepcopy

from ..parameters.model_param import Basic_Model_Param
from ..parameters.mpc_param import AMPC_Param
from ..parameters.nh_mpc_param import NH_AMPC_Param
from ..neural_horizon.neural_horizon import NN_for_casadi
from .mpc_dataclass import AMPC_data, SolveStepResults
from ..utils.decorators import enforce_call_order, log_memory_usage
from ..utils.errors import SingletonError



# ===================================================================================================
# BASIC ACADOS MPC
# ===================================================================================================
class Base_AMPC_class:
    """
    A class as a basic acados MPC implementation based on the give parameters.
    Only one instance can be uses at a time and the instance has to be deleted 
    after use with 'cleanup()' and also deleted.
    """
    _instance_exists = False

    def __new__(cls, *args, **kwargs) -> Self:
        """
        Ensures multiple instances are NOT called together, due to unexpected behavior of acados.
        """
        if not Base_AMPC_class._instance_exists:
            Base_AMPC_class._instance_exists = True
            return super(Base_AMPC_class, cls).__new__(cls)
        else:
            raise SingletonError(cls)
        
    def __init__(
            self, 
            model: AcadosModel,
            P: AMPC_Param | NH_AMPC_Param,  
            solver_verbose = False,
            ignore_status_errors: Optional[set[Literal[1, 2, 3, 4, 5, 6]]] = None,
    ) -> None:
        """
        Implements a acados MPC using a horizon of length ``P.N_MPC``, while using the default or 
        given acados options, for the solver.

        !!ATTENTION!! function ``cleanup()`` must be called after usage.

        Parameters
        ----------
        ``model`` : AcadosModel
            An acados model, either only continuous or with discrete implementation.
        ``P`` : MPC_Param
            The MPC_Param dataclass containing the MPC problem parameters.
        ``solver_options`` : dict, optional 
            A dictionary with acados options for ocp and ocp solver as well as cost type.
             See page -> https://docs.acados.org/python_interface/index.html 
              Default = None
        ``acados_name`` : str, optional
            A string defining the acados name used for ocp_solver.json and tqdm. 
             Default = ''
        ``solver_verbose`` : bool, optional
            A bool defining if during solving and broke solver the statistics are printet.
             Default = False,
        ``ignore_status_errors`` : set of [1, 2, 3, 4, 5], optional
            A set defining if during solving it does not raises an error 
             when {1, 2, 3, 4, 5} are the status returns of acados.
              Default = True
        ``horizon_name`` : str, optional
            A string used for ocp_solver.json and tqdm.
             Default = ''
        """
        self.P = deepcopy(P)
        self.acados_options = {} if P.acados_settings is None or not P.acados_settings else deepcopy(P.acados_settings) # deepcopy because of pops later.

        # Params and options
        self.acados_name = P.acados_name
        self.solver_verbose = solver_verbose
        self.ignore_status_errors = set() if ignore_status_errors is None else ignore_status_errors

        self.mpc_name = P.name

        self.max_rti_iters = self.acados_options.pop('max_rti_iters', 10) 
        self.rti_tol = self.acados_options.pop('rti_tol', 1e-4) 
        self.use_iter_rti_impl = self.acados_options.pop('use_iter_rti_impl', False)
        self.use_x_guess = self.acados_options.pop('use_x_guess', False)
        self.use_u_guess = self.acados_options.pop('use_u_guess', False)

        self.solver_status_meanings = {
            0 : 'ACADOS_SUCCESS',
            1 : 'ACADOS_NAN_DETECTED',
            2 : 'ACADOS_MAXITER',
            3 : 'ACADOS_MINSTEP',
            4 : 'ACADOS_QP_FAILURE',
            5 : 'ACADOS_READY',
            6 : 'ACADOS_UNBOUNDED'
        }
        self.current_status = 0

        if not self.ignore_status_errors.issubset(self.solver_status_meanings.keys()):
            raise ValueError(f'Parameter must be a set in {set(self.solver_status_meanings.keys())} -> got {self.ignore_status_errors}')
        
        # Model
        self.model = model

        # Dims
        self.N_MPC = self.P.N_MPC
        self.nx = self.P.model_param.nx
        self.nu = self.P.model_param.nu
        self.ny = self.P.ny
        self.ny_e = self.P.ny_e

        self._mpc_init()

        # Private variables
        self._yref_curr = None

        
    def _mpc_init(self) -> None:
        self.create_base_ocp()
        self.set_stage_cost()
        self.set_terminal_cost()
        self.set_constraints()
        self.set_nonlinear_constraints()
        self.set_solver_options()
        self._set_ocp_string()
        self.create_ocp_solver()


    @enforce_call_order('set_ocp')
    def create_base_ocp(self) -> None:
        """
        Sets the basic acados OCP filled with the model and the horizon.
        """
        self.ocp = AcadosOcp()
        self.ocp.model = self.model
        self.ocp.solver_options.N_horizon = self.N_MPC


    @enforce_call_order('set_solver_options')
    def set_solver_options(self) -> None:
        """
        Sets all given acados options by using the key as attribute of ocp.solver_options 
        and the value as value.
        """
        # Prediction horizon
        self.ocp.solver_options.tf = self.P.Ts * self.N_MPC

        # Solver options
        for key, value in self.acados_options.items():
            try:
                setattr(self.ocp.solver_options, key, value)
            except Exception as error:
                print('options {} cannot be set!\nException occurred: {} - {}'. format(key, type(error).__name__, error))

    
    @enforce_call_order('set_constraints')
    def set_constraints(self) -> None:
        """
        Sets the OCP constraints given by an instance MPC_param.
        """
        # U 
        lbu, ubu = self.P.lbu.reshape((-1,)), self.P.ubu.reshape((-1,))
        lbu_finite, ubu_finite, idxbu_finite = self._filter_finite_bounds(self._scale_u(lbu), self._scale_u(ubu))
        self.ocp.constraints.lbu = lbu_finite
        self.ocp.constraints.ubu = ubu_finite
        self.ocp.constraints.idxbu = idxbu_finite

        # X (States)
        lbx, ubx = self.P.lbx.reshape((-1,)), self.P.ubx.reshape((-1,))
        lbx_finite, ubx_finite, idxbx_finite = self._filter_finite_bounds(self._scale_x(lbx), self._scale_x(ubx))
        self.ocp.constraints.lbx = lbx_finite
        self.ocp.constraints.ubx = ubx_finite
        self.ocp.constraints.idxbx = idxbx_finite

        # X_e (Terminal States)
        lbx_e, ubx_e = self.P.lbx.reshape((-1,)), self.P.ubx.reshape((-1,))
        lbx_e_finite, ubx_e_finite, idxbx_e_finite = self._filter_finite_bounds(self._scale_x(lbx_e), self._scale_x(ubx_e))
        self.ocp.constraints.lbx_e = lbx_e_finite
        self.ocp.constraints.ubx_e = ubx_e_finite
        self.ocp.constraints.idxbx_e = idxbx_e_finite

        # X0
        if self.P.x0.size == 0:
            lbx_0, ubx_0 = self.P.lbx_0.reshape((-1,)), self.P.ubx_0.reshape((-1,))
            lbx_0_finite, ubx_0_finite, idxbx_0_finite = self._filter_finite_bounds(self._scale_x(lbx_0), self._scale_x(ubx_0))
            self.ocp.constraints.lbx_0 = lbx_0_finite
            self.ocp.constraints.ubx_0 = ubx_0_finite
            self.ocp.constraints.idxbx_0 = idxbx_0_finite
        else:
            self.ocp.constraints.x0 = self._scale_x(self.P.x0.reshape((-1,)))

        
    @enforce_call_order('set_constraints')
    def set_nonlinear_constraints(self) -> None:
        pass


    @enforce_call_order('set_cost')
    def set_stage_cost(self) -> None:
        """
        Sets the stage and terminal cost to the give type of cost.
        """
        self._set_LLS_stage_cost()


    @enforce_call_order('set_cost')
    def set_terminal_cost(self) -> None:
        """
        Sets the stage and terminal cost to the give type of cost.
        """
        self._set_LLS_terminal_cost()


    @enforce_call_order('set_cost')
    def _set_LLS_stage_cost(self) -> None:
        """
        Sets the stage cost to a basic linear least sqares cost that penalizes the states and inputs
        """
        ny = self.nu + self.nx
        if self.P.W.shape!= (ny, ny):
            raise ValueError('Weight must be of shape (nx+nu, nx+nu). Otherwise make a custom stage cost with the function "set_stage_cost"')

        # State X
        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx,:self.nx] = np.eye(self.nx)
        
        # U
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[self.nx:,:] = np.eye(self.nu)

        # Cost
        yref_sq = self.P.yref.squeeze() if self.P.yref.size > 1 else self.P.yref
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.W = self._scale_W(self.P.W)
        self.ocp.cost.yref = self._scale_y(yref_sq if len(yref_sq.shape) == 1 else yref_sq[0])

        
    @enforce_call_order('set_cost')
    def _set_LLS_terminal_cost(self) -> None:
        """
        Sets the terminal cost to a basic linear least sqares cost that penalizes the states. 
        """
        ny = self.nu + self.nx
        if self.P.W.shape!= (ny, ny):
            raise ValueError('Weight must be of shape (nx, nx). Otherwise make a custom terminal cost with the function "set_terminal_cost"')
        
        # Terminal X 
        self.ocp.cost.Vx_e = np.eye(self.ny_e)

        # Terminal Cost
        yref_e_sq = self.P.yref_e.squeeze() if self.P.yref_e.size > 1 else self.P.yref_e
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.W_e = self._scale_W_e(self.P.W_e)
        self.ocp.cost.yref_e = self._scale_y_e(yref_e_sq if len(yref_e_sq.shape) == 1 else yref_e_sq[0])


    def _set_acados_name(self, nh_str=None, join_str='_') -> None:
        """
        Sets the ``MPC_name``, by combining all acados option values that are strings
        and connect them with ``join_str``. Also add a the ``nh_str`` at the start, if not None.

        Keyword Parameters
        ------------------
        ``nh_str`` : str, optional
            A string that is placed at the start of the ``ocp_solver_string``, when not None.
            Default = None
        ``join_str`` : str, optional
            A string that connects all the acados option values. Default = '_'
        """
        name = [v for v in self.acados_options.values() if type(v) is str]
        name = join_str.join(name)

        if nh_str is not None:
            name = join_str.join((nh_str, name))

        self.acados_name = name


    @enforce_call_order('set_ocp_string')
    def _set_ocp_string(self) -> None:
        """
        Sets the ``ocp_solver_string``, by combining 'acados_ocp_' with the MPC_name 
        as well as the extension.
        """
        solver_dir = os.path.abspath('temp_solver_jsons')
        if not os.path.exists(solver_dir):
            os.mkdir(solver_dir)

        self.solver_json_path = os.path.join(
            solver_dir,
            f'ampc_solver_{self.acados_name}_{self.mpc_name}.json'
        )
        

    @enforce_call_order('set_ocp_solver')
    def create_ocp_solver(self) -> None:
        """
        Creates an acados OCP solver out of the OCP. 
        Stores the solver settings in a json named by the ``ocp_solver_string``.
        """
        assert not hasattr(self, 'ocp_solver'), 'creating multiple OCP SOLVERS leads to problems!'
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.solver_json_path, verbose=self.solver_verbose)

    
    @enforce_call_order('deleted')
    def cleanup(self) -> None:
        """
        Deletes first the ``ocp_solver``, then the ``ocp`` and then the ``model`` of this instance.
        Also resets the option to create a new instance of this class and collect garbage.
        """
        try:
            os.remove(self.solver_json_path)
            del self.ocp_solver
            del self.ocp
            del self.model
        except:
            pass
        finally:
            Base_AMPC_class._instance_exists = False
            gc.collect() 


    def reset_solver(self) -> None:
        self.ocp_solver.reset(reset_qp_solver_mem=1)

    
    def recreate_solver(self) -> None:
        del self.ocp_solver
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.solver_json_path, verbose=self.solver_verbose)
        

    def solve(
            self, 
            x0: np.ndarray,
            x_guess: Optional[np.ndarray] = None, 
            u_guess: Optional[np.ndarray] = None,
    ) -> SolveStepResults:
        """
        Solves the OCP with acados. Print statistics, if solver returns status that is not 0. <br>
        status:
        - 0 = ACADOS_SUCCESS
        - 1 = ACADOS_NAN_DETECTED
        - 2 = ACADOS_MAXITER
        - 3 = ACADOS_MINSTEP
        - 4 = ACADOS_QP_FAILURE
        - 5 = ACADOS_READY
        - 6 = ACADOS_UNBOUNDED
        
        Parameters
        ----------
        ``x0`` : numpy.ndarray
            An array that is the initial value for the open loop MPC.
        ``x_guess`` : numpy.ndarray
            An array that is the initial guess for all states the open loop MPC should predict.
        ``u_guess`` : numpy.ndarray
            An array that is the initial guess for all inputs the open loop MPC should predict.

        Returns
        -------
        ``solver_results`` : SolveStepResults
            A dataclass with solver times, solver iterations, x- and y-trajectories
        """
        solver_results = SolveStepResults.from_params(self.P)
        
        x0 = self._scale_x(x0)

        # initial guesses
        if x_guess is not None and self.use_x_guess:
            x_guess = self._scale_x(x_guess)
            for i_x in range(x_guess.shape[1]):
                self.ocp_solver.set(i_x, "x", x_guess[:, i_x])
        if u_guess is not None and self.use_u_guess:
            u_guess = self._scale_u(u_guess)
            for i_u in range(u_guess.shape[1]):
                self.ocp_solver.set(i_u, "u", u_guess[:, i_u])

        # SQP
        if self.ocp.solver_options.nlp_solver_type == 'SQP':
            # initial bounds
            self.ocp_solver.set(0, 'lbx', x0)
            self.ocp_solver.set(0, 'ubx', x0)

            # solve SQP
            start_time = time.perf_counter()
            status = self.ocp_solver.solve()
            solver_results.soltime_p = time.perf_counter() - start_time

            # check status
            self._solver_status_error(status)

            # save results
            solver_results.soltime_a = self.ocp_solver.get_stats('time_tot')
            solver_results.soliters = np.sum(self.ocp_solver.get_stats('qp_iter'), dtype=np.int64)
            
            
        # ACADOS RTI
        elif self.ocp.solver_options.nlp_solver_type == 'SQP_RTI' and not self.use_iter_rti_impl:
            # preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            start_time = time.perf_counter()
            status = self.ocp_solver.solve()
            prep_time = time.perf_counter() - start_time

            # save preperation results
            solver_results.prep_time = self.ocp_solver.get_stats('time_tot')
            solver_results.prep_iters = np.sum(self.ocp_solver.get_stats('qp_iter'), dtype=np.int64)

            # initial bounds
            self.ocp_solver.set(0, 'lbx', x0)
            self.ocp_solver.set(0, 'ubx', x0)

            # feedback phase
            self.ocp_solver.options_set('rti_phase', 2)
            start_time = time.perf_counter()
            status = self.ocp_solver.solve()
            solver_results.soltime_p = time.perf_counter() - start_time + prep_time

            # check status
            self._solver_status_error(status)

            # save feedback results
            solver_results.fb_time = self.ocp_solver.get_stats('time_tot')
            solver_results.fb_iters = np.sum(self.ocp_solver.get_stats('qp_iter'), dtype=np.int64)
            solver_results.add_prep_and_fb()


        # TO CONVERGENCE RTI
        elif self.ocp.solver_options.nlp_solver_type == 'SQP_RTI' \
            and self.use_iter_rti_impl and 4 == self.ocp.solver_options.as_rti_level:

            # initial bounds
            self.ocp_solver.set(0, 'lbx', x0)
            self.ocp_solver.set(0, 'ubx', x0)

            for i in range(self.max_rti_iters):
                # solve problem
                start_time = time.perf_counter()
                status = self.ocp_solver.solve()
                solver_results.soltime_p += time.perf_counter() - start_time

                # check status
                self._solver_status_error(status)

                # save results
                solver_results.soliters += np.sum(self.ocp_solver.get_stats('qp_iter'), dtype=np.int64)
                solver_results.soltime_a += self.ocp_solver.get_stats('time_tot')

                # check convergence
                start_time = time.perf_counter()
                residuals = self.ocp_solver.get_residuals()
                if max(residuals) < self.rti_tol:
                    break
                solver_results.soltime_p += time.perf_counter() - start_time

        
        # get open loop traj
        for i in range(self.N_MPC):
            solver_results.simX_traj[:, i] = self.ocp_solver.get(i, "x")
            solver_results.simU_traj[:, i] = self.ocp_solver.get(i, "u")
        solver_results.simX_traj[:, self.N_MPC] = self.ocp_solver.get(self.N_MPC, "x")

        solver_results.simX_traj = self._unscale_x(solver_results.simX_traj)
        solver_results.simU_traj = self._unscale_u(solver_results.simU_traj)

        return solver_results
    

    def get_trajectory(
            self, 
            x_init: np.ndarray | None = None,
            show_tqdm: bool=True,
            show_exception = True
    ) -> AMPC_data:
        mpc_results = AMPC_data(mpc_param=self.P)
        x_curr = self.P.x0.reshape((self.P.model_param.nx, )) if x_init is None else x_init

        x_guess = np.repeat(x_curr.reshape((self.P.model_param.nx, 1)), repeats=self.P.N_MPC, axis=1)
        u_guess = np.zeros((self.P.model_param.nu, self.P.N_MPC))

        for i in trange(self.P.N_sim, desc = self.mpc_name, mininterval=0.5, disable = not show_tqdm):
            self._presolver(i)

            try:
                solver_results = self.solve(x_curr, x_guess, u_guess)
            except Exception as e:
                if show_exception:
                    raise e
                msg = e.args[0]
                warnings.warn(f'MPC solver failed on step {i}, reason: {msg}',UserWarning)
                break
            time.sleep(0.001)

            x_curr = solver_results.simX_traj[:, 1]
            x_guess = np.hstack((solver_results.simX_traj[:, 1:], solver_results.simX_traj[:, -1:]))
            u_guess = np.vstack((solver_results.simU_traj[1:], solver_results.simU_traj[-1:]))
                
            # save results
            mpc_results = self._push_results(mpc_results, solver_results, i)
            
        mpc_results = self.postprocess_data(mpc_results)
        mpc_results.Cost = self.cost_calc(mpc_results)
        mpc_results.freeze()
        return mpc_results
    
    def _presolver(self, sim_step: int) -> None:
        is_yref_traj = len(self.P.yref.squeeze().shape) == 2
        has_p = self.ocp_solver.get(0, 'p') is not None

        if is_yref_traj:
            self._yref_curr = self._scale_y(self.P.yref[:, sim_step])
            for i_y in range(self.P.N_MPC):
                self.ocp_solver.set(i_y, 'yref', self._yref_curr)
            if has_p:
                self.ocp_solver.set(0, 'p', self._yref_curr)
    
    def postprocess_data(self, mpc_results: AMPC_data) -> AMPC_data:
        return mpc_results

    def cost_calc(self, mpc_results: AMPC_data) -> float:
        ny = self.nu + self.nx
        if self.P.W.shape!= (ny, ny):
            raise ValueError('Weight must be of shape (nx+nu, nx +nu). Otherwise make a custom trajectory cost calculation with the function "cost_calc"')

        y_curr = np.concatenate((mpc_results.X, mpc_results.U), axis=0) 
        y = self.P.yref - y_curr
        return y.T @ self.P.W @ y

    def _solver_status_error(self, status) -> None:
        self.current_status = status
        # solver failed print
        if self.solver_verbose and status != 0:
            self.ocp_solver.print_statistics()
            self._get_stats()
        if status not in self.ignore_status_errors and status != 0: 
            raise Exception('Solver returned status {} -> {}'.format(status, self.solver_status_meanings[status]))
        
    def _get_stats(self) -> None:
        residuals = self.ocp_solver.get_residuals()
        formatted_residuals = ", ".join([f"{r:.2f}" for r in residuals])
        print("="*50)
        print(f"{'Metric':<20} | {'Value'}")
        print("-"*50)
        print(f"{'Residuals':<20} | [{formatted_residuals}]")
        print(f"{'Total time':<20} | {(self.ocp_solver.get_stats('time_tot')*1e3):.3f} ms")
        print(f"{'QP time':<20} | {(self.ocp_solver.get_stats('time_qp')*1e3):.3f} ms")
        print(f"{'Linearisation time':<20} | {(self.ocp_solver.get_stats('time_lin')*1e3):.3f} ms")
        print(f"{'Simulation time':<20} | {(self.ocp_solver.get_stats('time_sim')*1e3):.3f} ms")
        print(f"{'OCP cost':<20} | {self.ocp_solver.get_cost():.3f}")
        print("="*50)
        
    def _scale_value(self, value: np.ndarray, factor: np.ndarray, offset: np.ndarray) -> np.ndarray:
        if not self.P.scale_model:
            return value
        if len(value.shape) == 2 and value.shape[1] != offset.shape[0]:
            return ((value.T - offset) / factor).T
        return (value - offset) / factor
    
    def _unscale_value(self, value: np.ndarray, factor: np.ndarray, offset: np.ndarray) -> np.ndarray:
        if not self.P.scale_model:
            return value
        if len(value.shape) == 2 and value.shape[1] != offset.shape[0]:
            return (value.T * factor + offset).T
        return value * factor + offset
    
    def _scale_weight(self, weight: np.ndarray, factor: np.ndarray) -> np.ndarray:
        if not self.P.scale_model:
            return weight
        scale_values = np.diag(factor)
        return scale_values @ weight @ scale_values
        
    def _scale_W(self, W: np.ndarray) -> np.ndarray:
        return self._scale_weight(W, self.P.y_scale_factor)
    
    def _scale_W_e(self, W_e: np.ndarray) -> np.ndarray:
        return self._scale_weight(W_e, self.P.ye_scale_factor)
        
    def _scale_x(self, x: np.ndarray) -> np.ndarray:
        return self._scale_value(x, self.P.x_scale_factor, self.P.x_scale_offset)
    
    def _scale_u(self, u: np.ndarray) -> np.ndarray:
        return self._scale_value(u, self.P.u_scale_factor, self.P.u_scale_offset)
    
    def _scale_h(self, h: np.ndarray) -> np.ndarray:
        return self._scale_value(h, self.P.h_scale_factor, self.P.h_scale_offset)
    
    def _scale_y(self, y: np.ndarray) -> np.ndarray:
        return self._scale_value(y, self.P.y_scale_factor, self.P.y_scale_offset)
    
    def _scale_y_e(self, y_e: np.ndarray) -> np.ndarray:
        return self._scale_value(y_e, self.P.ye_scale_factor, self.P.ye_scale_offset)
    
    def _unscale_x(self, x: np.ndarray) -> np.ndarray:
        return self._unscale_value(x, self.P.x_scale_factor, self.P.x_scale_offset)
    
    def _unscale_u(self, u: np.ndarray) -> np.ndarray:
        return self._unscale_value(u, self.P.u_scale_factor, self.P.u_scale_offset)
    
    def _unscale_y(self, h: np.ndarray) -> np.ndarray:
        return self._unscale_value(h, self.P.h_scale_factor, self.P.h_scale_offset)
    
    def _unscale_y(self, y: np.ndarray) -> np.ndarray:
        return self._unscale_value(y, self.P.y_scale_factor, self.P.y_scale_offset)
    
    def _unscale_y_e(self, y_e: np.ndarray) -> np.ndarray:
        return self._unscale_value(y_e, self.P.ye_scale_factor, self.P.ye_scale_offset)
    
    @staticmethod
    def _push_results(mpc_results: AMPC_data, solver_results: SolveStepResults, sim_step: int) -> AMPC_data:
        # timings
        mpc_results.Time[sim_step] = solver_results.soltime_p
        mpc_results.Acados_Time[sim_step] = solver_results.soltime_a
        mpc_results.Prep_Time[sim_step] = solver_results.prep_time
        mpc_results.Fb_Time[sim_step] = solver_results.fb_time

        # iterations
        mpc_results.Iterations[sim_step] = solver_results.soliters
        mpc_results.Prep_Iterations[sim_step] = solver_results.prep_iters
        mpc_results.Fb_Iterations[sim_step] = solver_results.fb_iters

        # trajectories
        mpc_results.X[:,sim_step] = solver_results.simX_traj[:, 0]
        mpc_results.U[:,sim_step] = solver_results.simU_traj[:, 0]
        mpc_results.X_traj[sim_step,:,:] = solver_results.simX_traj
        mpc_results.U_traj[sim_step,:,:] = solver_results.simU_traj
        return mpc_results
    
    @staticmethod
    def _filter_finite_bounds(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Helper function to filter out infinite bounds and return valid values and indices."""
        finite_mask = np.logical_and(np.isfinite(lower), np.isfinite(upper))
        return lower[finite_mask], upper[finite_mask], np.where(finite_mask)[0]
    
    @classmethod
    @log_memory_usage('Temp/log_ampc_mem.txt')
    def trajectory_and_save(
            cls, 
            ampc_param: AMPC_Param,
            model_fun: Callable[[Basic_Model_Param], AcadosModel],
            save_dir: str,
            always_overwrite: bool = True,
            x_init: np.ndarray | None = None,
            show_tqdm: bool=True,
            show_exception = True
    ) -> AMPC_data:
        model = model_fun(ampc_param.model_param)
        instance = cls(model, ampc_param)
        try:
            results = instance.get_trajectory(x_init=x_init, show_tqdm=show_tqdm, show_exception=show_exception)
        except Exception as e:
            raise e
        finally:
            instance.cleanup()
        
        results.save(file_dir=save_dir, always_overwrite=always_overwrite)
        return results

        


# ===================================================================================================
# ACADOS NH-MPC
# ===================================================================================================
class Base_NH_AMPC_class(Base_AMPC_class):
    """
    Class that uses ``Base_MPC_acados`` and instantiates a neural horizon acados MPC.
    """
    def __init__(
            self, 
            model: AcadosModel,
            NNmodel: NN_for_casadi, 
            solver_verbose: bool = False,
        ):
        """
        Implements a NH-AMPC using a horizon of length ``P.N_MPC`` for M and ``P.N_NN`` for N, 
        while using the default or given acados options, for the solver. 

        !!ATTENTION!! function ``cleanup()`` must be called after usage.

        Parameters
        ----------
        ``model`` : AcadosModel
            An acados model, either only continuous or with discrete implementation.
        ``NNmodel`` : NN_for_casadi
            A class containing the trained neural network model with the method
             NN_casadi(x0): a function that returns the optimal control sequence 
              for the horizon starting at x0, and the total cost of this sequence
               in the form of a list of CasADi symbolic variables.
        ``solver_options`` : dict, optional 
            A dictionary with acados options for ocp and ocp solver as well as cost type.
             See page -> https://docs.acados.org/python_interface/index.html 
              Default = None
        ``acados_name`` : str, optional
            A string defining the acados name used for ocp_solver.json and tqdm. 
             Default = ''
        ``solver_verbose`` : bool, optional
            A bool defining if during solving and broke solver the statistics are printet.
             Default = False,
        ``ignore_status_errors`` : set of [1, 2, 3, 4, 5], optional
            A set defining if during solving it does not raises an error 
             when {1, 2, 3, 4, 5} are the status returns of acados.
              Default = True
        ``horizon_name`` : str, optional
            A string used for ocp_solver.json and tqdm.
             Default = ''
        """          
        self.NNmodel = NNmodel
        Base_AMPC_class.__init__(
            self,
            model,
            NNmodel.P, 
            solver_verbose = solver_verbose, 
            ignore_status_errors = {0, 2, 3, 5},
        )
        
    def _mpc_init(self) -> None:
        self.create_base_ocp()
        self.set_stage_cost()
        self.set_nh_terminal_cost()
        self.set_constraints()
        self.set_nonlinear_constraints()
        self.set_solver_options()
        self._set_ocp_string()
        self.create_ocp_solver()

    @enforce_call_order('set_cost')
    def set_nh_terminal_cost(self) -> None:
        x = self.ocp.model.x
        nn_output = self.NNmodel.NN_casadi(x)

        # set cost typ to external
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # Cost
        yref_e = self.P.yref_e.squeeze()
        self.ocp.model.cost_y_expr_e = cs.vertcat(x, nn_output)
        self.ocp.cost.W_e = scipy.linalg.block_diag(self.P.W_e, *[self.P.W_NN]*self.P.N_NN)
        self.ocp.cost.yref_e = np.repeat(yref_e if len(yref_e.shape) == 1 else yref_e[0], 1 + nn_output.shape[0]//self.ny_e)

    @classmethod
    @log_memory_usage('Temp/log_nh_ampc_mem.txt')
    def trajectory_and_save(
            cls, 
            nh_ampc_param: NH_AMPC_Param,
            model_fun: Callable[[Basic_Model_Param], AcadosModel],
            save_dir: str, 
            nn_dir: str, 
            ds_dir: str, 
            always_overwrite: bool = True,
            x_init: np.ndarray | None = None,
            show_tqdm: bool = True,
            show_exception = True
    ) -> AMPC_data:
        device = torch.device('cpu')
        dtype = torch.float32
        NN_fc = NN_for_casadi.load_from_Param(nh_ampc_param, nn_dir, ds_dir, device, dtype)
        NN_fc.evaluate_NN()
        
        model = model_fun(nh_ampc_param.model_param)
        instance = cls(model, NN_fc)
        try:
            results = instance.get_trajectory(x_init=x_init, show_tqdm=show_tqdm, show_exception=show_exception)
        except Exception as e:
            raise e
        finally:
            instance.cleanup()

        results.save(file_dir=save_dir, always_overwrite=always_overwrite)
        return results