import numpy as np
import pandas as pd
import time
import os
import logging

from tqdm.auto import trange


from ..mpc.mpc_classes_acados import Base_AMPC_class
from ..mpc.mpc_dataclass import SolveStepResults
from ..parameters.dataset_param import Dataset_Param
from ..utils.errors import OutOfBoundsError



class AMPC_dataset_gen():
    """
    Methods
    -------  
    get_new_init_state(self, scale=np.array([[.75],[.25],[.25],[.25]]), bias=np.array([[0],[np.pi],[0],[0]])) -> numpy.ndarray:
        Generates a new initial state for the MPC problem.  
    get_new_guesses(self, x_curr: np.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        Generates new state and input guesses.
    save_df(df=None, filename=None, filedir='Results//MPC_data_gen', float_format='%.4f') -> PathLike:
        Saves the generated data to a CSV file.
    generate_data(show_tqdm=True, filename=None, filedir=os.path.abspath(os.path.join('Results', 'MPC_data_gen'))) -> PathLike:
        Generates the dataset.
    """
    def __init__(self, 
            P: Dataset_Param, 
            MPC: Base_AMPC_class,
            state_sample_space: np.ndarray | None = None,
            yref_sample_space: np.ndarray | None = None
        ) -> None:
        """
        Generates training dataset of model predictive control (MPC) predictions.
        
        Parameters
        ----------
        ``P`` : MPC_Param dataclass object, optional 
            An object containing the MPC parameters. Default = MPC_Param()
        ``n_samples`` : int, optional 
            The number of samples to generate. Default = 1000
        ``save_iter`` : int, optional 
            Determines how often to save intermediate results. Default = 1000
        ``chance_reset_sample`` : float, optional 
            The probability of resetting the initial state to a random value. Default = 0.25
        ``init_scale`` : numpy.ndarray, optional
            Range of the domain where initial point is sampled as the proportion of state bounds (P.xbnd).
            Default = np.array([[.75],[.25],[.25],[.25]])
        ``init_bias`` : numpy.ndarray, optional
            Bias term for the new starting point sampling.
            Default = np.array([[0],[np.pi],[0],[0]])
        """
        self.P = P
        self.mpc = MPC

        if state_sample_space is not None:
            self.P.state_sample_space = state_sample_space 

        if yref_sample_space is not None:
            self.P.yref_sample_space = yref_sample_space
        
        self.df = None

        self.filedir = os.path.abspath(os.path.join('Results', 'MPC_data_gen'))
        self.temp_dir = os.path.abspath(os.path.join('Temp', f'Datagen_log_{time.strftime("%d_%b_%Y_%H_%M")}'))
        self.store_iterate_filename = os.path.join(self.temp_dir, 'ocp_data_iterate.json')
        os.makedirs(os.path.dirname(self.store_iterate_filename), exist_ok=True)
        self.mpc.ocp_solver.store_iterate(self.store_iterate_filename, overwrite=True)
    

    def get_new_init_state(self) -> np.ndarray:
        """
        Generates a new initial state for the MPC problem.
        The new starting point is uniformly sampled from
            [-P.xbnd*scale,P.xbnd*scale] + bias

        
        Returns
        -------
        ``x_curr`` : numpy.ndarray
            The new initial state.
        """
        rand_num = np.random.random((self.P.mpc_param.model_param.nx,))

        if self.P.state_sample_space.size != 0:
            diff = self.P.state_sample_space[:, 1] - self.P.state_sample_space[:, 0]
            x_curr = self.P.state_sample_space[:, 0] + rand_num * diff
        else:
            diff = self.P.mpc_param.ubx.reshape((-1,)) - self.P.mpc_param.lbx.reshape((-1,))
            offset = self.P.mpc_param.lbx.reshape((-1,)) + diff/2
            x_curr = offset + self.P.init_bias + self.P.init_scale*diff*(rand_num - 0.5)

        return x_curr
        

    def get_new_guesses(self, x_curr: np.ndarray):
        """
        Generates new state and input guesses.
        State guess is a repeated ``x_curr`` and input guess is a zero array. 
        
        Parameters
        ----------
       ``x_curr`` : np.ndarray
            
        Returns
        -------
        ``x_guess`` : numpy.ndarray
            An array of initial state trajectory guess for the OCP solver.
        ``u_guess`` : numpy.ndarray
            An array of initial input trajectory guess for the OCP solver.
        """
        x_guess = np.repeat(x_curr.reshape((self.P.mpc_param.model_param.nx, 1)), repeats=self.P.mpc_param.N_MPC, axis=1)
        u_guess = np.zeros((self.P.mpc_param.model_param.nu, self.P.mpc_param.N_MPC))
        return x_guess, u_guess
    

    def get_new_yref(self) -> np.ndarray | None:
        if self.P.yref_sample_space is None:
            return None
        
        rand_num = np.random.random(self.P.mpc_param.yref.shape[-1], )
        diff = self.P.yref_sample_space[:, 1] - self.P.yref_sample_space[:, 0]
        return self.P.yref_sample_space[:, 0] + rand_num * diff


    def update_mpc_values(self):
        x_curr = self.get_new_init_state()
        x_guess, u_guess = self.get_new_guesses(x_curr)

        if self.P.yref_sample_space.size != 0:
            y_ref = self.get_new_yref()
            for i_ref in range(self.mpc.N_MPC):
                self.mpc.ocp_solver.set(i_ref, 'y_ref', y_ref)
        else:
            y_ref = self.P.mpc_param.yref

        return x_curr, x_guess, u_guess, y_ref
    

    def save_df(self, df=None, filename=None, filedir=None, float_format='%.6f'):
        """
        Saves the generated data to a CSV file.
        
        Parameters
        ----------
        ``df`` : pandas DataFrame, optional (default=None)
            The dataframe to save.
        ``filename`` : str, optional (default=None)
            The filename to use.
        ``filedir`` : str, optional (default='Results//MPC_data_gen')
            The directory to save the file in.
        ``float_format`` : str, optional (default='%.4f')
            The format string for floating point numbers.
        
        Returns
        -------
        ``filename`` : str
            The location of the saved file.
        """
        if filedir is None:
            filedir = self.filedir

        if len(filedir)>0:
            os.makedirs(filedir, exist_ok=True)
        
        if df is None:
            df = self.df
        
        if filename is None:
            filename = self.P.file

        filepath = os.path.join(filedir, filename)
        df.to_csv(filepath, index=False, float_format=float_format)
        self.P.save(filename=f'{os.path.basename(filepath)[:-3]}json', filedir=filedir)
        return filepath
    

    def check_bounds(self, solve_results: SolveStepResults) -> None:
        in_ubnd = np.all(solve_results.simU_traj.T <= self.P.mpc_param.ubu.squeeze()) \
                and np.all(solve_results.simU_traj.T >= self.P.mpc_param.lbu.squeeze())
        in_xbnd = np.all(solve_results.simX_traj.T <= self.P.mpc_param.ubx.squeeze()) \
                and np.all(solve_results.simX_traj.T >= self.P.mpc_param.lbx.squeeze())
        if not (in_ubnd and in_xbnd):
            raise OutOfBoundsError(
                (solve_results.simU_traj, solve_results.simX_traj), 
                (self.P.mpc_param.ubu, self.P.mpc_param.lbu, self.P.mpc_param.ubx, self.P.mpc_param.lbx))
    

    def generate_data(self, show_tqdm=True, filename=None, filedir=None):
        """
        Generates the dataset. Saves temporary data in the 'Temp' folder.
        
        Parameters
        ----------
        ``show_tqdm`` : bool, optional
            A flag to show a progress bar. Default = True
        ``filename`` : str, optional
            The filename where the data should be stored.
        ``filedir`` : PathLike, optional
            The directory where the data should be stored.
        
        Returns
        -------
        ``filename`` : str
            The location of the CSV file containing generated data.
        """

        ds_dict = {
            f'{yl}_ref': np.empty(self.P.samples)
                for yl in self.P.mpc_param.ylabel
        } | {
            f'{xul}_p{i}': np.empty(self.P.samples) 
                for i in range(self.P.mpc_param.N_MPC) 
                for xul in self.P.mpc_param.model_param.xlabel + self.P.mpc_param.model_param.ulabel
        } | {
            f'{xl}_p{self.P.mpc_param.N_MPC}': np.empty(self.P.samples) 
                for xl in self.P.mpc_param.model_param.xlabel
        }


        x_curr, x_guess, u_guess, y_ref = self.update_mpc_values()

        for i in trange(self.P.samples, disable = not show_tqdm, desc=f'MPC horizon of {self.P.mpc_param.N_MPC}', unit='Samples'):
            time.sleep(0.001)
            if np.random.random() < self.P.chance_reset_sample:
                x_curr, x_guess, u_guess, y_ref = self.update_mpc_values()

            soltry = True
            tries = 0
            while soltry:
                try:
                    tries += 1
                    solve_results = self.mpc.solve(x_curr, x_guess, u_guess)
                    self.check_bounds(solve_results)
                    soltry = False
                except Exception as e:
                    logging.debug(f'Iter {i} - {str(e)} :-> {x_curr}')
                    time.sleep(0.001)
                    x_curr, x_guess, u_guess, y_ref = self.update_mpc_values()
                    self.mpc.reset_solver()

                    # recreate bricked solver
                    if self.mpc.current_status in {1, 4} and tries > 10:
                        self.mpc.recreate_solver()
                
            x_curr = solve_results.simX_traj[:, 1]
            x_guess = np.hstack((solve_results.simX_traj[:, 1:], solve_results.simX_traj[:, -1:]))
            u_guess = np.vstack((solve_results.simU_traj[1:], solve_results.simU_traj[-1:]))

            # save references
            for yk, yl in enumerate(self.P.mpc_param.ylabel):
                ds_dict[f'{yl}_ref'][i] = y_ref[yk]

            for j in range(self.P.mpc_param.N_MPC+1):
                # save states
                for xk, xl in enumerate(self.P.mpc_param.model_param.xlabel):
                    ds_dict[f'{xl}_p{j}'][i] = solve_results.simX_traj[xk, j]
                # save inputs
                if j<self.P.mpc_param.N_MPC:
                    for uk, ul in enumerate(self.P.mpc_param.model_param.ulabel):
                        ds_dict[f'{ul}_p{j}'][i] = solve_results.simU_traj[uk, j]    

            if not i%self.P.save_iter and i>0:
                # save intermediate results
                self.save_df(df=pd.DataFrame(ds_dict).iloc[:i+1], filedir=self.temp_dir)

        # make dataframe
        self.df = pd.DataFrame(ds_dict)
        filename = self.save_df(filename=filename, filedir=filedir)
        
        print(f'Data generation complete.\nGenerated {self.df.shape[0]} data points for the baseline MPC with {self.P.mpc_param.N_MPC} steps.\nResult stored under   {filename}')
        
        return filename