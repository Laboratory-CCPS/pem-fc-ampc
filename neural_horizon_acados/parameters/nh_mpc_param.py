import numpy as np

from dataclasses import dataclass, field, Field
from typing import TypeVar, Any

from .mpc_param import MPC_Param, Acados_appendings
from .neural_network_param import NN_Param
from .model_param import Basic_Model_Param


##############################################################################################################
# Neural Horizon Casadi MPC
##############################################################################################################
@dataclass
class NH_MPC_Param(MPC_Param):
    """
    Data class to hold the parameters for a model predictive control (MPC) problem with a pendulum on a cart.

    Attributes
    ----------
    ``Q``: np.ndarray, optional
        State stage cost, default is np.diag([1, 1, 1e-5, 1e-5]).
    ``R``: np.ndarray, optional
        Input stage cost, default is np.diag([1e-5]).
    ``ubnd``: float, optional
        Bound on absolute value of input, default is 80.
    ``xbnd``: np.ndarray, optional
        Bound on absolute value of state, default is np.array([[2],[6*np.pi],[10],[10]]).
    ``N``: int, optional
        Length of the overall horizon (MPC horizon + Neural horizon), default is 30.
    ``N_MPC``: int, optional
        Length of the MPC horizon, default is 8.
    ``xinit``: np.ndarray, optional
        Starting point for the closed-loop trajectory, default is np.array([[0],[np.pi],[0],[0]]).
    ``Ts``: float, optional
        Sampling rate, default is 0.02.
    ``T_sim``: float, optional
        Simulation horizon (in seconds), default is 3.
    ``xlabel``: List[str]
        List of state names
    ``ulabel``: str
        Input name

    Notes
    -----
    The `Q_NN` attribute is set to the same value as `Q` for simplicity.
    The `P` attribute is set to the same value as `Q` for simplicity.
    The `N_NN` attribute is set to `N - N_MPC`.
    The `N_sim` attribute is set to `T_sim / Ts`.
    """
    nn_param : NN_Param

    N : int 
    W_NN : np.ndarray
    N_NN : int = field(init=False)

    def __post_init__(self) -> None:
        # if self.nn_param is None:
        #     raise ValueError('Neural Network parameters are None')
        self.N_NN = self.N - self.N_MPC
        super().__post_init__()

    def _set_default_file(self) -> None:
        """Default file setting."""
        self.file = f'NH_MPC_results_{self.name}.ph'

    def _set_default_name(self) -> None:
        _ds_begin = '' if '' == self.nn_param.train_dataset.begin else f'_{self.nn_param.train_dataset.begin}'
        _ds_features =  f'_{self.nn_param.train_dataset.feature}M' if 'fixed' == self.nn_param.train_dataset.begin else ''
        _nn_name = f'{self.nn_param.n_neurons}Nhid'
        _pruned = '' if self.nn_param.n_neurons_pruned is None else f'_prun_{self.nn_param.n_neurons_pruned}Nhid'
        self.name = f'{self.N_MPC}M_{self.N_NN}N_{self.nn_param.train_dataset.samples}ND_{self.nn_param.train_dataset.version}VD{_ds_begin}{_ds_features}_{_nn_name}{_pruned}'

    @staticmethod
    def _convert_to_type(
        f: Field, 
        param: Any, 
        model_cls: Basic_Model_Param = Basic_Model_Param,
    ) -> np.ndarray | Basic_Model_Param | Any:
        if f.type is np.ndarray:
            return np.array(param)
        elif f.name == 'model_param':
            return model_cls.from_dict(param)
        elif f.name == 'nn_param':
            return NN_Param.from_dict(param)
        else:
            return param
    


##############################################################################################################
# Neural Horizon Acados MPC
##############################################################################################################
@dataclass
class NH_AMPC_Param(NH_MPC_Param, Acados_appendings):
    """
    Data class to hold the parameters for a model predictive control (MPC) problem with a pendulum on a cart.

    MPC Parameters
    --------------
    ``Q``: np.ndarray, optional,
        State stage cost, default is np.diag([1, 1, 1e-5, 1e-5]).
    ``R``: np.ndarray, optional,
        Input stage cost, default is np.diag([1e-5]).
    ``ubnd``: float, optional,
        Bound on absolute value of input, default is 80.
    ``xbnd``: np.ndarray, optional,
        Bound on absolute value of state, default is np.array([[2],[6*np.pi],[10],[10]]).
    ``N``: int, optional,
        Length of the overall horizon (MPC horizon + Neural horizon), default is 30.
    ``N_MPC``: int, optional,
        Length of the MPC horizon, default is 8.
    ``xinit``: np.ndarray, optional,
        Starting point for the closed-loop trajectory, default is np.array([[0],[np.pi],[0],[0]]).
    ``Ts``: float, optional,
        Sampling rate, default is 0.02.
    ``T_sim``: float, optional,
        Simulation horizon (in seconds), default is 3.
    ``xlabel``: List[str],
        List of state names
    ``ulabel``: str
        Input name

    NN configuration Parameters
    ---------------------------
    ``get_states``: bool, optional,
        Whether to estimate the state with the neural network, default is True.
    ``get_state_bounds``: bool, optional,
        Whether to bounds the states estimated by the neural network, default is True.
    ``get_Jx``: bool, optional,
        Whether to estimate the state stage costs with the neural network, default is False.
    ``get_Ju``: bool, optional,
        Whether to estimate the input stage costs with the neural network, default is False.

    Model Parameters
    ----------------
    ``nx``: int, optional,
        Number of states, default is 4.
    ``nu``: int, optional,
        Number of inputs, default is 1.
    ``M``: float, optional,
        Cart weight in kilograms, default is 1.
    ``m``: float, optional,
        Pendulum weight in kilograms, default is 0.1.
    ``g``: float, optional,
        Gravity constant in m/s^2, default is 9.81.
    ``l``: float, optional,
        Pendulum length in meters, default is 0.8.

    Dataset Parameters
    ------------------
    ``N_DS`` : int, default = 0,
        Horizon of the dataset.
    ``TRAIN_V_DS`` : int, optional,
        Dataset version for training the NN. 
    ``TEST_V_DS`` : int, optional,
        Dataset version for NN testing. 
    ``DS_begin`` : Literal['begin', 'fixed', ''], default = '',
        Which dataset index as input of the network. 
         'begin' - always use the first index as the input to the NN. 
          'fixed' - always use the 'DS_feature' index as input to the NN.
           '' - uses the horizon P.N_MPC where the neural horizon starts.
    ``DS_feature`` : int, default = 8,
        Feature where to always start. Only used if 'DS_begin'='fixed'.
    ``DS_samples`` : int, default = 0,
        Number of trajectory samples in the dataset.  
    ``DS_opts_name`` : str, default = '',
        Dataset acados solver name for the postinit dataset name. 
    

    Notes
    -----
    The `Q_NN` attribute is set to the same value as `Q` for simplicity.

    The `P` attribute is set to the same value as `Q` for simplicity.

    The `N_NN` attribute is set to `N - N_MPC`.

    The `N_sim` attribute is set to `T_sim / Ts`.
    """
    def _set_default_file(self) -> None:
        """Default file setting."""
        self.file = f'NH_AMPC_results_{self.acados_name}_{self.name}.ph'


NH_MPC_PARAMS = TypeVar('NH_MPC_Params', NH_AMPC_Param, NH_MPC_Param)