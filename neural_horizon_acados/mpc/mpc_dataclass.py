import numpy as np
import pickle as pkl
import os
import logging

from dataclasses import dataclass, field, asdict, fields
from typing import Callable, TypeVar, Generator, Iterator, Any, Iterable, Self
from itertools import groupby


from ..parameters.mpc_param import AMPC_Param, MPC_Param, MPC_PARAMS
from ..parameters.nh_mpc_param import NH_AMPC_Param, NH_MPC_Param, NH_MPC_PARAMS


@dataclass
class SolveStepResults:
    """
    A dataclass of the results of one MPC step (open loop MPC results).

    Attributes
    ----------
    ``simX_traj`` : np.ndarray
        An array of the simulated open loop state trajectories.
         Shape -> (P.nx, P.N_MPC+1) 
    ``simU_traj`` : np.ndarray
        An array of the simulated open loop input trajectory.
         Shape -> (P.nx, P.N_MPC) 
    ``soltime_a`` : float, default = 0.0
        The acados measured solving time in seconds. 
    ``soltime_p`` : float, default = 0.0
        The python measured solving time in seconds.
    ``soliters`` : int, default = 0
        The qp iterations the solver took to solve the problems.
    ``prep_time`` : float, default = 0.0
        The acados measured solver preperation time in seconds. 
    ``fb_time`` : float, default = 0.0
        The acados measured solver feedback time in seconds. 
    ``prep_iters`` : int, default = 0
        The qp iterations the solver took in preperation phase.
    ``fb_iters`` : int, default = 0
        The qp iterations the solver took in feedback phase.
    """
    simX_traj : np.ndarray
    simU_traj : np.ndarray
    soltime_a : float = 0.0
    soltime_p : float = 0.0
    soliters : int = 0
    prep_time : float = 0.0
    fb_time : float = 0.0
    prep_iters : int = 0
    fb_iters : int = 0

    @classmethod
    def from_params(cls, P: MPC_Param) -> Self:
        return cls(
            np.full((P.model_param.nx, P.N_MPC+1), np.nan), 
            np.full((P.model_param.nu, P.N_MPC), np.nan)
        )

    def add_prep_and_fb(self) -> None:
        self.soltime_a = self.prep_time + self.fb_time
        self.soliters = self.prep_iters + self.fb_iters
    

@dataclass
class MPC_data:
    """
    A dataclass of the simulated closed loop MPC results.
    Can be set to frozen, so one can't change the attributes and values anymore.

    Attributes
    ----------
    ``P`` : MPC_Param
        The parameters to setup the MPC classes.
    ``name`` : str
        The name of the MPC data.
    ``X`` : np.ndarray
        The simulated closed loop state trajectories.
         Shape -> (P.nx, P.N_sim) 
    ``U`` : np.ndarray
        The simulated closed loop input trajectory.
         Shape -> (P.nu, P.N_sim) 
    ``Time`` : np.ndarray
        The solving time for each simulation step. 
         Shape -> (P.N_sim, ) 
    ``X_traj`` : np.ndarray
        The simulated open loop state trajectories for each closed loop step.
         Shape -> (P.N_sim, P.nx, P.N_MPC+1)     
    ``U_traj`` : np.ndarray
        The simulated open loop input trajectories for each closed loop step.
         Shape -> (P.N_sim, P.nx, P.N_MPC) 
    ``Cost`` : float
        The overall cost over the simulation time. 
    """
    mpc_param : MPC_Param | NH_MPC_Param
    
    X : np.ndarray = field(init=False)
    U : np.ndarray = field(init=False)
    Time : np.ndarray = field(init=False)
    X_traj : np.ndarray = field(init=False)
    U_traj : np.ndarray = field(init=False)
    Cost : float = field(init=False, default=None)

    _is_frozen : bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.X = np.full((self.mpc_param.model_param.nx, self.mpc_param.N_sim), np.nan)
        self.U = np.full((self.mpc_param.model_param.nu, self.mpc_param.N_sim), np.nan)
        self.Time = np.full(self.mpc_param.N_sim, np.nan)
        self.X_traj = np.full((self.mpc_param.N_sim, self.mpc_param.model_param.nx, self.mpc_param.N_MPC+1), np.nan)
        self.U_traj = np.full((self.mpc_param.N_sim, self.mpc_param.model_param.nu, self.mpc_param.N_MPC), np.nan)

    def __setattr__(self, name, value) -> None:
        if self._is_frozen:
            raise AttributeError(f"{self.__class__.__name__} is frozen. Cannot set attribute '{name}'")
        super().__setattr__(name, value)

    def freeze(self) -> None:
        """
        Freezes the dataclass, so one cannot change the attributes anymore.
        """
        self._is_frozen = True

    def save(
            self, *,
            file_path: str | None = None, 
            file_dir: str | None = None, 
            always_overwrite: bool = False
        ) -> None | str:
        """
        Save the dataclass as a pickle file in file_path.
        """
        if file_dir is None and file_path is not None:
            file_dir = os.path.dirname(file_path)
        elif file_dir is not None and file_path is None:
            file_path = os.path.join(file_dir, self.mpc_param.file)
        else:
            raise Exception('file path or dir need to be not None.')
        
        os.makedirs(file_dir, exist_ok = True)

        if os.path.exists(file_path) and not always_overwrite:
            inp = input('Overwrite existing file? [y/n]')
            
            if 'Y' != inp.capitalize():
                return None

        with open(file_path, 'wb') as handle:
            pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)

        logging.info(f'Saved data in {file_path}.')
        return file_path

    @classmethod
    def load(cls, file_path: os.PathLike) -> Self:
        """
        Classmethod to load the dataclass from a previously saved pickle file.
        """
        if not os.path.exists(file_path):
            raise ValueError(f'No such file path existent: {file_path}')

        with open(file_path, 'rb') as handle:
            obj: MPC_data = pkl.load(handle)   
        
        return obj
    
    @classmethod
    def load_from_param(cls, mpc_param: MPC_Param, file_dir: os.PathLike) -> Self:
        """
        Classmethod to load the dataclass from mpc param.
        """
        file_path = os.path.join(file_dir, mpc_param.file)
        return cls.load(file_path)
        
    @classmethod
    def from_MPC_data(cls, base_data: 'MPC_data') -> Self:
        new_instance = cls(base_data.mpc_param)
        cls_field_names = {field.name for field in fields(cls)}
        for field_name, value in asdict(base_data).items():
            if field_name in cls_field_names and field_name != 'mpc_param': 
                setattr(new_instance, field_name, value)
        return new_instance



@dataclass
class AMPC_data(MPC_data):
    """
    A dataclass of the simulated closed loop AMPC results.
    Can be set to frozen, so one can't change the attributes and values anymore.

    Attributes
    ----------
    ``P`` : AMPC_Param | NH_AMPC_Param | MPC_Param
        The parameters to setup the MPC classes.
    ``name`` : str
        The name of the MPC data.
    ``X`` : np.ndarray
        The simulated closed loop state trajectories.
         Shape -> (P.nx, P.N_sim) 
    ``U`` : np.ndarray
        The simulated closed loop input trajectory.
         Shape -> (P.nu, P.N_sim) 
    ``Time`` : np.ndarray
        The python measured solving time for each simulation step. 
         Shape -> (P.N_sim, ) 
    ``X_traj`` : np.ndarray
        The simulated open loop state trajectories for each closed loop step.
         Shape -> (P.N_sim, P.nx, P.N_MPC+1)     
    ``U_traj`` : np.ndarray
        The simulated open loop input trajectories for each closed loop step.
         Shape -> (P.N_sim, P.nx, P.N_MPC) 
    ``Cost`` : float
        The overall cost over the simulation time. 
    ``acados_name`` : str, default = ''
        The name used for the AMPC.
    ``acados_options`` : dict[str, Any]
        The dictionary containing the solver options for AMPC.
    ``Acados_Time`` : np.ndarray
        The acados measured solving time for each simulation step. 
         Shape -> (P.N_sim, ) 
    ``Iterations`` : np.ndarray
        The total number of qp iterations needed for each simulation step. 
         Shape -> (P.N_sim, ) 
    ``Prep_Time`` : np.ndarray
        The solver preperation time for each simulation step. 
         Shape -> (P.N_sim, ) 
    ``Fb_Time`` : np.ndarray
        The solver feedback time for each simulation step. 
         Shape -> (P.N_sim, ) 
    ``Prep_Iterations`` : np.ndarray
        The number of qp iterations needed for each solver preperation. 
         Shape -> (P.N_sim, ) 
    ``Fb_Iterations`` : np.ndarray
        The number of qp iterations needed for each solver feedback. 
         Shape -> (P.N_sim, ) 
    """
    mpc_param : AMPC_Param | NH_AMPC_Param
    
    Acados_Time : np.ndarray = field(init=False)
    Iterations : np.ndarray = field(init=False)
    Prep_Time : np.ndarray = field(init=False)
    Fb_Time : np.ndarray = field(init=False)
    Prep_Iterations : np.ndarray = field(init=False)
    Fb_Iterations : np.ndarray = field(init=False)

    
    def __post_init__(self) -> None:
        super().__post_init__()
        is_neural_horizon = isinstance(self.mpc_param, NH_AMPC_Param)
        self.name = f'{self.mpc_param.N_MPC}M_{self.mpc_param.N_NN}N' if is_neural_horizon else f'{self.mpc_param.N_MPC}M'
        self.Acados_Time = np.full(self.mpc_param.N_sim, np.nan)
        self.Iterations = np.full(self.mpc_param.N_sim, np.nan)
        self.Prep_Time = np.full(self.mpc_param.N_sim, np.nan)
        self.Fb_Time = np.full(self.mpc_param.N_sim, np.nan)
        self.Prep_Iterations = np.full(self.mpc_param.N_sim, np.nan)
        self.Fb_Iterations = np.full(self.mpc_param.N_sim, np.nan)



DATA = TypeVar('Data', MPC_data, AMPC_data)
DATA_PARAMS = TypeVar('Data_and_params', MPC_data, AMPC_data, MPC_Param, AMPC_Param, NH_AMPC_Param)

def dataclass_group_by(
        dataclass_list: Iterable[DATA_PARAMS], 
        by: Callable[[DATA_PARAMS], object]
    ) -> Generator[tuple[object, Iterator[DATA_PARAMS]], Any, None]:
    """
    Groupes the dataclass T by the given 'by' callable. 

     - T in {MPC_data, AMPC_data, MPC_Param, AMPC_Param, NH_AMPC_Param}

    Parameters
    ----------
    ``dataclass_list`` : Iterable[T]
        Any Iterable of dataclass T
    ``by`` : Callable[[T], object]
        A callable receiving an input T and returning a value that can be sorted (-> SupportsRichComparison).
        
    Returns
    -------
    ``grouped`` : Generator[tuple[str, Iterator[T]]]
        A generator,m where each item contain a tuple of string and an Iterator of given dataclasses T.
    """
    if not callable(by):
        raise TypeError('The "by" parameter must be callable!')

    dataclass_list = sorted(dataclass_list, key=by)

    for key, group in groupby(dataclass_list, key=by):
        yield key, group


def find_top_costs(nh_mpc_data_classes: Iterable[DATA], use_top_nns: int = 5) -> list[DATA]:
    return sorted(nh_mpc_data_classes, key=lambda nh_mpc_data: nh_mpc_data.Cost)[:use_top_nns]