import numpy as np
import json
import os
import warnings
import time

from typing import Self, Any, TypeVar
from dataclasses import dataclass, field, fields, is_dataclass, asdict, Field

from .model_param import Basic_Model_Param


@dataclass
class Acados_appendings:
    """
    Represents additional ACADOS-specific settings required for a control problem.

    Attributes
    ----------
    `acados_settings` : dict[str, object]
        Dictionary containing ACADOS configuration settings.
    `acados_name` : str
        Name identifier for the ACADOS configuration.
    """
    acados_settings: dict[str, object]
    acados_name: str


@dataclass
class MPC_Param:
    """
    Represents parameters for a Model Predictive Control (MPC) problem.

    Attributes
    ----------
    `model_param` : Basic_Model_Param | None
        Model parameters used in the control formulation.
    `scale_model` : bool
        Flag indicating whether the model should be scaled.
    `ny` : int
        Number of outputs in the control problem.
    `ny_e` : int
        Number of outputs at the terminal stage.
    `W` : np.ndarray
        Weight matrix for stage cost.
    `W_e` : np.ndarray
        Weight matrix for terminal cost.
    `N_MPC` : int
        Length of the MPC horizon.
    `ubu` : np.ndarray
        Upper bounds for control inputs.
    `lbu` : np.ndarray
        Lower bounds for control inputs.
    `ubx_0` : np.ndarray
        Initial state upper bounds.
    `lbx_0` : np.ndarray
        Initial state lower bounds.
    `ubx` : np.ndarray
        State upper bounds.
    `lbx` : np.ndarray
        State lower bounds.
    `ubh` : np.ndarray
        Upper bounds for nonlinear constraints.
    `lbh` : np.ndarray
        Lower bounds for nonlinear constraints.
    `x0` : np.ndarray
        Initial state.
    `yref` : np.ndarray
        Reference trajectory.
    `yref_e` : np.ndarray
        Terminal reference.
    `ylabel` : list[str]
        Labels for the stage cost values.
    `yelabel` : list[str]
        Labels for the terminal cost values.
    `x_scale_factor` : np.ndarray
        Scaling factor for states.
    `u_scale_factor` : np.ndarray
        Scaling factor for inputs.
    `h_scale_factor` : np.ndarray
        Scaling factor for nonlinear constraints.
    `y_scale_factor` : np.ndarray
        Scaling factor for outputs.
    `y_e_scale_factor` : np.ndarray
        Scaling factor for terminal outputs.
    `x_scale_offset` : np.ndarray
        Scaling offset for states.
    `u_scale_offset` : np.ndarray
        Scaling offset for inputs.
    `h_scale_offset` : np.ndarray
        Scaling offset for nonlinear constraints.
    `y_scale_offset` : np.ndarray
        Scaling offset for outputs.
    `y_e_scale_offset` : np.ndarray
        Scaling offset for terminal outputs.
    `Ts` : float
        Sampling time.
    `T_sim` : float
        Simulation time horizon.
    `version` : int
        Version of the MPC parameter configuration.
    `name` : str
        Identifier name for the MPC parameter configuration.
    `N_sim` : int
        Number of simulation steps, computed as T_sim / Ts.
    """
    model_param: Basic_Model_Param | None
    scale_model: bool
    
    ny:     int
    ny_e:   int
    W:      np.ndarray
    W_e:    np.ndarray
    N_MPC:  int

    ubu:    np.ndarray
    lbu:    np.ndarray

    ubx_0:  np.ndarray
    lbx_0:  np.ndarray
    ubx:    np.ndarray
    lbx:    np.ndarray

    ubh:    np.ndarray
    lbh:    np.ndarray  

    x0:     np.ndarray   
    yref:   np.ndarray
    yref_e: np.ndarray

    ylabel: list[str]
    yelabel: list[str]

    x_scale_factor:     np.ndarray
    u_scale_factor:     np.ndarray
    h_scale_factor:     np.ndarray
    y_scale_factor:     np.ndarray
    ye_scale_factor:   np.ndarray

    x_scale_offset:     np.ndarray
    u_scale_offset:     np.ndarray
    h_scale_offset:     np.ndarray
    y_scale_offset:     np.ndarray
    ye_scale_offset:   np.ndarray

    Ts:         float
    T_sim:      float

    version:    int
    name:       str
    file:       str
    N_sim:      int = field(init=False)
    
    def __post_init__(self) -> None:
        """Post-initialization to set derived attributes."""
        if self.model_param is None:
            raise ValueError('Model Parameters are None')
        self.N_sim = int(self.T_sim/self.Ts)
        if not self.name:
            self._set_default_name()
        if not self.file:
            self._set_default_file()

    def __eq__(self, other) -> bool:
        """Equality check of two instances."""
        if not isinstance(other, type(self)):
            return False
        
        if fields(self) != fields(other):
            return False
        
        for field in fields(self):
            self_val = getattr(self, field.name)
            other_val = getattr(other, field.name) 
            if isinstance(self_val, np.ndarray):
                if not np.array_equal(self_val, other_val):
                    return False
            elif self_val != other_val:
                return False
        return True
    
    def _set_default_file(self) -> None:
        """Default file setting."""
        self.file = f'MPC_results_{self.name}.ph'
    
    def _set_default_name(self) -> None:
        """Default name setting."""
        self.name = f'{self.N_MPC}M_{self.version}v'

    def diff(self,other) -> dict[str, Any]:
        """Difference of two instances"""
        differences = {}
        if isinstance(other, type(self)):
            for field in fields(self):
                self_val = getattr(self, field.name)
                other_val = getattr(other, field.name)

                if isinstance(self_val, np.ndarray):
                    if not np.array_equal(self_val, other_val):
                        differences[field.name] = (self_val, other_val)

                elif is_dataclass(self_val) and hasattr(self_val, 'diff'):
                    dc_diff = self_val.diff(other_val)
                    if dc_diff:
                        differences[field.name] = dc_diff

                elif self_val != other_val:
                    differences[field.name] = (self_val, other_val)
        return differences
    
    def make_arrays_1d(self) -> None:
        """Squeezes all fields if numpy arrays."""
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type == np.ndarray and v is not None and v.squeeze().ndim == 1:
                setattr(self, f.name, v.squeeze())
    
    def get_json(self) -> dict:
        """Gets serializable dict."""
        pjson = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if type(v) is np.ndarray:
                pjson[f.name] = v.tolist() 
            elif is_dataclass(v) and hasattr(v, 'get_json'):
                pjson[f.name] = v.get_json()
            elif is_dataclass(v) and not hasattr(v, 'get_json'):
                pjson[f.name] = asdict(v)
            else:
                pjson[f.name] = v
        return pjson

    def save(self, filename=None, filedir='Results//Parameters') -> None:
        """Save the """
        pjson = self.get_json()
        pjson_object = json.dumps(pjson, indent=4)
        if filename is None:
            filename=f'MPC_Param_{time.strftime("%d_%b_%Y_%H_%M")}.json'
        os.makedirs(filedir, exist_ok=True)
        with open(os.path.join(filedir, filename), 'w') as file:
            file.write(pjson_object)

    @staticmethod
    def _convert_to_type(
        f: Field, 
        param: Any, 
        model_cls: Basic_Model_Param = Basic_Model_Param
    ) -> np.ndarray | Basic_Model_Param | Any:
        if f.type is np.ndarray:
            return np.array(param)
        elif f.name == 'model_param':
            return model_cls.from_dict(param)
        else:
            return param

    @classmethod
    def from_dict(cls, params: dict, model_cls: Basic_Model_Param = Basic_Model_Param) -> Self:
        init_params = {}
        non_init_params = {}

        for f in fields(cls):
            if f.name in params and f.init:
                init_params[f.name] = cls._convert_to_type(f, params[f.name], model_cls)
            elif f.name in params and not f.init:
                non_init_params[f.name] = cls._convert_to_type(f, params[f.name], model_cls)
            else:
                warnings.warn(f'Parameter {f.name} not in dataclass {cls.__name__}', UserWarning)

        instance = cls(**init_params)
        for k, v in non_init_params.items():
            setattr(instance, k, v)
        
        return instance
    
    @classmethod
    def replace(cls, other_cls, **kwargs) -> Self:
        """
        Creates a new instance of the class with one or more parameters replaced.
        Ensures __post_init__ is called for the new instance.
        """
        current_values = {field.name: getattr(other_cls, field.name) for field in fields(other_cls) if field.init}
        updated_values = {**current_values, **kwargs}
        return cls(**updated_values)

    @classmethod
    def load(cls, filepath) -> None | Self:
        if not os.path.exists(filepath):
            warnings.warn(f'File {filepath} does not exist!', UserWarning)
            return None
        
        with open(filepath,'r') as file:
            pjson = json.load(file)

        return cls.from_dict(pjson)




@dataclass
class AMPC_Param(MPC_Param, Acados_appendings):
    """
    Represents parameters for a Model Predictive Control problem in acados.

    Inherits
    --------
    MPC_Param : Base class for standard MPC parameters.
    Acados_appendings : ACADOS-specific configuration.

    Attributes
    ----------
    Inherits all attributes from MPC_Param and Acados_appendings.
    """
    def _set_default_file(self) -> None:
        """Default file setting."""
        self.file = f'AMPC_results_{self.acados_name}_{self.name}.ph'
    
MPC_PARAMS = TypeVar('MPC_Params', MPC_Param, AMPC_Param)