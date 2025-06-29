import numpy as np
import json
import os
import warnings
import time

from dataclasses import dataclass, fields, field, is_dataclass, asdict, Field
from typing import Literal, Self, Any

from .mpc_param import MPC_Param, AMPC_Param


@dataclass
class Dataset_Param:
    """
    """
    mpc_param : MPC_Param | AMPC_Param | None = None

    version : int | None = None
    begin : Literal['begin', 'fixed', ''] = ''
    feature : int = 8
    samples : int = 10_000
    name : str = ''
    save_iter = 1000
    chance_reset_sample = 0.25
    init_scale : np.ndarray = field(default_factory=lambda: np.array([.75, .25, .25, .25])) 
    init_bias : np.ndarray  = field(default_factory=lambda: np.array([0, np.pi, 0, 0]))
    state_sample_space : np.ndarray = field(default_factory=lambda: np.array([]))
    yref_sample_space : np.ndarray = field(default_factory=lambda: np.array([]))

    file : str = field(init=False)

    def __post_init__(self) -> None:
        if self.mpc_param is None:
            raise ValueError(f'MPC parameters are None in Dataset_Param')
        self.set_file_name()

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
    
    def set_file_name(self) -> None:
        self.file = get_dataset_file(
            self.mpc_param.N_MPC, 
            self.samples, 
            self.mpc_param.acados_name if isinstance(self.mpc_param, AMPC_Param) else None, 
            self.version
        ) 

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
    
    def save(self,filename=None,filedir='Results//Parameters'):
        pjson = self.get_json()
        pjson_object = json.dumps(pjson, indent=4)
        if filename is None:
            filename=f'MPC_Param_{time.strftime("%d_%b_%Y_%H_%M")}.json'
        os.makedirs(filedir,exist_ok=True)
        with open(os.path.join(filedir,filename),'w') as file:
            file.write(pjson_object)

    @staticmethod
    def _convert_to_type(f: Field, param):
        if f.type is np.ndarray:
            return np.array(param)
        if f.name == 'mpc_param' and not 'acados_name' in param:
            return MPC_Param.from_dict(param)
        if f.name == 'mpc_param' and 'acados_name' in param:
            return AMPC_Param.from_dict(param)
        else:
            return param

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        init_params = {}
        non_init_params = {}

        for f in fields(cls):
            if f.name in params and f.init:
                init_params[f.name] = cls._convert_to_type(f, params[f.name])
            elif f.name in params and not f.init:
                non_init_params[f.name] = cls._convert_to_type(f, params[f.name])
            else:
                warnings.warn(f'Parameter {f.name} not in dataclass {cls.__name__}',UserWarning)

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
            warnings.warn(f'File {filepath} does not exist!',UserWarning)
            return None
        
        with open(filepath,'r') as file:
            pjson = json.load(file)

        return cls.from_dict(pjson)
    


def get_dataset_file(
    dataset_horizon: int,
    num_samples: int,
    acados_name: str | None,
    dataset_version: int | None,
) -> str:
    """
    Creates a standard dataset name out of the given parameters. 
    """
    _v_ds = '' if dataset_version is None else f'_{dataset_version}v'
    _acados_name = '' if acados_name is None else f'_{acados_name}'
    return f'MPC_data_{dataset_horizon}steps_{num_samples}datapoints{_acados_name}{_v_ds}.csv'