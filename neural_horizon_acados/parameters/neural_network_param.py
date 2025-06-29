import numpy as np
import json
import os
import warnings
import time
import torch.nn as nn

from dataclasses import dataclass, field, fields, is_dataclass, asdict, Field
from typing import Literal, Any, Self

from .mpc_param import AMPC_Param
from .dataset_param import Dataset_Param




@dataclass
class Train_Param:
    batch_size: int 
    shuffle: bool
    n_epochs: int
    noise: float
    lr: float
    weight_decay: float
    criterion: nn.modules.loss



@dataclass
class NN_Param: 
    train_dataset : Dataset_Param
    test_dataset : Dataset_Param

    train_param: Train_Param

    version: int = 0
    activation: str = 'tanh'
    n_layers: int = 3
    n_neurons: int = 32
    n_neurons_pruned : int | None = None

    # names
    file : str = field(init=False)
    pruned_file : str = field(init=False)

    def __post_init__(self) -> None:
        # network name
        self._set_new_NN_file()

        # pruned network name
        if self.n_neurons_pruned == self.n_neurons:
            raise ValueError(
                'n_neurons_pruned need to be None, if nothing is pruned, but' + \
                f'{self.n_neurons} == {self.n_neurons_pruned}'
            )
        self._set_new_pruned_NN_file()

    def _set_new_NN_file(self):
        self.file = get_nn_file(
            f'{self.train_dataset.mpc_param.N_MPC}M_{self.train_dataset.version}VD', 
            self.train_dataset.samples, 
            self.train_dataset.version,
            self.train_dataset.mpc_param.acados_name if isinstance(self.train_dataset.mpc_param, AMPC_Param) else None,
            self.n_neurons, 
            self.version
        )
    
    def _set_new_pruned_NN_file(self):
        if self.n_neurons_pruned is not None:
            self.pruned_file = get_pruned_nn_file(self.file, self.n_neurons_pruned)
        else:
            self.pruned_file = None

    def change_N_hidden(self, N_hidden: int):
        self.n_neurons = N_hidden
        if self.n_neurons_pruned == self.n_neurons:
            warnings.warn(
                'param N_hidden_end set to -> None because N_hidden is set to the same as N_hidden_end was before.', 
                UserWarning
            )
            self.n_neurons_pruned = None
        
        self._set_new_NN_file()
        self._set_new_pruned_NN_file()

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
    def _convert_to_type(
        f: Field, 
        param: Any
    ) -> np.ndarray | Any:
        if f.type is np.ndarray:
            return np.array(param)
        elif 'dataset' in f.name:
            return Dataset_Param.from_dict(param)
        elif f.name == 'train_param':
            return Train_Param(**param)
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
    def load(cls, filepath):
        if not os.path.exists(filepath):
            warnings.warn(f'File {filepath} does not exist!',UserWarning)
            return None
        
        with open(filepath,'r') as file:
            pjson = json.load(file)

        return cls.from_dict(pjson)





##############################################################################################################
# EXTRA FUNCTIONS
##############################################################################################################
def get_nn_file(
        param_name: str,
        dataset_samples: int,
        dataset_version: int,
        acados_name: str,
        n_hidden: int | None, 
        network_version: int | None
    ) -> str:
    """
    Creates a standard neural network name out of the given parameters. 
    """
    _v_nn = '' if network_version is None else f'_{network_version}v'
    _n_hidden = '' if n_hidden is None else f'_{n_hidden}Nhid'
    _acados_name = '' if acados_name is None else f'_{acados_name}'
    return f'NN_acados_{param_name}_{dataset_samples}ND_{dataset_version}VD{_acados_name}{_n_hidden}{_v_nn}.ph'


def get_pruned_nn_file(NN_file: str, end_hidden_size: int) -> str:
    """
    Creates a standard pruned neural network name out of the given parameters. 
    """
    setp_strs = NN_file.split('.')

    if len(setp_strs) > 2:
        raise ValueError(f'NN_name has more than one \".\" inside! {NN_file}')
    
    return f'{setp_strs[0]}_prun_{end_hidden_size}Nhid.{setp_strs[1]}'