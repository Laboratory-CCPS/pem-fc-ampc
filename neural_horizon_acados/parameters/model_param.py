import numpy as np
import json
import os
import warnings
import time

from typing import Self, Any
from dataclasses import dataclass, fields, is_dataclass, asdict


@dataclass
class Basic_Model_Param:
    nx: int     # number of states
    nu: int     # number of inputs

    xlabel: list[str]                       # list of state names
    ulabel: list[str]                       # list of input names
    xlabel_latex: list[str]                 # list of state names for latex
    ulabel_latex: list[str]                 # list of input names for latex
    hlabel: list[str] | None                       # list of input names
    hlabel_latex: list[str] | None                 # list of state names for latex
    name: str | None

    def __post_init__(self) -> None:
        if self.nx != len(self.xlabel):
            raise ValueError(f'nx ({self.nx}) and length of xlabel ({len(self.xlabel)}) do not match!')
        
        if self.nu != len(self.ulabel):
            raise ValueError(f'nu ({self.nu}) and length of ulabel ({len(self.ulabel)}) do not match!')

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
        pjson = {}
        for f in fields(self):
            v = getattr(self, f.name)
            pjson[f.name]=v.tolist() if type(v) is np.ndarray else v

        pjson_object = json.dumps(pjson, indent=4)
        if filename is None:
            filename=f'MPC_Param_{time.strftime("%d_%b_%Y_%H_%M")}.json'
        os.makedirs(filedir,exist_ok=True)
        with open(os.path.join(filedir,filename),'w') as file:
            file.write(pjson_object)

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        init_params = {}
        non_init_params = {}

        for f in fields(cls):
            if f.name in params and f.init:
                init_params[f.name] = params[f.name]
            elif f.name in params and not f.init:
                non_init_params[f.name] = params[f.name]
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
    def load(cls, filepath) -> Self:
        if not os.path.exists(filepath):
            warnings.warn(f'File {filepath} does not exist!',UserWarning)
            return None
        
        with open(filepath,'r') as file:
            pjson = json.load(file)

        return cls.from_dict(pjson)