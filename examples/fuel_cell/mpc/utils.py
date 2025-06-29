import numpy as np
import pandas as pd
import os
import casadi as cs

from dataclasses import dataclass


from neural_horizon_acados.parameters.dataset_param import Dataset_Param
from neural_horizon_acados.parameters.nh_mpc_param import NH_AMPC_Param

from model.model_src.model_suh.model_suh import model, params
from model.model_src.ode_modelling.ode_model import get_model


@dataclass
class Scalings:
    factor: np.ndarray
    offset: np.ndarray


def get_scaling() -> dict[str, Scalings]:
    _, scalings = model()
    x_factor = np.array([scalings[name].factor for name in ['p_O2', 'p_N2', 'w_cp', 'p_sm']])
    u_factor = np.array([scalings[name].factor for name in ['v_cm', 'I_st']])
    h_factor = np.array([scalings['lambda_O2'].factor])
    y_factor = np.array([scalings[name].factor for name in ['I_st', 'lambda_O2']])
    y_e_factor = np.array([scalings['lambda_O2'].factor])

    x_offset = np.array([scalings[name].offset for name in ['p_O2', 'p_N2', 'w_cp', 'p_sm']])
    u_offset = np.array([scalings[name].offset for name in ['v_cm', 'I_st']])
    h_offset = np.array([scalings['lambda_O2'].offset])
    y_offset = np.array([scalings[name].offset for name in ['I_st', 'lambda_O2']])
    y_e_offset = np.array([scalings['lambda_O2'].offset])
    return {
        'x': Scalings(x_factor, x_offset),
        'u': Scalings(u_factor, u_offset),
        'h': Scalings(h_factor, h_offset),
        'y': Scalings(y_factor, y_offset),
        'y_e': Scalings(y_e_factor, y_e_offset),
    }


def add_lambda_O2_to(df: pd.DataFrame, dataset_param: Dataset_Param) -> pd.DataFrame:
    assert len(df) == dataset_param.samples, "Mismatch between dataset_param.samples and number of rows in df."

    ode_model = get_model(model, params())
    lambda_O2_func = cs.Function('lambda_O2_func', [ode_model.states, ode_model.inputs], [ode_model.y])

    lambda_O2_dict = {f'lambda_O2_p{j}': np.full(dataset_param.samples, np.nan) for j in range(dataset_param.mpc_param.N_MPC)}
    for i, (_, row) in enumerate(df.iterrows()):
        for j in range(dataset_param.mpc_param.N_MPC):
            x = np.array([row[f'{xl}_p{j}'] for xl in dataset_param.mpc_param.model_param.xlabel])
            u = np.array([row[f'{ul}_p{j}'] for ul in dataset_param.mpc_param.model_param.ulabel])
            lambda_O2_dict[f'lambda_O2_p{j}'][i] = lambda_O2_func(x, u).full().item()
        
    lambda_O2_df = pd.DataFrame.from_dict(lambda_O2_dict)
    return pd.concat((df, lambda_O2_df), axis=1)


def scale_df(df: pd.DataFrame, dataset_param: Dataset_Param) -> pd.DataFrame:
    def scale_values(
            df: pd.DataFrame, 
            labels: list[str], 
            num: int, 
            factor: np.ndarray, 
            offset: np.ndarray,
            str_func = lambda l, j: f'{l}_p{j}'
    ) -> None:  
        for j in range(num):
            col_names = [str_func(l, j) for l in labels]
            if all(col_name in df for col_name in col_names):
                df[col_names] = (df[col_names] - offset) / factor

    scale_values(
        df, 
        dataset_param.mpc_param.model_param.xlabel,
        dataset_param.mpc_param.N_MPC+1,
        dataset_param.mpc_param.x_scale_factor,
        dataset_param.mpc_param.x_scale_offset
    )

    scale_values(
        df, 
        dataset_param.mpc_param.model_param.ulabel,
        dataset_param.mpc_param.N_MPC,
        dataset_param.mpc_param.u_scale_factor,
        dataset_param.mpc_param.u_scale_offset
    )

    scale_values(
        df, 
        dataset_param.mpc_param.model_param.hlabel,
        dataset_param.mpc_param.N_MPC,
        dataset_param.mpc_param.h_scale_factor,
        dataset_param.mpc_param.h_scale_offset
    )

    scale_values(
        df, 
        dataset_param.mpc_param.ylabel,
        1,
        dataset_param.mpc_param.y_scale_factor,
        dataset_param.mpc_param.y_scale_offset,
        str_func=lambda l, _: f'{l}_ref'
    )
    return df


def add_prep2file(file: str) -> str:
    base_name, ext = os.path.splitext(file)
    return f"{base_name}_prep{ext}"


def preprocess_data(dataset_param: Dataset_Param, filedir: os.PathLike):
    file_path = os.path.join(filedir, dataset_param.file)
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = add_lambda_O2_to(df, dataset_param)
    df = scale_df(df, dataset_param)

    scaled_file_name = add_prep2file(dataset_param.file)
    scaled_file_path = os.path.join(filedir, scaled_file_name)
    
    df.to_csv(scaled_file_path, index=False)


def get_labels_and_features_suh(nh_mpc_params: NH_AMPC_Param) -> tuple[list[str], list[str]]:
    if 'begin' == nh_mpc_params.nn_param.train_dataset.begin:
        features = [f'{x}_p{0}' for x in nh_mpc_params.model_param.xlabel] + [f'{x}_ref' for x in nh_mpc_params.ylabel]
        labels = [f'{x}_p{i+1}' for i in range(nh_mpc_params.N_NN) for x in nh_mpc_params.ylabel]
    elif 'fixed' == nh_mpc_params.nn_param.train_dataset.begin:
        features = [f'{x}_p{nh_mpc_params.nn_param.train_dataset.feature}' for x in nh_mpc_params.model_param.xlabel] + [f'{x}_ref' for x in nh_mpc_params.ylabel]
        labels = [f'{x}_p{i+1+nh_mpc_params.nn_param.train_dataset.feature}' for i in range(nh_mpc_params.N_NN) for x in nh_mpc_params.ylabel]
    elif '' == nh_mpc_params.nn_param.train_dataset.begin:
        features = [f'{x}_p{nh_mpc_params.N_MPC}' for x in nh_mpc_params.model_param.xlabel] + [f'{x}_ref' for x in nh_mpc_params.ylabel]
        labels = [f'{x}_p{i+1+nh_mpc_params.N_MPC}' for i in range(nh_mpc_params.N_NN) for x in nh_mpc_params.ylabel]
    else:
        raise ValueError('Unrecognized value for nh_mpc_params.nn_param.train_dataset.begin! Need to be "begin", "fixed" or "".')
    return features, labels