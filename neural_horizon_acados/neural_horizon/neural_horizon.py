import torch,os,warnings
import casadi as cs
import numpy as np
import pandas as pd

from typing import Self
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from ..parameters.nh_mpc_param import NH_AMPC_Param, NH_MPC_Param
from ..neural_horizon.neural_network import FFNN, train_NN

from ..pruning.prun_dataclasses import Prun_Param, Prun_Train_Param
from ..pruning.iter_pruning import iter_prun_nodes
from ..pruning.prun_utils import remove_nodes, remove_pruning_reparamitrizations




class NN_for_casadi():
    """
    A class for building and training a neural network for use with CasADi.

    Attributes
    ----------
    P : MPC_Param dataclass object
        Parameters of the model and the Neural Network
    device : torch.device
        The device on which to run the model.
    dtype : torch.dtype
        The data type of the model.
    labels : list of str
        The names of the columns in the data containing the labels.
    features : list of str
        The names of the columns in the data containing the features.
    states : list of str
        The names of the states of the model.
    inputs : list of str
        The names of the columns in the data containing the inputs.
    scale : dict
        A dictionary containing the means and standard deviations of the features and labels.
        The keys are 'xmean', 'xstd', 'ymean', and 'ystd'.
    trainset : torch.utils.data.TensorDataset
        The dataset used to train the model.
    NN : FFNN
        The neural network model.
    NN_casadi : casadi.Function
        The neural network model converted to a CasADi function.
    """
    def __init__(
            self, 
            P: NH_MPC_Param | NH_AMPC_Param, 
            dataset_dir: str,
            features: list[str], 
            labels: list[str], 
            device = torch.device('cpu'), 
            dtype = torch.float32
        ) -> None:
        """
        Constructor for the `NN_for_casadi` class.

        Parameters
        ----------
        df_file : str
            File path of the csv file containing the dataset.
        P : NH_MPC_Param, optional
            An instance of the `NH_MPC_Param` class containing parameters for the MPC problem, by default NH_MPC_Param().
        features : list of str, optional
            A list of strings containing the names of the columns to be used as features in the neural network, constructed by default from the states.
        labels : list of str, optional
            A list of strings containing the names of the columns to be used as labels in the neural network, constructed by default from the states.
        states : list of str, optional
            A list of strings containing the names of the columns that represent the states of the system, by default ['px', 'theta','v','omega'].
        inputs : list of str, optional
            A list of strings containing the names of the columns that represent the inputs to the system, by default ['u'].
        device : torch.device, optional
            The device on which the neural network will be trained, by default torch.device('cpu').
        dtype : torch.dtype, optional
            The data type for the neural network, by default torch.float32.
        """        
        self.P = P
        self.dataset_dir = dataset_dir
        
        # components of the NN
        self.NN = None
        self.scale = None
        self.NN_casadi = None
        self.prun_param = None
        self.device = torch.device(device)
        if self.device.type=='cpu' and dtype!=torch.float32:
            self.dtype=torch.float32
            warnings.warn(f'Only float32 precision supported for device type cpu, provided dtype={dtype} ignored.',UserWarning)
        else:
            self.dtype = dtype
        self.features = features
        self.labels = labels


    def gen_data(self, df_file: str, do_scaling=True) -> TensorDataset:
        """
        Generate training data for the neural network and scaling parameters.

        Parameters
        ----------
        do_scaling : bool, optional
            Whether to perform scaling on the data, by default True.

        Returns
        -------
        torch.utils.data.TensorDataset
            A tensor dataset containing the training data.
        """
        
        if self.features is None or self.labels is None:
            raise ValueError('Feautres or Labels are None. Must be a list of columns contained in the given DataFrame!')

                
        df = pd.read_csv(os.path.join(self.dataset_dir, df_file))

        tx, ty, self.scale = get_traindata_and_scaling(df, self.features, self.labels, do_scaling)
        return TensorDataset(torch.tensor(tx,device=self.device,dtype=self.dtype), torch.tensor(ty,device=self.device,dtype=self.dtype))
    

    def gen_training_data(self, do_scaling=True) -> TensorDataset:
        return self.gen_data(self.P.nn_param.train_dataset.file, do_scaling=do_scaling)


    def NNprun(self, prun_params: Prun_Param, show_tqdm: bool = False):
        """
        Prunes the pytorch network from the given pruning settings and 
        stores it in ``self.NN``.
        
        Parameters
        ----------
        ``prun_params`` : Prun_Param
            A dataclass that contains the pruning settings.

        ``show_tqdm`` : bool, optional
            A bool determining if the retraining process after pruning should be shown. 
             Default = False
        """
        trainset = self.gen_training_data()
        self.prun_param = prun_params
        pt_param = Prun_Train_Param(self.P.nn_param.train_param, self.prun_param, self.scale)
        pruned_model = iter_prun_nodes(self.NN, pt_param, trainset, show_tqdm=show_tqdm)
        self.NN = pruned_model



    def remove_nodes(self):
        """
        Sets the pruned parameters to zero, remove the prune mask and then remove the 
        parameters or nodes of the pytorch model. After that it stores it in self.NN.
        
        !!Attention!! 
            Only use after NNprun, otherwise no effect.
        """
        self.NN = remove_pruning_reparamitrizations(self.NN)
        self.NN = remove_nodes(self.NN)


    def NNprunCasadi(self, prun_params: Prun_Param, show_tqdm: bool = False):
        """
        Prunes the pytorch network from the given pruning settings. 
        Sets the pruned parameters to zero, remove the prune mask and 
        then remove the parameters or nodes of the pytorch model. 
        After that it calls the function ``transform_pytorch_to_casadi``

        Parameters
        ----------
        ``prun_params`` : Prun_Param
            A dataclass that contains the pruning settings.

        ``show_tqdm`` : bool, optional
            A bool determining if the retraining process after pruning should be shown. 
             Default = False
        """
        self.NNprun(prun_params, show_tqdm)
        self.remove_nodes()
        self.transform_pytorch_to_casadi()


    def evaluate_NN(self, df_file=None) -> tuple[float, float]:
        """
        Evaluate the neural network's accracy using R2-score.

        Parameters
        ----------
        df_file : str, optional
            The file path of the csv file containing the dataset to evaluate the neural network on, by default uses train set.
        """
        if df_file is None:
            df_file = self.P.nn_param.test_dataset.file

        train_x,train_y = self.gen_data(df_file).tensors

        y_scaled = (train_y.float().cpu().numpy() - self.scale['ymean']) / self.scale['ystd']
        with torch.no_grad():
            est_y = self.NN(train_x).float().cpu().numpy()

        r2s = r2_score(y_scaled,est_y)
        rel_err = np.abs((y_scaled - est_y) / self.scale['ystd'])

        print(
            f'''NN evaluation:
            NN: [{self.NN.nin}]->[{self.NN.nout}], {self.NN.n_layers} layer(s), {self.NN.n_neurons} neuron(s) per layer
            R2-score: {r2s:0.4f}
            Relative error: {100*rel_err.mean():0.2f}% mean, {100*rel_err.std():0.2f}% standard deviation'''
        )
        return r2s, rel_err
    


    def transform_pytorch_to_casadi(self, name_input='featues', name_output='predictions'):
        """
        Transform the PyTorch neural network to a CasADi function.

        Parameters
        ----------
        name_input : str, optional
            The name of the input variable for the CasADi function, by default 'features'.
        name_output : str, optional
            The name of the output variable for the CasADi function, by default 'predictions'.

        Returns
        -------
        casadi.Function
            A CasADi function representing the neural network.
        """
        ## Counter for layers and dicts for storage of weights, biases, and activation
        layer_counter, net_weights, net_biases = 1, {}, {}
        ## Get bias and weights in order of layers
        for name, module in self.NN.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, 'weight') and module.weight is not None:
                    net_weights[str(layer_counter)] = module.weight.float().cpu().detach().numpy()
                if hasattr(module, 'bias') and module.bias is not None:
                    net_biases[str(layer_counter)] = module.bias.float().cpu().detach().numpy()
                layer_counter += 1

        ## Define common activation functions
        def apply_act_fun(act,x):
            if act == 'relu':
                return cs.fmax(cs.SX.zeros(x.shape[0]),x)
            elif act == 'sigmoid':
                return 1/(1+cs.exp(-x))
            elif act == 'tanh':
                return cs.tanh(x)
            elif act == 'softplus':
                return cs.log(1+cs.exp(x))
            else:
                raise ValueError(f'Unknown activation function! Supported activations: [relu, sigmoid, tanh, softplus]; received: {act}.')

        ## Reconstruct network with activation
        # first layer is special since input shape is defined by feature size
        input_var = cs.SX.sym('input', net_weights['1'].shape[1])
        xmean = self.NN.xmean.float().cpu().numpy().T
        xstd = self.NN.xstd.float().cpu().numpy().T
        scaled_input_var = cs.if_else(cs.fabs(xstd) < 1e-8, input_var - xmean, (input_var - xmean) / xstd)
        output_var = cs.mtimes(net_weights['1'], scaled_input_var) + net_biases['1']
        
        output_var = apply_act_fun(self.NN.activation,output_var)
        # loop over layers and apply activation except for last one
        for l in range(2, layer_counter):
            output_var = cs.mtimes(net_weights[str(l)], output_var) + net_biases[str(l)]
            if l < layer_counter - 1:
                output_var = apply_act_fun(self.NN.activation,output_var)
        
        # unscale the outputs
        output_var = output_var*self.NN.ystd.float().cpu().numpy().T + self.NN.ymean.float().cpu().numpy().T
        
        self.NN_casadi = cs.Function('nn_casadi_function', [input_var], [output_var], [name_input],[name_output])
        


    def NNcompile(self, show_tqdm=True):
        """Compiles the data for training, runs it, and converts the result to CasADi.
        
        Parameters
        ----------
        ``show_tqdm`` : bool, optional
            Whether to show the progress bar during training (default is True).
        """   
        train_param = self.P.nn_param.train_param
        trainset = self.gen_training_data()
        trainloader = DataLoader(
            trainset, 
            batch_size=train_param.batch_size, 
            shuffle=train_param.shuffle
        )
        self.NN = FFNN(
            self.scale,
            activation=self.P.nn_param.activation,
            n_layers=self.P.nn_param.n_layers,
            n_neurons=self.P.nn_param.n_neurons,
            device=self.device,
            dtype=self.dtype
        )
        self.NN = train_NN(
            self.NN, 
            trainloader = trainloader, 
            criterion = train_param.criterion(), 
            n_epochs = train_param.n_epochs, 
            noise = train_param.noise, 
            lr = train_param.lr, 
            weight_decay = train_param.weight_decay, 
            show_tqdm = show_tqdm
        )
        self.transform_pytorch_to_casadi()
        


    def NNsave(self, file_dir: os.PathLike, use_pruned_file: bool = False) -> None | str:
        """Saves the current neural network to file.
        
        Parameters
        ----------
        file : str, optional
            The name of the file to save the neural network to. If not provided, the filename will be generated based on the neural network
            architecture and other parameters (default is None).
        filedir : str, optional
            The directory to save the file to (default is 'Results//Trained_Networks').
            
        Returns
        -------
        str or None
            The filepath of the saved neural network, or None if the neural network has not been compiled yet.
        """
        
        if self.NN is None:
            # NN wasn't initialized yet
            warnings.warn(f'Neural Network not initialized yet. Call NNcompile method to generate and train the Neural Network first.',UserWarning)
            return None
        
        # construct the file path
        os.makedirs(file_dir, exist_ok=True)
        file_name = self.P.nn_param.pruned_file if use_pruned_file else self.P.nn_param.file
        file_path = os.path.join(file_dir, file_name)    
        
        torch.save({
            'NNscale': self.scale,
            'features': self.features,
            'labels': self.labels,
            'n_layers': self.NN.n_layers,
            'n_neurons': self.NN.n_neurons[1:],
            'MPC_Param': self.P,
            'Prun_Param': self.prun_param,
            'model_state_dict': self.NN.state_dict(),
            'inital_model_state_dict': self.NN.init_state_dict,
            }, file_path)
        return file_path
        


    @classmethod
    def NNload(
        cls, 
        nn_path: str, 
        dataset_dir: str,
        new_nh_mpc_param: NH_MPC_Param | NH_AMPC_Param | None = None,
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32
    ) -> "NN_for_casadi":
        """Loads a trained neural network from file.
        
        Parameters
        ----------
        filepath : str
            The filepath of the file containing the trained neural network and all related parameters.
        kwargs: dict
            key-value pairs that pass through to __init__ of the new class instance. When provided, these
            will supercede the parameters stored in the filepath
        """
        # load a trained Neural Network from file
        loaded_NN = torch.load(nn_path, map_location=device)

        # get parameters for init
        nh_mpc_param: NH_AMPC_Param | NH_MPC_Param = loaded_NN['MPC_Param']
        features = loaded_NN['features']
        labels = loaded_NN['labels']

        # convert to dict and back
        nh_mpc_param_dict = nh_mpc_param.get_json()
        if isinstance(nh_mpc_param, NH_MPC_Param):
            nh_mpc_param = NH_MPC_Param.from_dict(nh_mpc_param_dict)
        if isinstance(nh_mpc_param, NH_AMPC_Param):
            nh_mpc_param = NH_AMPC_Param.from_dict(nh_mpc_param_dict)

        if (new_nh_mpc_param is not None) and not (new_nh_mpc_param == nh_mpc_param):
            warnings.warn(
                f'''Use given Parameters:
                Difference:
                {new_nh_mpc_param.diff(nh_mpc_param)}
                ''', 
                UserWarning
            )
            nh_mpc_param = new_nh_mpc_param
        
        # set all parameters
        NN_out = cls(nh_mpc_param, dataset_dir, features, labels, device, dtype)
        NN_out.scale = loaded_NN['NNscale']
        NN_out.prun_param = loaded_NN['Prun_Param'] if 'Prun_Param' in loaded_NN else None
        NN_out.NN = FFNN(NN_out.scale, activation=NN_out.P.nn_param.activation, n_layers=loaded_NN['n_layers'], n_neurons=loaded_NN['n_neurons'], device=NN_out.device, dtype=NN_out.dtype)
        NN_out.NN.load_state_dict(loaded_NN['model_state_dict'])
        NN_out.NN.init_state_dict = loaded_NN['inital_model_state_dict'] if 'inital_model_state_dict' in loaded_NN else None
        NN_out.transform_pytorch_to_casadi()
        
        # print model stats
        features_str = (
            str(NN_out.features) 
            if len(NN_out.features)<5 
            else f'[{NN_out.features[0]}, {NN_out.features[1]}, ..., {NN_out.features[-2]}, {NN_out.features[-1]}] ({len(NN_out.features)} elements)'
        )
        labels_str = (
            str(NN_out.labels) 
            if len(NN_out.labels)<5 
            else f'[{NN_out.labels[0]}, {NN_out.labels[1]}, ..., {NN_out.labels[-2]}, {NN_out.labels[-1]}] ({len(NN_out.labels)} elements)'
        )
        
        table = {
            "Feature Names": str(features_str),
            "Label Names": str(labels_str),
            "Activation Function": str(NN_out.NN.activation),
            "Input Size": str(NN_out.NN.nin),
            "Number of Hidden Neurons": str(NN_out.NN.n_neurons[1:]),
            "Outer Size": str(NN_out.NN.nout),
        }
        print(f'Model loaded from file: {nn_path}')
        print(f'{"Parameter":<30} | {"Value":<100}')
        print("-"*133)
        for key, value in table.items():
            print(f'{key:<30} | {value:<50}')
        return NN_out
    
    @classmethod
    def load_from_Param(
        cls,
        nh_mpc_params: NH_AMPC_Param, 
        nn_dir: str, 
        dataset_dir: str, 
        device: torch.device = torch.device('cpu'), 
        dtype: torch.dtype = torch.float32,
        force_load_unpruned: bool = False
    ) -> Self:
        nn_path = os.path.join(
            nn_dir, 
            nh_mpc_params.nn_param.file if nh_mpc_params.nn_param.n_neurons_pruned is None or force_load_unpruned else nh_mpc_params.nn_param.pruned_file
        )
        return cls.NNload(nn_path, dataset_dir, new_nh_mpc_param=nh_mpc_params, device=device, dtype=dtype)
    



def get_traindata_and_scaling(df: pd.DataFrame, features: list, labels: list, do_scaling: bool = True):
    """

    Parameters
    ----------

    Returns
    -------
    """
    wrong_labels = [x for x in labels if x not in df.columns]
    if len(wrong_labels) > 0:
        raise Exception(f'Expected labes not found in the dataframe: {wrong_labels}')
    
    tx, ty = df[features].to_numpy(), df[labels].to_numpy()
    nx, ny = len(features), len(labels)
    scale = {'xmean': np.ones(nx), 'xstd': np.ones(nx), 'ymean': np.ones(ny), 'ystd': np.ones(ny)}

    if do_scaling:
        # scaling to be used within NN, so that inputs and outputs remain the same
        sx = StandardScaler()
        sx.fit(tx) # fit scaler to features
        scale['xmean'], scale['xstd'] = sx.mean_, np.sqrt(sx.var_)

        sy = StandardScaler()
        sy.fit(ty)
        scale['ymean'], scale['ystd'] = sy.mean_, np.sqrt(sy.var_)
    return tx, ty, scale