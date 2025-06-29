# %%
import numpy as np
import os, sys
root_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_src_path not in sys.path:
    print('src added: ', root_src_path)
    sys.path.append(root_src_path)

from model_src.model_suh import model_suh
from model_src.ode_modelling.discrete_model import discretize_model
from model_src.ode_modelling.ode_model import get_linearized_matrices, get_model
from model_src.ode_modelling.simresult import plot_sim_results
from model_src.ode_modelling.simulator import sim_nl

#! %matplotlib qt

# %% Load model with standard parameters

p = model_suh.params()
model = get_model(model_suh.model, p)


# %% Simulate continuous model
Ts = 0.01
(u, x0) = model_suh.get_validation_input(Ts)

res = sim_nl(model, Ts, x0, u)
res.desc = f"cont., Ts = {Ts} s"

plot_sim_results(res, signal_infos=model_suh.signal_infos())


# %% Simulate discretized model

Ts = 0.001
dmodel = discretize_model(model, Ts, "rk4")

(u, x0) = model_suh.get_validation_input(Ts)

dres = sim_nl(dmodel, Ts, x0, u)
dres.desc = f"disc., Ts = {Ts} s"

plot_sim_results((dres, res), signal_infos=model_suh.signal_infos())


# %% Have a look at the "eigenvalues"

(A, _, _, _) = get_linearized_matrices(model, x0, u[:, 0])

# (x0, u) may not be a "good" stationary point. In this case the eigenvalues of A
# might not represent the dynamics very well.
print("eigenvalues of A evaluated at (x0, u(0)):", np.linalg.eigvals(A))
