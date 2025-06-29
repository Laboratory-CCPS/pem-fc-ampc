# %%
from model_src.model_suh import model_suh
from model_src.ode_modelling.model_helper import scale_model_signals
from model_src.ode_modelling.ode_model import get_model, scale_model, unscale_model
from model_src.ode_modelling.simresult import plot_sim_results
from model_src.ode_modelling.simulator import sim_nl

# !%matplotlib qt

# %% Load model with standard parameters

p = model_suh.params()
model = get_model(model_suh.model, p)

smodel = scale_model(model)
usmodel = unscale_model(smodel)

# %% Simulate continuous model

Ts = 0.01
(u, x0) = model_suh.get_validation_input(Ts)

res = sim_nl(model, Ts, x0, u)
res.desc = "original"

usres = sim_nl(usmodel, Ts, x0, u)
usres.desc = "unscale(scale(model))"

us = scale_model_signals(model, "u", u)
x0s = scale_model_signals(model, "x", x0)

sres = sim_nl(smodel, Ts, x0s, us)
sres.desc = "scale(model)"

sresu = sres.copy()
sresu.unscale()
sresu.desc = "scale(model) -> unscale"

plot_sim_results((res, usres, sres, sresu), signal_infos=model_suh.signal_infos())

# Normally the results are automatically unscaled when plotted.
# To prevent this auto_unscale can be set to False.
plot_sim_results(sres, signal_infos=model_suh.signal_infos(), auto_unscale=False)
