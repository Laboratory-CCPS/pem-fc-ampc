# PEM Fuel-Cell Model Predictive Control (MPC)

> **Status:** Classical nonlinear MPC only &nbsp;|&nbsp; Neural-Horizon layers **not** included

This repository contains a real-time capable Model Predictive Controller for a
Proton-Exchange-Membrane (PEM) fuel-cell stack.  
It re-uses the plant model, mpc class and evaluation harness from
[Laboratory-CCPS/Neural-Horizon-MPC](https://github.com/Laboratory-CCPS/Neural-Horizon-MPC),
but strips away the Neural-Horizon (NN-augmented) part.

## Features

* **Nonlinear MPC** formulated in CasADi and solved with **acados** for sub-millisecond iterations  
* Export of the generated solver as ANSI-C for embedded deployment  
* Jupyter notebooks for step-by-step design, tuning and closed-loop
  benchmarking  
* Continuous-integration workflow that unit-tests the controller on a
  synthetic drive cycle

## Installation

```bash
git clone https://github.com/Laboratory-CCPS/pem-fc-ampc.git
cd pem-fc-ampc
```

## Example
An example of the code on a working fuel-cell MPC can be found in [Fuel_Cell_AMPC.ipynb](Fuel_Cell_AMPC.ipynb)
A shorter test script can be found under [test_suh_mpc.py](test_suh_mpc.py).

## Acknowledgment
The Code in this repository was produced as part of the KI-Embedded project of the German Federal Ministry of Economic Affairs and Climate Action (BMWK).
The authors and maintainers acknowledge funding of the KI-Embedded project of the BMWK.
