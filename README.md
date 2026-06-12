# Bayesian Inference for Stochastic Volatility Models in Financial Markets

This repository contains the source code for the simulation appendix of my Bachelor's thesis at Bocconi University.

The notebook implements a simulated stochastic volatility model and a component-wise Metropolis--Hastings sampler for recovering the latent log-volatility path, with model parameters fixed at their true data-generating values.

The simulation is intended as an illustrative numerical exercise supporting the theoretical discussion in the thesis, not as a full empirical estimation procedure.

## Contents

- `sv_simulation_appendix.ipynb`: notebook containing the simulation, MCMC sampler, figures, and diagnostics.
- `figures/`: generated figures used in the appendix.
- `output/`: numerical summary outputs from the simulation.