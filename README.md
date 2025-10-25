# Stein Variational Model Predictive Control (SV-MPC) for Robot Navigation (JAX Implementation)

This repository provides a **starter implementation of Stein Variational Model Predictive Control (SV-MPC)** shown in this [paper](https://homes.cs.washington.edu/~bboots/files/SVMPC.pdf) using **JAX**, applied to a **multi-robot navigation problem** . Robots are initialized at random positions and iteratively update their states using SVGD to reach target locations while avoiding obstacles.  

---

## Features

- **JAX-based implementation** allowing fast, differentiable, and GPU-accelerated computation.  
- **Modular design** separating configuration, kernel definition, log probability, and visualization.  
- **Visualization utilities** for robot trajectories, goal locations, and obstacle fields.  
- **Easily extensible** to different kernels, log-probability functions, or dynamic environments.

## Project Structure

```text
├── config.py                 # Configuration file for simulation parameters
├── helper_functions.py       # Utility functions for visualization and plotting
├── kernels.py                # Median-based kernel implementation for SVGD
├── log_prob.py               # Log-probability (target distribution) definition
├── robot_navigation_sim.py   # Main script: runs the multi-robot navigation simulation
├── svgd.py                   # Core SVGD algorithm implemented in JAX
└── requirements.txt          # Dependencies (JAX, NumPy, Matplotlib, etc.)
```

## Usage
1. **Clone this repository**
2. **Download the dependencies**
3. **run `robot_navigation_sim.py`**
  


