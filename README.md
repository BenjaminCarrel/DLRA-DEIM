# DLRA-DEIM

Numerical experiments for the PRK-DEIM section of our paper

**Interpolatory Dynamical Low-Rank Approximation:
Theoretical Foundations and Algorithms**

*Link to arXiv:* TBD.

*Bibtex (temporary) template for citation:* TBD.

## Abstract

Dynamical low-rank approximation (DLRA) is a widely used paradigm for solving large-scale matrix differential equations, as they arise, for example, from the discretization of time-dependent partial differential equations on tensorized domains. 
Through orthogonally projecting the dynamics onto the tangent space of a low-dimensional manifold, DLRA achieves a significant reduction of the storage required to represent the solution. 
However, the need for evaluating the velocity field can make it challenging to attain a corresponding reduction of
computational cost in the presence of nonlinearities. 
In this work, we address this challenge by replacing orthogonal tangent space projections with oblique, data-sparse projections selected by a discrete empirical interpolation method (DEIM). 
At the continuous-time level, this leads to DLRA-DEIM, a well-posed differential inclusion (in the Filippov sense) that captures the discontinuities induced by changes in the indices selected by DEIM. 
We establish an existence result, exactness property and error bound for DLRA-DEIM that match existing results for DLRA. 
For the particular case of QDEIM, a popular variant of DEIM, we provide an explicit convex-polytope characterization of the differential inclusion. 
Building on DLRA-DEIM, we propose a new class of projected integrators, called PRK-DEIM, that combines explicit Runge–Kutta methods with DEIM-based projections. 
We analyze the convergence order of PRK-DEIM and show that it matches the accuracy of previously proposed projected Runge-Kutta methods, while being significantly cheaper. 
Extensions to exponential Runge–Kutta methods and low-order tensor differential equations demonstrate the versatility of our framework.

## Authors

- Carrel, Benjamin (Paul Scherrer Institute)
- Kressner, Daniel (EPFL)
- Lam, Hysan (EPFL)
- Vandereycken, Bart (University of Geneva)



## Installation

This package requires Python 3.12 or later, and numpy 2.0 or later.

### Clone the repository
```bash
git clone https://github.com/BenjaminCarrel/DLRA-DEIM.git
cd DLRA-DEIM
```

### Install the package

The recommended installation uses a virtual environment. You can use any environment manager (conda, venv, etc.).

#### Using conda (recommended)
```bash
# Create and activate environment
conda create -n dlra-deim python=3.12
conda activate dlra-deim

# For Apple Silicon Macs (optional but recommended for performance)
conda install numpy scipy "libblas=*=*accelerate"

# Install the project in editable mode
pip install -e .
```

#### Using venv
```bash
# Create and activate environment
python -m venv dlra-deim
source dlra-deim/bin/activate  # On Windows: dlra-deim\Scripts\activate

# Install the project in editable mode  
pip install -e .
```

#### Alternative: Manual dependency installation
If you prefer to manage dependencies manually:
```bash
# Install dependencies first
pip install -r requirements-dev.txt

# Then install the project in editable mode
pip install -e . --no-deps
```

This installs the source code in editable mode, allowing you to run experiments while making changes to the code.

### Verify Installation

After installation, run the validation script to ensure everything is working:
```bash
python validate_installation.py
```

This will test all dependencies and core functionality.



## Project Structure and Usage

This is a research project with the following structure:
- `src/`: Source code organized into modules:
  - `matrix_ode_toolbox/`: Matrix ODE solvers and utilities
  - `low_rank_toolbox/`: Low-rank matrix operations
  - `krylov_toolbox/`: Krylov subspace methods
- `experiments/`: Numerical experiments for the paper

### Running Experiments

After installation, navigate to any experiment folder and run the main script:
```bash
cd experiments/allen_cahn_motivation
python main.py
```

Each experiment folder contains:
- `main.py`: Main script that runs the complete experiment
- `shared_parameters.py`: Common parameters and configuration
- `simulation.py`: Generates numerical data
- `data_processing.py`: Processes and analyzes results  
- `plotting.py`: Creates figures and visualization

### Troubleshooting

**Import errors**: Make sure you've activated your virtual environment and installed the package in editable mode with `pip install -e .`

**Performance issues**: On Apple Silicon Macs, consider installing optimized BLAS libraries:
```bash
conda install "libblas=*=*accelerate"
```

**Missing dependencies**: If you encounter missing package errors, try reinstalling:
```bash
pip install -e . --force-reinstall
```


## License

MIT License - see LICENSE file for details

