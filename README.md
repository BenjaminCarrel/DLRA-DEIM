# DLRA-DEIM
Numerical experiments for the paper DLRA-DEIM. (TODO: change for official title)

(Add link to paper and bibtex citation template).

## Abstract

TBD

## Authors

- Carrel, Benjamin (Paul Scherrer Institute)
- Kressner, Daniel (EPFL)
- Lam, Hysan (EPFL)
- Vandereycken, Bart (University of Geneva)



## Installation

This package requires Python 3.12 or later.

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

