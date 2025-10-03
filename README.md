# DLRA-DEIM
Numerical experiments for the paper DLRA-DEIM

## Installation

This package requires Python 3.12 or later.

### Clone the repository
```bash
git clone https://github.com/BenjaminCarrel/DLRA-DEIM.git
cd DLRA-DEIM
```

### Install the package
```bash
pip install -e .
```

This will install the package in editable mode along with all required dependencies:
- numpy
- scipy
- matplotlib
- dill
- pandas
- tqdm

### Install development dependencies
```bash
pip install -e ".[dev]"
```

This will additionally install:
- pytest

## Project Structure

- `src/dlra_deim/`: Source code for the Python package
- `experiments/`: Experiment scripts that can be run after installation

## Usage

After installation, you can run experiment scripts from the `experiments/` folder:

```bash
python experiments/example_experiment.py
```

You can also import the package in your own scripts:

```python
import dlra_deim
```

## Development

### Running tests
```bash
pytest
```

## License

MIT License - see LICENSE file for details

