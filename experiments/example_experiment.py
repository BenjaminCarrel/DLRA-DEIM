"""
Example experiment file.
This file demonstrates how to run an experiment using the dlra_deim package.
"""

import numpy as np
from dlra_deim import __version__

def main():
    print(f"Running example experiment with dlra_deim version {__version__}")
    # Add your experiment code here
    data = np.random.rand(10, 10)
    print(f"Generated random data with shape: {data.shape}")

if __name__ == "__main__":
    main()
