#!/usr/bin/env python3
"""
Installation validation script for DLRA-DEIM
Run this script to verify that the installation is working correctly.
"""

import sys
import importlib

def test_import(module_name, description=""):
    """Test importing a module and return success status."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - Unexpected error: {e}")
        return False

def main():
    """Run installation validation tests."""
    print("DLRA-DEIM Installation Validation")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print()
    
    # Test core packages
    print("Testing core dependencies...")
    core_success = all([
        test_import("numpy", "(numerical computing)"),
        test_import("scipy", "(scientific computing)"),
        test_import("matplotlib", "(plotting)"),
        test_import("pandas", "(data analysis)"),
        test_import("dill", "(serialization)"),
        test_import("tqdm", "(progress bars)"),
        test_import("jupyter", "(notebooks)"),
    ])
    
    print()
    print("Testing DLRA-DEIM modules...")
    
    # Test project modules
    dlra_success = all([
        test_import("low_rank_toolbox", "(low-rank operations)"),
        test_import("matrix_ode_toolbox", "(matrix ODE solvers)"),
        test_import("krylov_toolbox", "(Krylov methods)"),
    ])
    
    print()
    print("Testing key functionality...")
    
    # Test specific functionality
    func_success = True
    try:
        from low_rank_toolbox import SVD
        from matrix_ode_toolbox.integrate import solve_matrix_ivp
        from matrix_ode_toolbox.dlra import solve_dlra
        print("✓ Key functions can be imported")
    except Exception as e:
        print(f"❌ Function imports failed: {e}")
        func_success = False
    
    print()
    print("=" * 40)
    
    if core_success and dlra_success and func_success:
        print("🎉 Installation validation PASSED!")
        print("You can now run experiments with:")
        print("   cd experiments/allen_cahn_motivation")
        print("   python main.py")
        return 0
    else:
        print("❌ Installation validation FAILED!")
        print("Please check the installation instructions in README.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())