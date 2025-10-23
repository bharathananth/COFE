"""Top-level package for Cyclic Ordering with Feature Extraction."""

__author__ = """Bharath Ananthasubramaniam"""
__email__ = 'bharath.ananthasubramaniam@hu-berlin.de'
__version__ = '1.3.0'

__all__ = ["preprocess_data", "cross_validate", "plot_markers", "print_markers",
           "estimate_phase", "plot_diagnostics", "plot_cv_run", "plot_circular_ordering"]


from COFE.analyse import preprocess_data, cross_validate, estimate_phase
from COFE.plot import plot_circular_ordering, plot_cv_run, plot_diagnostics, plot_markers, print_markers