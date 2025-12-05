# __init__.py
from .affine_subspace_ import affine_subspace, fixed_space, bg_space, write_spaces, read_spaces, ensure_vector_directions
from .EM import run_EM, run_EM_parallelized, fit_single_space, E_step

__version__ = "0.1.0"
__author__ = "Nicholas Markarian"
__email__ = "nmarkari@usc.edu"