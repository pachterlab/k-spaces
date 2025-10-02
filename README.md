# k-spaces

kspaces code is in the process of being refactored to match sklearn's model fitting function calls. functions are documented well with docstrings but a legitimate user guide has not been written yet. below is a quick overview of the kspaces modules with a rudimentary user guide.

examples will be written soon but intro_figure.ipynb (code for the kspaces paper introductory figure) has some basic usage examples.

functions intended for general usage:
EM.run_EM - fit a model
EM.E_step - given a fitted model and some data, perform assignments
model_selection.total_log_likelihood
model_selection.model_selection
model_selection.BIC
model_selection.ICL - to be uploaded in coming days
plotting.plot_spaces_2D - better version to be uploaded in coming days
plotting.plot_3D - better version to be uploaded in coming days
generate.generate - better version to be uploaded in coming days

**EM** - contains functions to fit a kspaces model using an EM algorithm. option to use deterministic annealing (Ueda and Nakano 1998) can improve the odds of finding the global maximum and is particularly helpful when many or most initializations with run_EM() fail. A buildup option is slow but also very useful for this (will be added in coming days)

**model_selection** - contains functions for calculating likelihoods and performing model selection with ICL, BIC, and Likelihood Ratio Test. The LRT function will be removed prior to release as it is not valid for GMMs due to nonidentifiability. recommendation is to use ICL, which penalizes BIC by the clustering entropy (it discourages selecting models with cluster components with overlapping density). (ICL not in file at the moment, will be added in coming days)

**affine_subspace_** - implements the affine_subspace class.

**generate** - generates synthetic data from an affine subspace. Will be updated soon to generate synthetic data given an affine subspace object or a kspaces model containing affine subspace objects.

**plotting** - visualization functions using matplotlib. Will be updated prior to release. view_3D() outputs static views of 3D data from different angles using matplotlib. A new function using plotly to generate an interactive 3D plot has been written and will be added shortly

docstrings copied and pasted below for now:
    
def run_EM(points, kd = [], assignment = 'hard', objective = 'L2', max_iter=50, tol=5e-2, initializations = 1, verbose = False, silent = False, print_solution = False, 
            randomize_init = False, batch_size = np.inf, batch_replace = True, print_ownerships = False,
          multiprocess_spaces = False, init_spaces = [], fixed_spaces = [], min_variance = 1e-10, return_if_failed = True,
          set_noise_equal = False, DA = False, beta_0 = 0.5, anneal_rate = 1.2):
    """ runs EM with multiple initializations and selects the maximum likelihood one.
    The first initialization uses kmeans to get centroids and then passes lines through those and the origin.
    
    returns: spaces (list of affine subspaces), probabilities (N x K np array of P(point | space))
    
    kd: 1 x k list containing dimensions (d) for subspaces. i.e. [1,1,1] or [0,2,1]
    assignment: default "hard". Other options: "soft" and "closest".
    fixed spaces: list of dicts {'vec':[basis],'tr':translation} where basis vectors and translation are all lists of length D
    init spaces: list of affine_subspaces (see affine_subspace_.py) to intialize with.
    objective: default (strongly recommended) is L2 but TLAD and L1 are options. TLAD minimizes total least absolute distance (as opposed to the sum of squared distances by L2, and L1 minimizes the manhattan distance. TLAD and L1 are slow as currently implemented and not recommended. Probabilistic model selection is not implemented for these and must be done manually.
    max_iter: maximum number of EM iterations
    tol: default 0.05. tolerance for determining EM convergence.
    initializations: default 1. 5-10 is recommended. Number of EM initializations to do. 
    verbose: default False (recommended). Optionally can be set to True to print out information about spaces in each EM iteration as EM converges.
    print_solution: default False. Print out the spaces. You can also print the spaces out with print(space), and the space's principal axes, translation, latent space standard deviations, complementary space noise standard deviation, and total ownership of points (prior) will be displayed.
    multiprocess_spaces = default False. Process each space in parallel in the M step of EM. Useful if fitting many spaces, but if doing many separate kspaces runs (i.e. running kspaces on 100 different pairs of genes) it will be faster to write a wrapper to run kspaces itself in parallel as multiprocessing in python has overhead.
    batch_size: default is np.inf (no batch; use full dataset) batch size for EM iterations. 
    batch_replace: default is True. Sample with/without replacement if using batches.
    min_variance: default is 1e-10. Minimum variance enforced to prevent singular covariance matrices in "soft" and "hard" assignment mode.
    return_if_failed: default True. Returns [spaces, probabilities] for last EM run if True. Returns [[],[]] if False.
    set_noise_equal: default False. If true, enforces equal sigma_noise for each space after each M step.
    """
note: objective will be removed as an argument and 'L2' will be hardcoded prior to release.
def E_step(points, spaces,assignment = 'hard',verbose = False, norm = 'L2'):
    """ caculates "ownership" of points by each space based on the probabilities of those spaces generating those points
    Noise is assumed to be orthogonal to spaces, gaussian, and homoscedastic. The variance is unique to each space.
    P(space_i | point) = P(point | space_i)*P(space_i)/ sum over k spaces ( P(point | space_j) * P(space_j))
    
    points: N x D np array (or less than N if EM is in batch mode)
    spaces: list of affine subspaces
    assignment: "hard" "closest" or "soft"
    verbose: bool
    norm: "L2" "L1" or "TLAD" (Total least absolute distance)
    
    returns: N x K matrix of probabilities P(space | point)"""
    
    
def total_log_likelihood(points, spaces, print_solution= False):
    """Calculate the Gaussian likelihood of the points given the lines using log sum exp.
        
    points: N x D array
    spaces: list of affine subspaces
    print_solution: whether to print the spaces
    
    returns: total log likelihood
    """
    
def BIC(df,num_points,log_likelihood):
    """returns BIC"""
    
