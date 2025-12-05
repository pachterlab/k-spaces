import numpy as np
from sklearn.cluster import MiniBatchKMeans
import copy
import time
import multiprocessing
from joblib import Parallel, delayed
from scipy.special import logsumexp
import warnings

from .affine_subspace_ import affine_subspace, fixed_space, bg_space, check_convergence
from .model_selection_ import total_log_likelihood

################################### INITIALIZATION ############################################
def add_fixed_spaces(k, spaces, fixed_spaces):
    """
    k: int
    spaces: list of affine subspaces
    fixed_spaces: list of fixed_spaces
    
    returns: list of affine subspaces
    """
    for s in fixed_spaces:
        prior_ = 1/k
        translation = s.translation #1 x D list
        vectors = s.vectors # d x D list of lists
        if s.sigma == -1:
            sigma = np.random.uniform(low = .01, high = 1.0)            
            spaces.append(fixed_space(vectors,translation,sigma,[1]*len(translation), prior_))
        else:
            sigma = s.sigma
            spaces.append(bg_space(vectors,translation,sigma,[1]*len(translation), prior_))
    return spaces

def smart_init1(points,kd, D, fixed_spaces = []):
    """
    First initialization is nonrandom.
    Initialize with lines (or subspaces) passing from the origin to the k-means centroids
    
    points: N x D np array
    D: int
    kd: list of ints
    fixed_spaces: a list of dicts [{tr:list, vec:list},...]
    
    returns: list of affine subspaces"""
    k = len(kd) + len(fixed_spaces) #if a kd matrix is passed in instead of k and d arguments
    kmeans = MiniBatchKMeans(n_clusters=sum(kd)+len(kd),
                             random_state=0,
                             max_iter=10,
                             n_init="auto").fit(points)
    centroids = kmeans.cluster_centers_
    spaces = []
    idx = 0
    for d_i in kd: #dimension for this subspace
        translation = centroids[idx] #passes through origin because translation is the vector pointing from origin to the first centroid
        vectors = centroids[idx:idx+d_i]
        vectors = [v if np.any(v) else np.ones(D) for v in vectors]
        
        idx += d_i
        sigma = np.random.uniform(low = 0.2, high = 1.0)
        prior_ = 1/k
        spaces.append(affine_subspace(vectors,translation,sigma,[1]*d_i,prior_))
        
   
    spaces = add_fixed_spaces(k, spaces, fixed_spaces)
    
    
    return spaces

def smart_init2(points,kd, D, fixed_spaces = []):
    """
    nonrandom initialization.
    do k means to partition points, then fit on each of those points to determine initial k-spaces.
    
    points: N x D np array
    D: int
    kd: list of ints
    fixed_spaces: a list of dicts [{tr:list, vec:list},...]
    
    returns: list of affine subspaces"""
    k = len(kd) + len(fixed_spaces) 
    kmeans = MiniBatchKMeans(n_clusters=len(kd),
                             random_state=0,
                             max_iter=10,
                             n_init="auto").fit(points)
    spaces = []
    for i, d_i in enumerate(kd): #dimension for this subspace
        s = fit_single_space(points[kmeans.labels_ == i],d_i, min_variance = 1e-10)
        s.prior = 1/k
        spaces.append(s)
        
    spaces = add_fixed_spaces(k, spaces, fixed_spaces)
    
    return spaces

def init(points,kd, D, fixed_spaces = []):
    """intialize k affine_subspaces with dimension d using random points as translation and basis vectors
    
    points: N x D np array
    D: int
    kd: list of ints
    fixed_spaces: a list of dicts [{tr:list, vec:list},...]
    
    returns: list of affine subspaces"""
    spaces = []

    k = len(kd) + len(fixed_spaces) #if a kd matrix is passed in instead of k and d arguments
    
    for d_i in kd: #dimension for this subspace
        translation = points[np.random.randint(low = 0,high = len(points),size = 1)[0]]
        vectors = [(np.random.uniform(low = -10,high = 10,size = D) - translation) for _ in range(d_i)]
        sigma = np.random.uniform(low = .01, high = 1.0)
        prior_ = 1/k
        spaces.append(affine_subspace(vectors,translation,sigma,[1]*d_i,prior_))
    
    spaces = add_fixed_spaces(k, spaces, fixed_spaces)
    return spaces

def check_inputs(points, kd, max_iter, tol, fixed_spaces, D, init_spaces, randomize_init, min_variance, silent = False):
    """Verify the following:
    1. kd is nonnegative
    2. tol and max_iter is valid
    3. dimensions of fixed spaces and init_spaces match D
    4. if init_spaces includes any fixed spaces, move them to fixed_spaces and readjust priors for init_spaces and fixed_spaces
    5. at least one of kd or init_spaces is nonempty. If both are nonemtpy, init_spaces overrides kd
    6. Count k, the number of fixed and free spaces
    7. Override randomize_init if init_spaces is passed in
    8. Verify min_variance is at least 10% of the range of data in each dimension
    
    
    points: N x D np array
    kd: list of ints
    max_iter: int
    tol: float
    fixed_spaces: a list of dicts [{tr:list, vec:list},...]
    D: int
    init_spaces: list of affine subspaces
    randomize_init: bool
    min_variance: float
    
    
    Returns: (potentially altered) kd, fixed_spaces, init_spaces, randomize_init, and k (not in order)"""
    # verify elements of kd are nonnegative
    for d in kd:
        if d < 0:
            raise ValueError('Failed check. Element in kd is negative')
        
    #verify tolerance and max iterations are valid
    if max_iter < 1 or tol < 0:
        raise ValueError('Failed check. max_iter < 1 or tol < 0')
    
    temp = []
    # verify dimensions match for fixed spaces and initialize
    for s in fixed_spaces:
        if len(s.get("tr")) != D:
            raise ValueError('D does not match ambient dimension (fixed_subspace.D) of a space in fixed_spaces)')
        sigma = -1
        if s.get('sigma') != None:
            sigma = s.get('sigma') #initializes a fixed space. sigma != -1 will indicate to add_fixed_spaces() to convert this to a bg_space
        temp.append(fixed_space(s.get("vec"),s.get("tr"),sigma,[1]*len(s.get('vec')),-1)) #initializing with sigma = 1. may want to change later
    fixed_spaces = temp
    
    # verify dimensions match for init_spaces and move any fixed spaces from init_spaces to fixed_spaces
    if len(init_spaces) > 0:
        #first adjust priors for spaces if an additional fixed space is passed in through fixed_spaces
        if len(fixed_spaces) > 0:
            # rescale priors for init_spaces
            adjustment = len(init_spaces)/(len(init_spaces) + len(fixed_spaces))
            for s in init_spaces:
                s.prior = s.prior*adjustment
            #now set priors for fixed spaces in fixed_spaces
            for s in fixed_spaces:
                s.prior = 1/(len(init_spaces) + len(fixed_spaces))
        
        #now check dimensions of init_spaces and move any fixed spaces from init_spaces to fixed_spaces
        if not silent:
            print('init_spaces passed in. Any fixed spaces in init_spaces will be appended to fixed_spaces. All other init_spaces will be used for only the first initialization')
        for s in init_spaces:
            if s.D != D:
                raise ValueError('D does not match ambient dimension (affine_subspace.D) of a space in init_spaces)')
            if isinstance(s, fixed_space):
                fixed_spaces.append(s)
                init_spaces.remove(s)
                print('fixed space moved from init_spaces to fixed_spaces')
              
    # verify there is at least one free space (non fixed) specified in the model         
    if len(kd) == 0 and len(init_spaces) == 0:
        raise RuntimeError('No free spaces to fit. len(kd) == 0 and len(init_spaces) == 0 after moving fixed spaces to fixed_spaces')
              
    if len(kd) > 0 and len(init_spaces) > 0:
        print('len(kd) > 0 and len(init_spaces) > 0. kd argument will be ignored')
        #override kd by dimensions of init_spaces (for use in initializations after the first one)

    
    if len(init_spaces) > 0: #override if kd is specified, or initialize kd if not
        kd = [] 
        for s in init_spaces:
              kd.append(s.d)

              
    k = len(kd) + len(fixed_spaces) #total number of spaces

    
    if len(init_spaces) > 0: #length may have changed due to above loop
        if not silent:
            print('First initialization will be done with init_spaces. Later inits will be random.')
        if randomize_init == True:
            print('Overriding and setting randomize_init = False because init_spaces passed into args')
            randomize_init = False
    
    #check min variance
    min_vals = np.min(points, axis = 0)
    max_vals = np.max(points, axis = 0)
    if np.any(max_vals - min_vals < 10*min_variance): 
        warnings.warn("Range of data in 1 or more dimensions is less than 10 * min_variance. Consider setting min_variance lower.")
    if min_variance == 0.0:
        warnings.warn("min_variance = 0 allows singular covariance matrices for subspaces and can produce errors in assignment.")
    if min_variance < 0:
        raise ValueError('min_variance cannot be negative')

        
    return kd, k, fixed_spaces, init_spaces, randomize_init

######################################### EM ############################################

def E_step(points, spaces,assignment = 'hard',verbose = False):
    """ caculates "ownership" of points by each space based on the probabilities of those spaces generating those points
    P(space_i | point) = P(point | space_i)*P(space_i)/ sum over k spaces ( P(point | space_j) * P(space_j))
    
    points: N x D np array (or less than N if EM is in batch mode)
    spaces: list of affine subspaces
    assignment: "hard" "closest" or "soft"
    verbose: bool
    
    returns: N x K matrix of probabilities P(space | point)"""
    if verbose:
        print('updated sigma noise:', [round(s.sigma,2) for s in spaces])
     # E-step: Assign probabilities to each point for each space
    probabilities = None
    
    if len(points.shape) == 1:
        points = points.reshape(1,-1) #fixes bug if E_step is run on a single point passed as an array with shape (D,)
    
    if assignment == 'closest':
        #assumes no distribution over latent space       
        distances = np.array([space.orthogonal_distance(points) for space in spaces]).T
        max_indices = np.argmin(distances, axis = 1) #find min distance among spaces for each points
        assignments = np.zeros_like(distances, dtype=float) #set to 1 while changing the others to 0
        rows = np.arange(distances.shape[0]) 
        assignments[rows, max_indices] = 1.0
        probabilities = assignments
        
    elif assignment in ('soft','hard'):
        #probabilistic soft assignment with distributions over latent and complementary space
        
        log_probabilities = np.array([(space.probability(points, log = True)+np.log(space.prior)) for space in spaces]).T
       
        if len(log_probabilities.shape) == 1: #fixes bug when only one point is passed in (axis = 1 below doesn't work otherwise)
            log_probabilities = np.array([log_probabilties])
            
        if assignment == 'hard':
            max_indices = np.argmax(log_probabilities, axis = 1) #find max probability among spaces for each points
            assignments = np.zeros_like(log_probabilities, dtype=float) #set to 1 while changing the others to 0
            rows = np.arange(log_probabilities.shape[0]) 
            assignments[rows, max_indices] = 1.0
            probabilities = assignments    
        
        if assignment == 'soft':
            log_probabilities  -= logsumexp(log_probabilities,axis = 1).reshape(len(log_probabilities),1)
            probabilities = np.exp(log_probabilities)    
            
   
            probabilities[probabilities < 1e-10] = 0.0 #np.cov() threw an error when weight was 1.57e-17 for an entry
            probabilities[probabilities > 1 - 1e-10] = 1.0
        
    if verbose:    
        print('updated ownerships: ',[round(p,2) for p in np.sum(probabilities,axis = 0)])
        print('----')
    return probabilities

def E_step_DA(points, spaces,beta,verbose = False):
    """ Analog to E_step for use in deterministic annealing EM. Do not use this to compute probabilities outside that function.
    
    points: N x D np array (or less than N if EM is in batch mode)
    spaces: list of affine subspaces
    beta: see 
    verbose: bool

    returns: N x K matrix of probabilities P(space | point)"""
    
    if verbose:
        print('updated sigma noise:', [round(s.sigma,2) for s in spaces])
     # E-step: Assign probabilities to each point for each space
    probabilities = None

    log_probabilities = np.array([(space.probability(points, log = True)+np.log(space.prior)) for space in spaces]).T

    if len(log_probabilities.shape) == 1: #fixes bug when only one point is passed in (axis = 1 below doesn't work otherwise)
        log_probabilities = np.array([log_probabilties])
    
    log_probabilities *= beta #deterministic annealing
    
    log_probabilities -= logsumexp(log_probabilities,axis = 1).reshape(len(log_probabilities),1)
    probabilities = np.exp(log_probabilities)    

    probabilities[probabilities < 1e-10] = 0.0 #np.cov() threw an error when weight was 1.57e-17 for an entry
    probabilities[probabilities > 1 - 1e-10] = 1.0

        
        
    if verbose:    
        print('updated ownerships: ',[round(p,2) for p in np.sum(probabilities,axis = 0)])
        print('----')
    return probabilities

def M_step(spaces, points, probabilities, multiprocess_spaces, verbose):
    """ decides whether to multiprocess and then fits spaces
    
    spaces: list of affine subspaces
    points: N x D np array (or less than N if EM is in batch mode)
    probabilities: N x K matrix of probabilities P(point | space) (or less than N if EM is in batch mode)
    multiprocess_spaces: bool
    verbose: bool
    
    returns: list of affine subspaces
    """
        
    if multiprocess_spaces:
        
        #fit each space on a separate core
        args = [(space,points,probabilities[:,i], verbose) for i,space in enumerate(spaces)]

        with multiprocessing.get_context("spawn").Pool() as pool:
            # Use pool.starmap to parallelize the fit_object function
            results = pool.starmap(fit_wrapper, args)
        spaces = [space for space, result in results]

    else:
        #dont fit spaces in parallel (could still use scipy.optimparallel to multiprocess fitting each space)
        spaces = [space.fit(points,probabilities[:,i], verbose) for i,space in enumerate(spaces)]
        
    return spaces

def expectation_maximization_DA(points, spaces, EM_times, max_iter=100, tol=5e-4, verbose = False, silent = False, 
                             print_ownerships = False, batch_size = np.inf, batch_replace = True, multiprocess_spaces = False, 
                             min_variance = 1e-10, num_fixed_spaces = 0, set_noise_equal = False, beta_0 = 0.5, anneal_rate = 1.2):
    """Modified expectation_maximization that uses deterministic annealing. Naonori Ueda and Ryohei Nakano. Deterministic annealing EM algorithm. Neural Networks, 11(2):271–282, March 1998. ISSN 08936080. doi: 10.1016/S0893- 6080(97)00133- 0.
    
    Notes: spaces must be sorted such that fixed spaces are at the end of the list or check_ownerships will be incorrect. This is currently done by the initialization functions in run_EM.
    
    points: N x D np array. 
    spaces: list of affine subspaces.
    EM_times: list of lists (shape: number of intializations, 2).
    max_iter: int.
    tol: float.
    verbose: bool. verbose print messages at each EM step.
    silent: bool. suppresses all print messages.
    print_ownerships: bool.
    batch_size: int (default is np.inf however).
    batch_replace: bool.
    multiprocess_spaces: bool.
    min_variance: float.
    num_fixed_spaces: int.
    set_noise_equal: bool.
    beta_0: default 0.5. Must be between 0 and 1. Inverse to initial annealing "temperature." Lower beta_0 is "hotter"
    anneal_rate: default 1.2. Must be > 1. Factor to cool down temperature by per round (multiplied to beta_0 successively to reach beta = 1).
    
    returns: spaces (list of affine subspaces), probabilities (N x K np array of P(space | point)), flag (int indicating success or failure)
    """
    if beta_0 <= 0:
        raise ValueError('beta_0 must be 0 < beta_0 <= 1.')
    if anneal_rate <= 1:
        raise ValueError('anneal_rate must be > 1.')
    if verbose:
        print('initialized sigmas:',[round(s.sigma,2) for s in spaces])
    if multiprocess_spaces:
        print('Multiprocessing is not optimized has significant overhead. multiprocessing=False may be much much faster.')
    prev = copy.deepcopy(spaces)
    flag = 0
    dims = [s.d for s in spaces]
    beta = beta_0
    stop = False
    while stop == False:
        
        for _ in range(max_iter):

            t0 = time.time()

            points_ = batch(points, batch_size)

            # E-step: Assign probabilities to each point for each space

            probabilities = E_step_DA(points_, spaces, beta, verbose= verbose)

            tE = time.time()

            if check_ownerships(probabilities, dims, num_fixed_spaces) == True: #sklearn pca raises an error if it gets no points (or one point)
                if not silent:
                    print(f'Failed on iteration {_ + 1}: a space was eliminated')
                flag = 1
                break

            update_ownerships(spaces, probabilities)

            # M-step: Update space parameters
            spaces = M_step(spaces, points_, probabilities, multiprocess_spaces, verbose)

            if set_noise_equal == True:
                set_sigmas_eq_noise(points, probabilities, spaces, verbose)

            enforce_min_variance(spaces, min_variance, verbose)
            tM = time.time()


            if check_convergence(prev,spaces, tolerance = tol) == True:
                probabilities = E_step_DA(points, spaces, beta) #note: points not points_
                if check_ownerships(probabilities, dims, num_fixed_spaces) == True: #check final M step
                    print(f'Beta {beta}: Failed on iteration {_ + 1}: a space was eliminated')
                    flag = 1
                    break
                if not silent:
                    print(f'Beta {beta}: Converged on iteration {_ + 1}')
                break
            else:
                prev = copy.deepcopy(spaces) 
            if _ == max_iter -1:
                if not silent:
                    print(f'Beta {beta}: Max iteration {max_iter} completed')

            EM_times.append([tE-t0,tM-tE])
        if beta == 1:
            stop = True
        beta = beta *anneal_rate
        if beta>1:
            beta = 1
    probabilities = E_step(points, spaces, assignment = 'soft')

    if print_ownerships:
        print('Final ownerships: ',[round(p,1) for p in np.sum(probabilities,axis = 0)])
    return spaces, probabilities, flag

def expectation_maximization(points, 
                             spaces, 
                             EM_times,
                             max_iter=100,
                             tol=5e-4, 
                             verbose = False, 
                             silent = False, 
                             assignment = 'hard',
                             print_ownerships = False, 
                             batch_size = np.inf, 
                             batch_replace = True, 
                             multiprocess_spaces = False, 
                             min_variance = 1e-10, 
                             num_fixed_spaces = 0, 
                             set_noise_equal = False):
    """Fit k spaces to a set of points using the EM algorithm.
    E step computes ownerships of points based on the spaces and their noise's standard deviations
    M step fits spaces based on those ownerships and minimizing the RSS
    standard deviations are updated based on what is implied by the M step
    
    Notes: spaces must be sorted such that fixed spaces are at the end of the list or check_ownerships will be incorrect. This is currently done by the initialization functions in run_EM.
    
    points: N x D np array. 
    spaces: list of affine subspaces.
    EM_times: list of lists (shape: number of intializations, 2).
    max_iter: int.
    tol: float.
    verbose: bool.
    multiprocess_spaces: bool.
    assignment: "hard" "closest" or "soft".

    print_ownerships: bool.
    batch_size: int (default is np.inf however).
    batch_replace: bool.
    multiprocess_spaces: bool.
    min_variance: float.
    num_fixed_spaces: int.
    set_noise_equal: bool.
    
    returns: spaces (list of affine subspaces), probabilities (N x K np array of P(space | point)), flag (int indicating success or failure)
    """
    
    if verbose:
        print('initialized sigmas:',[round(s.sigma,2) for s in spaces])
    if multiprocess_spaces:
        print('multiprocessing spaces has significant overhead. consider trying multiprocessing=False first')
    prev = copy.deepcopy(spaces)
    flag = 0
    dims = [s.d for s in spaces]
    for _ in range(max_iter):
        
        t0 = time.time()
        
        points_ = batch(points, batch_size)
        
        # E-step: Assign probabilities to each point for each space
        
        probabilities = E_step(points_, spaces, assignment = assignment, verbose= verbose)
        
        tE = time.time()
        
        if check_ownerships(probabilities, dims, num_fixed_spaces) == True: #sklearn pca raises an error if it gets no points (or one point)
            if not silent:
                print(f'Failed on iteration {_ + 1}: a space was eliminated')
            flag = 1
            break
        
        # M-step: Update space parameters
        update_ownerships(spaces, probabilities) #update space.prior

        spaces = M_step(spaces, points_, probabilities, multiprocess_spaces, verbose)
        
        if set_noise_equal == True:
            set_sigmas_eq_noise(points, probabilities, spaces, verbose)
        
        enforce_min_variance(spaces, min_variance, verbose)
        tM = time.time()
        
            
        if check_convergence(prev,spaces, tolerance = tol) == True:
            probabilities = E_step(points, spaces, assignment = assignment) #note: points not points_
            if check_ownerships(probabilities, dims, num_fixed_spaces) == True: #check final M step
                print(f'Failed on iteration {_ + 1}: a space was eliminated')
                flag = 1
                break
            if not silent:
                print(f'Converged on iteration {_ + 1}')
            break
        else:
            prev = copy.deepcopy(spaces) 
        if _ == max_iter -1:
            if not silent:
                print(f'max iteration {max_iter} completed')
        
        EM_times.append([tE-t0,tM-tE])
        
    probabilities = E_step(points, spaces, assignment = assignment )
    
    if assignment == 'closest':
        set_sigmas_eq_noise(points, probabilities, spaces, verbose) 
        enforce_min_variance(spaces, min_variance, verbose)
        
    if print_ownerships:
        print('final ownerships: ',[round(p,1) for p in np.sum(probabilities,axis = 0)])
    return spaces, probabilities, flag
    
################################### WRAPPER ############################################    
    
def run_EM(points, 
           kd, 
           assignment = 'soft', 
           max_iter=50, 
           tol=5e-2, 
           initializations = 10, 
           verbose = False, 
           silent = False, 
           print_solution = False, 
            randomize_init = False, 
           batch_size = np.inf, 
           batch_replace = True, 
           print_ownerships = False,
           multiprocess_spaces = False, 
           init_spaces = [], 
           fixed_spaces = [], 
           min_variance = 1e-10, 
           return_if_failed = True,
          set_noise_equal = False, 
           DA = False, 
           beta_0 = 0.5, 
           anneal_rate = 1.2,
           max_additional_init = 10):
    """ Runs EM with multiple initializations and selects the maximum likelihood one.
    The first initialization uses kmeans to get centroids and then passes lines through those and the origin.
    
    returns: spaces (list of affine subspaces), probabilities (N x K np array of P(space | point))
    
    kd: 1 x k list containing dimensions (d) for subspaces. i.e. [1,1,1] or [0,2,1]
    assignment: default "hard". Other options: "soft" and "closest".
    fixed spaces: list of dicts {'vec':[basis],'tr':translation} where basis vectors and translation are all lists of length D
    init spaces: list of affine_subspaces (see affine_subspace_.py) to intialize with.
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
    DA: default False. if True, use deterministic annealing EM (Naonori Ueda and Ryohei Nakano. Deterministic annealing EM algorithm. Neural Networks, 11(2):271–282, March 1998.) Will take longer to run. higher beta_0 and higher anneal_rate lead to faster convergence. 
    beta_0: default 0.5. ignored if DA = False. Must be between 0 and 1. Inverse to initial annealing "temperature." Lower beta_0 is "hotter"
    anneal_rate: default 1.2. ignored if DA = False. Must be > 1. Factor to cool down temperature by per round (multiplied to beta_0 successively to reach beta = 1).
    max_additional_init: default 10. maximum number of additional initializations to attempt if all of the requested initializations fail
    """
    init_times, EM_times, models, likelihoods = [],[],[],[]
    D = len(points[0])

    kd, k, fixed_spaces, init_spaces, randomize_init = check_inputs(points, kd, max_iter, tol, fixed_spaces, D, init_spaces, randomize_init, min_variance, silent = silent)
    
    ######### LOOP TO RUN EM ########

    max_init_ = int(max_additional_init)
    i = 0
    while i < initializations:
        t0 = time.time()
        
        ### INITIALIZE ###
        spaces = []
        if i < 2 and np.any(np.array(kd) > 0) and (randomize_init == False):
            if len(init_spaces) >0 and i ==0:
                spaces = init_spaces
                for f in fixed_spaces:
                    spaces.append(f)
            elif i == 0:
                spaces = smart_init1(points,kd,D=D, fixed_spaces = fixed_spaces )
                
            elif i == 1:
                try:
                    spaces = smart_init2(points,kd,D=D, fixed_spaces = fixed_spaces )
                except:
                    print('smart_init2 failed. catching exception and randomly initializing...')
                    spaces = init(points,kd,D=D, fixed_spaces = fixed_spaces)
                
        else:
            spaces = init(points,kd,D=D, fixed_spaces = fixed_spaces)
        
        if DA:
            spaces,probabilities, flag = expectation_maximization_DA(points, 
                                                        spaces,
                                                        EM_times,
                                                        max_iter=max_iter, 
                                                        tol=tol,
                                                        verbose = verbose,
                                                        silent = silent,
                                                        print_ownerships = (print_solution or print_ownerships),
                                                        batch_replace = batch_replace,
                                                        batch_size = batch_size,
                                                        multiprocess_spaces = multiprocess_spaces,
                                                        min_variance = min_variance,
                                                        num_fixed_spaces = len(fixed_spaces),
                                                        set_noise_equal = set_noise_equal,
                                                        beta_0 = beta_0,
                                                        anneal_rate = anneal_rate
                                                                     
                                                        )
        else:
            spaces,probabilities, flag = expectation_maximization(points, 
                                                        spaces,
                                                        EM_times,
                                                        max_iter=max_iter, 
                                                        tol=tol,
                                                        verbose = verbose,
                                                        silent = silent,
                                                        assignment = assignment, 
                                                        print_ownerships = (print_solution or print_ownerships),
                                                        batch_replace = batch_replace,
                                                        batch_size = batch_size,
                                                        multiprocess_spaces = multiprocess_spaces,
                                                        min_variance = min_variance,
                                                        num_fixed_spaces = len(fixed_spaces),
                                                        set_noise_equal = set_noise_equal
                                                        )
        t_EM = time.time()
        
        ### IF EM UNSUCESSFUL ###
        if flag != 0 and len(models) ==0 and i== (initializations -1): #if unsuccessful, all prior inits failed, and it is the last init, try again a few more times
            if max_additional_init > 0:
                randomize_init = True #in case randomize_init = False and initializations = 1
                max_additional_init -= 1
            else:
                if return_if_failed:
                    print(f'Maximum {max_init_} additional attempts failed. EM unsuccessful with k spaces. Returning last attempt')
                    return spaces, probabilities
                else:
                    print(f'Maximum {max_init_} additional attempts failed. EM unsuccessful with k spaces.')
                    return [[],[]]
            
        ### STORE RESULT IF SUCCESSFUL ###
        else:
            i+=1
            if flag == 0:
                models.append((spaces,probabilities))
                model_likelihood = total_log_likelihood(points, spaces, print_solution = print_solution) ###CHANGE BACK
                likelihoods.append(model_likelihood)
        
        init_times.append([t_EM-t0])
    
    if silent == False:
        print('time per EM run:',round(np.mean(init_times),3))
        print('time per E, M step :',np.round(np.mean(EM_times,axis = 0),4))
    
    return models[np.argmax(likelihoods)]


def run_EM_parallelized(points, 
    kd, 
    assignment='soft', 
    max_iter=50, 
    tol=5e-2, 
    initializations=10, 
    n_jobs=None,
    randomize_init=False, 
    batch_size=np.inf, 
    batch_replace=True,  
    init_spaces=[], 
    fixed_spaces=[], 
    min_variance=1e-10, 
    set_noise_equal=False, 
    DA=False, 
    beta_0=0.5, 
    anneal_rate=1.2,
    max_additional_init = 0,
    print_seeds = False):
    
    
    """ Multiprocessing for run_EM with joblib. Each worker calls `run_EM` separately, and any special initializations (i.e., non-random first initialization or init_spaces are passed only to one worker. Each worker is given a separate random seed. Certain arguments for `run_EM` are hard-coded, and the `multiprocess_spaces` argument is hard-coded to False to avoid oversubscription of resources. Memory is NOT shared by jobs, so very large datasets will be copied n_jobs times.


    returns: spaces (list of affine subspaces), probabilities (N x K np array of P(space | point))

    kd: 1 x k list containing dimensions (d) for subspaces. i.e. [1,1,1] or [0,2,1]
    assignment: default "hard". Other options: "soft" and "closest".
    fixed spaces: list of dicts {'vec':[basis],'tr':translation} where basis vectors and translation are all lists of length D
    init spaces: list of affine_subspaces (see affine_subspace_.py) to intialize with.
    max_iter: maximum number of EM iterations
    tol: default 0.05. tolerance for determining EM convergence.
    initializations: default 1. 5-10 is recommended. Number of EM initializations to do. 
    n_jobs: number of parallel processes requested. If None, n_jobs = min(4, initializations)
    batch_size: default is np.inf (no batch; use full dataset) batch size for EM iterations. 
    batch_replace: default is True. Sample with/without replacement if using batches.
    min_variance: default is 1e-10. Minimum variance enforced to prevent singular covariance matrices in "soft" and "hard" assignment mode.
    set_noise_equal: default False. If true, enforces equal sigma_noise for each space after each M step.
    DA: default False. if True, use deterministic annealing EM (Naonori Ueda and Ryohei Nakano. Deterministic annealing EM algorithm. Neural Networks, 11(2):271–282, March 1998.) Will take longer to run. higher beta_0 and higher anneal_rate lead to faster convergence. 
    beta_0: default 0.5. ignored if DA = False. Must be between 0 and 1. Inverse to initial annealing "temperature." Lower beta_0 is "hotter"
    anneal_rate: default 1.2. ignored if DA = False. Must be > 1. Factor to cool down temperature by per round (multiplied to beta_0 successively to reach beta = 1).
    max_additional_init: default 0. (NOTE THIS IS DIFFERENT FROM run_EM). maximum number of additional initializations to attempt for each `run_EM` worker if all of the requested initializations fail
    print_seeds: bool. default False. Whether to print the random seeds given to each worker `run_EM` run.
    """

    # hard-coded arguments for each run_EM call
    return_if_failed = True
    print_ownerships = False
    multiprocess_spaces = False
    verbose = False
    silent = True
    print_solution = False

    # Validate inputs
    kd, k, fixed_spaces, init_spaces, randomize_init = check_inputs(
        points, kd, max_iter, tol, fixed_spaces, points.shape[1], 
        init_spaces, randomize_init, min_variance, silent=silent
    )

    # -----------------------------------------------
    # 1. Decide number of workers
    # -----------------------------------------------
    if n_jobs is None:
        n_jobs = min(4, initializations)

    n_jobs = max(1, min(n_jobs, initializations))
    seeds = []
    # -----------------------------------------------
    # 2. Split initializations across workers
    # -----------------------------------------------
    base = initializations // n_jobs
    extras = initializations % n_jobs

    inits_per_worker = [base + (1 if j < extras else 0) for j in range(n_jobs)]
    print(inits_per_worker)
    # -----------------------------------------------
    # 3. Handle special initializations
    #    → only first worker gets init_spaces / randomize_init = False
    # -----------------------------------------------
    worker_args = []

    for j in range(n_jobs):
        n_inits = inits_per_worker[j]

        if n_inits == 0:
            continue

        # special initializations ONLY for first worker
        if j == 0:
            worker_init_spaces = init_spaces
            worker_randomize_init = randomize_init
        else:
            worker_init_spaces = []
            worker_randomize_init = False  # only random initializations

        seed = np.random.randint(0,1000,size = 1)[0]
        seeds.append(seed)
        
        worker_args.append(dict(
            points=points,
            kd=kd,
            assignment=assignment,
            max_iter=max_iter,
            tol=tol,
            initializations=n_inits,
            randomize_init=worker_randomize_init,
            batch_size=batch_size,
            batch_replace=batch_replace,
            init_spaces=worker_init_spaces,
            fixed_spaces=fixed_spaces,
            min_variance=min_variance,
            set_noise_equal=set_noise_equal,
            DA=DA,
            beta_0=beta_0,
            anneal_rate=anneal_rate,
            return_if_failed=return_if_failed,
            print_ownerships=print_ownerships,
            multiprocess_spaces=multiprocess_spaces,
            verbose=verbose,
            silent=silent,
            print_solution=print_solution,
            seed=seed,
            max_additional_init = max_additional_init
        ))
    if print_seeds:
        print(f'Random seeds for workers: {seeds}')
    # -----------------------------------------------
    # 4. Run workers in parallel via joblib
    # -----------------------------------------------
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_EM_worker_wrapper)(**args)
        for args in worker_args
    )

    # -----------------------------------------------
    # 5. Aggregate results, pick best model
    # -----------------------------------------------
    models = []
    likelihoods = []

    for spaces, probabilities in results:
        models.append((spaces, probabilities))
        ll = total_log_likelihood(points, spaces, print_solution=False)
        likelihoods.append(ll)

    return models[np.argmax(likelihoods)]


def _run_EM_worker_wrapper(seed, **kwargs):
    np.random.seed(seed)
    return run_EM(**kwargs)


############################################ EM HELPERS #################################################################
def batch(points, batch_size):
    """ Subsample points without replacements if a batch size is specified. otherwise return points as points_
    
    returns: min(N,batch_size) x D array of points"""
    points_ = None
    if batch_size < len(points):
        indices = np.random.choice(len(points), size=batch_size, replace=False) 
        points_ = points[indices]
    else:
        points_ = points
    return points_
    
def fit_single_space(points,d, min_variance = 1e-10):
    """ fits a single space with PCA
    points: N x D array
    d: int. dimension of space to fit
    min_variance: float. minimum variance added if variance along a dimension is zero to avoid a singular covariance matrix
    
    returns: affine_subspace"""
    D = len(points[0])
    vectors = [np.random.uniform(low = -10,high = 10,size = D) for i in range(d)]
    translation = points[np.random.randint(low = 0,high = len(points),size = 1)[0]]
    sigma = 1
    space = affine_subspace(vectors,translation,sigma,[1],1)
    space.fit(points,np.ones(len(points)), False)
    enforce_min_variance([space], min_variance, False)
    return space

def update_ownerships(spaces, probabilities):
    """global ownerships for each space (ie the prior for an arbitrary point belonging to a given space)
    
    returns: None
    """
    ownerships = np.sum(probabilities,axis = 0)/len(probabilities) #len(probabilities) = number of points
    for i,s in enumerate(spaces):
        s.prior = ownerships[i]

def check_ownerships(probabilities, dims, num_fixed_spaces):
    """return True if any of the subspaces were eliminated during optimization. else return false
    
    dims: kd extended to include any fixed spaces. [s.d for s in spaces]"""
    probabilities_non_fixed = probabilities
    if num_fixed_spaces > 0:
        probabilities_non_fixed = probabilities[:,:-num_fixed_spaces]
    nonzero_counts = np.count_nonzero(probabilities_non_fixed, axis=0)
    if np.any(nonzero_counts == 0):
        return True
    if np.any(nonzero_counts <= max(dims)):
        for i, s in enumerate(nonzero_counts):
            #if s ==1 and dims[i] != 0: #subspace is not a point but only has (partial) ownership of one point
            if s <= dims[i]: #subspace does not have enough points.
                return True
    return False    
            
    
def fit_wrapper(space, points_, probabilities_, verbose):
    """wrapper for using multiprocessing with affine_subspace.fit"""
    result =  space.fit(points_,probabilities_, verbose)
    return space, result

def set_sigmas_eq_noise(points, probabilities, spaces, verbose):
    """This function computes the correct shared variance and sets the standard deviation of noise for all spaces in the model. It computes expectation over points, spaces, and complementary dimensions for those spaces of squared distance from point to subspace. 
    This is the average variance per discarded dimension.
    
    returns: None"""
    spaces_ = [s for s in spaces if (isinstance(s, bg_space) == False)]
    probabilities_mask = [i for i,s in enumerate(spaces) if (isinstance(s, bg_space) == False)]
    probabilities_ = probabilities[:,probabilities_mask]
    N = np.sum(probabilities_) #expected number of points not attributed to the background
    distances = np.array([space.orthogonal_distance(points) for space in spaces_]).T
    # expectation over points, spaces, and complementary dimensions for those spaces of squared distance from point to subspace
    probability_weighted_sq_distances = np.multiply(probabilities_, distances**2)
    denom = len(points)*np.sum([(s.D-s.d)*s.prior for s in spaces_])
    mean_squared_distance = np.sum(probability_weighted_sq_distances)/denom
    new_std = np.sqrt(mean_squared_distance)
    
    invalid_latent_sigmas = []
    for i, s in enumerate(spaces):
        if s.d == 0 or s.latent_sigmas[-1] >= new_std:
            continue
        for a in np.argwhere(s.latent_sigmas < new_std):
            invalid_latent_sigmas.append([i,a[0], s.latent_sigmas[a[0]]])
            
    if len(invalid_latent_sigmas) > 0:
        new_std, spaces_ = adjust_std_and_update_latent(N, spaces_, np.array(invalid_latent_sigmas), mean_squared_distance)
        
    for s in spaces_: # do not modify preset sigma of background (bg_space)
        s.sigma = new_std
        

def adjust_std_and_update_latent(N, spaces, invalid_latent_sigmas, mean_squared_distance):
    """invalid_latent_sigmas 2D array. Each entry is [space's index, index within space.latent_sigmas, invalid value in space.latent_sigmas]."""
    denom = N*np.sum([(s.D-s.d)*s.prior for s in spaces])
    
    invalid_latent_sigmas = invalid_latent_sigmas[np.argsort(invalid_latent_sigmas[:,2])]
    incorporated_latent_sigmas = []
    for sigma in invalid_latent_sigmas:
        if sigma[2] >= mean_squared_distance: #if previous iteration of loop sufficiently decreased new_std...
            break
        s = spaces[int(sigma[0])]
        new_denom = denom + N*s.prior
        mean_squared_distance = (denom * mean_squared_distance + N*s.prior * sigma[2]**2)/new_denom
        incorporated_latent_sigmas.append(sigma)
        denom = new_denom
    new_std = np.sqrt(mean_squared_distance)
    for sigma in incorporated_latent_sigmas:
        spaces[int(sigma[0])].latent_sigmas[int(sigma[1])] = new_std
    
    return new_std, spaces
    
def enforce_min_variance(spaces, min_variance, verbose):
    """ enforces a minimum variance > 0 to avoid singular covariance matrices. """
    min_std = np.sqrt(min_variance)
    for i, s in enumerate(spaces):
        if s.sigma < min_std:
            if verbose:
                print(f"Out of subspace variance {s.sigma**2} < min_variance for space {i}, setting to minimum variance of {min_variance}")
            s.sigma = min_std
        for a in range(s.d):
            if s.latent_sigmas[a] < min_std:
                if verbose:
                    print(f"Latent variable #{a} variance {s.sigma**2} < min_variance for space {i}, setting to minimum variance of {min_variance}")
                s.latent_sigmas[a] = min_std
    
    
    
    
    
    
    
    