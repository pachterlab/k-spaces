import numpy as np
from scipy.stats import entropy
from scipy.special import logsumexp
from .affine_subspace_ import affine_subspace, fixed_space, bg_space

def total_log_likelihood_noLSE(points, spaces, print_solution= False):
    """Calculate the Gaussian likelihood of the points given the lines, without log sum exp trick.
    
    points: N x D array
    spaces: list of affine subspaces
    print_solution: whether to print the spaces
    
    returns: total log likelihood"""
    total_log_likelihood = 0
    if print_solution:
        print('spaces:')
        for s in spaces:
            print(s)
    spaces_ = []
    for s in spaces:
        if s.sigma != 0:
            spaces_.append(s)
  
    probabilities = np.array([s.probability(points) for s in spaces_]).T
    priors = np.array([s.prior for s in spaces_])
    total_log_likelihood = np.sum(np.log(np.sum(probabilities*priors,axis=1))) - len(points)*(spaces[0].D/2)*np.log(2*np.pi)
    
    return total_log_likelihood

def total_log_likelihood(points, spaces, print_solution= False):
    """Calculate the Gaussian likelihood of the points given the lines using log sum exp.
        
    points: N x D array
    spaces: list of affine subspaces
    print_solution: whether to print the spaces
    
    returns: total log likelihood
    """
    total_log_likelihood = 0
    if print_solution:
        print('spaces:')
        for s in spaces:
            print(s)
    spaces_ = []
    for s in spaces:
        if s.sigma != 0:
            spaces_.append(s)
  
    log_probabilities = np.array([s.probability(points, log = True) for s in spaces_]).T # N x k
    log_priors = np.log(np.array([s.prior for s in spaces_]).reshape(1,len(spaces_))) # 1 x k
    total_log_likelihood = np.sum(logsumexp(log_probabilities+ log_priors, axis =1)) - len(points)*(spaces[0].D/2)*np.log(2*np.pi)
    
    return total_log_likelihood

def get_df(spaces,eq_noise):
    """returns degrees of freedom. df larger model - df smaller model"""
    
    deg_spaces = 0
    sigma_df = 1
    if eq_noise:
        sigma_df = 0
        deg_spaces +=1 # 1 df for the whole model
    for s in spaces:
        if isinstance(s, bg_space):
            deg_spaces += 0
        elif isinstance(s, fixed_space):
            deg_spaces += sigma_df 
        else:
            deg_spaces += s.d*s.D - s.d*(s.d+1)/2
            deg_spaces += s.d #for the eigenvalues
            deg_spaces += s.D # translation vector
            deg_spaces += sigma_df 
    return deg_spaces
        
def model_selection(points,model,null, print_solution = False, eq_noise = False, test = 'BIC'):
    """Perform model selection with BIC or ICL. ICL penalizes BIC with the entropy of cluster assignments. Accepts a list of affine_subspaces or a single affine_subspace for model and null, but whether a list or single space is passed in, it should be a kspaces model because likelihoods need to be calculated. In other words, if the list is not a full model fit by kspaces, affine_subspace.prior should add up to 1 over the list or should be 1 for a single space.
    
    points: N x D array (observations x features).
    model: list of affine subspaces or single affine subspace.
    null: list of affine subspaces or single affine subspace.  
    eq_noise: bool. should be True if assignment is "closest" or set_noise_equal == True
    test: 'BIC' or 'ICL'. if ICL, assignments will be computed with a soft-assignment E_step as ICL with hard assignment is just BIC.
   
    returns: 'model' or 'null'.

    """
    valid_tests = ['BIC','ICL']
    if isinstance(model,list) == False:
        model = [model]
    if isinstance(null,list) == False:
        null = [null]
    if test not in valid_tests:
        print(f'Invalid argument for test: "{test}". Valid options are {valid_tests}. Defaulting to "BIC".')
    if test == 'BIC':
        if (get_df(model, eq_noise)-get_df(null, eq_noise)) == 0:
            likelihood_model = total_log_likelihood(points, model, print_solution = print_solution)
            likelihood_null = total_log_likelihood(points, null, print_solution = print_solution)
            if likelihood_model > likelihood_null:
                print('Same degrees of freedom: model has higher likelihood')
                return 'model'
            else:
                print('Same degrees of freedom: model does not have higher likelihood than null')
                return 'null'
        else: #do BIC
            likelihood_model = total_log_likelihood(points, model, print_solution = print_solution)
            likelihood_null = total_log_likelihood(points, null, print_solution = print_solution)
            N = len(points)
            BIC_model = get_BIC(get_df(model,eq_noise),N,likelihood_model)
            BIC_null = get_BIC(get_df(null, eq_noise),N,likelihood_null)
            if BIC_null > BIC_model:
                print('BIC model is lower')
                print(f'{BIC_model} < {BIC_null}')
                return 'model'
            else:
                print('BIC model is not lower')
                return 'null'
    elif test == 'ICL':
        from .EM import E_step

        probs_model = E_step(points, model,assignment = 'soft',verbose = False)
        ICL_model = get_ICL(probs_model, points, model, eq_noise)
        
        probs_null = E_step(points, null, assignment = 'soft',verbose = False)
        ICL_null =  get_ICL(probs_null, points, null, eq_noise)
        
        if ICL_null > ICL_model:
            print('ICL model is lower')
            print(f'{ICL_model} < {ICL_null}')
            return 'model'
        else:
            print('ICL model is not lower')
            return 'null'

    
def get_BIC(df,num_points,log_likelihood):
    """ returns Bayesian Information Criterion
    df: degrees of freedom. Can be obtained with 'get_df'
    num_points: N
    log_likelihood: observed log likelihood of data. Can be obtained with 'total_log_likelihood'
    
    returns: BIC
    """
    return df*np.log(num_points)-2*log_likelihood

def get_entropy(probs):
    return np.sum(entropy(probs,axis = 1))

def get_ICL(probs, points, spaces, eq_noise):
    """returns ICL
    probs: N x k array of assignment probabilities
    points: N x D array of points
    spaces: list of affine_subspace objects
    eq_noise: True/False, used to determine degrees of freedom. Was eq_noise set to True to fit the model?
    
    returns: ICL """
    N = probs.shape[0]
    k = probs.shape[1]
    LL = total_log_likelihood(points, spaces)
    BIC = get_BIC(get_df(spaces, eq_noise = eq_noise), N, LL)
    entropy = get_entropy(probs)
    ICL = BIC + entropy
    return ICL