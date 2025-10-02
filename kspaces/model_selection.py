import numpy as np
from scipy.stats import chi2
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
    """returns degrees of freedom for chi squared. df larger model - df smaller model"""
    
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

def check_validity_LRT(model,null):
    """verify that the likelihood ratio test is appropriate (ie the models are nested)
    must be able to drop spaces to match the dimensions of the null model 
    (ie each space in the null must have a separate matched partner in the alternative model)
    
    returns: bool"""
    d_s = np.sort([s.d for s in model])
    d_n = np.sort([n.d for n in null])
    i = 0
    for n in d_n:
        match = False
        for s in d_s[i:]:
            i+=1
            if n == s:
                match = True
                break
        if match == False:
            return False
    return True

        
def model_selection(points,spaces,null, print_solution = False, eq_noise = False, return_statistic = False, try_LRT = True):
    """Perform the likelihood ratio test or BIC.
    
    
    spaces: list of affine subspaces
    null: single line/space (or could be a smaller model). 
    eq_noise: bool. should be True if assignment is "closest" or set_noise_equal == True
   
    returns: p_value if LRT is appropriate. otherwise returns 'model' or 'null' if BIC is appropriate.

    """
    if isinstance(null,list) == False:
        null = [null]
    if (get_df(spaces, eq_noise)-get_df(null, eq_noise)) == 0:
        likelihood_model = total_log_likelihood(points, spaces, print_solution = print_solution)
        likelihood_null = total_log_likelihood(points, null, print_solution = print_solution)
        if likelihood_model > likelihood_null:
            print('model has higher likelihood')
            return 'model'
        else:
            print('model does not have higher likelihood than null')
            return 'null'
    elif check_validity_LRT(spaces,null) == True and try_LRT:    
        likelihood_ratio, p_value = likelihood_ratio_test(points, spaces, null, eq_noise, print_solution)
        if return_statistic:
            return likelihood_ratio, p_value
        return p_value
    else: #do BIC
        likelihood_model = total_log_likelihood(points, spaces, print_solution = print_solution)
        likelihood_null = total_log_likelihood(points, null, print_solution = print_solution)
        N = len(points)
        BIC_model = BIC(get_df(spaces,eq_noise),N,likelihood_model)
        BIC_null = BIC(get_df(null, eq_noise),N,likelihood_null)
        if BIC_null > BIC_model:
            print('BIC model is lower')
            print(f'{BIC_model} < {BIC_null}')
            return 'model'
        else:
            print('BIC model is not lower')
            return 'null'
def likelihood_ratio_test(points, spaces, null, eq_noise, print_solution = False):
    """Perform the likelihood ratio test.
    null: single line/space (or could be a smaller model)
    probabilities: K x N matrix (spaces x points)
    
    returns: test statistic, p value
    """
        
    log_likelihood_model = total_log_likelihood(points, spaces, print_solution = print_solution)
    log_likelihood_null = total_log_likelihood(points, null, print_solution = print_solution)
    
    statistic = -2 * (log_likelihood_null - log_likelihood_model)
    df = get_df(spaces, eq_noise)-get_df(null, eq_noise)
    p_value = chi2.sf(statistic, df=df)
    
    print(f'model log likelihood {round(log_likelihood_model,1)}, null model log likelihood {round(log_likelihood_null,1)}')
    print(f'difference in degrees of freedom: {df}')
    print(f'test statistic: {round(statistic,2)}, p-value {p_value}')
    
    return statistic, p_value
    
def BIC(df,num_points,log_likelihood):
    """returns BIC"""
    return df*np.log(num_points)-2*log_likelihood