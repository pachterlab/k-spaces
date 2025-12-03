# k-spaces

k-spaces fits mixtures of low dimensional Gaussian latent variable models for data modeling, dimension reduction, and subspace learning https://www.biorxiv.org/content/10.1101/2025.11.24.690254v1. Below is a quick overview of the kspaces modules with a rudimentary user guide.

more examples will be uploaded soon but model_fitting_examples has some basic usage examples and documentation.

functions intended for general usage:
|module             | function or class                  | short description                                             |
|-------------------|------------------------------------|---------------------------------------------------------------|
| `EM`              |`run_EM`                            | given data, construct and fit a model                         |
| `EM`              |`fit_single_space`                  | given data, construct and fit a single affine_subspace        |
| `EM`              |`E_step`                            | given a fitted model and some data, perform assignments       |
| `affine_subspace_`|`affine_subspace`                   | class defining an affine subspace                             |
| `affine_subspace_`|`affine_subspace.probability`       | compute P(points &#124; space)                                |
| `affine_subspace_`|`affine_subspace.transform`         | linear dimensionality reduction of points onto space          |
| `affine_subspace_`|`affine_subspace.projection`        | projection of points onto space, still in high dimension      |
| `affine_subspace_`|`fixed_space`                       | class that only updates noise and component weight in EM      |
| `model_selection` |`total_log_likelihood`              | compute the observed log likelihood of the data               |
| `model_selection` |`model_selection`                   | perform model selection using BIC or ICL                      |
| `model_selection` |`get_BIC`                           | compute BIC for a custom model selection pipeline             |
| `model_selection` |`get_ICL`                           | compute ICL for a custom model selection pipeline             |
| `generate`        |`generate`                          | generate synthetic data from a k-spaces model                 |

```bash
vk ref -h
vk count -h
...
```
# Modules
**EM** - contains functions to fit a kspaces model using an EM algorithm. option to use deterministic annealing (Ueda and Nakano 1998) can improve the odds of finding the global maximum and is particularly helpful when many or most initializations with run_EM() fail.

**model_selection** - contains functions for calculating likelihoods and performing model selection with ICL or BIC. BIC suggests a model based on the number of parameters and the observed likelihood of the data. ICL additionally penalizes BIC by the clustering entropy (it discourages selecting models with cluster components with overlapping density).

**affine_subspace_** - implements the affine_subspace class.

**generate** - generates synthetic data from an affine subspace.

**plotting** - visualization functions using matplotlib. view_3D() outputs static views of 3D data from different angles using matplotlib. 

# Documentation for commonly used functions:
All functions have docstrings. Here are some functions intended for general use

    ```python
    def run_EM(points, kd = [], assignment = 'hard', max_iter=50, tol=5e-2, initializations = 1, verbose = False, silent = False, print_solution = False, 
            randomize_init = False, batch_size = np.inf, batch_replace = True, print_ownerships = False,
          multiprocess_spaces = False, init_spaces = [], fixed_spaces = [], min_variance = 1e-10, return_if_failed = True,
          set_noise_equal = False, DA = False, beta_0 = 0.5, anneal_rate = 1.2):
        """ Runs EM with multiple initializations and selects the maximum likelihood one.
        The first initialization uses kmeans to get centroids and then passes lines through those and the origin.
        
        returns: spaces (list of affine subspaces), responsibilties (N x K np.array of P(space | point))
        
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
        """
    def model_selection(points,model,null, print_solution = False, eq_noise = False, test = 'BIC'):
        """Perform model selection with BIC or ICL. ICL penalizes BIC with the entropy of cluster assignments. Accepts a list of affine_subspaces or a single affine_subspace for model and null, but whether a list or single space is passed in, it should be a kspaces model because likelihoods need to be calculated. In other words, if the list is not a full model fit by kspaces, affine_subspace.prior should add up to 1 over the list or should be 1 for a single space.
        
        points: N x D array (observations x features).
        model: list of affine subspaces or single affine subspace.
        null: list of affine subspaces or single affine subspace.  
        eq_noise: bool. should be True if assignment is "closest" or set_noise_equal == True
        test: 'BIC' or 'ICL'. if ICL, assignments will be computed with a soft-assignment E_step as ICL with hard assignment is just BIC.
       
        returns: 'model' or 'null'.
    
        """
    def fit_single_space(points,d, min_variance = 1e-10):
        """ fits a single space with PCA
        points: N x D array
        d: int. dimension of space to fit
        min_variance: float. minimum variance added if variance along a dimension is zero to avoid a singular covariance matrix

        returns: affine_subspace"""
    
    def E_step(points, spaces,assignment = 'hard',verbose = False):
    """ caculates "ownership" of points by each space based on the probabilities of those spaces generating those points
    P(space_i | point) = P(point | space_i)*P(space_i)/ sum over k spaces ( P(point | space_j) * P(space_j))
    
    points: N x D np array (or less than N if EM is in batch mode)
    spaces: list of affine subspaces
    assignment: "hard" "closest" or "soft"
    verbose: bool
    
    returns: N x K matrix of probabilities P(space | point)"""
         
    
    def total_log_likelihood(points, spaces, print_solution= False):
        """Calculate the Gaussian likelihood of the points given the lines using log sum exp.
            
        points: N x D array
        spaces: list of affine subspaces
        print_solution: whether to print the spaces
        
        returns: total log likelihood
        """
         
    def get_BIC(df,num_points,log_likelihood):
        """returns BIC"""
        
    def get_ICL(probs, points, spaces, eq_noise):
        """returns Integrated Completed Likelihood C. Biernacki, G. Celeux, and G. Govaert. Assessing a mixture model for clustering with the integrated completed likelihood. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(7):719–725, July 2000. 
        probs: N x k array of assignment probabilities
        points: N x D array of points
        spaces: list of affine_subspace objects
        eq_noise: True/False, used to determine degrees of freedom. Was eq_noise set to True to fit the model?
        
        returns: ICL """
    
        
    def generate(spaces_,size = 1, seed = None):
        """generates points given a list of affine_subspaces
        
        spaces_: list of affine_subspaces
        size: number of data points
        """
          
    class affine_subspace:
        def __init__(self,vectors, translation, sigma, latent_sigmas, prior):
            """ initializes affine subspace
            vectors: d x D list of lists
            translation: list of length D
            sigma: nonnegative scalar
            latent_sigmas: list of length d
            prior: scalar from 0 to 1
            """
            self.vectors = self.vectors_to_orthonormal_basis(np.array(vectors)) #associated vector subspace is spanned by these basis vectors
            self.translation = np.array(translation) #translation vector for "origin" of the subspace
            self.sigma = sigma #standard deviation of orthogonal noise averaged over dimensions of complementary space
            self.latent_sigmas = np.array(latent_sigmas) #standard deviations for data along each eigenvector of the latent space
            self.D = len(translation) #dimensionality of ambient space
            self.d = len(vectors) #dimensionality of subspace
            self.prior = prior #mixture component weight for this subspace. All subspaces' priors add up to 1
        ```