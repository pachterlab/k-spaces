# k-spaces

k-spaces fits mixtures of low dimensional Gaussian latent variable models for data modeling, dimension reduction, and subspace learning https://www.biorxiv.org/content/10.1101/2025.11.24.690254v1. It provides a multipurpose toolkit for exploratory data analysis and probabilistic modeling.

More examples will be uploaded soon but model_fitting_examples.ipynb has some basic usage examples and documentation, and notebooks from the paper can be found at https://github.com/pachterlab/MEPSP_2024/tree/main/notebooks_paper, with minor differences in function calls.

functions intended for general usage:
|module             | function or class                  | short description                                              |
|-------------------|------------------------------------|----------------------------------------------------------------|
| `EM`              |`run_EM`                            | given data, construct and fit a model.                         |
| `EM`              |`E_step`                            | given a fitted model and some data, perform assignments.       |
| `EM`              |`fit_single_space`                  | given data, construct and fit a single affine_subspace.        |
| `EM`              |`run_EM_parallelized`               | parallelized version of run_EM using multiprocessing.          |
| `affine_subspace_`|`affine_subspace`                   | class defining an affine subspace with a probability density for data.|
| `affine_subspace_`|`affine_subspace.transform`         | linear dimensionality reduction of points onto space.          |
| `affine_subspace_`|`affine_subspace.projection`        | projection of points onto space, still in high dimension.      |
| `affine_subspace_`|`affine_subspace.probability`       | compute P(points &#124; space).                                |
| `affine_subspace_`|`fixed_space`                       | class that only updates noise and component weight in EM. Inherits from `affine_subspace`.      |
| `affine_subspace_`|`bg_space`                          | class even more fixed than `fixed_space` for modeling a fixed level of background noise and only updates component weight in EM. Inherits from `affine_subspace`.      |
| `affine_subspace_`|`ensure_vector_directions`          | PCA, SVD, and k-spaces vectors are sign indeterminate. Flips basis vectors to point in the positive direction.      |
| `affine_subspace_`|`write_spaces`                      | writes spaces to a csv file.                                   |
| `affine_subspace_`|`read_spaces`                       | reads spaces from a csv file.                                  |
| `model_selection_`|`total_log_likelihood`              | compute the observed log likelihood of the data.               |
| `model_selection_`|`model_selection`                   | perform model selection using BIC or ICL.                      |
| `model_selection_`|`get_BIC`                           | compute BIC for a custom model selection pipeline.             |
| `model_selection_`|`get_ICL`                           | compute ICL for a custom model selection pipeline.             |
| `generate_`       |`generate`                          | generate synthetic data from a k-spaces model.                 |
| `plotting`        |`view_3D_pca_mpl`                   | plots points and the projections of spaces in a 3D PCA space with matplotlib.|
| `plotting`        |`view_3D_pretransformed_mpl`        | plots pretransformed points in a 3D subspace and rescales sizes of points by distance with matplotlib.        |
| `plotting`        |`project_space`                     | projects one space onto another (or onto an sklearn PCA space) |


# Quick start

Given a data matrix of N observations x D features, construct and fit a model with `kspaces.EM.run_EM`.
`kspaces.EM.run_EM` requires specification of $k$, the number of spaces, and $d_i$ for each of the spaces. The argument `kd` accepts a list to specify them. For example, `[1,0,2,3]`, specifies $4$ subspaces of dimensions $d=1,0,2$ and $3$, or a line, a point, a plane, and a rectangular prism.

`kspaces.EM.run_EM` does not require specifying `assignment = 'soft'` or `assignment = 'hard'` but this choice matters. Soft assignment allows points to be probabilistically 'shared' by subspaces, as described in the paper, while hard assignment is all-or-nothing. In some contexts, hard assignment will be faster as it uses singular value decomposition. However, soft assignment has smoother convergence. `assignment = 'closest'` is a form of hard assignment that does not use a probabilistic model and just assigns points to their closest subspaces. It is faster but generally not recommended.

`kspaces.EM.run_EM` does not require specifying `initializations` or `max_iter` but depending on how hard it is to find the globally optimal solution, more initializations than the default of `10` would be needed, and decreasing the tolerance, `tol`, and increasing the number of allowed iterations per EM run, `max_iter` will help. 

Here is a basic example:

```python
spaces, responsibilities = kspaces.EM.run_EM(data, [1,1,2] assignment = 'soft', initializations = 50)
```
k-spaces fits two lines and a plane to the data, returns those `affine_subspace` objects in the list `spaces` and also returns a `N` x `k` array of probabilities that each subspace 'generated' that point with the fitted probability distributions. The `affine_subspace`s of `spaces` and the columns of `responsibilities` are in the same order as the spaces were specified in `kd`.

To use these spaces as dimension reductions for our data, we can use the `transform` function. While each space was optimized for a subset of the data, the function for the dimension reduction is defined everywhere, so we can project all the points onto any of the spaces for a dimension reduction if we want. Let's use the third space, the plane, so we can get 2-D dimension-reduced data:

```python
reduced_dim_data = spaces[2].transform(data)
```
We can visualize this $N$ x $d$ data with plotting libraries like matplotlib, plotly, bokeh, seaborn, etc. We can make our own or use two built-in functions from `k-spaces.plotting` for high dimensional data visualized in 3-D dimension reductions. `view_3D_pca_mpl` and `view_3D_pretransformed_mpl` both use matplotlib. `view_3D_pca_mpl` computes a dimension reduction for the data using 3D PCA and can optionally project our fitted subspaces into the plot too. `view_3D_pretransformed_mpl` assumes we have already transformed the data using one of our subspaces (as above, although we'd need d = 3) and want to visualize it.

If we want to project the points onto their subspaces in the high dimensional space, we use `project` instead.

```python
projected_data = spaces[2].project(data)
```

This data lies on a $d$-dimensional subspace of the $D$-dimensional space, but it is still located in the original feature space and is a $N$ x $D$ matrix.

We can save this with `write_spaces` and come back later to read it back in with `read_spaces`. The assignment matrix is not saved, only the model parameters, but we can use `E_step` to assign the points.

```python
kspaces.affine_subspace_.write_spaces(spaces, 'myfile.csv')
spaces = kspaces.affine_subspace_.read_spaces('myfile.csv')
probabilities = kspaces.EM.E_step(points, spaces, assignment = 'soft')
```
# Modules
**EM** - contains functions to fit a kspaces model using an EM algorithm. The option to use deterministic annealing (Ueda and Nakano 1998) can improve the odds of finding the global maximum and is particularly helpful when many or most initializations with `run_EM` fail.

**model_selection** - contains functions for calculating likelihoods and performing model selection with ICL or BIC. BIC suggests a model based on the number of parameters and the observed likelihood of the data. ICL additionally penalizes BIC by the clustering entropy (it discourages selecting models with cluster components with overlapping density).

**affine_subspace_** - implements the affine_subspace class.

**generate** - generates synthetic data from an affine subspace.

**plotting** - visualization functions using matplotlib. `view_3D` outputs static views of 3D data from different angles using matplotlib. 

# Documentation for commonly used functions:
All functions have docstrings that can be displayed in a notebook with <kbd>Shift</kbd> + <kbd>Tab</kbd>. Here are some functions intended for general use:

```python
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
       anneal_rate = 1.2):
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
    verbose: default False (recommended). Optionally can be set to True to print out 
        information about spaces in each EM iteration as EM converges.
    print_solution: default False. Print out the spaces. You can also print the spaces out with print(space), 
        and the space's principal axes, translation, latent space standard deviations, complementary space 
            noise standard deviation, and total ownership of points (prior) will be displayed.
    multiprocess_spaces = default False. Process each space in parallel in the M step of EM. Useful if fitting 
        many spaces, but if doing many separate kspaces runs (i.e. running kspaces on 100 different pairs of genes) 
        it will be faster to write a wrapper to run kspaces itself in parallel as multiprocessing in python has overhead.
    batch_size: default is np.inf (no batch; use full dataset) batch size for EM iterations. 
    batch_replace: default is True. Sample with/without replacement if using batches.
    min_variance: default is 1e-10. Minimum variance enforced to prevent singular covariance matrices in 
        "soft" and "hard" assignment mode.
    return_if_failed: default True. Returns [spaces, probabilities] for last EM run if True. Returns [[],[]] if False.
    set_noise_equal: default False. If true, enforces equal sigma_noise for each space after each M step.
    DA: default False. if True, use deterministic annealing EM (Naonori Ueda and Ryohei Nakano. 
        Deterministic annealing EM algorithm. Neural Networks, 11(2):271–282, March 1998.) Will take longer to run.
        higher beta_0 and higher anneal_rate lead to faster convergence. 
    beta_0: default 0.5. ignored if DA = False. Must be between 0 and 1. Inverse to initial annealing 
        "temperature." Lower beta_0 is "hotter"
    anneal_rate: default 1.2. ignored if DA = False. Must be > 1. Factor to cool down temperature 
        per round (multiplied to beta_0 successively to reach beta = 1).
    """
```

```python    
def E_step(points, spaces,assignment = 'hard',verbose = False):
    """ caculates "ownership" of points by each space based on the probabilities of those spaces generating those points
    P(space_i | point) = P(point | space_i)*P(space_i)/ sum over k spaces ( P(point | space_j) * P(space_j))
    
    points: N x D np array (or less than N if EM is in batch mode)
    spaces: list of affine subspaces
    assignment: "hard" "closest" or "soft"
    verbose: bool
    
    returns: N x K matrix of probabilities P(space | point)"""
         
 ```
```python
def fit_single_space(points,d, min_variance = 1e-10):
    """ fits a single space with PCA
    points: N x D array
    d: int. dimension of space to fit
    min_variance: float. minimum variance added if variance along a dimension is zero to avoid a singular covariance matrix

    returns: affine_subspace"""
```
```python
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
```
```python          
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
```python  
def transform(self, points):
    """Alias for self.displacement to match the call for sklearn's pca.transform.

    points: N x D array

    returns: N x d array"""
    return self.displacement(points)
```
```python
def projection(self,points):
    """project point onto subspace. The subspace is low dimensional but the points are still in high dimensional space.

    points: N x D array

    returns: N x D array"""
```
```python
def probability(self, points, log=False, true_val = False):
    """By default returns a value proportional to P(point | self) and ignores 1/sqrt(2 pi) term in normal pdf but can be made exact by multiplying result by 1/(2 pi)^ D/2 with true_val = True.

    points: N x D array
    log: bool. default False. Whether to return log probability or probability
    true_val: bool. default False. If True, multiply the constant 1/(2 pi)^ D/2 to get the exact probability rather than a proportional value.

    returns: (log) probability"""
```
```python
def model_selection(points,model,null, print_solution = False, eq_noise = False, test = 'BIC'):
    """Perform model selection with BIC or ICL. ICL penalizes BIC with the entropy of cluster assignments. Accepts a list of affine_subspaces or a single affine_subspace for model and null, but whether a list or single space is passed in, it should be a kspaces model because likelihoods need to be calculated. In other words, if the list is not a full model fit by kspaces, affine_subspace.prior should add up to 1 over the list or should be 1 for a single space.

    points: N x D array.
    model: list of affine subspaces or single affine subspace.
    null: list of affine subspaces or single affine subspace.  
    eq_noise: bool. should be True if assignment is "closest" or set_noise_equal == True
    test: 'BIC' or 'ICL'. default 'BIC'. if ICL, assignments will be computed with a soft-assignment E_step as ICL with hard assignment is just BIC.

    returns: 'model' or 'null'.

    """
```
```python
def ensure_vector_directions(spaces, inplace = True):
    """checks direction of each basis vector and flips them (reverses sign) as needed to point in a positive direction.
    Specifically, checks whether dot product of each basis vector with <1,1,1,...,1> is positive. Flips if negative.
    spaces: list of subspace objects
    inplace: bool. default True. Whether to modify the spaces in place or not

    returns: sorted spaces
    """
```
```python
def write_spaces(spaces, file):
    """writes spaces to a csv file
    spaces: list of subspace objects
    file: filename to write to"""
```
```python
def write_spaces(spaces, file):
    """writes spaces to a csv file
    spaces: list of subspace objects
    file: filename to write to"""
```
```python
def read_spaces(file):
    """reads in a file written by 'write_spaces'
    
    file: filename
    returns: list of subspace objects"""
```

```python   
def total_log_likelihood(points, spaces, print_solution= False):
    """Calculate the Gaussian likelihood of the points given the lines using log sum exp.

    points: N x D array
    spaces: list of affine subspaces
    print_solution: whether to print the spaces

    returns: total log likelihood
    """
```
```python         
def get_BIC(df,num_points,log_likelihood):
    """ returns Bayesian Information Criterion
    df: degrees of freedom. Can be obtained with 'get_df'
    num_points: N
    log_likelihood: observed log likelihood of data. Can be obtained with 'total_log_likelihood'
    
    returns: BIC
    """
```
```python        
def get_ICL(probs, points, spaces, eq_noise):
    """returns Integrated Completed Likelihood C. Biernacki, G. Celeux, and G. Govaert. Assessing a mixture model for clustering with the integrated completed likelihood. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(7):719–725, July 2000. 
    probs: N x k array of assignment probabilities
    points: N x D array of points
    spaces: list of affine_subspace objects
    eq_noise: True/False, used to determine degrees of freedom. Was eq_noise set to True to fit the model?

    returns: ICL """
    
```
```python        
def generate(spaces,size = 1, seed = None):
    """generates points given a list of affine_subspaces

    spaces: list of affine_subspaces (or a single affine subspace)
    size: default 1. number of data points
    seed: default None. random seed for numpy

    returns size x D array of points, where D is specified by the spaces.
    """
```

```python
def view_3D_pca_mpl(points, 
         aspect = 'equal', 
         subtypes = [],
         title = '', 
         axis_labels = ['PC 1', 'PC 2', 'PC 3'],
         color_dict = {}, 
         cmap_dict = {}, 
         s_points = 2, 
         alpha_points = 1, 
         elevation = [70,0], 
         azimuth = [20,90,160], 
         legend = True, 
         legend_loc = 'upper left',
         save = '', 
         show = True, 
         markerscale = 5,
         fontsize = 10,
         return_figure = False):
    """**will apply PCA to data**
    
    Plots points together in 3D PCA space with the projections of spaces (if space.d <= 3). 
    Optionally colors points by subtype if the subtypes argument and either color_dict or cmap_dict arguments are used.
    
    points: N x D array 
    aspect: passed to ax.set_aspect 
    subtypes: length N list or array of subtype labels for points
    title: passed to fig.suptitle 
    axis_labels: default ['basis 1', 'basis 2', 'basis 3']
    color_dict: dict. key, val pair is subtype_name:color. Overriden by cmap dict. Default of {} will plot black, while an 
     incomplete dictionary will color non included subtypes gray.
    cmap_dict: dict. key, val pair is subtype_name:(cmap_name,values_for_cmap)} Default of {} will plot black. 
    alpha_points: passed to ax.scatter as alpha
    elevation = list that sets viewing elevation(s). Elements passed to ax.view_init as elev
    azimuth = list that sets viewing azimuth(s). Elements passed to ax.view_init as azim
    legend = whether to plot legend. 
    legend_loc = passed to ax.legend.
    save = filename to save plot to. default '' does not save. 
    show = if True, triggers plt.show() 
    markerscale = passed to ax.legend
    fontsize = passed to ax.legend, fig.suptitle, axis labels
    return_figure: if True, return fig, axs"""
 ```
```python
def view_3D_pretransformed_mpl(points, 
             aspect = 'equal', 
             subtypes = [],
             title = '', 
             axis_labels = ['Basis 1', 'Basis 2', 'Basis 3'],
             color_dict = {}, 
             cmap_dict = {}, 
             s_points = 2, 
             scale_by_distance = 1,
             distances = None, 
             alpha_points = 1, 
             elevation = [70,0], 
             azimuth = [20,90,160], 
             legend = True, 
             legend_loc = 'upper left',
             save = '', 
             show = True, 
             markerscale = 5,
             fontsize = 10,
             return_figure = False):
    """**assumes pre-transformed data into a 3D space**
    
     Plots points in pretransformed 3D space with matplotlib. affine_subspace.transform can be used to do the dimension reduction. 
     Optionally rescales points to shrink with distance according to s_points /  ( log1p(distances[i]) ^ scale_by_distance ). 
         affine_subspace.orthogonal_distance can be used to obtain the distances.
     Optionally colors points by subtype if the subtypes argument and either color_dict or cmap_dict arguments are used.
     
     points: N x 3 array 
     aspect: passed to ax.set_aspect 
     subtypes: length N list or array of subtype labels for points
     title: passed to fig.suptitle 
     axis_labels: default ['basis 1', 'basis 2', 'basis 3']
     color_dict: dict. key, val pair is subtype_name:color. Overriden by cmap dict. Default of {} will plot black, while an 
         incomplete dictionary will color non included subtypes gray.
     cmap_dict: dict. key, val pair is subtype_name:(cmap_name,values_for_cmap)} Default of {} will plot black. 
     s_points: default 2. size for points before rescaling by distance. 
     scale_by_distance: default 1 means point size shrinks proportionally to logarithmic distance from projection. 
         size for point_i is s_points /  ( log1p(distances[i]) ^ scale_by_distance ) 
     distances: If None, all points will be plotted the same size. Else, distances is used to rescale points to reflect distance from the projection.
     alpha_points: passed to ax.scatter as alpha
     elevation = list that sets viewing elevation(s). Elements passed to ax.view_init as elev
     azimuth = list that sets viewing azimuth(s). Elements passed to ax.view_init as azim
     legend = whether to plot legend. 
     legend_loc = passed to ax.legend.
     save = filename to save plot to. default '' does not save. 
     show = if True, triggers plt.show() 
     markerscale = passed to ax.legend
     fontsize = passed to ax.legend, fig.suptitle, axis labels
     return_figure: if True, return fig, axs):
    
    """
 ```
```python
def project_space(s,vis):
    """projects k-spaces subspace into the 3D subspace of PCA or a 3D k-spaces subspace
    s: a k-spaces subspace object
    vis: a k-spaces subspace object or a sklearn.decomposition.PCA instance
    
    returns: affine_subspace object for the projection of 's' onto 'vis'"""
 ```
