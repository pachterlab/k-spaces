import numpy as np
from .affine_subspace_ import vectors_to_orthonormal_basis

def check_inputs(spaces):
    for s in spaces:
        if s.prior < 0:
            raise ValueError(f'Component weight s.prior {s.prior} < 0')
    D = spaces[0].D
    for s in spaces:
        if s.D != D:
            raise ValueError(f'Ambient data dimension s.D not equal for all spaces')       
            
def generate(spaces_,size = 1, seed = None):
    """generates points given a list of affine_subspaces
    
    spaces_: list of affine_subspaces
    size: number of data points
    """
    if seed is not None:
        np.random.seed(seed)
    spaces = spaces_
    if not isinstance(spaces_, list):
        spaces = [spaces_]
    check_inputs(spaces)
    
    weights = np.array([s.prior for s in spaces])
    weights = weights/np.sum(weights)
    z = np.random.choice(np.arange(len(spaces)),size = size, p = weights)
    X = np.zeros((size, spaces[0].D))
    for i in np.unique(z):
        m = z == i
        s = spaces[i]
        X[m] = s.generate(size = np.sum(m))
    return X 

def generate_points_subspace(translation, vectors, num_points, dist_std = 3, noise_std=0.1):
    """generates points from translation and basis vectors rather than using the affine_subspace objects. Using generate() instead with affine_subspace objects is recommended.
    
    generates normally distributed points in an affine subspace and then perturbs with isotropic normal noise (variance = sqrt(noise_std) in each of the D dimensions). note: isotropic noise is added to latent space as well, so fitting k-spaces will not return latent_sigma equal to dist_std. it will be sqrt(dist_std ^2 + noise_std^2)
    
    translation vector is also the mean
    
    dist_std: standard deviation for normal distribution of coefficients to multiply basis vectors by
    translation: list of length D
    vectors: k x D list of lists
    returns N x D """
    vectors = np.array(vectors)
    translation = np.array(translation)
    
    #Get the dimension of the ambient space
    D = len(translation)
    
    # Convert the basis vectors into a NumPy array
    basis = vectors_to_orthonormal_basis(vectors)
    
    # Get the dimension of the subspace
    subspace_dim = basis.shape[0]
    
    # Generate random coefficients for linear combinations within the subspace
    coefficients = np.random.normal(loc = 0, scale = dist_std, size = num_points*subspace_dim).reshape(num_points,subspace_dim)
    
    # Generate points in the subspace by taking linear combinations of the basis vectors
    points_in_subspace = np.dot(coefficients, basis)
    
    #fix bug for when d = 0 (ie a point) is passed in. Allows function to be called with args = (translation, [], num_points)
    if len(vectors) == 0:
        points_in_subspace = np.zeros(num_points*D).reshape(num_points,D)
        
    # Generate noise vectors orthogonal to the subspace
    isotropic_noise = np.random.normal(loc = 0, scale = noise_std, size = num_points*D).reshape(num_points,D)
    
    # Combine points in the subspace with orthogonal noise
    points_with_noise = points_in_subspace + isotropic_noise
    
    #translate points
    points_with_noise = points_with_noise + translation
    
    return points_with_noise
