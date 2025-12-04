import numpy as np
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans
from scipy.linalg import orth
import scipy.linalg
import sklearn
import copy
import csv


def extend_basis(Q, target_dim): #chatgpt... double check it
    """Helper function for initialization that adds (a) linearly dependent basis vector(s) to ensure that the shape of affine_subspace.vectors matches affine_subspace.d. 
    In rare cases, it is possible for the randomly initialized basis vectors to be linearly dependent,in which case scipy.linalg.orth will 
    return an array for the orthonormal basis with a different shape. This can cause a runtime error during matrix multiplication in the first E step."""
    Q_extended = Q.copy()
    d = Q.shape[1]
    counter = 0
    while Q_extended.shape[0] < target_dim:
        v = np.random.randn(d)
        # Project out the components in Q
        v_proj = v - Q_extended.T @ (Q_extended @ v)
        norm = np.linalg.norm(v_proj)
        if norm < 1e-10:
            continue  # Linearly dependent; try again
        v_proj /= norm
        Q_extended = np.vstack([Q_extended, v_proj])
        counter += 1
        if counter == 100:
            raise RuntimeError('breaking potentially infinite while loop... check extend_basis function in affine_subspace_.py')

    return Q_extended
    
class affine_subspace:
    def __init__(self,vectors, translation, sigma, latent_sigmas, prior):
        """ initializes affine subspace
        vectors: d x D list of lists
        translation: list of length D
        sigma: nonnegative scalar
        latent_sigmas: list of length d
        prior: scalar from 0 to 1
        """
        self.vectors = self.vectors_to_orthonormal_basis(np.array(vectors)) #associated vector subspace is spanned by self.vectors
        self.translation = np.array(translation) #translation vector for "origin"
        self.sigma = sigma #standard deviation of orthogonal noise averaged over dimensions of complementary space
        self.latent_sigmas = np.array(latent_sigmas) #standard deviations for data along each eigenvector of the latent space
        self.D = len(translation) #dimensionality of ambient space
        self.d = len(vectors) #dimensionality of subspace
        self.prior = prior
    
    def __str__(self):
        return f'affine subspace: \n basis: \n {np.round(self.vectors,3)} \n basis std. devs: \n {np.round(self.latent_sigmas,3)} \n translation: {np.round(self.translation,3)} \n sigma: {np.round(self.sigma,3)} \n prior: {np.round(self.prior,3)}'

    def copy(self):
        return copy.deepcopy(self)
    
   
    def vectors_to_orthonormal_basis(self, vectors):
        """given k vectors of dimension D spanning a vector subspace, return an orthonormal basis
        vectors: k x D array
        returns: k x D array"""
        if len(vectors) == 0:
            return np.array([])
        elif len(vectors) == 1: #np.linalg.orth expects a matrix not a single vector
            return (vectors/np.linalg.norm(vectors))
        else: 
            if np.allclose(np.linalg.norm(vectors, axis  = 1),1.0): #already an orthonormal basis constructed from space.fit
                return vectors 
            else: #np.linalg.orth finds orthonormal basis for the span of the column vectors (returns D x k matrix)
                basis = orth(vectors.T).T
                if basis.shape[0] != len(vectors):
                    basis = extend_basis(basis, len(vectors))
                return basis
    
    def update_vectors(self, vectors):
        """computes orthonormal basis from vectors and updates self.vectors
        vectors: d x D array"""
        basis = self.vectors_to_orthonormal_basis(vectors)
        self.vectors = basis
            
    def orthogonal_distance(self,points, squared = False):
        """point should be NxD array
        
        if squared == True, returns squared orthogonal distance"""
        projections  = self.projection(points)
        if squared == False:
            if len(points.shape) ==1: #single point
                return np.linalg.norm(points-projections)
            else:
                return np.linalg.norm(points-projections,axis = 1)
        if squared == True:
            if len(points.shape) ==1: #single point
                return np.sum((points-projections)**2)
            else:
                return np.sum((points-projections)**2, axis=1)

       
    
    def projection(self,points):
        """project point onto subspace. The subspace is low dimensional but the points are still in high dimensional space.
        
        points: N x D array
        
        returns: N x D array"""
        if self.d == 0:
            return self.translation
        else:
            diff = points - self.translation  # NxD
            projections = diff @ self.vectors.T @ self.vectors + self.translation
            return projections
    
   
    def transform(self, points):
        """Alias for self.displacement to match the call for sklearn's pca.transform.
        
        points: N x D array
        
        returns: N x d array"""
        return self.displacement(points)
    def displacement(self, points):
        """project point onto space and determine position in coordinate system defined by basis vectors and affine translation (the mean)"""
        if self.d == 0:
            raise RuntimeError('Displacement within latent space is undefined for 0-D space')
        
        return np.dot(points - self.translation, self.vectors.T)
   
        
    def probability(self, points, log=False, true_val = False):
        """By default returns a value proportional to P(point | self) and ignores 1/sqrt(2 pi) term in normal pdf but can be made exact by multiplying result by 1/(2 pi)^ D/2 with true_val = True.
        
        points: N x D array
        log: bool. default False. Whether to return log probability or probability
        true_val: bool. default False. If True, multiply the constant 1/(2 pi)^ D/2 to get the exact probability rather than a proportional value.
        
        returns: (log) probability"""
       
    
        N = points.shape[0]
        diff = points - self.translation  # NxD

        # Complementary space log-prob
        log_probs = -(self.D - self.d) * np.log(self.sigma) * np.ones(N, dtype=np.float64)

        if self.d > 0:
            # Precompute latent coordinates once
            coords = diff @ self.vectors.T  # NxD latent coordinates
            # Project back to space: proj = coords @ self.vectors
            proj = coords @ self.vectors    # NxD
        else:
            coords = None
            proj = np.zeros_like(diff)

        # Complementary space squared distance (orthogonal distance)
        orth_sq = np.sum((diff - proj)**2, axis=1)
        log_probs += -0.5 * orth_sq / self.sigma**2

        if self.d != 0:
            # Latent space contribution
            log_probs -= np.sum(np.log(self.latent_sigmas))
            log_probs -= 0.5 * np.sum((coords / self.latent_sigmas)**2, axis=1)
        if true_val:
            log_probs *= 1/(2*np.pi)**(self.D/2)
        if log:
            return log_probs
        else:
            return np.exp(log_probs)
    

        
    def fit(self,points,responsibilities, verbose):
        """ fits subspace to points using PCA/SVD if responsibilities are all 0 or 1, otherwise with np.covariance.
        """
        translation = None
        vectors = None
        latent_sigmas = None
        sigma = None
        alg = 'sklearn.decomposition.pca'
        if not np.all((responsibilities == 0) | (responsibilities == 1)): #for soft assignment
            alg = 'np.cov'
            
        mask = responsibilities.astype(bool) #in the future, this should be done in the M step to avoid copying data in multiprocessing

        if np.sum(mask) <= self.d:
             alg = 'np.cov'

        
        if self.d == 0 and np.sum(mask) == 1:
            translation = points[mask][0]
            vectors = []
            latent_sigmas = [0] #will trigger enforce_min_variance in EM.py
            
        elif alg == 'np.cov':
            
            
            #0-center the data
            mean = None
            
            mean = np.average(points, axis = 0, weights = responsibilities)
            
            points_ = points - mean
            translation = mean
            
            #weighted covariance matrix
            cov = np.cov(points_, rowvar = False, aweights = responsibilities) #np.cov is the sample covariance by default, while np.var is the population variance
            
            #eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eig(cov)
            
            #handle rare numerical error
            if np.iscomplexobj(eigvals):
                eigvals  = np.abs(np.real(eigvals))
                eigvecs  = np.real(eigvecs)
                
                
            #sort by eigenvalue
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs.T[order]
            
            #latent vector standard deviations are square roots of eigenvalues
            latent_sigmas = np.sqrt(eigvals[:self.d])
            
            #sigma is the mean discarded variance (also the per-dimension mean squared orthogonal residual length, but it is already almost computed here in the eigenvalues)
            sigma = np.sqrt(np.mean(eigvals[self.d:]))
            
            #latent vectors are the top d eigenvectors
            vectors = eigvecs[:self.d]
            
        elif alg == 'sklearn.decomposition.pca':
            pca = sklearn.decomposition.PCA(n_components = self.d, random_state = 0)
            r = pca.fit(points[mask])
            vectors = r.components_
            latent_sigmas = np.sqrt(r.explained_variance_)
            translation = r.mean_
            
        
            ###############
        else:
            raise ValueError(f'M step algorithm "{alg}" not recognized')
            return -1
        
        #update self
        
        self.translation = translation
        self.update_latent_sigmas(latent_sigmas)
        self.update_vectors(vectors)
        self.update_sigma(points,responsibilities, verbose, sigma = sigma)
        #self.prior is updated in EM.expectation_maximization
        return self


    def update_sigma(self,points,responsibilities_, verbose,sigma = None):
        """In the event of 0 total responsibility, a variance of 0 is assigned. This triggers enforce_min_variance to impose a minimum variance as 0 variance leads to an undefined PDF.
        """
        sum_responsibilities = np.sum(responsibilities_)
        if sum_responsibilities == 0.0:
            self.sigma = 0
            return
        if sigma is not None: #was precomputed
            self.sigma = sigma
            return
        
        squared_residuals = self.orthogonal_distance(points, squared = True)
        variance = np.dot(responsibilities_,squared_residuals)/sum_responsibilities
        self.sigma =  np.sqrt(variance/(self.D - self.d))
            
    def update_latent_sigmas(self, latent_sigmas):
        """In previous versions: Checks that the covariance matrix is nonsingular and if needed imposes a minimum variance is imposed
        as 0 variance leads to an undefined PDF. Note: this constitutes loss of a dimension for the subspace. I considered
        instead replacing the space with a new one of lower dimension. However, I did not as the subspace could in some cases
        regain the lost dimension in future iterations.
        
        Current version: checking for nonsingularity and minimum variance is enforced by a separate function. 
        """
        #latent_sigmas = np.array([s if s > np.sqrt(min_variance) else np.sqrt(min_variance) for s in latent_sigmas])
        self.latent_sigmas = latent_sigmas


    
    def generate(self, size = 1):
        """generate synthetic data"""
        rng = np.random.default_rng()
        d, D = self.d, self.D
        if d == 0:
            return rng.standard_normal((size, D)) * self.sigma + self.translation


        Z = rng.standard_normal((size, d)) * self.latent_sigmas  
        X_signal = Z @ self.vectors                                   

        noise = rng.standard_normal((size, D)) * self.sigma     

        X_noise = noise - ((noise @ self.vectors.T) @ self.vectors)

        return X_signal + X_noise + self.translation

################################## HELPERS TO CHECK CONVERGENCE ###########################

def check_convergence(prev_spaces,curr_spaces, tolerance = 5e-4):
    """checks if the algorithm has converged on a solution (ie no update occurs)"""
    for i in range(len(prev_spaces)):
        if check_subspace_equivalency(prev_spaces[i],curr_spaces[i],tolerance = tolerance) == False:
            return False
    return True

def check_subspace_equivalency(space_1, space_2, tolerance = 5e-4, check_sigma = True, verbose = False, point = []):
    """verify if subspaces are equivalent in terms of basis, translation, and noise parameters. Does not check distribution within subspace"""
    if space_1.D != space_2.D:
        raise ValueError('mismatched dimension of ambient space')
    if space_1.d != space_2.d:
        #could check if one space is a subspace of the other
        return False
   
    #the translation vectors are points on the subspaces that can be checked. 
    #there is no guarantee that the translation vectors are within the normal bounds of data, where approximately equivalent subspaces are very close
    #therefore, I project the origin onto both subspaces, unless another point is passed in
     
    if len(point) == 0:
        point = np.zeros(space_1.D) #origin
    t1 = space_1.projection(point)
    t2 = space_2.projection(point)
    
    if np.linalg.norm(t1-t2) > tolerance:
        return False
    
    for v in space_1.vectors:
        p = v + t1
        dist = np.linalg.norm(p - space_2.projection(p))
        if dist > tolerance:
            if verbose:
                print(f'distance too great between {p} and {space_2.projection(p)}: {dist}')
            return False
        
    for v in space_2.vectors:
        p = v + t2
        dist = np.linalg.norm(p - space_1.projection(p))
        if dist > tolerance:
            if verbose:
                print(f'distance too great between {p} and {space_2.projection(p)}: {dist}')
            return False  
        
    if check_sigma:
        if abs(space_1.sigma - space_2.sigma) > (space_1.sigma*.05): #less than 5% change in noise parameter and no significant change in positions of vectors (already tested above)
            return False
    return True



################################## HELPERS FOR OPTIMIZATION ###########################
# To use scipy optimize, convert the affine subspace class to a 1D vector
# some functions below are copies of their affine subspace counterparts, modified to 
#      work with the array instead
########################################################################################
#def unpack_subspace_array(subspace_array,D):
#    """accepts 1-D array of translation followed by direction vectors
#    reshapes and returns the translation and direction vectors"""
#    if len(subspace_array) % D != 0:
#        raise RuntimeError("Subspace array length not multiple of D")
#    reshaped_array = subspace_array.reshape(int(len(subspace_array)/D),D) #(k+1) x D array
#    translation = reshaped_array[0]
#    vectors = np.array([])
#    if len(reshaped_array) == 2: 
#        vectors = np.array([reshaped_array[1]])
#    elif len(reshaped_array) > 2: 
#        vectors = reshaped_array[1:]
#    return translation, vectors
#
#def pack_subspace_array(affine_subspace):
#    """converts the translation and direction vectors of an affine subspace object to a 1x((k+1)D) array"""
#    subspace_array = [affine_subspace.translation]
#    for v in affine_subspace.vectors:
#        subspace_array.append(v)
#    return np.array(subspace_array).flatten()
#
    
def orthogonal_distance(subspace_array,D,points):  
    """unused function to be deleted"""
    translation, vectors = unpack_subspace_array(subspace_array,D)

    basis = vectors_to_orthonormal_basis(vectors).T
    if len(basis) ==0:
        basis = np.zeros((D,1))
    #project
    #NxD points
    #DxK basis
    #NxD x DxK x KxD

    projections = np.matmul(np.matmul(points-translation, basis), basis.T) + translation

    if len(points.shape) ==1: #single point
        return np.linalg.norm(points-projections)
    else:
        return np.linalg.norm(points-projections,axis = 1)

def vectors_to_orthonormal_basis(vectors):
        """given k vectors of dimension D spanning a vector subspace, return an orthonormal basis
        **copy of affine_subspace.vectors_to_orthonomal_basis for use in data generation**
        vectors: k x D array
        returns: k x D array"""
        basis = np.array([])
        if len(vectors) == 0:
            return basis
        elif len(vectors) == 1: #np.linalg.orth expects a matrix not a single vector
            basis = (vectors/np.linalg.norm(vectors))
        else: 
            if np.allclose(np.linalg.norm(vectors, axis  = 1),1.0): #already an orthonormal basis constructed from space.fit
                basis = vectors 
            else: #np.linalg.orth finds orthonormal basis for the span of the column vectors (returns D x k matrix)
                basis = orth(vectors.T).T
        return basis

class fixed_space(affine_subspace):
    """basis vectors, translation, and latent_sigma parameters are fixed. Only sigma (for noise) and prior (proportion of total ownership of points) can be updated by fixed_space.fit()"""
    #overwrite fit from parent class to keep the space fixed
    def fit(self,points,responsibilities, verbose):
        self.update_sigma(points,responsibilities, verbose)

        return self
    def __str__(self):
        return f'fixed affine subspace: \n basis: \n {self.vectors} \n translation: {self.translation} \n sigma: {self.sigma}'

class bg_space(affine_subspace):
    "background 'bad data' model. fixed_space that is not only fixed in orientation but also is fixed in covariance"
    #overwrite fit from parent class to keep the space fixed
    def fit(self,points,responsibilities, verbose):
        #nothing
        return self
    def __str__(self):
        return f'background space: \n basis: \n {self.vectors} \n translation: {self.translation} \n sigma: {self.sigma}'
    
#@numba.njit
#def weighted_mean(points, w):
#    total = np.sum(w)
#    return np.sum(points * w[:, None], axis=0) / total


#@numba.njit
#def numba_probability(points, translation, D, d, sigma, latent_sigmas, vectors, log):
#        """proportional to P(point | self)
#        ignores 1/sqrt(2 pi) term in normal pdf
#        can be made exact by multiplying result by 1/(2 pi)^ D/2 (total_log_likelihood() function in model_selection.py does this). Old version was more readable and used affine_subspace methods but was slower"""
#        N = points.shape[0]
#        diff = points - translation  # NxD
#
#        # Complementary space log-prob
#        log_probs = -(D - d) * np.log(sigma) * np.ones(N, dtype=np.float64)
#
#        if d > 0:
#            # Precompute latent coordinates once
#            coords = diff @ vectors.T  # NxD latent coordinates
#            # Project back to space: proj = coords @ self.vectors
#            proj = coords @ vectors    # NxD
#        else:
#            coords = None
#            proj = np.zeros_like(diff)
#
#        # Complementary space squared distance (orthogonal distance)
#        orth_sq = np.sum((diff - proj)**2, axis=1)
#        log_probs += -0.5 * orth_sq / sigma**2
#
#        if d != 0:
#            # Latent space contribution
#            log_probs -= np.sum(np.log(latent_sigmas))
#            log_probs -= 0.5 * np.sum((coords / latent_sigmas)**2, axis=1)
#
#        if log:
#            return log_probs
#        else:
#            return np.exp(log_probs)
#            log_probs -= np.sum(np.log(latent_sigmas))
#            log_probs -= 0.5 * np.sum((coords / latent_sigmas)**2, axis=1)
#
#        if log:
#            return log_probs
#        else:
#            return np.exp(log_probs)

def ensure_vector_directions(spaces, inplace = True):
    """checks direction of each basis vector and flips them (reverses sign) as needed to point in a positive direction.
    Specifically, checks whether dot product of each basis vector with <1,1,1,...,1> is positive. Flips if negative.
    spaces: list of subspace objects
    inplace: bool. default True. Whether to modify the spaces in place or not
    
    returns: sorted spaces
    """
    
    spaces_ = spaces
    if inplace == False:
        spaces_ = [s.copy() for s in spaces]
    one_vector = [1]*spaces[0].D
    for s in spaces_:
        vecs = []
        for v in s.vectors:
            if np.dot(one_vector,v) < 0:
                vecs.append(-1*v)
            else:
                vecs.append(v)
        s.vectors = np.array(vecs)
    if inplace == False:
        return spaces_
    
def write_spaces(spaces, file):
    """writes spaces to a csv file
    spaces: list of subspace objects
    file: filename to write to"""
    data = []
    for s in spaces:
        data.append([f'new space', type(s).__name__])
        data.append([s.D, s.d,s.sigma,s.prior])
        data.append(s.latent_sigmas)
        for v in s.vectors:
            data.append(v)
        if len(s.vectors) == 0:
            data.append([])
        data.append(s.translation)
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    
def read_spaces(file):
    """reads in a file written by 'write_spaces'
    
    file: filename
    returns: list of subspace objects"""
    
    spaces = []
    with open(file, 'r') as f:
        csv_reader = csv.reader(f)
    
        space_array = []
        D = None
        d = None
        sigma = None
        prior = None
        latent_sigmas = None
        vectors = None
        translation = None
        
        for row in csv_reader:
            if len(row) > 0 and row[0] == 'new space':
                type_ = row[1]
                D = None
                d = None
                sigma = None
                prior = None
                latent_sigmas = None
                vectors = None
                translation = None
            elif D == None: #reading in a new space
                D = int(row[0])
                d = int(row[1])
                sigma = float(row[2])
                prior = float(row[3])
            elif latent_sigmas == None:
                latent_sigmas = [float(r) for r in row]
            elif vectors == None or len(vectors) < d:
                try:
                    vectors.append([float(r) for r in row])
                except:
                    vectors = []
                    if len(row) != 0:
                        vectors.append([float(r) for r in row])
            elif translation == None:
                translation = [float(r) for r in row]
                if type_ == 'affine_subspace':
                    s = kspaces.affine_subspace_.affine_subspace(vectors,
                        translation,
                        sigma,
                        latent_sigmas,
                        prior)
                elif type_ == 'fixed_space':
                    s = kspaces.affine_subspace_.fixed_space(vectors,
                        translation,
                        sigma,
                        latent_sigmas,
                        prior)
                elif type_ == 'bg_space':
                    s = kspaces.affine_subspace_.bg_space(vectors,
                        translation,
                        sigma,
                        latent_sigmas,
                        prior)
                else:
                    raise ValueError(f'Unrecognized type: {type_}')
                spaces.append(s)
                
    return spaces