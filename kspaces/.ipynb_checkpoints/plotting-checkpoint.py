import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
from .affine_subspace_ import affine_subspace

def plot_spaces(points, spaces, null = [], color_probabilities = [], scale = 10, alpha_s = 1):
    """Plot points and fitted lines."""
    D = len(points[0])
    if D == 2: 
        plot_2D(points,spaces, null = null, scale = scale, color_probabilities = color_probabilities, alpha_s = alpha_s)
    elif D == 3:
        plot_3D(points,spaces, null = null, color_probabilities = color_probabilities, alpha_s = alpha_s)
    else:
        print('plotting options for D > 3 not implemented')

def color_by_cluster(probabilities):
    """map points to colors based on argmax of probability
    return black if probabilities is empty"""
    if len(probabilities) == 0:
        return 'dimgray'
    else:
        # Determine the cluster with the highest probability
        cluster_assignment = np.argmax(probabilities, axis=1)

        # Define a color map (e.g., RGB tuples or hex codes)
        colors = np.array(['palevioletred', 'lightskyblue', 'sandybrown', 'mediumseagreen', 'mediumorchid','khaki'])

        # Assign colors based on the cluster with the highest probability
        point_colors = colors[cluster_assignment]

        return point_colors
##################################### 2D ##########################################

def plot_spaces_2D(ax,spaces, null = [], colors = [], scale = 10, labels = [], alpha_s = 1):
    """ plot subspaces if they are points or lines"""    
    if len(colors) == 0:
        colors = ['crimson','blue','orange','green','orchid','gold']
    
    def line_to_points(subspace, scale):
        """deprecated. removing soon"""
        v = subspace.vectors[0]
        point1 = subspace.translation - scale*subspace.latent_sigmas[0]*v
        point2 = subspace.translation + scale*subspace.latent_sigmas[0]*v
        return [point1, point2]

    if len(labels) == 0:
         for i, space in enumerate(spaces):
            if space.d == 0:
                labels.append(f'Space {i} (point)')
            if space.d == 1:
                labels.append(f'Space {i} (line)')
   
    for i, space in enumerate(spaces):
        if space.d == 0: #point
            ax.scatter(space.translation[0], space.translation[1], color=colors[i%len(colors)], label=labels[i], alpha = alpha_s)
        elif space.d == 1: #line
            points = line_to_points(space, scale = scale)
            ax.plot([p[0] for p in points], [p[1] for p in points], color=colors[i%len(colors)], label=labels[i], alpha = alpha_s)
        else:
            print('cannot plot d=3 in 2D')
            
    if type(null) != list:
        null = [null]
    for i, space in enumerate(null):
        if space.d == 0: #point
            ax.scatter(space.translation[0], space.translation[1], color='gray', label='null', alpah = alpha_s)
        elif space.d == 1: #line
            points = line_to_points(space, scale = scale)
            ax.plot([p[0] for p in points], [p[1] for p in points], linestyle='--', color='gray', label='null', alpha = alpha_s)
        else:
            print('cannot plot d=3 in 2D')


def plot_2D(points,spaces,null = [], scale = 10, color_probabilities = [], alpha_s = 1):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(points[:, 0], points[:, 1], s = 1, color=color_by_cluster(color_probabilities), marker='o', label='Points')

    plot_spaces_2D(ax,spaces, null, scale = scale, alpha_s = alpha_s)


    ax.set_xlim(np.min(points[:,0]), np.max(points[:,0]))
    ax.set_ylim(np.min(points[:,1]), np.max(points[:,1]))
    ax.legend()
    plt.show()
    plt.close()

############################################# 3D HELPERS #############################################################

def clip_bounds(xlim,ylim,zlim, normal, point):
    """ clips xy meshgrid to ensure the z dimension fits within the bounds"""

    # Calculate intersections with z boundaries (z = zlim)
    x_intersects = []
    y_intersects = []

    for z_boundary in zlim:
        for y_boundary in ylim:
            x_intersect = (-normal[2]*(z_boundary-point[2]) -normal[1]*(y_boundary-point[1]) )/ normal[0] + point[0]
            if xlim[0] <= x_intersect <= xlim[1]:
                x_intersects.append(x_intersect)                        
        
        for x_boundary in xlim:
            y_intersect = (-normal[2]*(z_boundary-point[2]) -normal[0]*(x_boundary-point[0]) )/ normal[1] + point[1]
            if ylim[0] <= y_intersect <= ylim[1]:
                y_intersects.append(y_intersect)
    
    # Update xlim and ylim based on intersections
    if x_intersects:
        xlim = (min(x_intersects), max(x_intersects))
    if y_intersects:
        ylim = (min(y_intersects), max(y_intersects))
    return xlim,ylim

def line_plane_intersection(point_on_line, direction_vector, point_on_plane, normal_vector):
    """intersection between a line and a plane"""
    # Calculate the dot product between the line's direction vector and the plane's normal vector
    dot_product = np.dot(direction_vector, normal_vector)

    # Check if the line is not parallel to the plane (dot product should not be zero)
    if abs(dot_product) > 1e-6:
        # Calculate a vector from a point on the line to a point on the plane
        line_to_plane_vector = point_on_plane - point_on_line
       
        # Calculate the scaling factor for the line's direction vector
        t = np.dot(line_to_plane_vector, normal_vector) / dot_product

        # Calculate the intersection point using the parameterization of the line
        intersection_point = point_on_line + t * direction_vector
        return intersection_point #not sure why it was returning a 2D array with only one element 4/30/24
    else:
        # If the line is parallel to the plane, return None indicating no intersection
        return np.array([])

def clip_line(line, xlim, ylim,zlim):
    """intersection of line with the boundaries of the plot"""
    bounds = np.concatenate([xlim, ylim,zlim])
    intersects = []
    for i in range(6):
        pos = int(i/2)
        point = [0,0,0]
        point[pos] = bounds[i]
        normal = [0,0,0]
        normal[pos] = 1
        intersect = line_plane_intersection(line.translation, line.vectors[0], point, normal)
        #adding tolerance of 1e-5 to fix a numerical issue
        tol = 1e-5
        if len(intersect) > 0:
            if (xlim[0] - tol <= intersect[0] <=xlim[1] + tol) and (ylim[0] - tol <= intersect[1] <= ylim[1]+ tol) and (zlim[0] - tol <= intersect[2] <=zlim[1]+ tol):
                intersects.append(intersect)
    return np.array(intersects)

################################## 3D ###########################################

def plot_point_3D(ax, xlim, ylim, zlim, space, color='crimson', alpha_s = 1):
    """plot point in 3D space
    xlim, ylim, zlim are (min,max) tuples"""
    
    ax.scatter(space.translation[0], space.translation[1], space.translation[2], s = 5, alpha = alpha_s, color=color)
def plot_line_3D(ax, xlim, ylim, zlim, line, color = 'crimson',linestyle = '-', alpha_s = 1, sd_length = None):
    """plot line in 3D space
    xlim, ylim, zlim are (min,max) tuples"""
    points = None
    if sd_length is not None:
        points = []
        tr = line.translation
        v = line.vectors[0]
        std_dev = line.latent_sigmas[0]
        points = np.array([tr - sd_length*std_dev*v,tr + sd_length*std_dev*v])
    else:
        points = clip_line(line, xlim, ylim,zlim)

    ax.plot(points[:,0], points[:,1], points[:,2], color=color, linewidth = 1, linestyle = linestyle, alpha = alpha_s)
    
def plot_plane(ax, xlim, ylim, zlim, space, color = 'crimson'):
    """plot plane in 3D space
    xlim, ylim, zlim are (min,max) tuples"""
    vertices = [space.translation]
    for v in space.vectors:
        vertices.append(v+space.translation)
    vertices = np.array(vertices)
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    point1 = vertices[0]
    point2 = vertices[1]
    point3 = vertices[2]
    # Calculate normal vector to the plane
    normal =  np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    # Create a mesh grid within extended x, y boundaries
    x_extended = np.linspace(xlim[0] - .1, xlim[1] + .1, 10) #slight extension required with clipping... shows up as flat region
    y_extended = np.linspace(ylim[0] - .1, ylim[1] + .1, 10) #could just add 1 instead and set the plot z limits
    
    xx, yy = np.meshgrid(x_extended, y_extended)
    # Calculate z values for the plane using the equation of the plane
    zz = (-normal[0] * (xx - point1[0]) - normal[1] * (yy - point1[1])) / normal[2] +point1[2]   
   
    ax.plot_surface(xx, yy, zz, color='crimson', alpha=0.1)
    #xlim, ylim = clip_bounds(xlim,ylim,zlim, normal, point1) #clips boundaries on very steep planes
    # Create a mesh grid within extended x, y boundaries
    x_extended = np.linspace(xlim[0] - .1, xlim[1] + .1, 10) #slight extension required with clipping... shows up as flat region
    y_extended = np.linspace(ylim[0] - .1, ylim[1] + .1, 10) #could just add 1 instead and set the plot z limits
    
    xx, yy = np.meshgrid(x_extended, y_extended)
    # Calculate z values for the plane using the equation of the plane
    zz = (-normal[0] * (xx - point1[0]) - normal[1] * (yy - point1[1])) / normal[2] +point1[2]
    ax.plot_wireframe(xx, yy, zz, color='crimson', alpha=0.2)



def plot_3D(points,spaces, null = [], axis_labels = ('X-axis','Y-axis','Z-axis'),title = 'title', equal_dims = True, color_probabilities = [], alpha_s = 1):
    """create a 3D plot with matplotlib"""
    # Create figure and subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 8), subplot_kw={'projection': '3d'})
    
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    
    spaces_ = copy.deepcopy(spaces)
    if len(null) > 0:
        for s in null:
            spaces_.append(s)
    
    colors = np.array(['crimson', 'blue', 'darkorange', 'green', 'violet','brown'])
    
    xlim = [min(x), max(x)]
    ylim = [min(y), max(y)]
    zlim = [min(z),max(z)]
    
    for i, el in enumerate([0,30,70]):
        for j, az in enumerate([20,90,160]):
            axs[i,j].set_xlabel(axis_labels[0])
            axs[i,j].set_ylabel(axis_labels[1])
            axs[i,j].set_zlabel(axis_labels[2])
            axs[i,j].set_title(title)
            axs[i,j].tick_params(axis='x', colors='crimson')
            axs[i,j].tick_params(axis='y', colors='green')
            axs[i,j].tick_params(axis='z', colors='blue')
            
            #plot points
            axs[i,j].scatter(x, y, z, color=color_by_cluster(color_probabilities), marker='o', alpha = 0.5, s = 1)
            
            if equal_dims:
                axs[i,j].set_aspect('equal')
            #plot the fitted spaces
            for idx, space in enumerate(spaces_):
                
                c = colors[idx%len(colors)]
                style = '-'
                if idx >= len(spaces):
                    c = 'black'
                    style = '--'
                if space.d == 0:
                    plot_point_3D(axs[i,j], xlim, ylim, zlim, space, color=c, alpha_s = alpha_s)
                elif space.d == 1:
                    plot_line_3D(axs[i,j], xlim, ylim, zlim, space, color=c, linestyle=style, alpha_s = alpha_s)
                elif space.d ==2:
                    plot_plane(axs[i,j], xlim, ylim, zlim, space, color=c) #not passing alpha_s because the wireframe is overwhelming when alpha is high
                else:
                    print(space.d)
                    print('plotting for spaces with d >= 3 not implemented')
                    
                    
            # Set different viewing angles
            axs[i,j].view_init(elev=el, azim=az)


    # Adjust layout and display
    fig.tight_layout()
    plt.show()
    plt.close()
    

def project_space(s,pca):
    """projects k-spaces subspace into the 3D subspace of PCA"""
    tr_rel = s.translation - pca.mean_#translation relative to pca
    tr = (tr_rel.reshape(1,len(tr_rel)) @ pca.components_.T)[0]
    vecs = []
    if len(s.vectors) > 0:
        vecs = s.vectors @ pca.components_.T
    space = affine_subspace(vecs, tr, 1,[1]*len(vecs),1)
    space.vectors = vecs #override affine_subspace.vectors_to_orthogonal_basis(), which is triggered by a numerical error and swaps order of vectors
    return space
#https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-a-3d-plot

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)
    
def plot_origin(pca, ax, origin_scale = 10):
    """plots origin along with a an arrow showing the direction of the <1,1,...1> vector"""
    
    D = len(pca.components_[0])
    origin = affine_subspace([[1]*D], [0]*D, 1,[1],1)
    origin_projected = project_space(origin,pca)
    v = origin_projected.translation + origin_scale*origin_projected.vectors[0] #may break
    a = Arrow3D([origin_projected.translation[0], v[0]], [origin_projected.translation[1], v[1]], 
                [origin_projected.translation[2], v[2]], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="black")
    ax.scatter(origin_projected.translation[0], origin_projected.translation[0], origin_projected.translation[0], 
               c='black', label='proj. origin & <1,1...,1>', marker='o', s = 2, alpha = 0.2)
    ax.add_artist(a)
    
def view_3D(points, aspect, spaces = [], subtypes = [], title = '', plot_origin_ = True, origin_scale = 10, print_PCs = False, color_dict = {}):
    if len(subtypes) == 0:
        subtypes = ['projected point']*len(points)
        
    #fit PCA on whole dataset
    pca = PCA(n_components=3)
    
    #project points onto 3D PCA
    points_rand = pca.fit_transform(points)
    if print_PCs:
        print('PC1: ',pca.components_[0])
        print('PC2: ',pca.components_[1])
        print('PC3: ',pca.components_[2])
    
    #project spaces onto 3D PCA
    spaces_projected = [project_space(s,pca) for s in spaces]
    
    #make figure
    fig, axs = plt.subplots(3, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
    fig.suptitle(title, fontsize = 30)
    x = points_rand[:,0]
    y = points_rand[:,1]
    z = points_rand[:,2]
    axis_labels = ['pc1','pc2','pc3']

    colors = ['green','violet','dodgerblue','purple','orange','hotpink','mediumspringgreen','navy','chocolate']
   
    xlim = (min(x), max(x))
    ylim = (min(y), max(y))
    zlim = (min(z), max(z))
    
    #plot with different viewing angles
    for i, el in enumerate([70,30,0]):
        for j, az in enumerate([20,90,160]):
            axs[i,j].set_xlabel(axis_labels[0])
            axs[i,j].set_ylabel(axis_labels[1])
            axs[i,j].set_zlabel(axis_labels[2])
            axs[i,j].tick_params(axis='x', colors='crimson')
            axs[i,j].tick_params(axis='y', colors='green')
            axs[i,j].tick_params(axis='z', colors='blue')
            
            if plot_origin_:
                plot_origin(pca,axs[i,j], origin_scale)
            #plot points by subtypes
            for idx, sub in enumerate(np.unique(subtypes)[np.argsort(np.unique(subtypes, return_counts = True)[1])[::-1]]):
                mask = subtypes == sub
                #### PATCH THAT NEEDS TO BE FIXED ####
                if sub == 'projected point':
                    mask = [True]*len(x)
                color = colors[idx %len(colors)]
                if sub == 'unknown':
                    color = 'gray'
                if len(color_dict) > 0:
                    color = color_dict[sub]
                axs[i,j].scatter(x[mask], y[mask], z[mask], c=color, label=sub, marker='o', s = 2, alpha = 0.3)
            
            #plot lines
            cmap = 'tab10'
            num_colors = len(spaces)
            if num_colors > 10:
                cmap = 'tab20'
            if num_colors > 20:
                print('max number of colors is 20. Using tab20')
            cmap = plt.cm.get_cmap(cmap, num_colors)  # Use 'tab10' for distinct colors
            space_colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]
            for idx, s in enumerate(spaces_projected):
                if s.d == 0:
                    plot_point_3D(axs[i,j], xlim, ylim, zlim, s, color=space_colors[idx % len(space_colors)])
    
                elif s.d == 1:
                    intersections = clip_line(s, xlim, ylim, zlim)
                    axs[i,j].plot(intersections[:,0], intersections[:,1], intersections[:,2], color=space_colors[idx % len(space_colors)],
                              linewidth = 1, linestyle = '-', label = f'line {idx}')
                elif s.d ==2:
                    plot_plane(axs[i,j], xlim, ylim, zlim, s, color = space_colors[idx % len(space_colors)])
            # Set different viewing angles
            axs[i,j].view_init(elev=el, azim=az)
            
            #add legend
            if i == 0 and j == 0:
                axs[i,j].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            
            #adjust limits
            axs[i,j].set_xlim3d(xlim)
            axs[i,j].set_ylim3d(ylim)
            axs[i,j].set_zlim3d(zlim)
            axs[i,j].set_aspect(aspect)

    # Adjust layout and display
    fig.tight_layout()
    plt.show()
    plt.close()