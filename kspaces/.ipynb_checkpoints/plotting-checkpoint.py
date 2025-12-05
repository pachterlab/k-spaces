import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import copy
from sklearn.decomposition import PCA
from .affine_subspace_ import affine_subspace, fixed_space, bg_space
import itertools



def project_space(s,vis):
    """projects k-spaces subspace into the 3D subspace of PCA or a 3D k-spaces subspace
    s: a k-spaces subspace object
    vis: a k-spaces subspace object or a sklearn.decomposition.PCA instance
    
    returns: affine_subspace object for the projection of 's' onto 'vis'"""
    tr_vis = None
    vecs_vis = None
    if np.any(isinstance(vis, affine_subspace), isinstance(vis, fixed_space), isinstance(vis, bg_space)):
        tr_vis = vis.translation
        vecs_vis = vis.vectors
        if vis.d == 0:
            raise ValueError('Projection onto a space with d = 0 unsupported')
    elif isinstance(vis, PCA):
        tr_vis = vis.mean_
        vecs_vis = vis.components_
    else:
        raise ValueError(f'Unrecognized type for vis: {type(vis)}.')
        
    tr_rel = s.translation - tr_vis#translation relative to pca
    tr = (tr_rel.reshape(1,len(tr_rel)) @ vecs_vis.T)[0]
    vecs = []
    if len(s.vectors) > 0:
        vecs = s.vectors @ vecs_vis.T
    space = affine_subspace(vecs, tr, 1,[1]*len(vecs),1)
    space.vectors = vecs #override affine_subspace.vectors_to_orthogonal_basis(), which is triggered by a numerical error and swaps order of vectors
    return space

def _set_plot_params(ax, x,y,z, axis_labels, aspect):
    """helper function for view_3D_proj"""
    #adjust limits
    xlim = (min(x), max(x))
    ylim = (min(y), max(y))
    zlim = (min(z), max(z))
    
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    
    ax.set_aspect(aspect)
    
    #set axes
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.tick_params(axis='x', colors='crimson')
    ax.tick_params(axis='y', colors='green')
    ax.tick_params(axis='z', colors='blue')
    
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
    if points.shape[1] != 3:
        raise ValueError (f'points.shape[1] = {points.shape[1]}. Must be equal to 3')
        
    if len(subtypes) == 0:
        subtypes = [None]*len(points)
            
    #make figure
    fig, axs = plt.subplots(len(elevation), len(azimuth), figsize=(6*len(azimuth), 4*len(elevation)), subplot_kw={'projection': '3d'})
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])
    if len(axs.shape) == 1:
        axs = np.array([axs])
    if len(azimuth) == 1:
        axs = axs.T

    fig.suptitle(title, fontsize = fontsize)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
   
   
    
    
    #plot with different viewing angles
    for i, el in enumerate(elevation):
        for j, az in enumerate(azimuth):
            ax = axs[i,j]
            
            
            #plot points by subtypes
            for idx, sub in enumerate(np.unique(subtypes)[np.argsort(np.unique(subtypes, return_counts = True)[1])[::-1]]):
                mask = subtypes == sub
                if sub is None:
                    mask = [True]*len(x)
                
                #choose color
                color = 'black'
                cmap = None
                if len(color_dict)>0:
                    color = np.array([color_dict.get(sub, 'gray')]*np.sum(mask)) #make 2D array with single row per warning from matplotlib
                
                #cmap_dict overrides color_dict
                if len(cmap_dict) > 0:
                    cmap = cmap_dict.get(sub)[0]
                    color = cmap_dict.get(sub)[1] 
                
                #determine sizes for points
                sizes = np.ones(len(x[mask]))*s_points
                if distances is not None:
                    sizes /= np.power(np.log1p(distances[mask]), scale_by_distance)
                
                ax.scatter(x[mask], y[mask], z[mask], c=color, label=sub, marker='o', cmap = cmap, edgecolor = 'none', s = sizes, alpha = alpha_points)
            
            
            # Set viewing angle
            ax.view_init(elev=el, azim=az)
            
            #add legend
            if i == 0 and j == 0 and legend:
                ax.legend(loc=legend_loc, bbox_to_anchor=(1.05, 1), markerscale = markerscale, fontsize = fontsize)
            
            _set_plot_params(ax, x,y,z, axis_labels, aspect)
            

    # Adjust layout and display
    fig.tight_layout()
    
    if save != '':
        plt.savefig(save)
    if show:
        plt.show()
    if return_figure:
        return fig, axs
    plt.close()
    


def view_3D_pca_mpl(points, 
         aspect = 'equal', 
         subtypes = [],
         spaces = [],
         title = '', 
         axis_labels = ['PC 1', 'PC 2', 'PC 3'],
         color_dict = {}, 
         cmap_dict = {}, 
         space_colors = [],
         s_points = 2, 
         alpha_points = 1, 
         alpha_spaces = 0.5,
         prism_scale = None,
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
    
    Plots points together in 3D PCA space with the projections of spaces (if space.d <= 3) with matplotlib. 
    Optionally colors points by subtype if the subtypes argument and either color_dict or cmap_dict arguments are used.
    
    points: N x D array 
    aspect: passed to ax.set_aspect 
    subtypes: length N list or array of subtype labels for points
    spaces: list of subspace objects to be projected into PCA space for visualization. spaces with d > 3 will be ignored. 
    title: passed to fig.suptitle 
    axis_labels: default ['basis 1', 'basis 2', 'basis 3']
    color_dict: dict. key, val pair is subtype_name:color. Overriden by cmap dict. Default of {} will plot black, while an 
     incomplete dictionary will color non included subtypes gray.
    cmap_dict: dict. key, val pair is subtype_name:(cmap_name,values_for_cmap)} Default of {} will plot black. 
    space_colors: list of colors to plot spaces. Default of [] triggers use of tableau color palette.
    alpha_points: transparency passed to ax.scatter as alpha
    alpha_spaces: transparency for spaces.
    prism_scale: scalar or None. passed to plot_prism. If None, prism side lengths 
        (in the ambient space before projection) are 2 s.d. Otherwise, a cube of side length 'scale'
    is constructed for each subspace in the ambient space.
    elevation = list that sets viewing elevation(s). Elements passed to ax.view_init as elev
    azimuth = list that sets viewing azimuth(s). Elements passed to ax.view_init as azim
    legend = whether to plot legend. 
    legend_loc = passed to ax.legend.
    save = filename to save plot to. default '' does not save. 
    show = if True, triggers plt.show() 
    markerscale = passed to ax.legend
    fontsize = passed to ax.legend, fig.suptitle, axis labels
    return_figure: if True, return fig, axs):"""
    if len(subtypes) == 0:
        subtypes = [None]*len(points)
        
    #fit PCA on whole dataset
    pca = PCA(n_components=3)
    
    #project points onto 3D PCA
    points_proj = pca.fit_transform(points)
    
    #make figure
    fig, axs = plt.subplots(len(elevation), len(azimuth), figsize=(6*len(azimuth), 4*len(elevation)), subplot_kw={'projection': '3d'})
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])
    if len(axs.shape) == 1:
        axs = np.array([axs])
    if len(azimuth) == 1:
        axs = axs.T
        
    fig.suptitle(title, fontsize = fontsize)
    x = points_proj[:,0]
    y = points_proj[:,1]
    z = points_proj[:,2]
    xlim = (min(x), max(x))
    ylim = (min(y), max(y))
    zlim = (min(z), max(z))
    

   
    
    #plot with different viewing angles
    for i, el in enumerate(elevation):
        for j, az in enumerate(azimuth):
            ax = axs[i,j]
            #########plot points########

            for idx, sub in enumerate(np.unique(subtypes)[np.argsort(np.unique(subtypes, return_counts = True)[1])[::-1]]):
                mask = subtypes == sub
                if sub is None:
                    mask = [True]*len(x)
                #choose color
                color = 'black'
                cmap = None
                if len(color_dict)>0:
                    color = np.array([color_dict.get(sub, 'gray')]*np.sum(mask)) #make 2D array with single row per warning from matplotlib
                
                #cmap_dict overrides color_dict
                if len(cmap_dict) > 0:
                    cmap = cmap_dict.get(sub)[0]
                    color = cmap_dict.get(sub)[1] 
                    
                ax.scatter(x[mask], y[mask], z[mask], color=color, label=sub, marker='o', s = s_points, edgecolor = 'none', alpha = alpha_points)
            
            ########plot spaces########
            #choose colors if not specified
            if len(space_colors) < len(spaces):
                print('len(space_colors) < len(spaces), defaulting to tableau color palette')
               
                space_colors = set_colors([],len(spaces))
            
            for idx, s in enumerate(spaces):
                if s.d == 0:
                    s_ = project_space(s,pca)
                    plot_point_3D(ax, xlim, ylim, zlim, s_, color=space_colors[idx % len(space_colors)], alpha = alpha_spaces)
    
                elif s.d == 1:
                    s_ = project_space(s,pca)
                    intersections = clip_line(s_, xlim, ylim, zlim)
                    ax.plot(intersections[:,0], intersections[:,1], intersections[:,2], color=space_colors[idx % len(space_colors)],
                              linewidth = 1, linestyle = '-', alpha = alpha_spaces, label = f'line {idx}')
                elif s.d ==2:
                    s_ = project_space(s,pca)
                    plot_plane(ax, xlim, ylim, zlim, s_, color = space_colors[idx % len(space_colors)], alpha = alpha_spaces)
                elif s.d ==3:
                      #plot prism handles projection
                    plot_prism(s, pca, axs[i,j], color=space_colors[idx % len(space_colors)], alpha = alpha_spaces, scale = prism_scale)

            
            # Set viewing angle
            ax.view_init(elev=el, azim=az)
            
            #add legend
            if i == 0 and j == 0 and legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), markerscale = 5)
            
            _set_plot_params(ax, x,y,z, axis_labels, aspect)

    # Adjust layout and display
    fig.tight_layout()
    if save != '':
        plt.savefig(save)
    if show:
        plt.show()
    if return_figure:
        return fig, axs
    plt.show()
    
    
    
def plot_prism(s, pca, ax, color='k', alpha = 1, scale = None):
    """
    Plot a rectangular prism representing affine_subspace `s` in the PCA space. `pca` can also be a d = 3 affine_subspace

    Parameters
    ----------
    s : affine_subspace
        3D affine subspace in D-dimensional ambient space.
    pca : sklearn PCA object or d = 3 affine_subspace object
        Fitted PCA with n_components=3 in the same D-dimensional ambient space.
    color : str
        Color for the prism edges.
    scale: scalar or None. If None, prism side lengths (in the ambient space before projection) are 2 s.d. Otherwise, a cube of side length 'scale'
    is constructed for each subspace in the ambient space.
    """
    if s.d != 3:
        raise ValueError(f"This plotting function currently only supports s.d = 3, got {s.d}.")

    # Generate all combinations of +/-1 for each of the 3 vectors
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))

    # Build vertices in ambient D space
    vertices = []
    for sign in signs:
        point = s.translation.copy()
        for i in range(3):
            if scale is None:
                scale = s.latent_sigmas[i]
            else:
                scale = scale
            point += sign[i] * s.vectors[i] * scale
        vertices.append(point)
    vertices = np.array(vertices)  # shape (8, D)

    # Project vertices into PCA space (3D)
    projected_vertices = pca.transform(vertices)  # shape (8, 3)

    # Define edges of a cube (pairs of vertex indices)
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Prepare edge coordinates
    edge_lines = [
        [projected_vertices[i], projected_vertices[j]]
        for i, j in edges
    ]

    # Draw prism as wireframe
    lc = Line3DCollection(edge_lines, colors=color, alpha = alpha)
    ax.add_collection3d(lc)
    

########################################################################################################################
########################################################################################################################
############################## LEGACY FUNCTIONS FOR COMPATIBILITY WITH NOTEBOOKS FROM PAPERS ###########################
########################################################################################################################
def plot_spaces(points, spaces, null = [], color_probabilities = [], scale = 10, alpha_s = 1):
    """(legacy function for reproducing figures from papers) Plot points and fitted lines."""
    D = len(points[0])
    if D == 2: 
        plot_2D(points,spaces, null = null, scale = scale, color_probabilities = color_probabilities, alpha_s = alpha_s)
    elif D == 3:
        plot_3D(points,spaces, null = null, color_probabilities = color_probabilities, alpha_s = alpha_s)
    else:
        print('plotting options for D > 3 not implemented')

def color_by_cluster(probabilities, colors = []):
    """(helper for plot_spaces_2D) map points to colors based on argmax of assignment probabilities
    returns 'dimgray' if probabilities is empty"""
    if len(probabilities) == 0:
        return 'dimgray'
    else:
        # Determine the cluster with the highest probability
        cluster_assignment = np.argmax(probabilities, axis=1)

        # Assign colors based on the cluster with the highest probability
        point_colors = colors[cluster_assignment]

        return point_colors

def set_colors(colors, num_colors):
    """helper function that checks if `colors` is empty and returns tab10 or tab20 colors based on the number needed"""
    if len(colors) == 0:
        cmap = 'tab10'
        if num_colors > 10:
            cmap = 'tab20'
        
        cmap = plt.cm.get_cmap(cmap, num_colors)  # Use 'tab10' for distinct colors
        colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]
    return colors
##################################### 2D ##########################################

def plot_spaces_2D(ax,spaces, null = [], colors = [], scale = 10, labels = [], alpha_s = 1):
    """ plot subspaces if they are points or lines"""    
    
    colors = set_colors(colors, len(spaces))
    def line_to_points(subspace, scale):
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


def plot_2D(points,spaces,null = [], scale = 10, assignment_probabilities = [], alpha_s = 1):
    """plot points and spaces' projections in 2D with matplotlib. Optionally specify a null model to be plotted in gray.
    
    points: N x 2 array
    spaces: list of subspace objects
    null: default []. list of subspace objects to be plotted in gray
    scale: number of standard deviations to extend lines from means. passed to plot_spaces_2D. 
    assignment_probabilities: assignment probabilities for coloring by cluster
    alpha_s: transparency for plotted spaces."""
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    colors = set_colors([], len(spaces))
    
    ax.scatter(points[:, 0], points[:, 1], s = 1, color=color_by_cluster(assignment_probabilities, colors), marker='o', label='Points')

    plot_spaces_2D(ax,spaces, null, scale = scale, alpha_s = alpha_s)


    ax.set_xlim(np.min(points[:,0]), np.max(points[:,0]))
    ax.set_ylim(np.min(points[:,1]), np.max(points[:,1]))
    ax.legend()
    plt.show()
    plt.close()

############################################# 3D HELPERS #############################################################

def clip_bounds(xlim,ylim,zlim, normal, point):
    """ clips xy meshgrid for a plane to ensure the z dimension fits within the bounds"""

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
    """(helper for clip_line) intersection between a line and a plane"""
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
    """returns intersection of line with the boundaries of the plot
    
    line: d = 1 affine_subspace
    xlim: tuple
    ylim: tuple
    zlim: tuple"""
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
    """(matplotlib helper) plot point in 3D space
    xlim, ylim, zlim: (min,max) tuples
    space: d = 0 subspace object"""
    
    ax.scatter(space.translation[0], space.translation[1], space.translation[2], s = 5, alpha = alpha_s, color=color)
def plot_line_3D(ax, xlim, ylim, zlim, line, color = 'crimson',linestyle = '-', alpha_s = 1, sd_length = None):
    """(matplotlib helper) plot line in 3D space
    
    xlim, ylim, zlim: (min,max) tuples
    line: d = 1 subspace object
    color: color
    linestyle: passed to ax.plot
    alpha_s: passed to ax.plot
    sd_length: default None. If none, use clip_line to extend line to plot boundaries. 
        Otherwise, the number of standard deviations (s.latent_sigmas[0]) to extend the line in either direction
        from the mean.
    """
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
    """(matplotlib helper) plot plane in 3D space
    
    xlim, ylim, zlim: (min,max) tuples
    space: d = 2 subspace object
    color: color"""
    
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
   
    ax.plot_surface(xx, yy, zz, color=color, alpha=0.1)
    #xlim, ylim = clip_bounds(xlim,ylim,zlim, normal, point1) #clips boundaries on very steep planes
    # Create a mesh grid within extended x, y boundaries
    x_extended = np.linspace(xlim[0] - .1, xlim[1] + .1, 10) #slight extension required with clipping... shows up as flat region
    y_extended = np.linspace(ylim[0] - .1, ylim[1] + .1, 10) #could just add 1 instead and set the plot z limits
    
    xx, yy = np.meshgrid(x_extended, y_extended)
    # Calculate z values for the plane using the equation of the plane
    zz = (-normal[0] * (xx - point1[0]) - normal[1] * (yy - point1[1])) / normal[2] +point1[2]
    ax.plot_wireframe(xx, yy, zz, color=color, alpha=0.2)



def plot_3D(points,spaces, null = [], axis_labels = ('X-axis','Y-axis','Z-axis'),title = 'title', equal_dims = True, color_probabilities = [], alpha_s = 1):
    """(legacy function for reproducing figures from paper) create a 3D plot with matplotlib"""
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
    """(legacy function for reproducing figures from paper) plots origin along with a an arrow showing the direction of the <1,1,...1> vector"""
    
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
    """(legacy function for reproducing figures from paper)"""
    if len(subtypes) == 0:
        subtypes = ['projected point']*len(points)
        
    #fit PCA on whole dataset
    pca = PCA(n_components=3)
    
    #project points onto 3D PCA
    points_proj = pca.fit_transform(points)
    if print_PCs:
        print('PC1: ',pca.components_[0])
        print('PC2: ',pca.components_[1])
        print('PC3: ',pca.components_[2])
    
    #project spaces onto 3D PCA
    spaces_projected = [project_space(s,pca) for s in spaces]
    
    #make figure
    fig, axs = plt.subplots(3, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
    fig.suptitle(title, fontsize = 30)
    x = points_proj[:,0]
    y = points_proj[:,1]
    z = points_proj[:,2]
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