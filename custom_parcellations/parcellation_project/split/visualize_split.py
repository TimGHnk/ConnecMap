import numpy
from voxcell import VoxelData
from parcellation_project.analyses import flatmaps as fm_analyses
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from parcellation_project.split import decide_split as splt
from parcellation_project.split.reversal_detector import reversal_detector
from parcellation_project.split.cosine_distance_clustering import extract_gradients, cosine_distance_clustering



def binary_classification_for_visualization(parc_level, r, **kwargs):  
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    annotations = parc_level.region_volume
    coords3d, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r, hemisphere=parc_level._config["hemisphere"])
    deg_arr = fm_analyses.degree_matrix_from_parcellation(parc_level, r, normalize=True)
    results = splt.binary_classification(deg_arr, coords2d)
    return results


def HDBSCAN_classification_for_visualization(parc_level, r, **kwargs):
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    fm1 = parc_level.flatmap
    annotations = parc_level.region_volume
    coords3d, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r, hemisphere=parc_level._config["hemisphere"])
    coords2d = numpy.unique(coords2d, axis=0)
    x1, y1, x2, y2 = fm_analyses.gradient_map(fm0, fm1, annotations, r, show=False)
    results = splt.HDBSCAN_classification(x1, y1, x2, y2, coords2d, **kwargs)
    return results
    
    
def quadri_classification_for_visualization(parc_level, r, **kwargs):
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    fm1 = parc_level.flatmap
    annotations = parc_level.region_volume
    coords3d, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r, hemisphere=parc_level._config["hemisphere"])
    coords2d = numpy.unique(coords2d, axis=0)
    x1, y1, x2, y2 = fm_analyses.gradient_map(fm0, fm1, annotations, r, show=False)
    results = splt.quadri_classification(x1, y1, x2, y2, coords2d)
    return results


def reversal_detection_for_visualization(parc_level, r, pre_filter_sz=5, post_filter_sz=1,
                                         min_seed_cluster_sz=4, border_thresh=0.05, **kwargs):
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    fm1 = parc_level.flatmap
    annotations = parc_level.region_volume
    solution = reversal_detector(r.data["acronym"],
                                fm0, fm1,
                                annotations, r, component = kwargs["component"],
                                pre_filter_sz=pre_filter_sz,
                                post_filter_sz=post_filter_sz,
                                min_seed_cluster_sz=min_seed_cluster_sz,
                                border_thresh=border_thresh)
    
    return solution
    

def cosine_distance_clustering_for_visualization(parc_level, r, alpha, eps, min_cluster_size, min_samples, N):
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    fm1 = parc_level.flatmap
    annotations = parc_level.region_volume
    char = parc_level.characterization
    gXs, gYs = extract_gradients(fm0, fm1, annotations, r)
    lambdas = [i for i in char if i["region"] == r.data["acronym"]][0]["lambdas"]
    _, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r, hemisphere=parc_level._config["hemisphere"])
    coords2d = numpy.unique(coords2d, axis=0)
    solution = cosine_distance_clustering(gXs, gYs, coords2d, lambdas, alpha, eps, min_cluster_size, min_samples, N)
    return solution



def viz_with_SVM(coords_clf, **kwargs):
    if isinstance(coords_clf, list):
        out_C1 = splt.split_with_SVM(coords_clf[0], kwargs["C"], kwargs["gamma"], thres_accuracy=0, show=kwargs["show"])
        out_C2 = splt.split_with_SVM(coords_clf[1], kwargs["C"], kwargs["gamma"], thres_accuracy=0, show=kwargs["show"])
        solution = numpy.column_stack((out_C1[:,:-1], out_C1[:, -1] + 2 * out_C2[:, -1]))
    else:
        solution = splt.split_with_SVM(coords_clf, kwargs["C"], kwargs["gamma"], thres_accuracy=0, show=kwargs["show"])
    return solution


def viz_split_region(parc_level, r, func1, func2, extend=False, **kwargs):
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    fm1 = parc_level.flatmap  # Diffusion fm
    annotations = parc_level.region_volume
    coords_2d_clf = func1(parc_level, r, **kwargs)
    solution0 = func2(coords_2d_clf, show=True, **kwargs)
    new_subregions = splt.extract_subregions(solution0, kwargs['t'])
    new_subregions = splt.unflattening(parc_level, r.data["acronym"], new_subregions)

    if kwargs["merge"] is True:
        new_subregions = splt.merge_lonely_voxels(new_subregions, kwargs["thres_size"])
    # plot 2d classification
    viz_gradient_split(solution0, fm0, fm1, annotations, r)
    # Adding argument to display the whole process on gradient map
    if extend is True:
        if isinstance(coords_2d_clf, list):
            viz_gradient_split(coords_2d_clf[0], fm0, fm1, annotations, r, title="sign sine", **kwargs)
            viz_gradient_split(coords_2d_clf[1], fm0, fm1, annotations, r, title="sign cosine", **kwargs)       
        else:
            viz_gradient_split(coords_2d_clf, fm0, fm1, annotations, r, title="1st classification", **kwargs)
        viz_gradient_split(solution0, fm0, fm1, annotations, r, title="2nd classification", **kwargs)
        solution_viz = numpy.delete(solution0, numpy.where(solution0[:,0] == -1)[0], 0)
        solution2_viz = splt.extract_subregions(solution_viz, kwargs['t'])
        viz_gradient_split(solution2_viz, fm0, fm1, annotations, r, title="Hierarchical Clustering", **kwargs)
        solution3_viz = splt.merge_lonely_voxels(solution2_viz, kwargs["thres_size"])
        viz_gradient_split(solution3_viz, fm0, fm1, annotations, r, title="Merging process", **kwargs)
    return new_subregions


def viz_gradient_split(coords2d, fm0, fm1, annotations, hierarchy_root, title=None, **kwargs):
    '''Plot the classification and splitting result on the anatomical flatmap along with the 
    gradients.
    If overlay in kwargs, plot also the borders of the regions specified in the overlay.
    The overlay must be N * 3 coordinates array where the last column corresponds to an index 
    of the regions (use the function overlay_img to generate it).
    Can set a title to described to which function or step this result corresponds.
    '''
    # Generate gradients
    x1, y1, x2, y2 = fm_analyses.gradient_map(fm0, fm1, annotations, hierarchy_root, show=False)
    # Sort classes between 0 and N class
    classes = numpy.unique(coords2d[:,-1][~numpy.isnan(coords2d[:,-1])])
    for i in range(len(classes)):
        coords2d[:,-1][coords2d[:,-1]==classes[i]] = i
    # Generate pixel image
    img =  numpy.empty((int(numpy.amax(coords2d[:,0])+1), int(numpy.amax(coords2d[:,1])+1)))
    img[:] = numpy.nan
    for i in range(len(coords2d)):
            img[int(coords2d[i][0]), int(coords2d[i][1])] = coords2d[i,-1]
   
    # Set axis limit
    y_min = min(m for m in coords2d[:,0] if m >= 0)
    y_max = max(m for m in coords2d[:,0] if m >= 0)
    x_min = min(m for m in coords2d[:,1] if m >= 0)
    x_max = max(m for m in coords2d[:,1] if m >= 0)
    
    ax = plt.figure(figsize=(20, 20)).gca()
    X_axis, Y_axis = numpy.meshgrid(numpy.arange(x1.shape[1]), numpy.arange(x1.shape[0]))
    image = ax.imshow(img, cmap=plt.cm.get_cmap('tab20b', len(classes)))
    # Scale gradients arrows
    scale_arrows = ((x_max-x_min+6)*1.285714)
    ax.quiver(X_axis, Y_axis, x1, y1, color='yellow', scale_units='width', headaxislength=110,headlength=110, headwidth=60, scale=scale_arrows)
    ax.quiver(X_axis, Y_axis, x2, y2, color='red', scale_units='width', headaxislength=110,headlength=110, headwidth=60, scale=scale_arrows)
    colors = {'\u03B1-axis':'yellow', '\u03B2-axis':'red'}           
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]   
    ax.legend(handles, labels, prop={'size': 20})
    ax.set_ylim(sorted(ax.get_ylim()))
    if "overlay" in kwargs:
        idx = numpy.unique(kwargs["overlay"][~numpy.isnan(kwargs["overlay"])])
        for i in idx:
            borders = kwargs["overlay"].copy()
            borders[borders != i] = 0
            plot_outlines(borders.T, lw=2, color='black')
    if title is not None:
        plt.title(f"Split Gradient map after {title}", fontsize=30)
    # Scale the colorbar
    scale_cbar = (x_max - x_min) / 30
    if scale_cbar > 1: scale_cbar = 1
    plt.colorbar(image, ax=ax, ticks=numpy.arange(len(classes)), shrink=scale_cbar, orientation="horizontal", pad=0.05)
    plt.axis((x_min-3,x_max+3,y_min-3,y_max+5))
    plt.show()




def overlay_img(fm0, annotations, hierarchy_root, lst_regions):   
    coords_lst = []
    i = 1
    for reg in lst_regions:
        hierarchy_reg = hierarchy_root.find('acronym', reg)[0]
        _, two_d_coords = fm_analyses.flatmap_to_coordinates(annotations, fm0, hierarchy_reg)
        idx = numpy.zeros((len(two_d_coords), 1)) + i
        coords_lst.append(numpy.column_stack((two_d_coords, idx)))
        i += 1
    out_coords = numpy.concatenate(coords_lst)
    img =  numpy.empty((int(numpy.amax(out_coords[:,0])+1), int(numpy.amax(out_coords[:,1])+1)))
    img[:] = numpy.nan
    for xy in out_coords:
        img[int(xy[0]),int(xy[1])] = 0
    img = img.astype('float64')
    bounds = numpy.unique(out_coords[:,-1][~numpy.isnan(out_coords[:,-1])])
    for i in range(len(bounds)):
        out_coords[:,-1][out_coords[:,-1]==bounds[i]] = i+1
    bounds = numpy.unique(out_coords[:,-1][~numpy.isnan(out_coords[:,-1])])
    two_d = out_coords.copy()
    for i in range(len(two_d)):
            img[int(two_d[i][0]), int(two_d[i][1])] = two_d[i,-1]
    return img



def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = numpy.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            edges.append(numpy.array([[i, j+1],
                                   [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            edges.append(numpy.array([[i+1, j],
                                   [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            edges.append(numpy.array([[i, j],
                                   [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            edges.append(numpy.array([[i, j],
                                   [i, j+1]]))

    if not edges:
        return numpy.zeros((0, 2, 2))
    else:
        return numpy.array(edges)


def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:

        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = numpy.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = numpy.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                # loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = numpy.delete(edges, i, axis=0)

        loop_list.append(numpy.array(loop))

    return loop_list


def plot_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = get_all_edges(bool_img=bool_img)
    edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
    outlines = close_loop_edges(edges=edges)
    cl = LineCollection(outlines, **kwargs)
    ax.add_collection(cl)