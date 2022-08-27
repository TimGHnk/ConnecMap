import numpy
from voxcell import VoxelData
from parcellation_project.analyses import flatmaps as fm_analyses
from parcellation_project.project import ParcellationLevel
from sklearn import svm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from parcellation_project.split.reversal_detector import reversal_detector
from parcellation_project.split.cosine_distance_clustering import extract_gradients, cosine_distance_clustering
 

def binary_classification(deg_arr, two_d_coords):
    '''Classify pixels into two classes based on the angle between alpha and beta gradients.
    '''
    degree = []
    for i in range(len(two_d_coords)):
        degree.append(deg_arr[two_d_coords[i,0], two_d_coords[i,1]])
    degree = numpy.array(degree)
    grad_clf = numpy.column_stack((two_d_coords, degree)).astype('float64')
    grad_clf[grad_clf[:, -1] <= 90 , -1] = 1
    grad_clf[grad_clf[:, -1] > 90, -1] = 2
    return grad_clf


def binary_classification_from_parcellation(parc_level, **kwargs):
    """Apply binary classification to the current parcellation.
    """
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    annotations = parc_level.region_volume
    results = {}
    for region_name in parc_level.regions:
        r = parc_level.hierarchy_root.find("acronym", region_name)
        assert len(r) == 1
        r = r[0]
        _, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r)
        coords2d = numpy.unique(coords2d, axis=0)
        deg_arr = fm_analyses.degree_matrix_from_parcellation(parc_level, r, normalize=True)
        gradient_dev = numpy.mean(fm_analyses.gradient_deviation_from_parcellation(parc_level, r, plot=False))
        reversal_idx = fm_analyses.reversal_index_from_parcellation(parc_level, r)
        if (reversal_idx > kwargs["thres_reversal_index"]) | (gradient_dev > kwargs["thres_gradient_deviation"]):
            split_is_required = True
        else: split_is_required = False
        if split_is_required:
            results[region_name] = binary_classification(deg_arr, coords2d)
    return results

def quadri_classification(x1, y1, x2, y2, two_d_coords):
    ''' Classifies gradients based on the sign of sine and cosine of
    the angle between alpha and beta gradients.
    '''
    da = numpy.column_stack((x1[two_d_coords[:,0], two_d_coords[:,1]], y1[two_d_coords[:,0], two_d_coords[:,1]]))
    db = numpy.column_stack((x2[two_d_coords[:,0], two_d_coords[:,1]], y2[two_d_coords[:,0], two_d_coords[:,1]]))
    da_angle = numpy.arctan2(da[:, 0], da[:, 1])
    db_angle = numpy.arctan2(db[:, 0], db[:, 1])
    angle_difference = numpy.mod(db_angle - da_angle, 2*numpy.pi)
    angle_img = angle_difference.copy()
    cls_sign = numpy.sign(numpy.sin(angle_img)) + 1
    cls_inversion = numpy.sign(numpy.cos(angle_img)) + 1
    out_ssin = numpy.column_stack((two_d_coords, cls_sign))
    out_scos = numpy.column_stack((two_d_coords, cls_inversion))
    return [out_ssin, out_scos]

def quadri_classification_from_parcellation(parc_level, **kwargs):
    """"Apply quadri classification on the current parcellation.
    """
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    fm1 = parc_level.flatmap  # Diffusion fm
    annotations = parc_level.region_volume
    results = {}
    for region_name in parc_level.regions:
        r = parc_level.hierarchy_root.find("acronym", region_name)
        assert len(r) == 1
        r = r[0]
        _, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r)
        coords2d = numpy.unique(coords2d, axis=0)
        gradient_dev = numpy.mean(fm_analyses.gradient_deviation_from_parcellation(parc_level, r, plot=False))
        reversal_idx = fm_analyses.reversal_index_from_parcellation(parc_level, r)
        if (reversal_idx > kwargs["thres_reversal_index"]) | (gradient_dev > kwargs["thres_gradient_deviation"]):
            split_is_required = True
        else: split_is_required = False
        if split_is_required:    
            x1,y1,x2,y2 = fm_analyses.gradient_map(fm0, fm1, annotations, r, show=False) 
            results[region_name] = quadri_classification(x1, y1, x2, y2, coords2d)
    return results

def reversal_detector_from_parcellation(parc_level, **kwargs):
    """Apply the reversal detector on the current parcellation.
    """
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    fm1 = parc_level.flatmap  # Diffusion fm
    annotations = parc_level.region_volume
    results = {}
    for region_name in parc_level.regions:
        r = parc_level.hierarchy_root.find("acronym", region_name)
        assert len(r) == 1
        r = r[0]
        _, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r)
        coords2d = numpy.unique(coords2d, axis=0)
        gradient_dev = numpy.mean(fm_analyses.gradient_deviation_from_parcellation(parc_level, r, plot=False))
        reversal_idx = fm_analyses.reversal_index_from_parcellation(parc_level, r)
        if (reversal_idx > kwargs["thres_reversal_index"]) | (gradient_dev > kwargs["thres_gradient_deviation"]):
            split_is_required = True
        else: split_is_required = False
        if split_is_required:
            results[region_name] = reversal_detector(region_name,
                                                    fm0, fm1,
                                                    annotations, r,
                                                    pre_filter_sz=kwargs["pre_filter_sz"],
                                                    post_filter_sz=kwargs["post_filter_sz"],
                                                    min_seed_cluster_sz=kwargs["min_seed_cluster_sz"],
                                                    border_thresh=kwargs["border_thresh"])
    return results
    

def cosine_distance_clustering_from_parcellation(parc_level, **kwargs):
    """Apply the cosine distance clustering on the current parcellation.
    """
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    fm1 = parc_level.flatmap  # Diffusion fm
    annotations = parc_level.region_volume
    char = parc_level.characterization
    results = {}
    for region_name in parc_level.regions:
        r = parc_level.hierarchy_root.find("acronym", region_name)
        assert len(r) == 1
        r = r[0]
        _, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r)
        coords2d = numpy.unique(coords2d, axis=0)
        gradient_dev = numpy.mean(fm_analyses.gradient_deviation_from_parcellation(parc_level, r, plot=False))
        reversal_idx = fm_analyses.reversal_index_from_parcellation(parc_level, r)
        if (reversal_idx > kwargs["thres_reversal_index"]) | (gradient_dev > kwargs["thres_gradient_deviation"]):
            split_is_required = True
        else: split_is_required = False
        if split_is_required:
            gXs, gYs = extract_gradients(fm0, fm1, annotations, r)
            lambdas = [i for i in char if i["region"] == region_name][0]["lambdas"]
            results[region_name] = cosine_distance_clustering(gXs, gYs, coords2d, lambdas, **kwargs)
    return results

def split_with_SVM(gradient_clf, c, gamma, thres_accuracy, show=True):
    '''Apply Support Vector Machine to classify every voxels of the 3d map.
    Show = True will display the classification metric.
    Last column of the coordinates array must be the class.
    ADDED: Accuracy threshold, evaluate the classification as valid or not based 
    on that threshold.
    '''
    two_d_coords = gradient_clf[:,:-1]
    idx = numpy.isnan(gradient_clf[:,-1])
    idx = numpy.where(idx == False)
    X_set = gradient_clf[idx]
    X = X_set[:,:-1]
    Y = X_set[:,-1]
    # Train SVM
    clf = svm.SVC(C=c, kernel='rbf', gamma=gamma)
    clf = clf.fit(X, Y)
    # confusion matrix
    if show is True:
        matrix = plot_confusion_matrix(clf, X, Y,
                                         cmap=plt.cm.Blues,
                                         normalize='true')
        plt.title('Confusion matrix for our classifier')
        plt.show(matrix)
    # Predict the unknown data set
    Y_pred = clf.predict(two_d_coords)
    # Inject prediction and build new 3d coordinates array
    clf_extrapolation = numpy.column_stack((two_d_coords, Y_pred))
    # NEW: Using classification of SVM as validation metric
    if show is False:
        matrix = plot_confusion_matrix(clf, X, Y,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
        plt.close()
    if any(matrix.confusion_matrix[n,n] < thres_accuracy for n in range(len(matrix.confusion_matrix))):
        clf_extrapolation[:,-1] = 0 #Returns one class, which will be rejected in validation process
    # Voxel Extrapolation
    return clf_extrapolation


def extract_subregions(coords, t):
    '''Apply hierarchical clustering based on the euclidian distance of the voxels/pixels
    within the two classes using the Nearest point algorithm.
    Then returns the subregions.
    '''
    coords = coords[~(coords == -1).any(axis=1)] # if using pixels, remove invalid pixels
    new_subs = numpy.zeros(coords.shape[0], dtype=int)
    offset = 0
    for i in numpy.unique(coords[:, -1]):
        bitmask = coords[:, -1] == i
        subregion = coords[bitmask, :-1]
        dist = pdist(subregion)
        try:
            linked = linkage(dist, 'single')
        except: continue #empty distance matrix, continue
        else:
            labels = fcluster(linked, t=t, criterion='distance')
            new_subs[bitmask] = labels + offset
            offset += numpy.max(labels) + 1
    coords[:, -1] = new_subs
    return coords


def merge_lonely_voxels(solution, thres_size):
    '''Merges regions that are smaller than the size threshold with its closest neighbour.
    If more than one closest neighbour, computes the frontier with each close neighbour and 
    merges with the one that shares the most voxels.
    '''
    lst_subregions = [solution[:,:-1][solution[:,-1] == sub] for sub in numpy.unique(solution[:,-1])]
    i = 0
    while i < len(lst_subregions):
        if len(lst_subregions[i]) < thres_size:
            minDist = numpy.array([numpy.amin(cdist(lst_subregions[i], lst_subregions[k])) for k in range(len(lst_subregions))])
            minDist[minDist == 0] = 1e10 # Distance to itself, invalid
            if len(numpy.where(minDist == numpy.amin(minDist))[0]) < 2:
                idx = numpy.where(minDist == numpy.amin(minDist))[0][0]
            else: 
                try:
                    neigh = numpy.where(minDist == numpy.amin(minDist))[0]                 
                    boundaries = numpy.array([len(extract_frontier(lst_subregions[i], lst_subregions[j])) for j in neigh])               
                    idx = neigh[numpy.where(boundaries == numpy.amax(boundaries))][0]
                except:
                    idx = numpy.where(minDist == numpy.amin(minDist))[0][0]
            lst_subregions[idx] = numpy.row_stack((lst_subregions[idx], lst_subregions[i]))
            lst_subregions.pop(i)
            i = 0
        else: i +=1       
    idx = 1
    new_lst = []
    for subs in lst_subregions:
        new_lst.append(numpy.column_stack((subs, numpy.zeros((subs.shape[0], 1))+idx)))
        idx += 1
    new_subs = numpy.concatenate(new_lst)    
    return new_subs


# Added separated SVM splitter and validation process, coded in a way to handle 
# 2 SVMs classifications for the quadri classification
def splitter(parc_level, initial_solution, validation_function,  **kwargs):
    '''Function wrapping all the processes to split regions into subregions.
    '''
    out_solution = {}
    for region_name, solution in initial_solution.items():
        solution = splitting_process(parc_level, region_name, solution, validation_function, **kwargs)
        if solution is not None:
            solution = unflattening(parc_level, region_name, solution, only_sort=True)
            out_solution[region_name] = solution
    return out_solution

        
def splitting_process(parc_level, region_name, solution, validation_function, **kwargs):
    '''All steps for the splitting, gradient classification, Machine learned extrapolation, and post-process.
    '''
    split_is_validated = None
    ## If 1st classification is quadri clf, performs 2 SVMs
    if isinstance(solution, list):
        if any(len(numpy.unique(solution[i][:,-1][~numpy.isnan(solution[i][:,-1])])) < 2 for i in range(len(solution))):
            split_is_validated = False
        else: 
            out_C1 = split_with_SVM(solution[0], kwargs['C'], kwargs['gamma'], kwargs['thres_accuracy'], show=False)
            out_C2 = split_with_SVM(solution[1], kwargs['C'], kwargs['gamma'], kwargs['thres_accuracy'], show=False)
            solution0 = numpy.column_stack((out_C1[:,:-1], out_C1[:, -1] + 2 * out_C2[:, -1]))
            if len(numpy.unique(solution0[:,-1])) == 1:
                split_is_validated = False
    else: 
        if len(numpy.unique(solution[:,-1][~numpy.isnan(solution[:,-1])])) < 2:
                split_is_validated = False
        else:
            solution0 = split_with_SVM(solution, kwargs['C'], kwargs['gamma'], kwargs['thres_accuracy'], show=False)
            if len(numpy.unique(solution0[:,-1])) == 1:
                split_is_validated = False
    if split_is_validated is not False:    # Hierarchical clustering to differentiate regions  
        try:
            solution = extract_subregions(solution0, kwargs['t'])
        except: split_is_validated = False
    if split_is_validated is not False:
        solution = unflattening(parc_level, region_name, solution) # Unflattening to voxels
        if kwargs["merge"] == 1: # Merging small regions with closest neighbour
            solution = merge_lonely_voxels(solution, kwargs["thres_size"])
            if len(numpy.unique(solution[:,-1])) == 1:
                split_is_validated = False
            elif validation_function is not None: # Perform validation function if specified
                split_is_validated = validation_function(solution, **kwargs)
            else: split_is_validated = True
    if split_is_validated: return solution
    else: return None

def unflattening(parc_level, region, solution, only_sort=False):
    '''Unflatten the pixel classification to voxels.
    Add a sorting coordinates process to keep consistency.
    Set only_sort=True if you want to skip the unflattening process.
    '''
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    annotations = parc_level.region_volume
    hier = parc_level.hierarchy_root.find("acronym", region)[0]
    three_d_coords, two_d_coords = fm_analyses.flatmap_to_coordinates(annotations, fm0, hier)
    if only_sort == False:
        sort_coords = numpy.column_stack((two_d_coords, numpy.zeros((len(two_d_coords),1))))
        for i in range(len(sort_coords)):
            sort_coords[i,-1] = solution[numpy.where((sort_coords[i,:-1] == solution[:,:-1]).all(1))[0][0],-1]
        voxel_clf = numpy.column_stack((three_d_coords, sort_coords[:,-1]))
    elif only_sort == True:
        voxel_clf = numpy.column_stack((three_d_coords, numpy.zeros((len(three_d_coords),1))))
        for i in range(len(voxel_clf)):
            voxel_clf[i,-1] = solution[numpy.where((voxel_clf[i,:-1] == solution[:,:-1]).all(1))[0][0],-1]
    return voxel_clf

def extract_frontier(regionA, regionB):
    dist = cdist(regionA, regionB)
    min_dist = numpy.column_stack(numpy.where(dist==1)) #extract touching pixels/voxels
    frontier = numpy.vstack((numpy.unique(regionA[min_dist[:,0]], axis=0), numpy.unique(regionB[min_dist[:,1]], axis=0)))
    return frontier