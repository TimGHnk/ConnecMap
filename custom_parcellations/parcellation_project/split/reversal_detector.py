'''Reveral detector for gradients' classification.
'''


import numpy
import pandas
from scipy import stats, signal, sparse
from scipy.spatial.distance import pdist, squareform
import parcellation_project.analyses.flatmaps as fm_analyses
from parcellation_project.plotting import connectivity_structure

from ..tree_helpers import region_map_at

def convolve(A, B):
        return signal.convolve2d(A, B, "same") / signal.convolve2d(numpy.ones_like(A), B, "same")

def kernel(kernel_sz):
    """
    Just a quick gaussian 2d kernel. I think there are built-in methods for this..?
    But I could not find them, so I hacked this together
    """
    kernel = (
        stats.norm.pdf(
            numpy.sqrt(
                (numpy.dstack(numpy.meshgrid(numpy.linspace(-1, 1, kernel_sz),
                                            numpy.linspace(-1, 1, kernel_sz))) ** 2)
            .sum(axis=2)
            )
        )
    )
    return kernel


def gradient_on_filtered(img, kernel_sz):
    """
    Calculates the gradient, but does some smoothing first for a more robust result
    """
    assert img.ndim == 2
    img_cp = img.copy()
    img_cp[numpy.isnan(img_cp)] = numpy.nanmean(img_cp)  # Avoids the NaN sections growing
    img_flt = convolve(img_cp, kernel(kernel_sz))
    img_flt[numpy.isnan(img)] = numpy.NaN  # Put the NaNs back in ;)

    gradient = numpy.gradient(img_flt)
    gradient_abs = numpy.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
    DY, DX = gradient
    return DY, DX, gradient_abs


def interpolate_gradient(DY, DX):
    from scipy import interpolate
    X, Y = numpy.meshgrid(range(DY.shape[1]), range(DY.shape[0]))
    dy_flat = DY.flat; dx_flat = DX.flat; x_flat = X.flatten(); y_flat = Y.flatten()
    
    valid = ~numpy.isnan(dy_flat) & ~numpy.isnan(dx_flat)
    ip_y = interpolate.LinearNDInterpolator(numpy.vstack([x_flat[valid], y_flat[valid]]).transpose(),
                                            dy_flat[valid])
    ip_x = interpolate.LinearNDInterpolator(numpy.vstack([x_flat[valid], y_flat[valid]]).transpose(),
                                            dx_flat[valid])
    dy_flat[~valid] = ip_y(numpy.vstack([x_flat[~valid], y_flat[~valid]]).transpose())
    dx_flat[~valid] = ip_x(numpy.vstack([x_flat[~valid], y_flat[~valid]]).transpose())


def borderness(DY, DX):
    border_kernels = (kernel(5), kernel(5))

    tst1 = convolve(DY, numpy.sign(border_kernels[0]))
    tst2 = convolve(DX, numpy.sign(border_kernels[1]))
    borderness = numpy.sqrt((tst1 ** 2) + (tst2 ** 2))
    
    return borderness


def flatten_borderness(borderness_vals):
    X, Y = numpy.meshgrid(range(borderness_vals.shape[1]), range(borderness_vals.shape[0]))
    b_flat = borderness_vals.flatten(); x_flat = X.flatten(); y_flat = Y.flatten()
    valid = ~numpy.isnan(b_flat)

    b_flat = b_flat[valid]; x_flat = x_flat[valid]; y_flat = y_flat[valid]
    return y_flat, x_flat, b_flat

def weighted_neighbor_graph(y_flat, x_flat, b_flat):
    adj = squareform(pdist(numpy.vstack([x_flat, y_flat]).transpose())) < 2
    sparse_adj = sparse.coo_matrix(adj)

    sparse_adj.data = numpy.maximum(b_flat[sparse_adj.row], b_flat[sparse_adj.col])
    return sparse_adj


def connected_components_from_borders(cmat, magic_thresh, min_clst_sz=4):
    cmat = cmat.tocoo()
    islt = cmat.data < magic_thresh
    fltr_cmat = sparse.coo_matrix((cmat.data[islt], (cmat.row[islt], cmat.col[islt])), shape=cmat.shape)
    raw_cc = sparse.csgraph.connected_components(fltr_cmat)[1]
    cc_counts = pandas.Series(raw_cc).value_counts()
    active_components = cc_counts.index[cc_counts > min_clst_sz].values.tolist()
    out_clst = numpy.array([active_components.index(x) if x in active_components else -1 for x in raw_cc])
    quality = len(active_components) * cc_counts[active_components].sum()
    return out_clst, quality

def interpolate_missing_clusters(clst, pw_dist):
    clst = clst.copy()
    valid = clst >= 0
    valid_idx = numpy.nonzero(valid)[0]
    nneighbors = numpy.argmin(pw_dist[numpy.ix_(~valid, valid)], axis=1)
    clst[clst < 0] = clst[valid_idx[nneighbors]]

    return clst


def trace_fill_extrapolate(img, pre_filter_sz=5, post_filter_sz=1,
                           min_seed_cluster_sz=4, border_thresh=0.05,
                           message=False):
    if message == True:
        print("Filtering image and calculating gradient...")
    DY, DX, gradient_abs = gradient_on_filtered(img, kernel_sz=pre_filter_sz)
    DY = DY / gradient_abs
    DX = DX / gradient_abs
    if message == True:
        print("Interpolating invalid gradient values...")
    interpolate_gradient(DY, DX)
    if message == True:
        print("Done! Calculating the border parameter of the gradient...")
    borderness_vals = borderness(DY, DX)
    borderness_vals = convolve(borderness_vals, kernel(post_filter_sz))
    if message == True:    
        print("Done! Building neighbor graph...")
    y_flat, x_flat, b_flat = flatten_borderness(borderness_vals)
    adj_mat = weighted_neighbor_graph(y_flat, x_flat, 1.0 - b_flat)
    
    if message == True:
        print("Done! Calculating pairwise distances...")
    pw_dist_mat = sparse.csgraph.dijkstra(adj_mat, directed=False)
    pw_dist_mat = 0.5 * (pw_dist_mat + pw_dist_mat.transpose())
    if message == True:
        print("Done! Finding connected graph components...")
    clst, quality = connected_components_from_borders(adj_mat, border_thresh, min_clst_sz=min_seed_cluster_sz)
    if message == True:
        print("Done! Filling in missing cluster associations...")
    clst = interpolate_missing_clusters(clst, pw_dist_mat)
    if message == True:
        print("Done!")
    midx = pandas.MultiIndex.from_arrays([y_flat, x_flat], names=["Y", "X"])
    return pandas.Series(clst, index=midx), quality, DX, DY



def reversal_detector_tuner0(image, pre_filter_sz, post_filter_sz, min_seed_cluster_sz,
                            border_thresh):
    
    scores = dict()
    n_components = image.shape[2]
    for comp in range(n_components):
        try:
            _, quality, _, _ = trace_fill_extrapolate(image[:, :, comp],
                                                         pre_filter_sz=pre_filter_sz,
                                                         post_filter_sz=post_filter_sz,
                                                         min_seed_cluster_sz=min_seed_cluster_sz,
                                                         border_thresh=border_thresh)
            scores[comp] = quality
        except: continue
    if len(scores) == 0:
        print("Reversal detector failed, cancel splitting")
        return None
    else:
        component_star = max(scores, key=scores.get)
    return component_star


def reversal_index_for_quality(result, img, component, method="reversal"):
    """Compute the reversal index of one region, i.e. a measure of how much the connectivity
    gradients reverse for each regions of the results.
    """
    img_cp = img[:, :, component].copy()
    gradient = numpy.gradient(img_cp)
    DY, DX = gradient
    gradient_abs = numpy.sqrt(DX ** 2 + DY ** 2)
    DX = DX / gradient_abs
    DY = DY / gradient_abs
    rev_index = []
    sol = numpy.column_stack((numpy.vstack(result.index.to_numpy()), result.to_numpy())) 
    for i in numpy.unique(sol[:,-1]):
        vectors = numpy.column_stack((DX[sol[sol[:,-1] == i][:,0], sol[sol[:,-1] == i][:,1]],
                                      DY[sol[sol[:,-1] == i][:,0], sol[sol[:,-1] == i][:,1]]))
        vectors = numpy.delete(vectors, numpy.where(numpy.isnan(vectors))[0], 0)
        dist = pdist(vectors, metric='cosine')
        dist = dist / 2 * 180 # normalize between 0 and 180
        if method == "reversal":
            factor_reverse = numpy.count_nonzero(dist > 90) / len(dist)
        elif method == "variance":
            factor_reverse = numpy.var(dist)
        rev_index.append(factor_reverse)
    quality = numpy.mean(rev_index)
    return quality


def reversal_detector_tuner(image, pre_filter_sz, post_filter_sz, min_seed_cluster_sz,
                            border_thresh):
    '''Use the reversal index for each region of the output and select the component
    that minimize this value.
    '''
    
    scores = dict()
    n_components = image.shape[2]
    for comp in range(n_components):
        try:
            result, _, _, _ = trace_fill_extrapolate(image[:, :, comp],
                                                         pre_filter_sz=pre_filter_sz,
                                                         post_filter_sz=post_filter_sz,
                                                         min_seed_cluster_sz=min_seed_cluster_sz,
                                                         border_thresh=border_thresh)
            if len(numpy.unique(result)) < 2:
                score = 1e5 # will be rejected
            else:
                score = reversal_index_for_quality(result, image, comp)
            scores[comp] = score
        except: continue
    if len(scores) == 0:
        print("Reversal detector failed, cancel splitting")
        return None
    else:
        component_star = min(scores, key=scores.get)
    return component_star

def reversal_detector(region, fm0, fm1, annotations, hierarchy_root, component="optimized",
                       pre_filter_sz=5, post_filter_sz=1, min_seed_cluster_sz=4, border_thresh=0.05):
    
    hierarchy_reg = region_map_at(hierarchy_root, region)
    three_d_coords, two_d_coords = fm_analyses.flatmap_to_coordinates(annotations, fm0, hierarchy_reg)
    two_d_coords = numpy.unique(two_d_coords, axis=0)
    
    img = connectivity_structure(region, fm0, fm1, annotations, hierarchy_root)
    if component == "optimized":
        print("Searching best component...")
        comp = reversal_detector_tuner(img, pre_filter_sz=pre_filter_sz,
                                                 post_filter_sz=post_filter_sz,
                                                 min_seed_cluster_sz=min_seed_cluster_sz,
                                                 border_thresh=border_thresh)
        if comp is None:
            output = numpy.column_stack((two_d_coords, numpy.zeros(len(two_d_coords))))
            return output
        else: 
            print(f"Done! Reversal detection on component {comp}")
    else: comp = component
    result, quality, DX, DY = trace_fill_extrapolate(img[:, :, comp],
                                                     pre_filter_sz=pre_filter_sz,
                                                     post_filter_sz=post_filter_sz,
                                                     min_seed_cluster_sz=min_seed_cluster_sz,
                                                     border_thresh=border_thresh, message=True)

    coords = numpy.vstack(result.index.to_numpy())
    clf = result.to_numpy()
    mask = (coords[:, None] == two_d_coords).all(-1).any(-1)
    grad_clf = numpy.column_stack((coords[mask], clf[mask]))
    solution = numpy.column_stack((two_d_coords, numpy.empty(len(two_d_coords))))
    solution[:,-1] = numpy.NaN
    for xy in grad_clf:
        idx = numpy.where((solution[:,0] == xy[0]) & (solution[:,1] == xy[1]))[0][0]
        solution[idx,-1] = xy[-1]
    return solution

