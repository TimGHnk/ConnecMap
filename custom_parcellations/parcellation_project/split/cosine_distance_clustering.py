import hdbscan
import numpy
from voxel_maps import coordinates_to_image
from scipy.spatial.distance import pdist, squareform


def extract_gradients(fm0, fm1, annotations, hierarchy_root):
    """Compute connectivity gradients of one region and plot them onto its anatomical
    flatmap.
    """
    region = hierarchy_root.data['acronym']
    mask = numpy.all((~numpy.isnan(fm1.raw)) & (fm1.raw > -1), axis=3)
    ann_vals = annotations.raw[mask]
    xy = fm0.raw[mask]
    ab = fm1.raw[mask]

    def normalize_gradient(gX, gY):
        l = numpy.sqrt(gX ** 2 + gY ** 2)
        return gX / l, gY / l

    tgt_region_ids = list(hierarchy_root.collect('acronym', region, 'id'))
    bitmask = numpy.in1d(ann_vals, tgt_region_ids)
    sub_xy = xy[bitmask]
    sub_ab = ab[bitmask]
    tmp_img = coordinates_to_image(sub_ab, sub_xy)
    gXs = []; gYs = []
    for i in range(tmp_img.shape[2]):
        gX, gY = numpy.gradient(tmp_img[:, :, i])
        gXs.append(gX); gYs.append(gY)
    for i in range(len(gXs)):
        gXs[i][gXs[i] == 0] = 1e-10
        gYs[i][gYs[i] == 0] = 1e-10
        gXs[i], gYs[i] = normalize_gradient(gXs[i], gYs[i])
    return gXs, gYs

def sum_of_cos_distances(coords, gXs, gYs, lambdas, N):
    if N > len(gXs):
        N = len(gXs)
    for i in range(N):
        arr = numpy.column_stack((gXs[i][coords[:,0], coords[:,1]].flatten(),
                                  gYs[i][coords[:,0], coords[:,1]].flatten()))
        arr = arr[~numpy.isnan(arr)].reshape(-1,2)
        mat = squareform(pdist(arr, metric='cosine'))
        if i == 0:
            distanceMatrix = mat * lambdas[i]
        else:
            distanceMatrix += (mat * lambdas[i]) 
    distanceMatrix = distanceMatrix / N
    return distanceMatrix

def cosine_distance_clustering(gXs, gYs, two_d_coords, lambdas, **kwargs):
    ''' Uses HDBSCAN clustering for gradients classification.
    '''
    if numpy.count_nonzero(gXs[0][~numpy.isnan(gXs[0])]) < 2:
        print("N gradients < 2, not enough !")
        return numpy.column_stack((two_d_coords, numpy.zeros(len(two_d_coords)))) # not enough gradients to work with.
    
    if "N" in kwargs:
        N = kwargs["N"]
    else:
        N = len(gXs)
    distanceMatrix = sum_of_cos_distances(two_d_coords, gXs, gYs, lambdas, N=N)
    clusterer = hdbscan.HDBSCAN(algorithm='best',
                                alpha=kwargs["alpha"],
                                metric='precomputed',
                                cluster_selection_epsilon=kwargs["eps"],
                                min_cluster_size=kwargs["min_cluster_size"],
                                min_samples=kwargs["min_samples"],
                                cluster_selection_method = "eom")   
    
    clusterer.fit(distanceMatrix)
    clf = clusterer.labels_
    vec = gXs[0][two_d_coords[:,0], two_d_coords[:,1]] # to extract Nan
    X = numpy.delete(two_d_coords, numpy.where(numpy.isnan(vec))[0], 0)
    X_clf = numpy.column_stack((X, clf)).astype("float64")
    X_clf[:,-1][X_clf[:,-1] == -1] = numpy.NaN
    X_nan = two_d_coords[numpy.unique(numpy.where(numpy.isnan(vec))[0])]
    labels = numpy.empty((len(X_nan),1))
    labels[:] = numpy.NaN
    X_nan = numpy.column_stack((X_nan, labels))
    grad_clf = numpy.row_stack((X_clf, X_nan))
    return grad_clf