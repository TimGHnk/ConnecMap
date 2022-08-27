"""For analyses of flatmaps
"""
import numpy
import random
import voxcell
import matplotlib.pyplot as plt
from voxel_maps import coordinates_to_image
import os
import pandas as pd
from scipy.spatial.distance import pdist, squareform




# Function below provides a wrapper that turns a function with multiple inputs into one that looks up most info from
# the ParcellationLevel object.
# The region_root input is a voxcell.Hierarchy object representing the region that is to be analyzed. If not provided
# the root region is used, i.e. most likely Isocortex
def from_parcellation(base_func):
    def returned_function(parc_level, region_root=None, **kwargs):
        fm0_fn = parc_level._config["anatomical_flatmap"]
        fm0 = voxcell.VoxelData.load_nrrd(fm0_fn)
        annotations = parc_level.region_volume
        fm1 = parc_level.flatmap
        if region_root is None:
            region_root = parc_level.hierarchy_root  # Perform analysis at root level. You might not want to do this
        return base_func(fm0, fm1, annotations, region_root, **kwargs)
    return returned_function



def flatmap_to_coordinates(ann, fm, hierarchy_root):
    '''
    Allows to visualize the shape of region with 3d and 2d coordinates from a flatmap
    without using cache, you can theoritically visualize any regions from a given
    annotation file.
    '''
    lst_ids = list(hierarchy_root.get("id"))
    coords = []
    for x in range(int(ann.raw.shape[0])):
        for y in range(int(ann.raw.shape[1])):
            for z in range(int(ann.raw.shape[2]/2),int(ann.raw.shape[2])):
                if ann.raw[x,y,z] in lst_ids:
                    coords.append([x,y,z])
    coords_3d = numpy.vstack(coords)
    coords = []
    flatmap = numpy.round(fm.raw)
    x_fm = flatmap[:,:,:,0]
    y_fm = flatmap[:,:,:,1]
    for xyz in coords_3d:
        coords.append([x_fm[xyz[0], xyz[1], xyz[2]], y_fm[xyz[0], xyz[1], xyz[2]]])
    coords_2d = numpy.vstack(coords)
    return coords_3d, coords_2d

def region_image(region, pxl, ann, hierarchy_root):
    ann_vals = ann.raw.reshape((-1,))
    xy = pxl.raw.reshape((-1, 2))
    img_xy = numpy.empty(shape=(0, 2))
    img_vals = numpy.empty(shape=(0, 1))
    tgt_region_ids = list(hierarchy_root.collect('acronym', region, 'id'))
    sub_xy = xy[numpy.in1d(ann_vals, tgt_region_ids), :]
    img_xy = numpy.vstack([img_xy, sub_xy])
    img_vals = numpy.vstack([img_vals, 0 * numpy.ones((len(sub_xy), 1))])
    return coordinates_to_image(img_vals, img_xy)


def gradient_map(fm0, fm1, annotations, hierarchy_root, show=True, normalize=True, **kwargs):
    """Compute connectivity gradients of one region and plot them onto its anatomical
    flatmap. Only considers the 2 first gradients.
    """
    region = hierarchy_root.data['acronym']
    if show is True:
        ax = plt.figure(figsize=(20, 20)).gca()
        ax.imshow(region_image(region, fm0, annotations, hierarchy_root), cmap="tab20b")
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
    gY1, gX1 = numpy.gradient(tmp_img[:, :, 0])
    gY2, gX2 = numpy.gradient(tmp_img[:, :, 1])
    # remove zero values
    gX1[gX1 == 0] = 1e-10
    gY1[gY1 == 0] = 1e-10
    gX2[gX2 == 0] = 1e-10
    gY2[gY2 == 0] = 1e-10

    if normalize:
        gX1, gY1 = normalize_gradient(gX1, gY1)
        gX2, gY2 = normalize_gradient(gX2, gY2)
    if show is True:
        y_min = min(m for m in sub_xy[:,0] if m >= 0)
        y_max = max(m for m in sub_xy[:,0] if m >= 0)
        x_min = min(m for m in sub_xy[:,1] if m >= 0)
        x_max = max(m for m in sub_xy[:,1] if m >= 0)
        plot_gradient_map(gX1, gY1, gX2, gY2, ax=ax, scale_arrows=((x_max-x_min+6)*1.285714))
        plt.title(f'Gradient map of {region}', fontsize = 30)
        plt.axis((x_min-3,x_max+3,y_min-3,y_max+5))
        if 'output_root' in kwargs:
            plt.savefig(kwargs['output_root']+f'/gradient_map_{region}.png')
        plt.show()
    return gX1, gY1, gX2, gY2


def plot_gradient_map(gX1, gY1, gX2, gY2, ax, scale_arrows):
    X, Y = numpy.meshgrid(numpy.arange(gX1.shape[1]), numpy.arange(gX1.shape[0]))
    ax.quiver(X, Y, gX1, gY1, color='yellow', scale_units='width', headaxislength=110,headlength=110, headwidth=60, scale=scale_arrows);
    ax.quiver(X, Y, gX2, gY2, color='red', scale_units='width', headaxislength=110,headlength=110, headwidth=60, scale=scale_arrows)
    colors = {'\u03B1-axis':'yellow', '\u03B2-axis':'red'}           
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ax.legend(handles, labels, prop={'size': 20})
    ax.set_ylim(sorted(ax.get_ylim()))
    


def degree_matrix(fm0, fm1, annotations, hierarchy_root, **kwargs):
    gX1, gY1, gX2, gY2 = gradient_map(fm0, fm1, annotations, hierarchy_root, show=False)
    deg_arr = gX1.copy()
    for x,y in numpy.ndindex(deg_arr.shape):
        if deg_arr[x,y] != numpy.isnan:
            deg_arr[x,y] = _compute_angle(gX1[x,y], gY1[x,y], gX2[x,y], gY2[x,y], ret_degree=True)
    return deg_arr


def _compute_angle(x1, y1, x2, y2, ret_degree=False):
    vector_1 = [x1, y1]
    vector_2 = [x2, y2]
    unit_vector_1 = vector_1 / numpy.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / numpy.linalg.norm(vector_2)
    dot_product = numpy.dot(unit_vector_1, unit_vector_2)
    angle = numpy.arccos(dot_product)
    if ret_degree is True:
        angle = numpy.rad2deg(angle)
    return angle



def gradient_deviation(fm0, fm1, annotations, hierarchy_root, plot=True, **kwargs):
    """Compute the gradient deviations for each pixels, i.e. how much the angle between
    the 2 connectivity gradients deviates from 90°.
    If 'output_root' is in kwargs, save the mean gradient deviation for this region in a .csv file.
    """
    deg_arr = degree_matrix(fm0, fm1, annotations, hierarchy_root, **kwargs)
    bitmask = numpy.invert(numpy.isnan(deg_arr))
    deviations = [abs(deg_arr[bitmask].flatten()[i] - 90) for i in range(len(deg_arr[bitmask].flatten()))]
    if plot is True:
        fig1, ax1 = plt.subplots()
        ax1.boxplot(deviations, showfliers = True)
        ax1.set_xticklabels([hierarchy_root.data['acronym']])
        ax1.set_title("Gradients' deviation")
        if 'output_root' in kwargs:
            plt.savefig(kwargs['output_root']+f"/gradient_deviation_{hierarchy_root.data['acronym']}.png")
        plt.show()
    if 'output_root' in kwargs:
        save_results('gradient_deviation', numpy.mean(deviations), hierarchy_root.data['acronym'], kwargs['output_root'])
    return deviations
    

def reversal_index(fm0, fm1, annotations, hierarchy_root, **kwargs):
    """Compute the reversal index of one region, i.e. a measure of how much the connectivity
    gradients reverse.
    """
    three_d_coords, two_d_coords = flatmap_to_coordinates(annotations, fm0, hierarchy_root)
    two_d_coords = numpy.unique(two_d_coords, axis=0)
    x1,y1,x2,y2 = gradient_map(fm0, fm1, annotations, hierarchy_root, show=False) 
    if numpy.count_nonzero(~numpy.isnan(x1)) < 2: # 1 or less gradient, won't work
        factor_reverse = float(0)
    else:
        vecX = numpy.column_stack((x1[two_d_coords[:,0], two_d_coords[:,1]], y1[two_d_coords[:,0], two_d_coords[:,1]]))
        vecY = numpy.column_stack((x2[two_d_coords[:,0], two_d_coords[:,1]], y2[two_d_coords[:,0], two_d_coords[:,1]]))
        x_arr = numpy.delete(vecX, numpy.where(numpy.isnan(vecX))[0], 0)
        y_arr = numpy.delete(vecY, numpy.where(numpy.isnan(vecY))[0], 0)
        distX = pdist(x_arr, metric='cosine')
        distY = pdist(y_arr, metric='cosine')
        distX = distX / 2 * 180 # normalize between 0 and 180
        distY = distY / 2 * 180 # normalize between 0 and 180
        reversal_x = numpy.count_nonzero(distX > 90) / len(distX)
        reversal_y = numpy.count_nonzero(distY > 90) / len(distY)
        factor_reverse = (reversal_x + reversal_y)
    if "output_root" in kwargs:
        save_results("reversal_index", factor_reverse, hierarchy_root.data["acronym"], kwargs["output_root"])
    return factor_reverse

def abs_cos_angle(v1, v2):
    l1 = numpy.sqrt(numpy.sum(v1 ** 2, axis=1, keepdims=True))
    l2 = numpy.sqrt(numpy.sum(v2 ** 2, axis=1, keepdims=True))
    ret = numpy.arccos(numpy.sum((v1 / l1) * (v2 / l2), axis=1))
    return numpy.abs(ret - numpy.pi / 2) / (numpy.pi / 2)

def banana_factor(fm0, fm1, annotations, hierarchy_root, plot=True, **kwargs):
    x1, y1, x2, y2 = gradient_map(fm0, fm1, annotations, hierarchy_root, show=False, **kwargs)
    g1 = numpy.dstack([x1, y1]).reshape((-1, 2))
    g2 = numpy.dstack([x2, y2]).reshape((-1, 2))
    banana = abs_cos_angle(g1, g2).reshape(x1.shape)
    if plot is True:
        ax = plt.figure(figsize=(10, 6)).gca()
        ax.imshow(banana)
        plt.colorbar(ax.imshow(banana), ax=ax, orientation='vertical')
        plt.title(f"Banana factor_{hierarchy_root.data['acronym']}")
        if 'output_root' in kwargs:
            plt.savefig(kwargs['output_root']+f"/banana_factor_{hierarchy_root.data['acronym']}.png")
        plt.show()
    if 'output_root' in kwargs:
        save_results('banana_factor', numpy.nanmean(banana), hierarchy_root.data['acronym'], kwargs['output_root'])
    return banana

def find_pre_images(fm0, fm1, annotations, hierarchy_root, method, n=10):
    '''Returns a list of the preimages of random pixels (n=number of random pixels,
    defaults is set to 10) of either the anatomical of the diffusion flatmap.
    method = anatomical OR diffusion. If n > number of pixels returns the pre images
    of every pixels.
    Returns also coordinates of the random pixels selected and the 3d coordinates of
    the region if needed.
    '''
    if method == 'anatomical':
        flatmap = fm0
    elif method == 'diffusion':
        flatmap = fm1
    else:
        raise ValueError("flatmap not recognized, use 'anatomical' or 'diffusion'")
    coords_3d, coords_2d = flatmap_to_coordinates(annotations, flatmap, hierarchy_root)
    coords_2d = numpy.round(numpy.unique(coords_2d, axis=0))
    fm = numpy.round(flatmap.raw)
    x_fm = fm[:,:,:,0]
    y_fm = fm[:,:,:,1]
    pre_images = []
    pxl_idx = []
    if n > len(coords_2d):
        n = len(coords_2d)
    for i in range(n):
        if n == len(coords_2d):
            idx = i
        else:
            idx = random.randrange(0,len(coords_2d))
        pos_x = coords_2d[idx][0]
        pos_y = coords_2d[idx][1]
        pxl_idx.append(numpy.array([pos_x, pos_y]))
        pre_img = numpy.column_stack(numpy.where((x_fm == pos_x) & (y_fm == pos_y)))
        pre_images.append(numpy.vstack(pre_img))        
    return pre_images, pxl_idx, coords_3d


def save_results(analysis_name, results, region, analysis_root):
    if os.path.isfile(analysis_root+f'/{analysis_name}.csv') is False:
        df = pd.DataFrame(columns=['region', analysis_name])
    else:
        df = pd.read_csv(rf'{analysis_root}/{analysis_name}.csv', sep = ";")
    df.at[len(df)+1, 'region'] = region # TODO: Improve code
    df.at[len(df), analysis_name] = results
    df.to_csv(analysis_root+f'/{analysis_name}.csv', sep = ";", index=False)
    
    


gradient_map_from_parcellation = from_parcellation(gradient_map)
degree_matrix_from_parcellation = from_parcellation(degree_matrix)
banana_factor_from_parcellation = from_parcellation(banana_factor)
reversal_index_from_parcellation = from_parcellation(reversal_index)
gradient_deviation_from_parcellation = from_parcellation(gradient_deviation)
pre_image_of_flatmap = from_parcellation(find_pre_images)