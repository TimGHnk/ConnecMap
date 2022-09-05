''' Make a null connectivity dataset under the assumption of a 
    continuous and homogenous connectivity.
    NOTE: Currently only the regular architecture is supported, the irregular
    one is still in development.
'''
import numpy
import voxcell
from voxel_maps import coordinates_to_image
from parcellation_project.analyses.flatmaps import gradient_map
from parcellation_project.plotting import connectivity_structure
from scipy.spatial.distance import cdist, pdist
import copy
import matplotlib.pyplot as plt
from parcellation_project.tree_helpers import leaves, normalization_spread, normalization_offsets
from parcellation_project.tree_helpers import at_depth, max_depth
import pandas as pd
from scipy.cluster.hierarchy import ward, fcluster
import hdbscan
from parcellation_project.split.decide_split import HDBSCAN_classification
from parcellation_project.split import reversal_detector
from parcellation_project.split.decide_split import split_with_SVM, extract_subregions, merge_lonely_voxels
from parcellation_project.split.visualize_split import viz_gradient_split
from parcellation_project.analyses.flatmaps import gradient_deviation, reversal_index
from parcellation_project.split.cosine_distance_clustering import extract_gradients, cosine_distance_clustering

    
# from .embed import compute_diffusion_map # part of this repo. originally: https://github.com/satra/mapalign
# # if import of compute_diffusion_map fails, run this

def compute_diffusion_map(L, alpha=0.5, n_components=None, diffusion_time=0,
                          skip_checks=False, overwrite=False,
                          eigen_solver=None, return_result=False):

    import scipy.sparse as sps

    use_sparse = False
    if sps.issparse(L):
        use_sparse = True

    if not skip_checks:
        from sklearn.manifold.spectral_embedding_ import _graph_is_connected
        if not _graph_is_connected(L):
            raise ValueError('Graph is disconnected')


    ndim = L.shape[0]
    if overwrite:
        L_alpha = L
    else:
        L_alpha = L.copy()

    if alpha > 0:
        # Step 2
        d = numpy.array(L_alpha.sum(axis=1)).flatten()
        d_alpha = numpy.power(d, -alpha)
        if use_sparse:
            L_alpha.data *= d_alpha[L_alpha.indices]
            L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())
            L_alpha.data *= d_alpha[L_alpha.indices]
            L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())
        else:
            L_alpha = d_alpha[:, numpy.newaxis] * L_alpha 
            L_alpha = L_alpha * d_alpha[numpy.newaxis, :]

    # Step 3
    d_alpha = numpy.power(numpy.array(L_alpha.sum(axis=1)).flatten(), -1)
    if use_sparse:
        L_alpha.data *= d_alpha[L_alpha.indices]
    else:
        L_alpha = d_alpha[:, numpy.newaxis] * L_alpha

    M = L_alpha

    from scipy.sparse.linalg import eigs, eigsh
    if eigen_solver is None:
        eigen_solver = eigs

    # Step 4
    func = eigen_solver
    if n_components is not None:
        lambdas, vectors = func(M, k=n_components + 1)
    else:
        lambdas, vectors = func(M, k=max(2, int(numpy.sqrt(ndim))))
    del M

    if func == eigsh:
        lambdas = lambdas[::-1]
        vectors = vectors[:, ::-1]
    else:
        lambdas = numpy.real(lambdas)
        vectors = numpy.real(vectors)
        lambda_idx = numpy.argsort(lambdas)[::-1]
        lambdas = lambdas[lambda_idx]
        vectors = vectors[:, lambda_idx]

    return _step_5(lambdas, vectors, ndim, n_components, diffusion_time,
                   return_result)


def _step_5(lambdas, vectors, ndim, n_components, diffusion_time, return_result):
    psi = vectors/vectors[:, [0]]
    diffusion_times = diffusion_time
    if diffusion_time == 0:
        diffusion_times = numpy.exp(1. -  numpy.log(1 - lambdas[1:])/numpy.log(lambdas[1:]))
        lambdas = lambdas[1:] / (1 - lambdas[1:])
    else:
        lambdas = lambdas[1:] ** float(diffusion_time)
    lambda_ratio = lambdas/lambdas[0]
    threshold = max(0.05, lambda_ratio[-1])
    n_components_auto = numpy.amax(numpy.nonzero(lambda_ratio > threshold)[0])
    n_components_auto = min(n_components_auto, ndim)
    if n_components is None:
        n_components = n_components_auto
    embedding = psi[:, 1:(n_components + 1)] * lambdas[:n_components][None, :]

    if return_result:
        result = dict(lambdas=lambdas, vectors=vectors,
                      n_components=n_components, diffusion_time=diffusion_times,
                      n_components_auto=n_components_auto)
        return embedding, result
    else:
        return embedding

# Functions

def normalize_fm_coordinates(fm_coords, normalize_args=-1, multiply=numpy.NaN):
    fm_coords = fm_coords - numpy.nanmin(fm_coords, axis=0, keepdims=True)
    # normalize_spread = numpy.array(normalize_args['normalize_spread'])
    normalize_spread = numpy.repeat(normalize_args['normalize_spread'][0], fm_coords.shape[1])
    # offsets = numpy.array(normalize_args['normalize_offsets'])
    offsets = numpy.repeat(normalize_args['normalize_offsets'][0], fm_coords.shape[1])
    multiply = numpy.array(multiply)
    if numpy.all(normalize_spread > 0):
        fm_coords = numpy.array(normalize_spread) * fm_coords / (numpy.nanmax(fm_coords, axis=0, keepdims=True) -
                                                                 numpy.nanmin(fm_coords, axis=0, keepdims=True))
        fm_coords = offsets + fm_coords
    if not numpy.any(numpy.isnan(multiply)):
        fm_coords = numpy.array(multiply) * fm_coords
    return fm_coords

def gaussian_func(x, mu, sig):
    return numpy.exp(-numpy.square(x - mu) / (2 * numpy.square(sig)))

def crisscross(volume, grid, higher_grid):
    width = volume.shape[0]
    height = volume.shape[1]
    ids = numpy.unique(volume)
    reg_id = numpy.amax(volume) + 1
    for reg in ids:
        idxx = numpy.where(volume == reg)
        offset_x = numpy.amin(idxx[0])
        for x in range(int(grid[0] / higher_grid[0])):
            offset_y = numpy.amin(idxx[1])
            for y in range(int(grid[1] / higher_grid[1])):
                volume[int(offset_x):int(offset_x+width/grid[0]),int(offset_y):int(offset_y+height/grid[1]),:] = reg_id
                offset_y += height /grid[1]
                reg_id += 1
            offset_x += width /grid[0]
            
                
                
def reverseGradient(model, width, height, grid):
    fm1 = copy.deepcopy(model)
    comp = numpy.arange(1,width/grid[0]+1)
    rev_comp = numpy.flip(comp)
    i = 0
    n = 0
    rev = False
    while i < fm1.raw.shape[0]:  
        if rev == False:
            fm1.raw[i,:,:,0] = comp[n]
        else:
            fm1.raw[i,:,:,0] = rev_comp[n]
        if n == len(comp) - 1:
            n = 0
            rev = not rev  
        else:
            n += 1
        i += 1
        
    comp = numpy.arange(1,height/grid[1]+1)
    rev_comp = numpy.flip(comp)
    i = 0
    n = 0
    rev = False
    while i < fm1.raw.shape[1]:     
        if rev == False:
            fm1.raw[:,i,:,1] = comp[n]
        else:
            fm1.raw[:,i,:,1] = rev_comp[n]
        if n == len(comp) - 1:
            n = 0
            rev = not rev  
        else:
            n += 1
        i += 1
    return fm1

def normalize01(x):
    return (x - numpy.amin(x)) / (numpy.amax(x)-numpy.amin(x))
            
def path_distance(r0, idA, idB):
    if idA == idB:
        return 1
    else:
        distance = 1
        level = max_depth(r0) - 1
        foundit = False
        while foundit == False:
            leaves = at_depth(r0, level, "id")
            for leaf in leaves:
                if all(item in list(r0.collect("id", leaf, "id")) for item in [idA,idB]) == True:
                    foundit = True
                    return 1 / (distance*2)
            distance += 1
            level -= 1
    
def all_nodes_distance(r0):
    nodes = at_depth(r0, max_depth(r0), property="id")
    distance_frame = pd.DataFrame(index=nodes, columns=nodes)
    for source in distance_frame.index:
        for target in distance_frame.columns:
            distance_frame.loc[source, target] = path_distance(r0, source, target)
    return distance_frame

def make_anatomical_fm(self):
    pxl_y, pxl_x = numpy.meshgrid(range(self.height), range(self.width))
    pixel_x = pxl_x.reshape((self.width, self.height, 1)) + 1
    pixel_y = pxl_y.reshape((self.width, self.height, 1)) + 1
    if self.depth > 1:
        for i in range(self.depth-1):
            pixel_x = numpy.dstack((pixel_x, pxl_x + 1))
            pixel_y = numpy.dstack((pixel_y, pxl_y + 1))
    fm0 = voxcell.VoxelData(numpy.stack((pixel_x, pixel_y), axis=3), (self.width,self.height,self.depth))
    return fm0

from scipy import stats, signal, sparse

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

def random_split_initial(width, height, depth, N):
    vol = numpy.zeros((width,height,depth))
    coords = numpy.column_stack(numpy.where(vol != numpy.nan))
    linkage = ward(coords)
    clusters = fcluster(linkage, N, criterion="maxclust") + 1
    res = numpy.column_stack((coords, clusters)).astype("int")
    vol[res[:,0], res[:,1], res[:,2]] = res[:,-1]
    return vol

def random_split(vol, coords, N):
    linkage = ward(coords)
    id_off = numpy.amax(vol)
    clusters = fcluster(linkage, N, criterion="maxclust") + id_off
    res = numpy.column_stack((coords, clusters)).astype("int")
    vol[res[:,0], res[:,1], res[:,2]] = res[:,-1]
    return vol

def reversing_gradients_initial(rubix, masked_annotation, kernel_sz=1):
    fm01 = copy.deepcopy(rubix.anatomical_flatmap)
    bitmask = numpy.in1d(masked_annotation, numpy.amax(masked_annotation)).reshape(*masked_annotation.shape)
    if sum([len(numpy.unique(masked_annotation[i,:]))-1 for i in range(masked_annotation.shape[0])]) < sum([len(numpy.unique(masked_annotation[:,i]))-1 for i in range(masked_annotation.shape[1])]):
        to_reverse = fm01.raw[:,:,:,0]
        to_reverse[bitmask] = numpy.negative(to_reverse[bitmask]) + numpy.amax(to_reverse[bitmask]) +1
    else:
        to_reverse = fm01.raw[:,:,:,1]
        to_reverse[bitmask] = numpy.negative(to_reverse[bitmask]) + numpy.amax(to_reverse[bitmask]) +1
    return fm01

def reversing_gradients(rubix, annotation, fm01, mask, kernel_sz=1):
    masked_annotation = numpy.ma.masked_array(annotation, numpy.invert(mask))
    bitmask = numpy.in1d(masked_annotation, numpy.amax(masked_annotation)).reshape(*masked_annotation.shape)
    if sum([len(numpy.unique(masked_annotation[i,:]))-1 for i in range(masked_annotation.shape[0])]) > sum([len(numpy.unique(masked_annotation[:,i]))-1 for i in range(masked_annotation.shape[1])]):
        to_reverse = numpy.ma.masked_array(fm01.raw[:,:,:,0], numpy.invert(mask))
        to_reverse[bitmask] = numpy.negative(to_reverse[bitmask]) + numpy.amax(to_reverse[bitmask]) +1
    else:
        to_reverse = numpy.ma.masked_array(fm01.raw[:,:,:,1], numpy.invert(mask))
        to_reverse[bitmask] = numpy.negative(to_reverse[bitmask]) + numpy.amax(to_reverse[bitmask]) +1
    return fm01

def irregularRubix(self, n_level, n_split=2): #TODO: finish implementation
    self.anatomical_flatmap = make_anatomical_fm(self)
    hier = voxcell.Hierarchy({"name": "Rubix",
                          "acronym": "Rubix",
                          "id": 1})
    vol = random_split_initial(self.width, self.height, self.depth, n_split)
    for i in range(len(numpy.unique(vol))):
        hier.children.append(voxcell.Hierarchy({"name": "Module_" + str(i+1),
                                        "acronym": "Module_" + str(i+1),
                                        "id": int(numpy.unique(vol)[i])}))
    fm1s = []
    fm = reversing_gradients_initial(self, vol, self.kernel_size)
    fm1s.append(copy.deepcopy(fm))
    for n in range(n_level-1):
        for i in numpy.unique(vol):
            r = hier.find("id", i)[0]
            reg = r.data["acronym"]
            coords = numpy.column_stack(numpy.where(vol == i))
            masked_mat = numpy.in1d(vol, i).reshape(*vol.shape)
            vol = random_split(vol, coords, n_split)
            fm = reversing_gradients(self, vol, fm, masked_mat, self.kernel_size)
            for j in range(len(numpy.unique(vol[masked_mat]))):
                r.children.append(voxcell.Hierarchy({"name": reg + "_" + str(j+1),
                        "acronym": reg + "_" + str(j+1),
                        "id": int(numpy.unique(vol[masked_mat])[j])}))
        fm1s.append(copy.deepcopy(fm))
    if self.hierarchy_method == "reversing_hierarchy":
        fm1 = fm1s[-self.levels:]
    elif self.hierarchy_method == "node_distance":
        fm1 = fm1s[-1]
    ann = voxcell.VoxelData(vol, (self.width,self.height,self.depth))
    return ann, hier, self.anatomical_flatmap, fm1
        
def regularRubix(self):
    width = self.width
    height = self.height
    depth = self.depth
    grid = self.grid
    assert numpy.mod(width,grid[0]) == 0, "Wrong dimensions for desired number of regions"
    assert numpy.mod(height,grid[1]) == 0, "Wrong dimensions for desired number of regions"
    assert numpy.mod(width*height,grid[0]*grid[1]) == 0, "Wrong dimensions for desired number of regions"
    assert (width*height*depth) / self.number_regions >= 100, "Regions must be at least 100 voxels"
    rubix = numpy.zeros((width,height,depth))
    reg_id = 2
    offset_x = 0
    for x in range(grid[0]):
        offset_y = 0
        for y in range(grid[1]):
            rubix[int(offset_x):int(offset_x+width/grid[0]),int(offset_y):int(offset_y+height/grid[1]),:] = reg_id
            offset_y += height/grid[1]
            reg_id += 1
        offset_x += width/grid[0]
    ann = voxcell.VoxelData(rubix, (width,height,depth))
    hier = voxcell.Hierarchy({"name": "Rubix",
                              "acronym": "Rubix",
                              "id": 1})
    for i in range(grid[0]*grid[1]):
        hier.children.append(voxcell.Hierarchy({"name": "Module_" + str(i+1),
                                                "acronym": "Module_" + str(i+1),
                                                "id": i+2}))
    if self._hierarchical == True:
        higher_grid = grid
        for level in self.hierarchical:
            sub_grid = level["grid"]
            assert numpy.mod(sub_grid[0]*sub_grid[1], higher_grid[0]*higher_grid[1]) == 0, "Wrong hierarchical organization"
            regions = leaves(hier, property="acronym")
            for reg in regions:
                child = int((sub_grid[0]*sub_grid[1]) / (higher_grid[0]*higher_grid[1]))
                for i in range(child):
                    id_offset = int(numpy.max(list(hier.get("id"))) + 1)
                    r = hier.find("acronym", reg)[0]
                    r.children.append(voxcell.Hierarchy({"name": reg + "_" + str(i+1),
                                            "acronym": reg + "_" + str(i+1),
                                            "id": id_offset}))
            crisscross(ann.raw, sub_grid, higher_grid)
            higher_grid = sub_grid

    fm0 = make_anatomical_fm(self)
    if self._hierarchical == False:
        fm1 = reverseGradient(fm0, width, height, grid)
    else:
        if self.hierarchy_method == "node_distance":
            fm1 = reverseGradient(fm0, width, height, self.hierarchical[-1]["grid"])
        elif self.hierarchy_method == "reversing_hierarchy":
            fm = reverseGradient(fm0, width, height, grid)
            fm1 = [fm]
            for level in self.hierarchical:
                fm1.append(reverseGradient(fm0, width, height, level["grid"]))
    return ann, hier, fm0, fm1

def unflattening(ann, fm0, hier, solution, only_sort=False):
    '''Unflatten the pixel classification to voxels.
    Add a sorting coordinates process to keep consistency.
    Set only_sort=True if you want to skip the unflattening process.
    '''
    three_d_coords, two_d_coords = flatmap_to_coordinates1(ann, fm0, hier)
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

def flatmap_to_coordinates1(ann, fm, hierarchy_root):
    '''
    Allows to visualize the shape of region with 3d and 2d coordinates from a flatmap
    without using cache, you can theoritically visualize any regions from a given
    annotation file.
    '''
    lst_ids = list(hierarchy_root.get("id"))
    coords = []
    for x in range(int(ann.raw.shape[0])):
        for y in range(int(ann.raw.shape[1])):
            for z in range(int(ann.raw.shape[2])):
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

# OBJECT
class RubixBrain(object):
    def __init__(self, config):
        assert (config.get("method") == "node_distance") | (config.get("method") == "reversing_hierarchy"), "This method doesn't exist"
        assert (config["architecture"].get("architecture") == "regular") | (config["architecture"].get("architecture") == "irregular"), "This architecture is not recognized"
        assert (config.get("split_with") == "reversal_detector") | (config.get("split_with") == "hdbscan"), "This splitting method is not recognized"

        self.width = config.get("width")
        self.height = config.get("height")
        self.depth = config.get("depth")
        self.noise_amplitude = config["connectivity"].get("noise")
        self.hierarchy_method = config.get("method")
        self._architecture = config["architecture"].get("architecture")
        self.split_with = config.get("split_with")

        if self._architecture == "regular":
            self.grid = config["architecture"].get("grid")
            if config["architecture"].get("hierarchy") is not None:
                self._hierarchical = True
                self.hierarchical = config["architecture"].get("hierarchy")
                self.number_regions = numpy.prod(self.hierarchical[-1]["grid"])
                self.levels = len(self.hierarchical) + 1
            else:
                self._hierarchical = False
                self.number_regions = self.grid[0] * self.grid[1]
                self.levels = 1
        elif self._architecture == "irregular":
            self.hierarchical = config["architecture"].get("hierarchy")
            self.levels = self.hierarchical["levels"]
            self.n_split = self.hierarchical["n_split"]
            self.n_by_split = self.hierarchical["n_by_split"]
            self.kernel_size = config["architecture"].get("kernel_size")
            self.number_regions = self.n_by_split ** (self.n_split)
            if config["architecture"].get("hierarchy") is not None:
                self._hierarchical = True
            else:
                self._hierarchical = False

            
        self.__initialize__()
        
    def __initialize__(self):
        if self._architecture == "regular":
            ann, hier, fm0, fm1 = regularRubix(self)
        elif self._architecture == "irregular":
            ann, hier, fm0, fm1 = irregularRubix(self, self.n_split, self.n_by_split)
        
        self.hierarchy = hier
        self.annotation = ann
        self.anatomical_flatmap = fm0
        self.null_diffusion_flatmap = fm1
        # custom
        self.custom_annotation = copy.deepcopy(ann)
        offset = int(numpy.max(self.custom_annotation.raw))
        self.custom_annotation.raw[:] = offset
        self.custom_hierarchy = voxcell.Hierarchy({"name": "Rubix",
                              "acronym": "Rubix",
                              "id": offset})
            
    def reinitialize(self):
        '''Reinitialize custom parcellation and erase analyses.
        '''
        self.custom_annotation = copy.deepcopy(self.annotation)
        offset = int(numpy.max(self.custom_annotation.raw))
        self.custom_annotation.raw[:] = offset
        self.custom_hierarchy = voxcell.Hierarchy({"name": "Rubix",
                              "acronym": "Rubix",
                              "id": offset})
        del self.df_gd
        del self.df_ri
    
    def build_connectivity_matrix(self, diffusion_coords):       
        dist = cdist(diffusion_coords,diffusion_coords,metric="sqeuclidean")
        max_distance = numpy.amax(dist) + 1
        x = numpy.linspace(0, 1, num = int(max_distance))
        sigma = self.scaling_distance / max_distance * 0.1
        wave = numpy.flip(gaussian_func(x, 1, sigma))

        
        connectivity_matrix = numpy.zeros((len(diffusion_coords), len(diffusion_coords)))
        connectivity_matrix[:] = numpy.nan
        for source in range(len(diffusion_coords)):
            for target in range(len(diffusion_coords)):
                connectivity_matrix[source,target] = wave[int(dist[source,target])]  
        # add noise
        noise_amp = self.noise_amplitude
        connectivity_matrix = connectivity_matrix + noise_amp * numpy.random.rand(*connectivity_matrix.shape)
        return connectivity_matrix

    def hierarchical_connectivity(self):
        three_d_coords, _ = flatmap_to_coordinates1(self.annotation, self.anatomical_flatmap,
                                                    self.hierarchy)
        nodes_distance = all_nodes_distance(self.hierarchy)
        connectivity_matrix = numpy.zeros((len(three_d_coords), len(three_d_coords)))
        connectivity_matrix[:] = numpy.nan
        for source in range(len(three_d_coords)):
            for target in range(len(three_d_coords)):
                idA = int(self.annotation.raw[three_d_coords[source][0], three_d_coords[source][1], three_d_coords[source][2]])
                idB = int(self.annotation.raw[three_d_coords[target][0], three_d_coords[target][1], three_d_coords[target][2]])
                connectivity_matrix[source,target] = nodes_distance.loc[idA, idB] + numpy.random.rand() *0.1
        return connectivity_matrix
           
    
    def build_connectivity_data(self):        
        if self.hierarchy_method == "reversing_hierarchy":
            if self._hierarchical == False:
                # get scaling distance constant:
                three_d_coords, diff_coords = flatmap_to_coordinates1(self.annotation,
                                                              self.null_diffusion_flatmap,
                                                              self.hierarchy)
                self.scaling_distance = numpy.amax(cdist(diff_coords,diff_coords,metric="sqeuclidean")) + 1
                three_d_coords, diff_coords = flatmap_to_coordinates1(self.annotation,
                                                                      self.null_diffusion_flatmap,
                                                                      self.hierarchy)
    
                connectivity_matrix = self.build_connectivity_matrix(diff_coords)
            else:
                # get scaling distance constant:
                three_d_coords, diff_coords = flatmap_to_coordinates1(self.annotation,
                                                              self.null_diffusion_flatmap[0],
                                                              self.hierarchy)
                self.scaling_distance = numpy.amax(cdist(diff_coords,diff_coords,metric="sqeuclidean")) + 1
                con_matrices = []
                for diffusion_fm in self.null_diffusion_flatmap:
                    three_d_coords, diff_coords = flatmap_to_coordinates1(self.annotation,
                                                          diffusion_fm,
                                                          self.hierarchy)
                    con_matrices.append(self.build_connectivity_matrix(diff_coords))
                connectivity_matrix = con_matrices[-1].copy()
                for m in con_matrices[:-1]:
                    connectivity_matrix *= m
            null_data = numpy.hstack((three_d_coords, connectivity_matrix))
            self.connectivity_data = null_data
        elif self.hierarchy_method == "node_distance":
            # get scaling distance constant:
            three_d_coords, diff_coords = flatmap_to_coordinates1(self.annotation,
                                                          self.null_diffusion_flatmap,
                                                          self.hierarchy)
            self.scaling_distance = numpy.amax(cdist(diff_coords,diff_coords,metric="sqeuclidean")) + 1
            three_d_coords, diff_coords = flatmap_to_coordinates1(self.annotation,
                                                                  self.null_diffusion_flatmap,
                                                                  self.hierarchy)
    
            connectivity_matrix = self.build_connectivity_matrix(diff_coords)
            if self._hierarchical is True:
                hierarchical_connectivity_matrix = self.hierarchical_connectivity()
                connectivity_matrix *= hierarchical_connectivity_matrix
            null_data = numpy.hstack((three_d_coords, connectivity_matrix))
            self.connectivity_data = null_data
            
        
    def make_flattening_config(self, depth=0, component_to_use=[0,1], initial=True):
        assert depth <= self.levels, "This level doesn't exist"
        if initial is True:
            r = self.hierarchy
            ann = self.annotation
            to_consider = at_depth(r, depth, property="acronym")
            to_flatten = at_depth(r, depth, property="acronym")
        else:
            r = self.custom_hierarchy
            ann = self.custom_annotation
            to_consider = leaves(r, "acronym")
            to_flatten = leaves(r, "acronym")
        flat_config = []
        lst_spread = []
        for region in to_flatten:
            norm_arg = normalization_spread(ann.raw, r, region)
            lst_spread.append(norm_arg)
            flat_config.append({
                "connectivity_target": {
                    "considered_regions": to_consider,
                    "direction": "both"},
                "normalization_args": {
                    "normalize_spread": [norm_arg, norm_arg]},
                "flatten": [region],
                "components": component_to_use})
        flat_config = normalization_offsets(lst_spread, flat_config)
        return flat_config
    
    def diffusion_mapping(self, flattening_config, initial=True):
        assert hasattr(self, "connectivity_data"), "Connectivity data has not been built yet"
        vxl_shape = (self.width, self.height, self.depth)
        vxl_offset = numpy.array([0., 0., 0.])
        vxl_dims = numpy.array([100., 100., 100.])
        fm1 = numpy.NaN * numpy.ones(vxl_shape + (len(flattening_config[0]["components"]),), dtype=float)
        if initial is True:
            r = self.hierarchy
            ann = self.annotation
        else:
            r = self.custom_hierarchy
            ann = self.custom_annotation
        lambdas_val = []
        for n in range(len(flattening_config)):
            to_flatten = flattening_config[n]["flatten"][0]
            reg_coords, _ = flatmap_to_coordinates1(ann, self.anatomical_flatmap, r.find("acronym", to_flatten)[0])
            # From the whole connectivity dataset, extract rows corresponding to the region to flatten
            c = cdist(self.connectivity_data[:,:3], reg_coords)==0
            connectivity_matrix = self.connectivity_data[:,3:][c.any(axis=1)]
            normalize_vals = numpy.linalg.norm(connectivity_matrix, axis=1, keepdims=True)
            normalize_vals[normalize_vals == 0] = 1E-9
            C_norm = connectivity_matrix / normalize_vals  # Still zero for zero connected voxels
            Cosine_similarity = numpy.dot(C_norm, C_norm.transpose())  # rows and cols of zero connected voxels are zero
            cs_sum = Cosine_similarity.sum(axis=1, keepdims=True)  # L x 1, zero for zero connected voxels
            cs_sum[cs_sum == 0] = 1E-9
            cs_sum_ratios = numpy.hstack([cs_sum / _x for _x in cs_sum[:, 0]])
            Cosine_similarity_norm = (Cosine_similarity / cs_sum) * numpy.sqrt(cs_sum_ratios)
            _regions = flattening_config[n]
            components_to_use = _regions["components"]
            n_components = numpy.amin([numpy.count_nonzero(ann.raw == i) for i in numpy.unique(ann.raw)]) - 1 
        
            vxl_is_valid = Cosine_similarity_norm.sum(axis=1) != 0
            embed_coords = numpy.NaN * numpy.ones((Cosine_similarity_norm.shape[0], n_components), dtype=float)
            embed_coords[vxl_is_valid, :], embed_res = compute_diffusion_map(Cosine_similarity_norm[numpy.ix_(vxl_is_valid, vxl_is_valid)],
                                                                             return_result=True,
                                                                             diffusion_time=1,
                                                                             n_components=n_components)
            if embed_res["n_components_auto"] >= (embed_res["n_components"] - 2):
                # If this is true then even the last component is stronger than 0.05 of the first.
                print(f"Warning: {embed_res['n_components']} components were apparently not enough to characterize the full variance. Resulting variance fractions not valid")
                component_lambda_ratios = embed_res["lambdas"][components_to_use] / embed_res["lambdas"].sum()
            embed_coords_final = embed_coords[vxl_is_valid, :].copy()
            embed_coords_final = embed_coords_final[:, components_to_use]
            embed_coords_final = normalize_fm_coordinates(embed_coords_final, _regions.get("normalization_args", {}))
            fm1[reg_coords[:, 0], reg_coords[:, 1], reg_coords[:, 2]] = embed_coords_final
            print(f"Writing flatmap for region {to_flatten} via {', '.join(sorted(_regions['connectivity_target'].get('considered_regions')))}")
            lambdas_val.append(dict({"region": to_flatten,
                                     "lambdas": embed_res["lambdas"]}))
        fm1 =  voxcell.VoxelData(fm1, voxel_dimensions=tuple(vxl_dims),
                               offset=tuple(vxl_offset))    
        return fm1, lambdas_val, embed_coords_final
    
    def show_gradient_maps(self, flattening_config, diffusion_fm, ret_results=False):
        regions = flattening_config[0]["connectivity_target"]["considered_regions"]
        for n in range(len(regions)):
            hierarchy_reg = self.hierarchy.find("acronym", regions[n])[0]
            x1,y1,x2,y2 = gradient_map(self.anatomical_flatmap, diffusion_fm, self.annotation, hierarchy_reg, show=True)
            if ret_results == True:
                return x1, y1, x2, y2
            
    def show_connectivity_structure(self, flattening_config, diffusion_fm, ret_results=False):
        regions = flattening_config[0]["connectivity_target"]["considered_regions"]
        for n in range(len(regions)):
            r = self.hierarchy.find("acronym", regions[n])[0]
            img = connectivity_structure(regions[n], self.anatomical_flatmap, diffusion_fm, self.annotation, r, show=True)
            if ret_results == True:
                return img
            

    def update_parcellation(self, solution0, region):          
        offset = int(numpy.max(self.custom_annotation.raw) + 1)
        r = self.custom_hierarchy.find("acronym", region)[0]
        solution = unflattening(self.custom_annotation, self.anatomical_flatmap, r, solution0)
        curr_ids = list(r.get('id'))
        idxx = numpy.in1d(self.custom_annotation.raw.flat, curr_ids).reshape(self.custom_annotation.raw.shape)
        self.custom_annotation.raw[idxx] = solution[:, -1] + offset
        for sub_id in numpy.unique(solution[:, -1]):
            r.children.append(voxcell.Hierarchy({
                "name": r.data["name"] + "_" + str(int(sub_id)+1),
                "acronym": r.data["acronym"] + "_" + str(int(sub_id)+1),
                "id": int(sub_id + offset)
            }))

    def split_by_HDBSCAN(self, flattening_config, diffusion_fm, char,
                         alpha=1.0, eps=0.0, min_cluster_size=30, min_samples=None,
                         C=0.5, gamma=0.05, t=1, thresh_size=20, show=False, test=False, **kwargs):
        for reg in flattening_config[0]["connectivity_target"]["considered_regions"]:
            try:
                lambdas = [i for i in char if i["region"] == reg][0]["lambdas"]
                hierarchy_reg = self.custom_hierarchy.find("acronym", reg)[0]
                _, two_d_coords = flatmap_to_coordinates1(self.custom_annotation, self.anatomical_flatmap, hierarchy_reg)
                two_d_coords = numpy.unique(two_d_coords, axis=0).astype("int")
                gXs, gYs = extract_gradients(self.anatomical_flatmap, diffusion_fm, 
                                             self.custom_annotation, hierarchy_reg)
                grad_clf = cosine_distance_clustering(gXs, gYs, two_d_coords, lambdas, alpha=alpha,
                                                  eps=eps, min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs)
                solution2 = split_with_SVM(grad_clf, C, gamma, thres_accuracy=0, show=False)
                solution3 = extract_subregions(solution2, t)
                solution4 = merge_lonely_voxels(solution3, thresh_size)
                if show:
                    viz_gradient_split(grad_clf, self.anatomical_flatmap,
                                                 diffusion_fm, self.custom_annotation,
                                                 hierarchy_reg)
                    viz_gradient_split(solution4, self.anatomical_flatmap,
                                                 diffusion_fm, self.custom_annotation,
                                                 hierarchy_reg)                
                if test is False:
                    if len(numpy.unique(solution4[:,-1])) > 1:
                        self.update_parcellation(solution4, reg)
            except: continue
        
    def split_by_reversalDetector(self, flattening_config, diffusion_fm, component="optimized",
                                   C=0.5, gamma=0.05, t=1, thresh_size=20, show=False, test=False):
        for reg in flattening_config[0]["connectivity_target"]["considered_regions"]:
            try:
                hierarchy_reg = self.custom_hierarchy.find("acronym", reg)[0]
                grad_clf = reversal_detector(reg, self.anatomical_flatmap,
                                             diffusion_fm, self.custom_annotation,
                                             hierarchy_reg, component=component)
                solution2 = split_with_SVM(grad_clf, C, gamma, thres_accuracy=0, show=False)
                solution3 = extract_subregions(solution2, t)
                solution4 = merge_lonely_voxels(solution3, thresh_size)
                if show:
                    viz_gradient_split(grad_clf, self.anatomical_flatmap,
                                                 diffusion_fm, self.custom_annotation,
                                                 hierarchy_reg)
                    viz_gradient_split(solution4, self.anatomical_flatmap,
                                                 diffusion_fm, self.custom_annotation,
                                                 hierarchy_reg)
                if test is False:
                    if len(numpy.unique(solution4[:,-1])) > 1:
                        self.update_parcellation(solution4, reg)
            except: continue
        
    def viz_custom_parcellation(self, fm1):
        three_d_coords, two_d_coords = flatmap_to_coordinates1(self.custom_annotation,
                                                                self.anatomical_flatmap,
                                                                self.custom_hierarchy)
        label = numpy.zeros((len(three_d_coords), 1))
        for i in range(len(three_d_coords)):
            label[i,0] = self.custom_annotation.raw[three_d_coords[i,0], three_d_coords[i,1], three_d_coords[i,2]]
            
        coords2d = numpy.column_stack((two_d_coords, label))
        viz_gradient_split(coords2d, self.anatomical_flatmap, fm1,
                           self.custom_annotation,
                           self.custom_hierarchy)
        
    def analysis_gd(self, fm1):
        if hasattr(self, "df_gd") is False:
            regions = leaves(rubix.custom_hierarchy, "acronym")
            self.df_gd = pd.DataFrame(columns=("region", "gradient_deviation"))
            for reg in regions:
                hier = self.custom_hierarchy.find("acronym", reg)[0]
                gd = gradient_deviation(self.anatomical_flatmap, fm1,
                                        self.custom_annotation, hier, plot=False)     
                self.df_gd.loc[len(self.df_gd), self.df_gd.columns] = reg, numpy.mean(gd)
        else:
            regions = leaves(rubix.custom_hierarchy, "acronym")
            for reg in regions:
                hier = self.custom_hierarchy.find("acronym", reg)[0]
                gd = gradient_deviation(self.anatomical_flatmap, fm1,
                                        self.custom_annotation, hier, plot=False)     
                self.df_gd.loc[len(self.df_gd), self.df_gd.columns] = reg, numpy.mean(gd)
                self.df_gd = self.df_gd.drop_duplicates(subset='region', keep='first', inplace=False)

    def analysis_ri(self, fm1):
        if hasattr(self, "df_ri") is False:
            regions = leaves(rubix.custom_hierarchy, "acronym")
            self.df_ri = pd.DataFrame(columns=("region", "reversal_index"))
            for reg in regions:
                hier = self.custom_hierarchy.find("acronym", reg)[0]
                ri = reversal_index(self.anatomical_flatmap, fm1,
                                        self.custom_annotation, hier)     
                self.df_ri.loc[len(self.df_ri), self.df_ri.columns] = reg, ri
        else:
            regions = leaves(rubix.custom_hierarchy, "acronym")
            for reg in regions:
                hier = self.custom_hierarchy.find("acronym", reg)[0]
                ri = reversal_index(self.anatomical_flatmap, fm1,
                                        self.custom_annotation, hier)     
                self.df_ri.loc[len(self.df_ri), self.df_ri.columns] = reg, ri
                self.df_ri = self.df_ri.drop_duplicates(subset='region', keep='first', inplace=False)
                
        def results_last_parc(self, df=None, save=False):
            if df is None:
                df = pd.DataFrame(columns=("method", "noise", "algo", "gd", "ri"))
            gd = self.df_gd.copy()
            ri = self.df_ri.copy()
            max_split = max([gd.loc[i, "region"].count("_") for i in gd.index])
            idx = [i for i in gd.index if gd.loc[i, "region"].count("_") == max_split]
            self.df_gd = gd.loc[idx]
            self.df_ri = ri.loc[idx]
            method = self.hierarchy_method 
            algo = self.split_with
            noise = self.noise_amplitude
            dftemp = self.df_gd
            dftemp["reversal_index"] = self.df_ri["reversal_index"]
            dftemp["method"] = method
            dftemp["algo"] = algo
            dftemp["noise"] = noise
            df = pd.concat((df, dftemp))                
            return df
# 
def run_splitting(rubix, n_comp, method, C=0.1, gamma=0.01, t=1, thresh_size=30,
                  save=False, show=False,**kwargs):
    # make components
    components = list(numpy.arange(n_comp))
    # make flattening config
    flat_config = rubix.make_flattening_config(depth=0, component_to_use=components,
                                               initial=True)
    # run the first diffusion mapping
    print("Running first diffusion flatmap ...")
    fm1, diffusion_res, embed_coords_final = rubix.diffusion_mapping(flat_config,
                                                                     initial=True)
    print("Done !")
    # update analysis
    rubix.analysis_gd(fm1)
    rubix.analysis_ri(fm1)
    # start the loop
    complete = False
    parc_t1 = leaves(rubix.custom_hierarchy, "acronym")
    print("Starting the loop ...")
    while complete is False:
        print("Splitting parcellation ...")
        if method == "hdbscan":
            rubix.split_by_HDBSCAN(flat_config, fm1, diffusion_res,
                                    alpha=kwargs["alpha"], eps=kwargs["eps"], 
                                    min_cluster_size=kwargs["min_cluster_size"], min_samples=kwargs["min_samples"],
                                    C=C, gamma=gamma, t=t, thresh_size=thresh_size, show=show)
            
        elif method == "reversal_detector":
            rubix.split_by_reversalDetector(flat_config, fm1, component="optimized",
                                            C=C, gamma=gamma, t=t, thresh_size=thresh_size, show=show)
        else: 
            print("method not recognized")
            return
        print("Done !")
        # new flattening config
        flat_config = rubix.make_flattening_config(component_to_use=components,
                                                   initial=False)
        # diffusion mapping
        print("Running diffusion mapping ...")
        fm1, diffusion_res, embed_coords_final = rubix.diffusion_mapping(flat_config,
                                                                         initial=False)
        print("Done !")
        rubix.analysis_gd(fm1)
        rubix.analysis_ri(fm1)
        # plot and save
        rubix.viz_custom_parcellation(fm1)
        parc_t2 = leaves(rubix.custom_hierarchy, "acronym")
        if parc_t2 == parc_t1:
            complete = True
            print("Parcellation complete !")
        else:
            parc_t1 = parc_t2
            
def results_last_parc(rubix, df=None, save=False):
    if df is None:
        df = pd.DataFrame(columns=("method", "noise", "algo", "gd", "ri"))
    gd = rubix.df_gd.copy()
    ri = rubix.df_ri.copy()
    max_split = max([gd.loc[i, "region"].count("_") for i in gd.index])
    idx = [i for i in gd.index if gd.loc[i, "region"].count("_") == max_split]
    df_gd = gd.loc[idx]
    df_ri = ri.loc[idx]
    method = rubix.hierarchy_method 
    algo = rubix.split_with
    noise = rubix.noise_amplitude
    dftemp = df_gd
    dftemp["reversal_index"] = df_ri["reversal_index"]
    dftemp["method"] = method
    dftemp["algo"] = algo
    dftemp["noise"] = noise
    df = pd.concat((df, dftemp))                
    return df
#%%
# 4 levels
rubix_config = dict({"width": 40,
                      "height": 40,
                      "depth": 2,
                      "architecture": dict({"architecture": "regular",
                                            "grid": [2,1],
                                            "hierarchy": [dict({"level": "level_2",
                                                                "grid": [2,2]}),
                                                          dict({"level": "level_3",
                                                                "grid": [4,4]})]}),
                      "connectivity": dict({"noise": 5.0}),
                      "method": "reversing_hierarchy",
                      "split_with": "reversal_detector"})


rubix = RubixBrain(rubix_config)
#check rubix organization
plt.imshow(rubix.annotation.raw[:,:,0])
plt.axis('off')
#%% make connectivity data
rubix.build_connectivity_data()
plt.imshow(rubix.connectivity_data[:,3:]) # first 3 columns are coordinates
plt.colorbar()
plt.show()
#%%
method = "hdbscan"
run_splitting(rubix, n_comp=20, method=method, alpha=1.0, eps=0.0, min_cluster_size=30, min_samples=30)
res = results_last_parc(rubix)



#%% reset
rubix.reinitialize()



#%% Full analysis
modes = ["reversing_hierarchy", "node_distance"]
noises = [0.1, 5.0]
splitters = ["reversal_detector", "hdbscan"]


import itertools
conditions = list(itertools.product(modes, noises, splitters))

res = pd.DataFrame(columns=("method", "noise", "algo", "gd", "ri"))

for i in range(len(conditions)):
    rubix_config = dict({"width": 40,
                          "height": 40,
                          "depth": 2,
                          "architecture": dict({"architecture": "regular",
                                                "grid": [2,1],
                                                "hierarchy": [dict({"level": "level_2",
                                                                    "grid": [2,2]}),
                                                              dict({"level": "level_3",
                                                                    "grid": [4,4]})]}),
                          "connectivity": dict({"noise": conditions[i][1]}),
                          "method": conditions[i][0],
                          "split_with": conditions[i][2]})


    rubix = RubixBrain(rubix_config)
    #check rubix organization
    plt.imshow(rubix.annotation.raw[:,:,0])
    plt.axis('off')
    # make connectivity data
    rubix.build_connectivity_data()
    plt.imshow(rubix.connectivity_data[:,3:]) # first 3 columns are coordinates
    plt.colorbar()
    plt.show()
    
    # split
    method = rubix.split_with
    run_splitting(rubix, n_comp=20, method=method, alpha=1.0, eps=0.0, min_cluster_size=30, min_samples=30)
    res = results_last_parc(rubix, res)
#%% plot results (barplots)
import seaborn as sns
df0 = res.copy()
df0['algo.noise'] = df0.algo.astype(str).str.cat([df0.noise.astype(str)], sep='.')
df0 = df0.sort_values(by=["algo.noise"])

plt.figure(figsize=(5,6)).gca()
sns.barplot(x="method", y="gradient_deviation", hue="algo.noise", data=df0, ci="sd", capsize=0.03)
plt.show()
plt.figure(figsize=(5,6)).gca()
sns.barplot(x="method", y="reversal_index", hue="algo.noise", data=df0, ci="sd", capsize=0.03)
plt.ylim(0, max(df0["reversal_index"]))
plt.show()