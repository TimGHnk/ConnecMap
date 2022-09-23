import os
import h5py
import numpy

# from read_projection_volumes import read_volume, read_coordinates # part of this repo
from .embed import compute_diffusion_map # part of this repo. originally: https://github.com/satra/mapalign


def coordinate_overlap_bitmask(coords_view1, coords_view2):
    is_valid_view1 = numpy.any(numpy.vstack([numpy.all(coords_view1 == _x, axis=1)
                                             for _x in coords_view2]), axis=0)
    is_valid_view2 = numpy.any(numpy.vstack([numpy.all(coords_view2 == _x, axis=1)
                                             for _x in coords_view1]), axis=0)
    return is_valid_view1, is_valid_view2


def make_relevant_connectivity_profile(cache, lst_to_flatten, lst_to_consider, direction='both'):
    """Returns a connection matrix between the listed regions to flatten
    and the listed regions whose connectivity is to be considered.
    If direction='afferent': Connection from right hemi of the list to be considered to the right hemi of the list
    to be flattened
    If direction='efferent': Connection from right hemi of the list to be flattened to the right hemi of the list
    to be considered
    If direction='both': horizontal concatenation of both.
    voxels to be flattened are always along the first axis returned!
    """
    C_eff, to_flatten_coords_eff, consider_coords_eff = cache.projection(lst_to_flatten, lst_to_consider)
    C_aff, consider_coords_aff, to_flatten_coords_aff = cache.projection(lst_to_consider, lst_to_flatten)
    C_aff = C_aff.transpose()  # Now both matrices are to_flatten X to_consider

    # It is not guaranteed that afferent and efferent connectivity consider the same hemisphere.
    # We define the overlap of what's considered (most probably the right hemisphere) as the valid region
    valid_flatten_aff, valid_flatten_eff = coordinate_overlap_bitmask(to_flatten_coords_aff, to_flatten_coords_eff)
    valid_consider_aff, valid_consider_eff = coordinate_overlap_bitmask(consider_coords_aff, consider_coords_eff)

    C_eff = C_eff[numpy.ix_(valid_flatten_eff, valid_consider_eff)]
    C_aff = C_aff[numpy.ix_(valid_flatten_aff, valid_consider_aff)]
    to_flatten_coords_eff = to_flatten_coords_eff[valid_flatten_eff]
    to_flatten_coords_aff = to_flatten_coords_aff[valid_flatten_aff]

    assert len(to_flatten_coords_aff) == len(to_flatten_coords_eff)
    assert numpy.all(to_flatten_coords_aff == to_flatten_coords_eff)

    if direction == 'efferent':
        return C_eff, to_flatten_coords_eff
    elif direction == 'afferent':
        return C_aff, to_flatten_coords_aff
    elif direction == 'both':
        return numpy.hstack([C_eff, C_aff]), to_flatten_coords_eff
    else:
        raise ValueError("Unknown value for direction: {0}".format(direction))


def similarity_matrix(C):
    """Turns a L x N connectivity matrix into a L x L similarity matrix. I.e. evaluates how similar the
    individual _rows_ of the input matrix are.
    Additionally returns a normalized version of the similarity that is relevant for the
    diffusion embedding process."""
    normalize_vals = numpy.linalg.norm(C, axis=1, keepdims=True)
    normalize_vals[normalize_vals == 0] = 1E-9
    C_norm = C / normalize_vals  # Still zero for zero connected voxels
    Cosine_similarity = numpy.dot(C_norm, C_norm.transpose())  # rows and cols of zero connected voxels are zero
    cs_sum = Cosine_similarity.sum(axis=1, keepdims=True)  # L x 1, zero for zero connected voxels
    cs_sum[cs_sum == 0] = 1E-9
    cs_sum_ratios = numpy.hstack([cs_sum / _x for _x in cs_sum[:, 0]])
    Cosine_similarity_norm = (Cosine_similarity / cs_sum) * numpy.sqrt(cs_sum_ratios)
    return Cosine_similarity, Cosine_similarity_norm


def flatten_pathway(cache, lst_to_flatten, lst_to_consider, diffusion_time=1, n_components=3, **kwargs):
    """Loads connectivity between the specified regions, calculates the similarity matrix
    and performs diffusion embedding on it. Returns everything, just in case...

    Additional parameters: diffusion_time: kind of arbitrary. In the literature they often use 0. Here, I use 1.
    play around with it...
                           n_components: The dimensionality of the embedding, i.e. the number of returned
                           coordinates for each thalamic voxel
    """
    print("Reading connectivity profile...")
    C, coords = make_relevant_connectivity_profile(cache, lst_to_flatten, lst_to_consider, **kwargs)
    print("...done! Calculating similarity...")
    S, S_norm = similarity_matrix(C)
    print("...done! Performing diffusion mapping...")
    #  Treat voxels with zero connectivity, which would be considered disconnected from the rest of the graph
    vxl_is_valid = S_norm.sum(axis=1) != 0
    embed_coords = numpy.NaN * numpy.ones((S_norm.shape[0], n_components), dtype=float)
    embed_coords[vxl_is_valid, :], embed_res = compute_diffusion_map(S_norm[numpy.ix_(vxl_is_valid, vxl_is_valid)],
                                                                     return_result=True,
                                                                     diffusion_time=diffusion_time,
                                                                     n_components=n_components)
    print("...done!")
    return C, S, S_norm, coords, embed_coords, embed_res

