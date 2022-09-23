import numpy

from .cached_projection import CachedProjections


class AibsTreeWrapper(object):
    """
    Helper class that wraps the "structure_tree" of the aibs mcm package to expose a region_ids method, as required by
    the CachedProjections class.
    """

    def __init__(self, aibs_tree):
        """
        Args:
            aibs_tree: A structure tree returned by the get_structure_tree method of a VoxelModelCache object
            (see mouse_connectivity_models)
        """
        self.tree = aibs_tree

    def region_ids(self, regions):
        """
        Args:
            regions (list): list of strings of brain regions of interest
        
        Returns:
            (numpy.array): list of integers that are associated with those regions. That is, finding these integers in
                the annotation_volume returned from the same VoxelModelCache object will yield all voxels that are
                associated with any of the specified regions.
        """
        resolve_to_leaf = True
        if not isinstance(regions, list) or isinstance(regions, numpy.ndarray):
            regions = [regions]
        r_struc = self.tree.get_structures_by_acronym(regions)
        r_ids = numpy.array([_x['id'] for _x in r_struc])

        def resolver(r_ids):
            rslvd = [resolver(_chldr) if len(_chldr) else _base
                     for _base, _chldr in
                     zip(r_ids, self.tree.child_ids(r_ids))]
            return numpy.hstack(rslvd)

        if resolve_to_leaf:
            return resolver(r_ids)
        return r_ids

class AibsMcmProjections(CachedProjections):
    """"
    A version of CachedProjections that is specific for the AIBS mouse_connectivity_models as a data source.
    Simplified constructor.
    """

    def __init__(self, vmc, voxel_sizes=(100.0, 100.0, 100.0), cache_file=None, grow_cache=True):
        """
        Args:
            vmc: A VoxelModelCache object (see mouse_connectivity_models)
            
            voxel_sizes (tuple): Tuple of length 3. Specifies the resolution of the voxelized connectivity, i.e. the size of each
               voxel in x, y, z dimensions
            
            cache_file (string): See parent class

            grow_cache (string): See parent class
        """
        voxel_array, source_mask, target_mask = vmc.get_voxel_connectivity_array()
        source_3d = source_mask.coordinates
        target_3d = target_mask.coordinates  # TODO: Option to limit to right hemisphere..?
        region_annotation_args = (vmc.get_cache_path(None, vmc.ANNOTATION_KEY, vmc.reference_space_key, vmc.resolution),)
        # vol, _ = vmc.get_annotation_volume()
        hierarchy_tree_args = (vmc.get_structure_tree(),)
        super().__init__(voxel_array, source_3d, target_3d, region_annotation_args, hierarchy_tree_args, cache_file, grow_cache)
        self._voxel_sizes = voxel_sizes
    
    def _three_d_indices_to_output_coords(self, idx, direction):
        return numpy.array(idx) * numpy.array([self._voxel_sizes])
    
    @classmethod
    def _initialize_hierarchy(cls, *args):
        assert len(args) == 1
        aibs_tree = args[0]
        return AibsTreeWrapper(aibs_tree)
