import numpy
import voxcell


class RegionHierarchyWrapper(object):
    def __init__(self, hierarchy_fn):
        self._tree = voxcell.RegionMap.load_json(hierarchy_fn)
    
    def region_ids(self, lst_regions):
        region_id_set = {}
        for region in lst_regions:
            region_id_set.update(self._tree.find(region, "acronym", with_descendants=True))
        return list(region_id_set)


class CachedProjections(object):
    """
    Main class for accessing a data source of voxelized projection strengths.
    While this class is fully functional and can be used on its own, the idea is to build derived classes that are 
    specific to different sources of voxelized projections strengths, e.g. one for the AIBS mouse_connectivity_models,
    another for human functional connectivity, etc.

    It accesses the requested data, reformats it into a format that better serves our purposes, caches the result
    locally into an hdf5 file, and returns the result.

    The purpose is as follows:
    1. Derived classes are to be created for different data sources. Thus, it serves as interface to those data sources.
    2. Data sources such as the AIBS mouse_connectome_models are online. Local caching will make future lookup faster.
    """

    def __init__(self, voxel_array, source_coords_3d, target_coords_3d,
                 region_annotation_args, hierarchy_args,
                 cache_file=None, grow_cache=True):
        """
        Args:
            voxel_array: An object that provides access to the voxel-to-voxel connectivity data of the brain that is studied.
                It must provide a two-dimensional indexing operator (voxel_array[x, y], where x and y are numpy.arrays)
                that returns the connectivity strengths _from_ the voxels at indices x _to_ the voxels at indices y in the form
                of a 2d numpy.array. That is, it behaves like a 2d numpy.array itself, although for larger connectomes you might
                want to implement something cleverer to avoid holding all data in memory at once. The order of the voxel in this
                structure is given by source_coords_3d and target_coords_3d.

            source_coords_3d (numpy.array): An N x 3 array specifying the 3d coordinates of source voxels in voxel_array.
                These coordinates are integers that can be used as indices into model_annotations.
            
            target_coords_3d (numpy.array): An N x 3 array specifying the 3d coordinates of target voxels in voxel_array.
                That is, voxel_array[x, y] is the connection strengths from the voxels at source_coords_3d[x] to target_coords_3d[y].
                Note that voxel_array does _not_ have to cover all brain voxels and the coverage does not have to be the same
                on the source and target side!
            
            region_annotation_args (tuple): A tuple of arguments that will be fed into the ._initialize_annotations function.
                In this case, a tuple of length 1 containing the path to a .nrrd file that defines the region annotations of
                the brain that is studied. That is, each entry represents a brain voxel, and its value is an integer specifying
                the brain region the voxel belongs to.
            
            hierarchy_args (tuple): A tuple of arguments that will be fed into the ._initialize_hierarchy function.
                In this case, a tuple of lenght 1 containing a path to a .json file that specifies the region hierarchy of the brain
                that is studied. Will be read as a voxcell.RegionMap. Check the voxcell documentation for specifications of the
                exact format.
            
            cache_file (string): Name / location of the hdf5 file to use as a local cache. If it does not exist and grow_cache is set
                to True, it will be created.
            
            grow_cache (bool): Whether to add new, previously uncached results to the cache file. If you don't want to cache at all, 
                set this to False and cache_file to None.

        """
        self.voxel_array = voxel_array
        self.source_3d = source_coords_3d
        self.target_3d = target_coords_3d

        self.vol = self.__class__._initialize_annotations(*region_annotation_args)
        self.tree = self.__class__._initialize_hierarchy(*hierarchy_args)

        self._shape3d = self.vol.shape
        assert len(self._shape3d) == 3, "Must provide a three-dimensional brain!"
        self._make_indices()

        self._cache_only = False  # There will be a derived class that sets this to true.
        if cache_file is None:
            self._cache_fn = "projection_cache.h5"
        else:
            self._cache_fn = cache_file
        self._grow_cache = grow_cache
    

    @classmethod
    def _initialize_hierarchy(cls, *args):
        assert len(args) == 1
        fn_hierarchy = args[0]
        return RegionHierarchyWrapper(fn_hierarchy)
    

    @classmethod
    def _initialize_annotations(cls, *args):
        assert len(args) == 1
        fn_annotations = args[0]
        annotations = voxcell.VoxelData.load_nrrd(fn_annotations)
        return annotations.raw


    def _make_indices(self):
        """"
        Initialization helper that transforms the source and target 3d coordinates into flattened 1d coordinates.
        See _three_d_to_three_d_flat for details.

        Args:
            
        Returns:

        """
        self.source_3d_flat = self._three_d_to_three_d_flat(self.source_3d, self._shape3d)
        self.target_3d_flat = self._three_d_to_three_d_flat(self.target_3d, self._shape3d)

    @staticmethod
    def _three_d_to_three_d_flat(idx, reference_shape):
        """
        Transforms 3d coordinates into flattened 1d indices. 
        That is, if the input is used as indices into 3d array A, then the output can be used as indices into A.flat to return
        the same entries.

        Args:
            idx (numpy.array): N x 3 array of integers to be used as indices into an array.
            reference_shape (tuple): Length 3 tuple of the shape of the array to index.
        
        Returns:
            (numpy.array): The equivalent flat indices.
        """
        assert len(reference_shape) == 3
        return idx[:, 0] * numpy.prod(reference_shape[1:]) + \
               idx[:, 1] * reference_shape[-1] + idx[:, 2]

    def _three_d_indices_to_output_coords(self, idx, direction):
        """
        Transforms 3d indices into the voxelized brain into "real" coordinates. That is, looks up the coordinates of the voxel centers.
        For example, locations in um in a brain coordinate system.
        The output coordinate system does not have to be 3d. Also, one can use a different one for voxels depending on whether thet serve
        as a source or target.

        This implementation just returns the unchanged indices. To be overridden in derived classes.
        """
        return numpy.array(idx)  # Will have more functionality in derived classes

    def _three_d_flat_to_array_flat(self, three_d_flat, index_as, strict=False):
        """
        Translates from flat 3d indices into the flattened brain voxels (see _three_d_to_three_d_flat) into indices that can be used with
        voxel_array. They are potentially different because voxel_array does not have to cover all voxels.

        Args:
            three_d_flat (numpy.array): 1d array of flattened 3d indices to look up.
            
            index_as (string): One of "source" or "target". Specifies whether the output is to be used as the _first_ ("source") or
            _second_ ("target") index into voxel_array.

            strict (bool): If strict==True and three_d_flat contains voxels that are _not_ covered by voxel_array, raise an exception.
            Otherwise, not covered voxels will just be silently ignored / removed.
        
        Returns:
            (numpy.array): 1d array of indices that can be used as specified for voxel_array. Note: May change the order of voxels.
            The output will be in the order in which the voxels are listed in source_3d_flat or target_3d_flat.
        """
        if strict:
            assert numpy.all(numpy.diff(three_d_flat) > 0)
        if index_as == 'source':
            if strict:
                assert numpy.all(numpy.in1d(three_d_flat, self.source_3d_flat))
            return numpy.nonzero(numpy.in1d(self.source_3d_flat, three_d_flat))[0]
        elif index_as == 'target':
            if strict:
                assert numpy.all(numpy.in1d(three_d_flat, self.target_3d_flat))
            return numpy.nonzero(numpy.in1d(self.target_3d_flat, three_d_flat))[0]
        else:
            raise Exception("Need to index as either 'source', or 'target'")

    def _make_volume_mask(self, idxx):
        """
        Create 3d boolean mask of voxels that belong to the regions identified by region ids.

        Args:
            idxx: list of integers associated with the target regions.
        
        Returns:
            (numpy.array): a 3d boolean array of the same shape as model_annotations (that is, the same shape as the voxelized brain
            that is studied). Entries are True for all voxels that beling to the list of regions. 
        """
        return numpy.isin(self.vol, idxx)

    def make_volume_mask(self, regions):
        """
        Create 3d boolean mask of voxels that belong to the regions identified by a list of region names (strings).

        Args:
            idxx: list of strings of region names.
        
        Returns:
            (numpy.array): a 3d boolean array of the same shape as model_annotations (that is, the same shape as the voxelized brain
            that is studied). Entries are True for all voxels that beling to the list of regions. 
        """
        idxx = self.tree.region_ids(regions)
        return self._make_volume_mask(idxx)

    def mask_to_indices(self, vol_mask, index_as):
        """
        Turns a 3d boolean mask of brain voxels into two types of indices that denote the same voxels.

        Args:
            vol_mask (numpy.array): a 3d boolean array that is True for the brain voxels of interest.

            index_as (string): One of "source" or "target". Specifies whether the second output is to be used as the
                _first_ ("source") or _second_ ("target") index into voxel_array.
        
        Returns:
            (numpy.array): a N x 3 array of integers. Can be used as indices into model_annotations, i.e. into the 3d voxelized brain.
                Denotes the same voxels as the input mask (but see the caveats below!).
            
            (numpy.array): a 1d array of length N. Can be used as indices into voxel_array.
            
            Caveats: Only returns indices for the voxels that are covered by voxel_array on the specified side ("source" or "target").
            Entries that are not covered are silently removed. Also the output may be in a different order than the input: Voxels will
            be in the order in which they appear in source_3d_flat or target_3d_flat. Both caveats are applied to both outputs equally,
            so the two indices returned will be mutually consistent. 
        """
        mask_idx = numpy.nonzero(vol_mask)
        mask_idx = numpy.vstack(mask_idx).transpose()
        mask_idx_flat = self._three_d_to_three_d_flat(mask_idx, self._shape3d)
        if index_as == 'source':
            some_3d_flat = self.source_3d_flat
        elif index_as == 'target':
            some_3d_flat = self.target_3d_flat
        else:
            raise Exception("Need to index as either 'source', or 'target'")
        valid = numpy.in1d(mask_idx_flat, some_3d_flat)
        mask_idx = mask_idx[valid]
        mask_idx_flat = mask_idx_flat[valid]
        return mask_idx, \
               self._three_d_flat_to_array_flat(mask_idx_flat, index_as)

    def indices_for_region(self, regions, index_as):
        """
        Returns two types of indices that denote the voxels associated with the specified list of brain regions.

        Args:
            regions (list): List of strings of brain region names.

            index_as (string): One of "source" or "target". Specifies whether the second output is to be used as the
                _first_ ("source") or _second_ ("target") index into voxel_array.
        
        Returns:
            (numpy.array): a N x 3 array of integers. Can be used as indices into model_annotations, i.e. into the 3d voxelized brain.
                Denotes the same voxels as the input mask (but see the caveats below!).
            
            (numpy.array): a 1d array of length N. Can be used as indices into voxel_array.
            
            See mask_to_indices for caveats.
        """
        mask = self.make_volume_mask(regions)
        return self.mask_to_indices(mask, index_as)

    def _uncached_projection(self, src_regions, tgt_regions):
        """
        Return strengths of connections from voxels in the specified source regions to voxels in the specified target regions
        without using the cache. Does _not_ translate returned source / target locations to the source / target coordinate systems
        (see _three_d_indices_to_output_coords).

        Args:
            src_regions (list): List of strings of region names. Sepcifies the source regions of projections to look up.

            tgt_regions (list): List of strings of region names. Sepcifies the target regions of projections to look up.

        Returns:

            (numpy.array): A 1d numpy.array of projection strengths between the voxels associated with the source and target
                regions.
            
            (numpy.array): A N x 3 array of the indices of brain voxels that the _sources_ of the entries in the first output.

            (numpy.array): A N x 3 array of the indices of brain voxels that are the _target_ of the entries in the first output.
        """
        if self._cache_only:
            raise ValueError("Working in cache-only mode. Please use .projection!")
        src_3d, src_array = self.indices_for_region(src_regions, 'source')
        tgt_3d, tgt_array = self.indices_for_region(tgt_regions, 'target')
        values = self.voxel_array[src_array, tgt_array]
        return values, src_3d, tgt_3d

    def uncached_projection(self, src_regions, tgt_regions):
        """
        Return strengths of connections from voxels in the specified source regions to voxels in the specified target regions
        without using the cache. Translates returned source / target locations to the source / target coordinate systems
        (see _three_d_indices_to_output_coords).

        Args:
            src_regions (list): List of strings of region names. Sepcifies the source regions of projections to look up.

            tgt_regions (list): List of strings of region names. Sepcifies the target regions of projections to look up.

        Returns:

            (numpy.array): A 1d numpy.array of projection strengths between the voxels associated with the source and target
                regions.
            
            (numpy.array): A N x 3 array of the coordinates of brain voxels that the _sources_ of the entries in the first output.

            (numpy.array): A N x 3 array of the coordinates of brain voxels that are the _target_ of the entries in the first output.
        """
        values, src_coord, tgt_coord = self._uncached_projection(self, src_regions, tgt_regions)
        src_coord = self._three_d_indices_to_output_coords(src_coord, 'source')
        tgt_coord = self._three_d_indices_to_output_coords(tgt_coord, 'target')
        return values, src_coord, tgt_coord

    def _cached_single_projection(self, src_region, tgt_region, coordinate_name='3d', write_cache=True):
        """
        Return strengths of connections from voxels in the specified source region to voxels in the specified target region
        using the cache.
        Does _not_ translate returned source / target locations to the source / target coordinate systems
        (see _three_d_indices_to_output_coords).

        Args:
            src_region (String): A single brain region name. Sepcifies the source region of projections to look up.

            tgt_regions (String): A single brain region name. Sepcifies the target region of projections to look up.

        Returns:

            (numpy.array): A 1d numpy.array of projection strengths between the voxels associated with the source and target
                region.
            
            (numpy.array): A N x 3 array of the indices of brain voxels that the _sources_ of the entries in the first output.

            (numpy.array): A N x 3 array of the indices of brain voxels that are the _target_ of the entries in the first output.
        """
        import h5py
        expected_grp = "{0}/{1}".format(src_region, tgt_region)
        with h5py.File(self._cache_fn, "a") as h5:
            if expected_grp in h5:
                grp = h5[expected_grp]
                return (grp['data'][:],
                        grp[coordinate_name]['source_coordinates'][:],
                        grp[coordinate_name]['target_coordinates'][:])
            elif self._cache_only:
                raise ValueError("Working in cache-only mode, but {0} to {1} is not in cache!".format(src_region,
                                                                                                      tgt_region))
            else:
                V, src, tgt = self._uncached_projection([src_region], [tgt_region])
                if write_cache:
                    grp = h5.require_group(expected_grp)
                    grp.create_dataset('data', data=V)
                    grp = grp.create_group(coordinate_name)
                    grp.create_dataset('source_coordinates', data=src)
                    grp.create_dataset('target_coordinates', data=tgt)
                return V, src, tgt

    def projection(self, src_regions, tgt_regions, coordinate_name='3d'):
        """
        Return strengths of connections from voxels in the specified source regions to voxels in the specified target regions
        using the cache.
        Translate returned source / target locations to the source / target coordinate systems
        (see _three_d_indices_to_output_coords).

        Args:
            src_regions (list): List of strings of region names. Sepcifies the source regions of projections to look up.

            tgt_regions (list): List of strings of region names. Sepcifies the target regions of projections to look up.

            coordinate_name (str): Not used.

        Returns:

            (numpy.array): A 1d numpy.array of projection strengths between the voxels associated with the source and target
                regions.
            
            (numpy.array): A N x 3 array of the coordinates of brain voxels that the _sources_ of the entries in the first output.

            (numpy.array): A N x 3 array of the coordinates of brain voxels that are the _target_ of the entries in the first output.
        """
        assert len(src_regions) > 0 and len(tgt_regions) > 0
        all_src_coord = []
        all_V = []
        for src in src_regions:
            V, src_coord, tgt_coord = zip(*[self._cached_single_projection(src, tgt, coordinate_name=coordinate_name,
                                                                           write_cache=self._grow_cache)
                                            for tgt in tgt_regions])
            V = numpy.hstack(V)
            all_src_coord.append(src_coord[0])  # TODO: Check consistency of src_coord
            all_V.append(V)
        #  TODO: Check consistency of tgt_coord
        src_coord = self._three_d_indices_to_output_coords(numpy.vstack(all_src_coord), 'source')
        tgt_coord = self._three_d_indices_to_output_coords(numpy.vstack(tgt_coord), 'target')
        return numpy.vstack(all_V), src_coord, tgt_coord

