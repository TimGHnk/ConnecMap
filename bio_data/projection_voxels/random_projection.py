import string
import numpy

from .cached_projection import CachedProjections

"""
This file implements an exemplary test class, derived from CachedProjections that returns completely random connectivity data.
It mainly serves as an illustration of how such a derived class for a specific data source could be implemented, and for technical
tests of the tool chain.
"""

class RandomTree(object):

    def __init__(self, n_regions):
        self._n_regions = n_regions

    @staticmethod
    def _region_ids(lst_regions):
        return numpy.array(
            [
                string.ascii_lowercase.index(_reg[0].lower())
                for _reg in lst_regions
            ]
        ) + 1
    
    def region_ids(self, lst_regions):
        ret = self._region_ids(lst_regions)
        if numpy.any(ret > self._n_regions):
            raise ValueError("Unknown region!")
        return ret

class RandomVoxelArray(object):

    def __init__(self, tgt_shape, n_regions):
        self._shape = tgt_shape
        self.__setup__(n_regions)
        self._meta_seed = numpy.random.randint(2000)
    
    def __setup__(self, n_regions):
        shape_arr = numpy.array(self._shape)
        r = numpy.min(shape_arr) / 2.0
        cutoffs = numpy.linspace(0.0, r, n_regions + 1)
        c = shape_arr / 2.0
        ann = numpy.zeros(self._shape, dtype=int)

        for i in numpy.arange(c[0] - r, c[0] + r):
            for j in numpy.arange(c[1] - r, c[1] + r):
                for k in numpy.arange(c[2] - r, c[2] + r):
                    pt = numpy.array([i, j, k]).astype(int)
                    d = numpy.linalg.norm(pt - c)
                    idxx = numpy.nonzero(cutoffs > d)[0]
                    if len(idxx) > 0:
                        ann[pt[0], pt[1], pt[2]] = idxx[0]
        
        self.model_annotations = ann
        nz = numpy.vstack(numpy.nonzero(ann)).transpose()
        self.source_coords_3d = nz
        self.target_coords_3d = nz
    
    def __getitem__(self, args):
        assert len(args) == 2
        arg1, arg2 = args
        tgt_shape = ()
        if isinstance(arg1, slice):
            raise NotImplementedError()
        elif hasattr(arg1, "__len__"):
            tgt_shape = tgt_shape + (len(arg1),)

        if isinstance(arg2, slice):
            raise NotImplementedError()
        elif hasattr(arg2, "__len__"):
            tgt_shape = tgt_shape + (len(arg2),)
        
        numpy.random.seed(int(numpy.prod(tgt_shape)) + self._meta_seed)
        return numpy.random.rand(*tgt_shape)


class RandomProjections(CachedProjections):

    def __init__(self, tgt_shape, n_regions, voxel_sizes, cache_file=None, grow_cache=True):
        """
        Args:
            tgt_shape (tuple): Tuple of length 3. The shape of the voxelized structure to emulate.

            n_regions(int): Number of individual brain regions to emulate within the voxelized structure

            voxel_sizes (tuple): Tuple of length 3. Specifies the resolution of the voxelized connectivity, i.e. the size of each
               voxel in x, y, z dimensions

            cache_file (string): See parent class

            grow_cache (string): See parent class
        
        Will return random connectivity strength. Emulates a voxelized brain that is a sphere within the specified shape. Places
        the specified number of brain regions as concentric shells of that sphere.
        """
        voxel_array = RandomVoxelArray(tgt_shape, n_regions)
        model_annotations = voxel_array.model_annotations
        source_coords_3d = voxel_array.source_coords_3d
        target_coords_3d = voxel_array.target_coords_3d
        tree = RandomTree(n_regions)
        super().__init__(voxel_array, source_coords_3d, target_coords_3d, 
                         (voxel_array, ), (n_regions, ), cache_file, grow_cache)
        self._voxel_sizes = voxel_sizes
    
    def _three_d_indices_to_output_coords(self, idx, direction):
        return numpy.array(idx) * numpy.array([self._voxel_sizes])
    
    @classmethod
    def _initialize_annotations(cls, *args):
        assert len(args) == 1
        vxl_array = args[0]
        return vxl_array.model_annotations
    
    @classmethod
    def _initialize_hierarchy(cls, *args):
        assert len(args) == 1
        n_regions = args[0]
        return RandomTree(n_regions)
