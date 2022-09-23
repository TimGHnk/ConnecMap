#from .cached_projection import CachedProjections
from .aibs_mcm_projection import AibsMcmProjections
import voxcell


class FlattenedProjections(AibsMcmProjections):
    """
    A version of CachedProjections that additionally has flat maps attached to it that affect the output coordinates
    For more info see CachedProjections
    """

    def __init__(self, allen_data=None, cache_file=None, flatmap_source=None, flatmap_target=None):
        super(FlattenedProjections, self).__init__(allen_data=allen_data, cache_file=cache_file)
        if isinstance(flatmap_source, str):
            self._fm_src = voxcell.VoxelData.load_nrrd(flatmap_source)
        else:
            self._fm_src = flatmap_source
        if isinstance(flatmap_target, str):
            self._fm_tgt = voxcell.VoxelData.load_nrrd(flatmap_target)
        else:
            self._fm_tgt = flatmap_target

    def _three_d_indices_to_output_coords(self, idx, which_direction):
        coords = super(FlattenedProjections, self)._three_d_indices_to_output_coords(idx, which_direction)
        if which_direction == 'source' and self._fm_src is not None:
            coords = self._fm_src(coords)
        if which_direction == 'target' and self._fm_tgt is not None:
            coords = self._fm_tgt(coords)
        return coords
