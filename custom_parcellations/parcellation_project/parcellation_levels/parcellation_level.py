import os
import voxcell
import json
import numpy


class ParcellationLevel(object):

    def __init__(self, root, config, hierarchy, region_volume, overwrite=False):
        self._root = root
        if not os.path.isdir(root):
            os.makedirs(root)
        self._config = config

        self.child = None
        self._hierarchy = hierarchy
        if self._hierarchy is not None and (not os.path.isfile(self.hierarchy_fn) or overwrite) :
            self.write_hierarchy()
        self._region_volume = region_volume
        self._voxel_resolution = int(region_volume.voxel_dimensions[0])  # TODO: Stop assuming isotropic voxels?
        if self._region_volume is not None and (not os.path.isfile(self.region_volume_fn) or overwrite):
            self.write_region_volume()
        

        self.region_map = self.max_depth_region_map(self._hierarchy, self._config["root_region"])
        self.regions = list(self.region_map.keys())
        self.ids = [self.region_map[r] for r in self.regions]
        self.id_map = dict(zip(self.ids, self.regions))

        self._flatmap = None
        self._characterization = None

        self.is_flattenend = os.path.isfile(self.flatmap_fn)
        self.is_split = False

    @property
    def region_volume_fn(self):
        return os.path.join(self._root, self._config["paths"]["region_volume"])

    @property
    def hierarchy_fn(self):
        return os.path.join(self._root, self._config["paths"]["hierarchy"])

    @property
    def next_level_root(self):
        return os.path.join(self._root, self._config["paths"]["lower_level"])

    @property
    def flattening_fn(self):
        return os.path.join(self._root, self._config["paths"]["flatten_cfg"])

    @property
    def cache_cfg_fn(self):
        return os.path.join(self._root, self._config["paths"]["cache_cfg"])

    @property
    def flatmap_fn(self):
        return os.path.join(self._root, self._config["paths"]["flatmap"])

    @property
    def characterization_fn(self):
        return os.path.join(self._root, self._config["paths"]["characterization"])

    @property
    def analysis_root(self):
        a_root = os.path.join(self._root, self._config["paths"]["analyses"])
        if not os.path.exists(a_root):
            os.makedirs(a_root)
        return a_root
    
    @property
    def cache_cfg(self):
        # TODO: Specify resolution to use (self._voxel_resolution) here as an input to AibsMcmProjections!
        cache_cfg = {
            "class": "CachedProjections",
            "args": {
                "H5Cache": os.path.join(os.path.split(self._config["inputs"]["voxel_model_manifest"])[0],
                                        "projections_h5_cache.h5")
            }
        }
        return cache_cfg

    def _load_region_volume(self):
        return voxcell.VoxelData.load_nrrd(self.region_volume_fn)

    def _load_hierarchy(self):
        return voxcell.RegionMap.load_json(self.hierarchy_fn)

    def _load_flatmap(self):
        if not os.path.isfile(self.flatmap_fn):
            raise RuntimeError("This level has not been flattened yet.")
        return voxcell.VoxelData.load_nrrd(self.flatmap_fn)

    def _load_characterization(self):
        if not os.path.isfile(self.flatmap_fn):
            raise RuntimeError("This level has not been flattened yet.")
        with open(self.characterization_fn, "r") as fid:
            char = json.load(fid)
        return char

    @property
    def region_volume(self):
        if self._region_volume is None:
            self._region_volume = self._load_region_volume()
        return self._region_volume

    @property
    def hierarchy(self):
        if self._hierarchy is None:
            self._hierarchy = self._load_hierarchy()
        return self._hierarchy

    @property
    def flatmap(self):
        if self._flatmap is None:
            self._flatmap = self._load_flatmap()
        return self._flatmap
    
    @property
    def characterization(self):
        if self._characterization is None:
            self._characterization = self._load_characterization()
        return self._characterization

    @property
    def hierarchy_root(self):
        from ..tree_helpers import region_map_at
        return region_map_at(self.hierarchy, self._config["root_region"])


    def find_next_level(self):
        if os.path.isdir(self.next_level_root):
            try:
                next_level = ParcellationLevel.from_file_system(self.next_level_root, self._config)
                self.child = next_level
            except:
                pass
        if self.child is not None:
            self.is_split = True

    def write_region_volume(self):
        reg_vol = self.region_volume
        if not os.path.isdir(os.path.split(self.region_volume_fn)[0]):
            os.makedirs(os.path.split(self.region_volume_fn)[0])
        reg_vol.save_nrrd(self.region_volume_fn)
    
    def write_hierarchy(self):
        from parcellation_project.tree_helpers import region_map_to_dict
        hier = region_map_to_dict(self.hierarchy)  # voxcell.RegionMap
        if not os.path.isdir(os.path.split(self.hierarchy_fn)[0]):
            os.makedirs(os.path.split(self.hierarchy_fn)[0])
        with open(self.hierarchy_fn, 'w') as fid:
            json.dump(hier, fid)

    def write_flattening_config(self, components_to_use, overwrite=False, **kwargs):
        from parcellation_project.tree_helpers import leaves, normalization_spread, normalization_offsets
        if os.path.isfile(self.flattening_fn) and not overwrite:
            return
        to_flatten = leaves(self.hierarchy_root)["acronym"].values
        if kwargs["consider_connectivity"] == 'inter':
            to_consider = list(leaves(self.hierarchy_root)["acronym"].values)
        direction = kwargs["connectivity_direction"]
        flat_config = []
        lst_spread = []
        for region in to_flatten:
            norm_arg = normalization_spread(self.region_volume.raw, self.hierarchy, region)
            lst_spread.append(norm_arg)
            if kwargs["consider_connectivity"] == 'intra':
                to_consider = [region]
            flat_config.append({
                "connectivity_target": {
                    "considered_regions": to_consider,
                    "direction": direction},
                "normalization_args": {
                    "normalize_spread": [norm_arg, norm_arg]},
                "flatten": [region],
                "components": components_to_use})
        
        flat_config = normalization_offsets(lst_spread, flat_config)
        with open(self.flattening_fn, "w") as fid:
            json.dump(flat_config, fid, indent=2)
    
    def write_cache_config(self, overwrite=False):
        if os.path.isfile(self.cache_cfg_fn) and not overwrite:
            return

        with open(self.cache_cfg_fn, "w") as fid:
            json.dump(self.cache_cfg, fid, indent=2)
            
    @staticmethod
    def max_depth_region_map(hierarchy, root_region):
        from ..tree_helpers import region_map_at, leaves

        r = leaves(region_map_at(hierarchy, root_region))
        return dict(zip(r["acronym"].values, r.index))
    
    @staticmethod
    def find_inputs_from_config(config, initial_parcellation):
        from ..tree_helpers import region_map_to_dict, truncate_hierarchy_dict_at, find_node_in_hierarchy_dict

        annotations = voxcell.VoxelData.load_nrrd(config["inputs"]["anatomical_parcellation"])
        hierarchy = voxcell.RegionMap.load_json(config["inputs"]["anatomical_hierarchy"])
        
        hierarchy_dict = truncate_hierarchy_dict_at(region_map_to_dict(hierarchy), "acronym", config["root_region"])
        # Not a copy. Changes to this will be reflected in hierarchy_dict
        hierarchy_root = find_node_in_hierarchy_dict(hierarchy_dict, "acronym", config["root_region"])

        id_offset = int(hierarchy.as_dataframe().index.values.max() + 1)        

        for i, items in enumerate(initial_parcellation.items()):
            module_name, module_regions = items
            module_id = id_offset + i

            target_ids = []
            for region in module_regions:
                reg = hierarchy.find(region, "acronym", with_descendants=True)
                if len(reg) < 1:
                    print("Warning: No region found for {0}".format(region))
                target_ids.extend(reg)

            module_mask = numpy.in1d(annotations.raw.flat, target_ids).reshape(annotations.raw.shape)
            ### Build mask, consider one or both hemispheres and discard non flat voxels
            if config["hemisphere"] == "right":                         
                module_mask[:,:,:int(module_mask.shape[2]/2)] = False
            elif config["hemisphere"] == "left":
                module_mask[:,:,int(module_mask.shape[2]/2):] = False
            # Remove non flat pixels    
            anat_fm = voxcell.VoxelData.load_nrrd(config["inputs"]["anatomical_flatmap"])
            mask_fm = anat_fm.raw[:,:,:,0] == -1
            module_mask[mask_fm == True] = False
            ###
            annotations.raw[module_mask] = module_id

            hierarchy_root.setdefault("children", []).append(
                {
                    "name": module_name,
                    "acronym": module_name,
                    "id": int(module_id),
                    "children": []
                }
            )
        custom_hierarchy = voxcell.RegionMap.from_dict(hierarchy_dict)

        return custom_hierarchy, annotations
    
    @staticmethod
    def find_inputs_from_file_system(root, config):
        fn_h = os.path.join(root, config["paths"]["hierarchy"])
        h = voxcell.RegionMap.load_json(fn_h)
        fn_v = os.path.join(root, config["paths"]["region_volume"])
        v = voxcell.VoxelData.load_nrrd(fn_v)
        
        return h, v

    @classmethod
    def from_file_system(cls, root, config):
        args = cls.find_inputs_from_file_system(root, config)

        ret = cls(root, config, *args)
        ret.find_next_level()
        return ret

    @classmethod
    def initial_parcellation(cls, root, config, initial_parcellation):
        args = cls.find_inputs_from_config(config, initial_parcellation)

        return cls(root, config, *args, overwrite=True)
