import os
import shutil
import json
import numpy

from .parcellation_level import ParcellationLevel


class AibsLevel(ParcellationLevel):

    def __init__(self, root, config, hierarchy, structures, region_volume, overwrite=False):
        super().__init__(root, config, hierarchy, region_volume, overwrite=overwrite)
        self._structures = structures
        if self._structures is not None and (not os.path.isfile(self.structures_fn) or overwrite):
            self.write_structures()
        if not os.path.isfile(self.manifest_fn) or overwrite:
            self.write_manifest()

    @property
    def structures_fn(self):
        return os.path.join(self._root, self._config["paths"]["structures"])

    @property
    def manifest_fn(self):
        return os.path.join(self._root, self._config["paths"]["manifest"])
    
    @property
    def cache_cfg(self):
        # TODO: Specify resolution to use (self._voxel_resolution) here as an input to AibsMcmProjections!
        cache_cfg = {
            "class": "AibsMcmProjections",
            "args": {
                "AllenCache": os.path.abspath(self.manifest_fn),
                "H5Cache": os.path.join(os.path.split(self._config["inputs"]["voxel_model_manifest"])[0],
                                        "projections_h5_cache.h5")
            }
        }
        return cache_cfg

    def _load_structures(self):
        with open(self.structures_fn, "r") as fid:
            struc = json.load(fid)
        return struc

    @property
    def structures(self):
        if self._structures is None:
            self._structures = self._load_structures()
        return self._structures

    def write_structures(self):
        struc = self.structures
        if not os.path.isdir(os.path.split(self.structures_fn)[0]):
            os.makedirs(os.path.split(self.structures_fn)[0])
        with open(self.structures_fn, "w") as fid:
            json.dump(struc, fid, indent=2, sort_keys=False)

    def write_manifest(self):
        """
        Files to override: 
            structures (custom)
            annotations (custom)
        """
        str_custom_dir = "CUSTOMDIR"
        import os

        region_vol_for_mcm = self.copy_files_for_mcmodels()

        path_in = os.path.abspath(self._config["inputs"]["voxel_model_manifest"])
        path_out = os.path.abspath(self.manifest_fn)
        root_in = os.path.split(path_in)[0]
        root_out = os.path.split(path_out)[0]

        with open(path_in, "r") as fid:
            manifest_in = json.load(fid)
        manifest_lst = manifest_in["manifest"]

        for entry in manifest_lst:
            if entry["type"] in ["dir", "file"]:
                if entry["key"] == "ANNOTATION":
                    entry["spec"] = os.path.relpath(region_vol_for_mcm, root_out)
                    entry["parent_key"] = str_custom_dir
                elif entry["key"] == "STRUCTURE_TREE":
                    entry["spec"] = os.path.relpath(self.structures_fn, root_out)
                    entry["parent_key"] = str_custom_dir
                elif "parent_key" not in entry:
                    entry["spec"] = os.path.relpath(os.path.join(root_in, entry["spec"]), root_out)
        
        manifest_lst.insert(1,
            {
                "key": str_custom_dir,
                "type": "dir",
                "spec": "."
            }
        )
        
        with open(self.manifest_fn, 'w') as fid:
            json.dump(manifest_in, fid, indent=2, sort_keys=False)


    def copy_files_for_mcmodels(self):
        from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
        MCM_FRAG = "%s/%d"
        res_args = (MouseConnectivityApi.CCF_VERSION_DEFAULT, self._voxel_resolution)
        vol_root, vol_fn = os.path.split(self.region_volume_fn)
        out_root = os.path.join(vol_root, MCM_FRAG)
        if not os.path.exists(out_root % res_args):
            os.makedirs(out_root % res_args)
        shutil.copy(self.region_volume_fn, out_root % res_args)

        out_fn = os.path.join(out_root, vol_fn)
        return out_fn
    
    @staticmethod
    def find_inputs_from_file_system(root, config):
        h, v = ParcellationLevel.find_inputs_from_file_system(root, config)
        fn_s = os.path.join(root, config["paths"]["structures"])
        
        with open(fn_s, "r") as fid:
            s = json.load(fid)
        
        return h, s, v

    @staticmethod
    def find_inputs_from_config(config, initial_parcellation):
        custom_hierarchy, annotations = ParcellationLevel.find_inputs_from_config(config, initial_parcellation)
        hier_series = custom_hierarchy.as_dataframe().reset_index().set_index("acronym")["id"]

        with open(config["inputs"]["voxel_model_structures"], "r") as fid:
            structures = json.load(fid)
        highest_graph_order = numpy.max([struc["graph_order"] for struc in structures])
        struc_root = [x for x in structures if x["acronym"] == config["root_region"]]
        assert len(struc_root) == 1, "Root region {0} not found in structures file".format(config["root_region"])
        root_id_path = struc_root[0]["structure_id_path"]
        root_struc_set_ids = struc_root[0]["structure_set_ids"]

        for items in initial_parcellation.items():
            module_name, module_regions = items
            module_id = hier_series[module_name]
            structures.append({
                "acronym": module_name,
                "name": module_name,
                "id": int(module_id),
                "rgb_triplet": [255, 255, 255],
                "graph_id": 1,
                "graph_order": int(highest_graph_order + 1),
                "structure_id_path": root_id_path + [int(module_id)],
                "structure_set_ids": root_struc_set_ids.copy()
            })
            highest_graph_order += 1
        
        return custom_hierarchy, structures, annotations
