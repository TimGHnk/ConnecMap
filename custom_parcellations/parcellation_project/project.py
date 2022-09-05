import os
import shutil
import voxcell
import json
import numpy


class ParcellationProject(object):
    _path_config = "input/configuration/config.json"

    def __init__(self, root, resume=False):
        """
        root: Path to project root. A configuration file must exist under root/input/configuration/config.json
        resume: Set to true if you are resuming an existing flattening project, for example to add another split to
        an existing project. Otherwise a new project will be initialized, potentially overwriting anything in the
        project directory.
        """
        self._root = root
        self.__read_config__()
        if resume:
            self.__resume__()
        else:
            self.__initialize__()

    def __read_config__(self):
        # Find configuration file at expected location and load it
        with open(os.path.join(self._root, ParcellationProject._path_config), "r") as fid:
            self.config = json.load(fid)
        # Hyperparameters, such as initial module parcellation and other
        self.parameters = self.config["parameters"]
        # Configuration of the individual parcellation level
        self.lvl_cfg = self.config["level_configuration"]

    def __initialize__(self):  # Use this when starting a new ParcellationProject
        root = os.path.join(self._root, self.lvl_cfg["paths"]["lower_level"])
        # Create initial ParcellationLevel object according to the "initial_parcellation"
        self.root_level = ParcellationLevel.initial_parcellation(root,
                                                                 self.lvl_cfg,
                                                                 self.config["parameters"]["initial_parcellation"])
        # Note: Creating this object will automatically write the custom hierarchy and volume to files

        self.current_level = self.root_level  # Level that is currently worked on

    def __resume__(self):  # Use this when resuming an existing project
        root = os.path.join(self._root, self.lvl_cfg["paths"]["lower_level"])
        self.root_level = ParcellationLevel.from_file_system(root,
                                                             self.lvl_cfg)
        self.current_level = self.root_level
        while self.current_level.child is not None:
            self.current_level = self.current_level.child

    def flatten_current_level(self, components_to_use=[0, 1], overwrite=False):
        connectivity = self.config['parameters']['Diffusion_mapping']
        self.current_level.write_flattening_config(components_to_use, overwrite=overwrite, **connectivity)
        print("""Flattening configuration has been written to {0}.
        Execute flattening algorithm using that file!""".format(self.current_level.flattening_fn))
    
    def allen_model_config(self, version):
        """Write voxel model manifest into current level and copy manifest, annotation file
        and structures file into the allen connectivity model directory.
        """
        with open(self.lvl_cfg["voxel_model_manifest"], "r") as read_file:
            original_manifest = json.load(read_file)
        self.current_level.write_manifest(version=version, model_manifest=original_manifest)
        shutil.copyfile(self.current_level.structures_fn,
                        f'{self.lvl_cfg["allen_model_path"]}/structures_{version}.json')
        shutil.copyfile(self.current_level.manifest_fn,
                        f'{self.lvl_cfg["allen_model_path"]}/voxel_model_manifest_{version}.json')
        shutil.copyfile(self.current_level.region_volume_fn,
                        f'{self.lvl_cfg["allen_model_path"]}/annotation/ccf_2017/annotation_100_{version}.nrrd')
        print(f'Copying allen model configuration files for Trial {version} into {self.lvl_cfg["allen_model_path"]}')
    

    def analyze_current_parcellation(self):
        """Call specified functions that analyze the current parcellation scheme. Function are specified in the configuration
        Each function should take a ParcellationLevel object as input and write some results (text file, plot, etc.) under
        ParcellationLevel.analysis_root.
        """
        from parcellation_project.analyses import parcellation  # module for all analyses of parcellations
        for analysis_name in self.lvl_cfg["analyses"]["parcellation"]:
            function = parcellation.__dict__[analysis_name]
            function(self.current_level, output_root=self.current_level.analysis_root, hemisphere=self.current_level._config["hemisphere"])


    def analyze_current_flatmap(self):
        """
        Like "analyze_current_parcellation, but analyzing the flatmap at the current level instead
        """
        from parcellation_project.analyses import flatmaps  # module for all analyses of flatmaps
        for analysis_name in self.lvl_cfg["analyses"]["flatmap"]:
            for region_name in self.current_level.regions:  # Analyze regions separately
                # The input specifying the region to be analyzed is conveniently a Hierarchy object
                region_root = self.current_level.hierarchy_root.find("acronym", region_name)
                assert len(region_root) == 1
                function = flatmaps.__dict__[analysis_name]
                function(self.current_level, region_root[0], output_root=self.current_level.analysis_root, hemisphere=self.current_level._config["hemisphere"])

    def __apply_split__(self, split_solution):
        from parcellation_project.tree_helpers import deep_copy
        curr = self.current_level
        hc = deep_copy(curr.hierarchy)
        ann_vol = curr.region_volume.raw.copy()
        structures = curr.structures.copy()
        offset = numpy.max(ann_vol) + 1

        for region_name, solution in split_solution.items():
            r = hc.find("acronym", region_name)[0]
            assert len(r.children) == 0, "Region to be split should be clean"
            curr_ids = list(r.get("id"))
            idxx = numpy.in1d(ann_vol.flat, curr_ids).reshape(ann_vol.shape)
            idxx[:,:,:int(idxx.shape[2]/2)] = False
            assert numpy.count_nonzero(idxx) == len(solution), "This should never be triggered!"
            ann_vol[idxx] = solution[:, -1] + offset
            for sub_id in numpy.unique(solution[:, -1]):
                r.children.append(voxcell.Hierarchy({
                    "name": r.data["name"] + "_" + str(int(sub_id)),
                    "acronym": r.data["acronym"] + "_" + str(int(sub_id)),
                    "id": int(sub_id + offset)
                }))
                highest_graph_order = int(numpy.max([struc["graph_order"] for struc in structures]))
                struc_reg = [x for x in structures if x["acronym"] == region_name][0]
                structures.append({
                    "acronym": r.data["acronym"] + "_" + str(int(sub_id)),
                    "name": r.data["name"] + "_" + str(int(sub_id)),
                    "id": int(sub_id + offset),
                    "rgb_triplet": [255, 255, 255],
                    "graph_id": int(1),
                    "graph_order": int(highest_graph_order + 1),     
                    "structure_id_path": struc_reg.get('structure_id_path') + [int(sub_id + offset)],
                    "structure_set_ids": struc_reg.get('structure_set_ids').copy()})
            offset += (numpy.max(solution) + 1)
        return hc, voxcell.VoxelData(ann_vol, curr.region_volume.voxel_dimensions,
                                     offset=curr.region_volume.offset), structures

    def split_current_level(self):
        from parcellation_project.split import decide_split
        # Step 1
        cfg = self.config['parameters']["splitting"]["step_1"]
        initial_func = decide_split.__dict__[cfg["function"]]
        initial_solution = initial_func(self.current_level, *cfg["args"], **cfg["kwargs"])
        # Step 2
        cfg = self.config['parameters']["splitting"]["step_2"]
        second_func = decide_split.__dict__[cfg["function"]]
        validation_with_ = cfg["validation"]
        if validation_with_['function'] != 0:
            validation_func = decide_split.__dict__[validation_with_['function']]
        else: validation_func = None
        second_solution = second_func(self.current_level, initial_solution, validation_func, *cfg["args"], **cfg["kwargs"], **validation_with_["kwargs"])
        split_hierarchy, split_regions, split_structures = self.__apply_split__(second_solution)
        new_level = ParcellationLevel(self.current_level.next_level_root,
                                      self.lvl_cfg,
                                      split_hierarchy,
                                      split_structures,
                                      split_regions)
        self.current_level.child = new_level
        self.current_level = new_level

    def tune_splitting_algorithm(self, parameters_grid, **kwargs):
        from parcellation_project.split import decide_split, tuning
        cfg = self.config['parameters']["splitting"]["step_1"]
        initial_func = decide_split.__dict__[cfg["function"]]
        initial_solution = initial_func(self.current_level, *cfg["args"], **cfg["kwargs"])
        cfg = self.config['parameters']["splitting"]["step_2"]
        tuner = tuning.__dict__[cfg["tuning"]]
        gridSearch = tuner(self.current_level, initial_solution, parameters_grid, **kwargs, **cfg["validation"]["kwargs"])
        return gridSearch
        
        
        
    def viz_split(self, region, **kwargs):
        from parcellation_project.split import visualize_split
        hierarchy_root = self.current_level.hierarchy_root.find('acronym', region)[0]
        cfg = self.config["parameters"]["visualization"]
        firstFunc = visualize_split.__dict__[cfg["step_1"]]
        secondFunc = visualize_split.__dict__[cfg["step_2"]]
        new_subregions = visualize_split.viz_split_region(self.current_level, hierarchy_root, firstFunc, secondFunc, **kwargs)
        return new_subregions
    

        

class ParcellationLevel(object):

    def __init__(self, root, config, hierarchy, structures, region_volume, overwrite=False):
        self._root = root
        if not os.path.isdir(root):
            os.makedirs(root)
        self._config = config

        self.child = None
        self._hierarchy = hierarchy
        if self._hierarchy is not None and (not os.path.isfile(self.hierarchy_fn) or overwrite) :
            self.write_hierarchy()
        self._structures = structures
        if self._structures is not None and (not os.path.isfile(self.structures_fn) or overwrite):
            self.write_structures()
        self._region_volume = region_volume
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
    def structures_fn(self):
        return os.path.join(self._root, self._config["paths"]["structures"])

    @property
    def next_level_root(self):
        return os.path.join(self._root, self._config["paths"]["lower_level"])

    @property
    def manifest_fn(self):
        return os.path.join(self._root, self._config["paths"]["manifest"])

    @property
    def flattening_fn(self):
        return os.path.join(self._root, self._config["paths"]["flatten_cfg"])

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

    def _load_region_volume(self):
        return voxcell.VoxelData.load_nrrd(self.region_volume_fn)

    def _load_hierarchy(self):
        return voxcell.Hierarchy.load_json(self.hierarchy_fn)

    def _load_structures(self):
        with open(self.structures_fn, "r") as fid:
            struc = json.load(fid)
        return struc

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
    def structures(self):
        if self._structures is None:
            self._structures = self._load_structures()
        return self._structures

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
        h = self.hierarchy
        r = h.find("acronym", self._config["root_region"])
        assert len(r) == 1, "Root region {0} not found!".format(self._config["root_region"])
        return r[0]

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
        from parcellation_project.tree_helpers import hierarchy_to_dict
        hier = hierarchy_to_dict(self.hierarchy)  # voxcell.Hierarchy
        if not os.path.isdir(os.path.split(self.hierarchy_fn)[0]):
            os.makedirs(os.path.split(self.hierarchy_fn)[0])
        with open(self.hierarchy_fn, 'w') as fid:
            json.dump(hier, fid)

    def write_structures(self):
        struc = self.structures
        if not os.path.isdir(os.path.split(self.structures_fn)[0]):
            os.makedirs(os.path.split(self.structures_fn)[0])
        with open(self.structures_fn, "w") as fid:
            json.dump(struc, fid, indent=2, sort_keys=False)

    def write_manifest(self, version, model_manifest):
        model_manifest['manifest'][2]['spec'] = f'structures_{version}.json'
        model_manifest['manifest'][4]['spec'] = f'annotation_%d_{version}.nrrd'
        with open(self.manifest_fn, 'w') as fid:
            json.dump(model_manifest, fid, indent=1, sort_keys=False)

    def write_flattening_config(self, components_to_use, overwrite=False, **kwargs):
        from parcellation_project.tree_helpers import leaves, normalization_spread, normalization_offsets
        if os.path.isfile(self.flattening_fn) and not overwrite:
            return
        to_flatten = leaves(self.hierarchy_root, property="acronym")
        if kwargs["consider_connectivity"] == 'inter':
            to_consider = leaves(self.hierarchy_root, property="acronym")
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
            json.dump(flat_config, fid)
            
            
    @staticmethod
    def max_depth_region_map(hierarchy, root_region):
        from parcellation_project.tree_helpers import at_max_depth, leaves
        r = hierarchy.find("acronym", root_region)
        assert len(r) == 1, "Root region {0} not found!".format(root_region)
        r = r[0]
        # leaves = at_max_depth(r)
        leaves = leaves(r)
        ids = [l.data["id"] for l in leaves]
        regions = [l.data["acronym"] for l in leaves]
        region_map = dict(zip(regions, ids))
        return region_map

    @classmethod
    def from_file_system(cls, root, config):
        fn_h = os.path.join(root, config["paths"]["hierarchy"])
        h = voxcell.Hierarchy.load_json(fn_h)
        fn_v = os.path.join(root, config["paths"]["region_volume"])
        v = voxcell.VoxelData.load_nrrd(fn_v)
        fn_s = os.path.join(root, config["paths"]["structures"])
        with open(fn_s, "r") as fid:
            s = json.load(fid)

        ret = cls(root, config, h, s, v)
        ret.find_next_level()
        return ret

    @classmethod
    def initial_parcellation(cls, root, config, initial_parcellation):
        from parcellation_project.tree_helpers import deep_copy
        annotations = voxcell.VoxelData.load_nrrd(config["anatomical_parcellation"])
        hierarchy = voxcell.Hierarchy.load_json(config["anatomical_hierarchy"])
        with open(config["voxel_model_structures"], "r") as fid:
            structures = json.load(fid)
        custom_hierarchy = deep_copy(hierarchy)
        id_offset = int(numpy.max(list(hierarchy.get("id"))) + 1)        
        r = custom_hierarchy.find("acronym", config["root_region"])
        assert len(r) == 1, "Root region {0} not found!".format(config["root_region"])
        r = r[0]
        ids_to_remove = []
        for child in r.children:
            ids_to_remove.extend(list(child.get("id")))

        r.children = []  # truncate hierarchy
        highest_graph_order = numpy.max([struc["graph_order"] for struc in structures])
        struc_root = [x for x in structures if x["acronym"] == config["root_region"]]
        assert len(struc_root) == 1, "Root region {0} not found in structures file".format(config["root_region"])
        root_id_path = struc_root[0]["structure_id_path"]
        root_struc_set_ids = struc_root[0]["structure_set_ids"]

        for i, items in enumerate(initial_parcellation.items()):
            module_name, module_regions = items
            module_id = id_offset + i

            target_ids = []
            for region in module_regions:
                reg = hierarchy.find("acronym", region)
                if len(reg) < 1:
                    print("Warning: No region found for {0}".format(region))
                for _r in reg:
                    target_ids.extend(_r.get("id"))

            module_mask = numpy.in1d(annotations.raw.flat, target_ids).reshape(annotations.raw.shape)
            # Build mask, consider one or both hemispheres and discard non flat voxels
            if config["hemisphere"] == "right":                         
                module_mask[:,:,:int(module_mask.shape[2]/2)] = False
            elif config["hemisphere"] == "left":
                module_mask[:,:,int(module_mask.shape[2]/2):] = False
            # Remove non flat pixels    
            anat_fm = voxcell.VoxelData.load_nrrd(config["anatomical_flatmap"])
            mask_fm = anat_fm.raw[:,:,:,0] == -1
            module_mask[mask_fm == True] = False
            
            annotations.raw[module_mask] = module_id
            r.children.append(voxcell.Hierarchy({
                "name": module_name,
                "acronym": module_name,
                "id": int(module_id)
            }))
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
        return cls(root, config, custom_hierarchy, structures, annotations, overwrite=True)

