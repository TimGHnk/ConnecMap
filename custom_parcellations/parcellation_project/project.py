import os
import voxcell
import json
import numpy

# from .parcellation_levels import AibsLevel
from . import parcellation_levels


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
        fn_cfg = os.path.join(self._root, ParcellationProject._path_config)
        path_cfg = os.path.split(os.path.abspath(fn_cfg))[0]
        with open(fn_cfg, "r") as fid:
            self.config = json.load(fid)

        # Hyperparameters, such as initial module parcellation and other
        self.parameters = self.config["parameters"]

        # Which data-source specific class to use
        self.level_class = parcellation_levels.__dict__[self.config["level_class"]]

        # Configuration of the individual parcellation level
        self.lvl_cfg = self.config["level_configuration"]
        for k, k_path in self.lvl_cfg["inputs"].items():
            if not os.path.isabs(k_path):
                self.lvl_cfg["inputs"][k] = os.path.join(path_cfg, k_path)

    def __initialize__(self):  # Use this when starting a new ParcellationProject
        root = os.path.join(self._root, self.lvl_cfg["paths"]["lower_level"])
        # Create initial ParcellationLevel object according to the "initial_parcellation"
        self.root_level = self.level_class.initial_parcellation(root,
                                                                 self.lvl_cfg,
                                                                 self.config["parameters"]["initial_parcellation"])
        # Note: Creating this object will automatically write the custom hierarchy and volume to files

        self.current_level = self.root_level  # Level that is currently worked on

    def __resume__(self):  # Use this when resuming an existing project
        root = os.path.join(self._root, self.lvl_cfg["paths"]["lower_level"])
        self.root_level = self.level_class.from_file_system(root,
                                                             self.lvl_cfg)
        self.current_level = self.root_level
        while self.current_level.child is not None:
            self.current_level = self.current_level.child

    def flatten_current_level(self, components_to_use=[0, 1], overwrite=False):
        # TODO: components should be in configuration instead.
        connectivity = self.config['parameters']['Diffusion_mapping']
        self.current_level.write_flattening_config(components_to_use, overwrite=overwrite, **connectivity)
        self.current_level.write_cache_config(overwrite=overwrite)

        print(
            """
            Preparations for running the diffusion embedding process done. Please run:

            make_diffusion_flatmap.py --cache_config {0} --output_path {1} --region_file {2} --characterize {3}
            """.format(self.current_level.cache_cfg_fn,
            self.current_level.flatmap_fn,
            self.current_level.flattening_fn,
            self.current_level.characterization_fn)
        )
    

    def analyze_current_parcellation(self):
        """Call specified functions that analyze the current parcellation scheme. Function are specified in the configuration
        Each function should take a ParcellationLevel object as input and write some results (text file, plot, etc.) under
        ParcellationLevel.analysis_root.
        """
        from parcellation_project.analyses import parcellation  # module for all analyses of parcellations
        for analysis_name in self.lvl_cfg["analyses"]["parcellation"]:
            function = parcellation.__dict__[analysis_name]
            function(self.current_level, output_root=self.current_level.analysis_root)


    def analyze_current_flatmap(self):
        """
        Like "analyze_current_parcellation, but analyzing the flatmap at the current level instead
        """
        from parcellation_project.analyses import flatmaps  # module for all analyses of flatmaps
        from .tree_helpers import region_map_at
        for analysis_name in self.lvl_cfg["analyses"]["flatmap"]:
            for region_name in self.current_level.regions:  # Analyze regions separately
                region_root = region_map_at(self.current_level.hierarchy_root, region_name)
                function = flatmaps.__dict__[analysis_name]
                function(self.current_level, region_root, output_root=self.current_level.analysis_root)

    def __apply_split__(self, split_solution):
        from .tree_helpers import deep_copy, region_map_to_dict, find_node_in_hierarchy_dict
        curr = self.current_level
        hc = region_map_to_dict(curr.hierarchy)
        # hc = deep_copy(curr.hierarchy)
        ann_vol = curr.region_volume.raw.copy()
        structures = curr.structures.copy()
        offset = numpy.max(ann_vol) + 1

        for region_name, solution in split_solution.items():
            # r = hc.find("acronym", region_name)[0]
            r = find_node_in_hierarchy_dict(hc, "acronym", region_name)
            assert len(r.get("children", [])) == 0, "Region to be split should be clean"
            curr_ids = voxcell.RegionMap.from_dict(r).as_dataframe().index.values
            idxx = numpy.in1d(ann_vol.flat, curr_ids).reshape(ann_vol.shape)
            idxx[:,:,:int(idxx.shape[2]/2)] = False
            assert numpy.count_nonzero(idxx) == len(solution), "This should never be triggered!"
            ann_vol[idxx] = solution[:, -1] + offset
            for sub_id in numpy.unique(solution[:, -1]):
                r.setdefault("children", []).append(
                    {
                        "name": r["name"] + "_" + str(int(sub_id)),
                        "acronym": r["acronym"] + "_" + str(int(sub_id)),
                        "id": int(sub_id + offset),
                        "children": []
                    }
                )

                highest_graph_order = int(numpy.max([struc["graph_order"] for struc in structures]))
                struc_reg = [x for x in structures if x["acronym"] == region_name][0]
                structures.append({
                    "acronym": r["acronym"] + "_" + str(int(sub_id)),
                    "name": r["name"] + "_" + str(int(sub_id)),
                    "id": int(sub_id + offset),
                    "rgb_triplet": [255, 255, 255],
                    "graph_id": int(1),
                    "graph_order": int(highest_graph_order + 1),     
                    "structure_id_path": struc_reg.get('structure_id_path') + [int(sub_id + offset)],
                    "structure_set_ids": struc_reg.get('structure_set_ids').copy()})
            offset += (numpy.max(solution) + 1)
        return voxcell.RegionMap.from_dict(hc), voxcell.VoxelData(ann_vol, curr.region_volume.voxel_dimensions,
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
        new_level = self.level_class(self.current_level.next_level_root,
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
