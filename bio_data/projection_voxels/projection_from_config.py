import json

def projection_from_json(fn):
    with open(fn, "r") as fid:
        cfg = json.load(fid)
    
    assert "class" in cfg, "Must specify Cache class!"
    assert "args" in cfg, "Must parameterize Cache class!"
    cache_cls = cfg["class"]
    cache_args = cfg["args"]

    if cache_cls == "CachedProjections":
        assert "H5Cache" in cache_args, "Must specify h5 Cache to use CachedProjections base class!"
        from .cached_projection import CachedProjections
        cache = CachedProjections(None, None, None, (), (),
        cache_file=cache_args["h5Cache"], grow_cache=False)
        cache._cache_only = True
        return cache

    elif cache_cls == "AibsMcmProjections":
        try:
            import mcmodels
            vmc = mcmodels.core.VoxelModelCache(manifest_file=cache_args.get("AllenCache", None))
            from .aibs_mcm_projection import AibsMcmProjections
            cache = AibsMcmProjections(vmc, cache_args.get("H5Cache", None),
                                       grow_cache=(cache_args.get("H5Cache", None) is not None))
        except ImportError:
            raise ImportError("Must install mouse_connectivity_models to use the AibsMcmProjections class!")

    else:
        raise ValueError("Unknown Cache class: {0}".format(cache_cls))
