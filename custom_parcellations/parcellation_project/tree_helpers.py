import numpy
import rpack


def max_depth(t):
    if len(t.children) == 0:
        return 0
    return numpy.max([max_depth(c) for c in t.children]) + 1


def at_depth(t, depth, property=None):
    if depth == 0:
        return [t.data.get(property, t)]
    ret = []
    for c in t.children:
        ret.extend(at_depth(c, depth - 1, property=property))
    return ret


def at_max_depth(t, property=None):
    return at_depth(t, max_depth(t), property=property)


def leaves(t):
    r = t.as_dataframe()
    leaf_ids = numpy.setdiff1d(r.index, r["parent_id"])
    leaves = r.loc[leaf_ids]
    return leaves


def deep_copy(t):
    from voxcell import Hierarchy
    h_out = Hierarchy(t.data.copy())
    for c in t.children:
        h_out.children.append(deep_copy(c))
    return h_out


def hierarchy_to_dict(h):
    ret = {}
    ret.update(h.data)
    ret["children"] = [hierarchy_to_dict(c) for c in h.children]
    return ret


def region_map_to_dict(h, root=None):
    if root is None:
        roots = [k for k, v in h._parent.items() if v is None]
        assert len(roots) == 1
        root = roots[0]
    elif isinstance(root, str):
        root_list = list(h.find(root, "acronym"))
        assert len(root_list) == 1, "Region {0} not found!".format(root)
        root = root_list[0]

    def _recursive_part(node_id):
        ret = {}
        ret.update(h._data[node_id])
        ret["children"] = [_recursive_part(c) for c in h._children[node_id]]
        return ret

    return _recursive_part(root)


def region_map_at(h, root):
    from voxcell import RegionMap
    return RegionMap.from_dict(region_map_to_dict(h, root=root))


def find_node_in_hierarchy_dict(hier_dict, prop_name, prop_value):
    def _recursive_part(h):
        if h.get(prop_name, None) == prop_value:
            return h
        for c in h["children"]:
            tst = _recursive_part(c)
            if tst is not None: return tst
    return _recursive_part(hier_dict)


def truncate_hierarchy_dict_at(hier_dict, prop_name, prop_value):
    def _recursive_part(h):
        ret = {}
        for k, v in h.items():
            if k != "children":
                ret[k] = v
        if ret.get(prop_name, None) != prop_value:
            ret["children"] = [_recursive_part(c) for c in h["children"]]
        return ret
    return _recursive_part(hier_dict)


def normalization_spread(annotation, hierarchy, region):
    constant = 12
    voxel_index = list(hierarchy.find(region, "acronym", with_descendants=True))
    count = 0
    for i in range(len(voxel_index)):
        count += numpy.count_nonzero(annotation == voxel_index[i])
    norm_arg = round(numpy.sqrt(count/constant))
    return int(norm_arg)      

def normalization_offsets(lst_spread, configuration_json):
    squares = []
    for spread in lst_spread:
        squares.append((spread, spread))
    positions = rpack.pack(squares)
    for i in range(len(configuration_json)):
        configuration_json[i]['normalization_args']['normalize_offsets'] = list(
            positions[i])
    return configuration_json