"""
For analyses and plots of region annotation volumes
"""
import numpy
from scipy import stats
import pandas as pd
from math import log2
import matplotlib.pyplot as plt
import copy
import re
from voxcell import VoxelData, RegionMap



def from_parcellation(base_func):
    def returned_function(parc_level, **kwargs):
        vol0_fn = parc_level._config["anatomical_parcellation"]
        vol0 = VoxelData.load_nrrd(vol0_fn)
        vol1 = copy.deepcopy(parc_level.region_volume)
        r0_fn = parc_level._config['anatomical_hierarchy'] 
        r0 = RegionMap.load_json(r0_fn)
        r1 = parc_level.id_map
        return base_func(vol0, vol1, r0, r1, **kwargs)
    return returned_function

def _mi_implementation(distribution_initial, distribution_split):
    """Computes information gain of a given distribution. Also returns the prior entropy.
    """
    voxel_subs = []
    for n in range(len(distribution_split)):
        voxel_subs.append(sum(distribution_split[n]['count']))
    voxel_total = sum(distribution_initial['count'])
    
    distrib_parcel = numpy.array(distribution_initial['distrib'])
    prior_entropy = stats.entropy(distrib_parcel, base = 2) / log2(len(distrib_parcel))
    entropies = []
    for n in range(len(distribution_split)):
        distrib_split = numpy.array(distribution_split[n]['distrib'])
        entropy_sub = stats.entropy(distrib_split, base = 2) / log2(len(distrib_split))
        entropies.append(voxel_subs[n] / voxel_total * entropy_sub)
    info_gain = prior_entropy - sum(entropies)
    return info_gain, prior_entropy



def mutual_information_regions(vol0, vol1, hierarchy0, modules, plot=True, right_hemisphere=True, **kwargs):
    '''Computes the information gain of the regions' distribution of the 
    traditional parcellation scheme (CCF, v3) depending on the custom
    parcellation scheme.
    '''
    distrib0, distrib1 = distribution_regions(vol0, vol1, hierarchy0, modules, right_hemisphere=right_hemisphere)
    info_gain, prior_entropy =  _mi_implementation(distrib0, distrib1)
    if plot is True:
        plot_info_gain(prior_entropy, info_gain, method='regions')
        if 'output_root' in kwargs:
            plt.savefig(kwargs['output_root']+"/Information_gain_regions.png")
    if 'output_root' in kwargs:
        res = pd.DataFrame({})
        res['Prior_entropy'] = [prior_entropy]
        res['Information_gain'] = [info_gain]
        res.to_csv(kwargs['output_root']+"/Information_gain_regions.csv", sep = ';', index = False)
    else:
        if plot is True:
            plt.show()
        return prior_entropy, info_gain
    
def mutual_information_layers(vol0, vol1, hierarchy0, modules, plot=True, right_hemisphere=True, **kwargs):
    '''Computes the information gain of the layers' distribution of the 
    traditional parcellation scheme (CCF, v3) depending on the custom
    parcellation scheme.
    '''
    distrib0, distrib1 = distribution_layers(vol0, vol1, hierarchy0, modules, right_hemisphere=right_hemisphere)
    info_gain, prior_entropy =  _mi_implementation(distrib0, distrib1)
    if plot is True:
        plot_info_gain(prior_entropy, info_gain, method='layers')
        if 'output_root' in kwargs:
            plt.savefig(kwargs['output_root']+"/Information_gain_layers.png")
    if 'output_root' in kwargs:
        res = pd.DataFrame({})
        res['Prior_entropy'] = [prior_entropy]
        res['Information_gain'] = [info_gain]
        res.to_csv(kwargs['output_root']+"/Information_gain_layers.csv", sep = ';', index = False)
    else:
        if plot is True:
            plt.show()
        return prior_entropy, info_gain
        


def plot_info_gain(prior_entropy, lst_info_gain, method, lst_names=None):
    if isinstance(lst_info_gain, list) is False:
        lst_info_gain = [lst_info_gain]
    ax = plt.figure(figsize=(10, 6)).gca()
    ax.bar(range(1,len(lst_info_gain)+1),lst_info_gain, color = 'royalblue')
    ax.bar(range(1),prior_entropy, color = 'darkorange')
    ax.set_title(f'Information gain of {method} distribution (in bits)')
    if lst_names is not None:
        name_labels = lst_names.copy()
        name_labels.insert(0, 'Prior_entropy')
        ax.set_xticks(range(len(name_labels)))
        ax.set_xticklabels(name_labels, rotation='vertical')
    else:
        ax.set_xticks(range(len(lst_info_gain)+1))
        ax.set_xticklabels(['Prior_entropy', 'Split'], rotation='vertical')
    colors = {'Prior_entropy':'darkorange', 'information_gain':'royalblue'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    

def plot_overlap(overlap, iso_regions, new_regions, **kwargs):    
    ax = plt.figure(figsize=(12, 12)).gca()
    ax.imshow(overlap)
    ax.set_xticks(range(len(new_regions)))
    ax.set_xticklabels(new_regions, rotation='vertical')
    ax.set_yticks(range(len(iso_regions)))
    ax.set_yticklabels(iso_regions)
    plt.colorbar(ax.imshow(overlap), ax=ax, orientation='horizontal')
    plt.title('Overlap of the 2 parcellations', fontsize = 20)



def overlap_two_parcellations(vol0, vol1, hierarchy0, modules, plot=True, right_hemisphere=True, **kwargs):
    ''' Returns the ratio of each region from parcellation A contained in each region 
    from parcellation B.
    '''
    curr_modules = list(modules.values())
    reg_names, _ = regions_and_layers(hierarchy0, vol0)
    _, distrib1 = distribution_regions(vol0, vol1, hierarchy0, modules, right_hemisphere=right_hemisphere)
    df_ratio = pd.DataFrame({})
    df_ratio['name'] = reg_names
    for j in range(len(distrib1)):
        df_ratio[f'{curr_modules[j]}'] = distrib1[j]['ratio']
    if plot is True:
        overlap = df_ratio.loc[:, df_ratio.columns != 'name'].to_numpy()
        plot_overlap(overlap, reg_names, curr_modules)
        if 'output_root' in kwargs:
            plt.savefig(kwargs['output_root']+"/Overlap.png")
    if 'output_root' in kwargs:
        df_ratio.to_csv(kwargs['output_root']+"/overlap.csv", sep = ';', index=False)
    else:
        if plot is True:
            plt.show()
        return df_ratio
    

def distribution_regions(vol0, vol1, hierarchy0, modules, right_hemisphere=True):
    ''' Returns one dictionary of the number of voxels of each regions of the 
    traditional parcellation scheme (CCF,v3) in the isocortex and its distribution.
    Returns another dictionary correponding of the
    number of voxels, distribution and the ratio of these regions within the regions
    of the custom parcellation scheme.
    '''
    if right_hemisphere is True:
        vol0.raw = vol0.raw[:,:,int(vol0.raw.shape[2]/2) : int(vol0.raw.shape[2])].copy()
        vol1.raw = vol1.raw[:,:,int(vol1.raw.shape[2]/2) : int(vol1.raw.shape[2])].copy()
    reg_names, _ = regions_and_layers(hierarchy0, vol0)
    valid_ids  = [list(hierarchy0.find(reg, 'acronym', with_descendants=True)) for reg in reg_names]
    valid_ids = [idx for sublist in valid_ids for idx in sublist]
    bitmask = numpy.in1d(vol0.raw.flatten(), valid_ids).reshape(vol0.raw.shape)
    vol = vol0.raw[bitmask]
    # Distribution established parcellation
    distrib0 = {'count': [],
                'distrib': []}
    for reg_name in reg_names:
        reg_ids = list(hierarchy0.find(reg_name, 'acronym', with_descendants=True))
        count = sum([numpy.count_nonzero(vol == idx) for idx in reg_ids])
        distrib0['count'].append(count)
        distrib0['distrib'].append(count / len(vol))
    # Distribution within custom parcellation
    distrib1 = []
    for mod_id in modules.keys():
        bitmask = numpy.in1d(vol1.raw.flatten(), mod_id).reshape(vol1.raw.shape)
        vol = vol0.raw[bitmask]
        distrib_mod = {'count': [],
                       'distrib': [],
                       'ratio': []}
        for i in range(len(reg_names)):
            count = sum([numpy.count_nonzero(vol == idx) for idx in list(hierarchy0.find(reg_names[i], 'acronym', with_descendants=True))])
            distrib_mod['count'].append(count)
            distrib_mod['distrib'].append(count / len(vol))
            distrib_mod['ratio'].append(count / distrib0['count'][i])
        distrib1.append(distrib_mod)
    return distrib0, distrib1


def distribution_layers(vol0, vol1, hierarchy0, modules, right_hemisphere=True):
    ''' Returns one dictionary of the number of voxels of each layers of the 
    traditional parcellation scheme (CCF,v3) in the isocortex and its distribution.
    Returns another dictionary correponding of the
    number of voxels, distribution and the ratio of these layers within the regions
    of the custom parcellation scheme.
    '''
    if right_hemisphere is True:
        vol0.raw = vol0.raw[:,:,int(vol0.raw.shape[2]/2) : int(vol0.raw.shape[2])].copy()
        vol1.raw = vol1.raw[:,:,int(vol1.raw.shape[2]/2) : int(vol1.raw.shape[2])].copy()
    reg_names, layers = regions_and_layers(hierarchy0, vol0)
    valid_ids  = [list(hierarchy0.find(reg, "acronym", with_descendants=True)) for reg in reg_names]
    valid_ids = [idx for sublist in valid_ids for idx in sublist]
    bitmask = numpy.in1d(vol0.raw.flatten(), valid_ids).reshape(vol0.raw.shape)
    vol = vol0.raw[bitmask]
    vxl_ids = numpy.unique(vol)
    distrib0 = {'count': [],
                'distrib': []}
    for lay in layers:
        count = sum([numpy.count_nonzero(vol == idx) for idx in vxl_ids if lay in hierarchy0.get(idx, "acronym")])
        distrib0['count'].append(count)
        distrib0['distrib'].append(count / len(vol))
    distrib1 = []
    for mod_id in modules.keys():
        bitmask = numpy.in1d(vol1.raw.flatten(), mod_id).reshape(vol1.raw.shape)
        vol = vol0.raw[bitmask]
        distrib_mod = {'count': [],
                       'distrib': [],
                       'ratio': []}
        for i in range(len(layers)):
            count = sum([numpy.count_nonzero(vol == idx) for idx in vxl_ids if layers[i] in hierarchy0.get(idx, "acronmym")])
            distrib_mod['count'].append(count)
            distrib_mod['distrib'].append(count / len(vol))
            distrib_mod['ratio'].append(count / distrib0['count'][i])
        distrib1.append(distrib_mod)
    return distrib0, distrib1

def mutual_information_two_parcellations(parcA, parcB):
    ''' Compute mutual information between 2 sets of labelled coordinates.
    '''
    idsA, idsB = numpy.unique(parcA[:,-1]), numpy.unique(parcB[:,-1])
    distrib0 = {'count': [],
                'distrib': []}
    for idx in idsA:
        count = numpy.count_nonzero(parcA[:,-1] == idx)
        distrib0["count"].append(count)
        distrib0["distrib"].append(count/len(parcA))
    distrib1 = []
    for idx in idsB:
        distrib_mod = {'count': [],
               'distrib': []}
        mask = numpy.where(parcB[:,-1] == idx)[0]
        for idxx in idsA:
            count = numpy.count_nonzero(parcA[mask,-1] == idxx)
            distrib_mod['count'].append(count)
            distrib_mod['distrib'].append(count / len(parcA[mask]))
        distrib1.append(distrib_mod) 
    return distrib0, distrib1
        
def regions_and_layers(hier, annotations, root='Isocortex'):
    '''For a given cortical area (default is the Isocortex), returns a list
    of the regions and layers that are contained in this area.
    '''
    region_names = []
    layer_names = []
    valid_ids = list(hier.find(root, 'acronym', with_descendants=True))   
    bitmask = numpy.in1d(annotations.raw.flatten(), valid_ids).reshape(annotations.raw.shape)
    vxl_ids = numpy.unique(annotations.raw[bitmask])
    all_names = [hier.get(i, "acronym") for i in vxl_ids]
    for name in all_names:
        region_names.append(re.split(r'(?=\d)', name, maxsplit=1)[0])
        layer_names.append(re.split(r'(?=\d)', name, maxsplit=1)[1])
    region_names = list(set(region_names))
    layer_names = list(set(layer_names))
    region_names.sort()
    layer_names.sort()
    return region_names, layer_names


mutual_information_from_parcellation_regions = from_parcellation(mutual_information_regions)
mutual_information_from_parcellation_layers = from_parcellation(mutual_information_layers)
overlap_with_parcellation = from_parcellation(overlap_two_parcellations)

