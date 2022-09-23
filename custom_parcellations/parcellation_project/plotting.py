"""
To generate some plots.
"""


import json, numpy, voxcell, os
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from parcellation_project.analyses import flatmaps as fm_analyses
from parcellation_project.analyses.parcellation import _mi_implementation
from parcellation_project.analyses.parcellation import mutual_information_two_parcellations
from voxel_maps import coordinates_to_image
import seaborn as sns

from .tree_helpers import region_map_at

def grow_tree(graph, root, df, stat):
    '''Grow a hierarchical tree from the whole isocortex to the last parcellation scheme.
    Node colour is a given statistics.
    '''
    for i in range(len(root.children)):
        graph.add_node(root.data["acronym"], value=float(df.loc[df['region'] == root.data["acronym"], stat]))
        graph.add_node(root.children[i].data["acronym"], value=float(df.loc[df['region'] == root.children[i].data["acronym"], stat]))
        graph.add_edge(root.data["acronym"], root.children[i].data["acronym"])
        if len(root.children[i].children) != 0:
            graph = grow_tree(graph, root.children[i], df, stat=stat)
    return graph
            
def viz_tree(root, df, stat, color='viridis', node_size=30, width=1):
    G = nx.Graph()
    G = grow_tree(G, root, df, stat=stat)
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos=pos, with_labels=False, node_size=node_size,
                   width=width, node_color=list(nx.get_node_attributes(G, 'value').values()), cmap=color)
    value = df[stat].to_numpy()
    cbar = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin = min(value), vmax=max(value)))
    cbar._A = []
    plt.colorbar(cbar, orientation='horizontal', pad=0.05)
    plt.title(f'Tree plot with {stat}')
    plt.show()

def final_analyses(root, analysis, save=True):
    '''Returns a data frame for analysis of every regions at every steps.
    '''
    dfs = []
    initial_root = root
    while any([entry.path.endswith('split') for entry in os.scandir(root)]) is True:
        root = root+'/split'
        df = pd.read_csv(root+f'/output/analyses/{analysis}.csv', sep=";")
        dfs.append(df)
    results = pd.concat(dfs)
    results = results.drop_duplicates(subset='region', keep='first', inplace=False)
    results.to_csv(initial_root+f'/{analysis}.csv', sep = ";", index=False)
    return results



def connectivity_structure(region, fm0, fm1, annotations, hierarchy_root, show=False):
    '''Return a 2D image of the connectivity structure of a given region
    (Diffusion flatmap on Anatomical flatmap).
    '''
    mask = numpy.all((~numpy.isnan(fm1.raw)) & (fm1.raw > -1), axis=3)
    ann_vals = annotations.raw[mask]
    xy = fm0.raw[mask]
    ab = fm1.raw[mask]
    tgt_region_ids = list(hierarchy_root.find(region, "acronym", with_descendants=True))
    sub_xy = xy[numpy.in1d(ann_vals, tgt_region_ids)]
    sub_ab = ab[numpy.in1d(ann_vals, tgt_region_ids)]
    tmp_img = coordinates_to_image(sub_ab, sub_xy)
    if show == True:
        y_min = min(m for m in sub_xy[:,0] if m >= 0)
        y_max = max(m for m in sub_xy[:,0] if m >= 0)
        x_min = min(m for m in sub_xy[:,1] if m >= 0)
        x_max = max(m for m in sub_xy[:,1] if m >= 0)
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(tmp_img[:, :, 0], cmap='jet')
        fig.colorbar(im, ax=ax1, orientation='vertical')
        ax1.set_title("Mean \u03B1")
        ax1.axis((x_min-2,x_max+2,y_min-2,y_max+2))
        ax1.set_ylim(sorted(ax1.get_ylim()))
        ax1.axis('off')
        ax2 = fig.add_subplot(1, 2, 2)
        im1 = ax2.imshow(tmp_img[:, :, 1], cmap='jet')
        fig.colorbar(im1, ax=ax2, orientation='vertical')
        ax2.set_title("Mean \u03B2")
        ax2.axis((x_min-2,x_max+2,y_min-2,y_max+2))
        ax2.set_ylim(sorted(ax2.get_ylim()))
        ax2.axis('off')
        plt.suptitle(f'Connectivity structure of {region}', y = 1.05)
        plt.show()
    return tmp_img


from parcellation_project.split import decide_split as splt
from parcellation_project.split.tuning import noisy_quadri_classification


def stability_measure(fm0, fm1, annotations, hierarchy_reg, SVM=True, c=0.1, gamma=0.01, N=10, noise_amplitude=0.3, show=True):
    '''Computes the stability against noise of the quadri classification results.
    Set SVM = True to add the SVM extrapolation process.
    '''
    x1,y1,x2,y2 = fm_analyses.gradient_map(fm0, fm1, annotations, hierarchy_reg, show=False)  
    three_d_coords, two_d_coords = fm_analyses.flatmap_to_coordinates(annotations, fm0, hierarchy_reg)
    solution0 = splt.quadri_classification(x1, y1, x2, y2, two_d_coords)
    if SVM is True:    
        out_C1 = splt.split_with_SVM(solution0[0], c, gamma, thres_accuracy=0, show=False)
        out_C2 = splt.split_with_SVM(solution0[1], c, gamma, thres_accuracy=0, show=False)
        real_solution = numpy.column_stack((out_C1[:,:-1], (out_C1[:, -1] + 2 * out_C2[:, -1])/2))
    else: 
        real_solution = numpy.column_stack((two_d_coords, (solution0[0][:, -1] + 2 * solution0[1][:, -1])/2))
    lst_info_gain = []
    noisy_solutions = []
    for j in range(N):
        solution_noise = noisy_quadri_classification(x1, y1, x2, y2, two_d_coords, noise_amplitude=noise_amplitude)
        if SVM is True:
            out_C1 = splt.split_with_SVM(solution_noise[0], c, gamma, thres_accuracy=0, show=False)
            out_C2 = splt.split_with_SVM(solution_noise[1], c, gamma, thres_accuracy=0, show=False)
            noisy_results = numpy.column_stack((out_C1[:,:-1], (out_C1[:, -1] + 2 * out_C2[:, -1])/2))
        else:
            noisy_results = numpy.column_stack((two_d_coords, (solution_noise[0][:, -1] + 2 * solution_noise[1][:, -1])/2))
        noisy_solutions.append(noisy_results)
        distrib0, distrib1 = mutual_information_two_parcellations(real_solution[~numpy.isnan(real_solution[:,-1])], noisy_results[~numpy.isnan(noisy_results[:,-1])])
        info_gain, prior_entropy = _mi_implementation(distrib0, distrib1)
        lst_info_gain.append(info_gain)
    res = lst_info_gain / prior_entropy * 1 # Normalize between 0 and 1
    avg_stability = numpy.mean(res)
    error = numpy.std(res)
    if show is True:
        fig, ax = plt.subplots(figsize=(2,5))
        df = pd.DataFrame({"name": "info_gain_grd",
                            "values": res})
        sns.barplot(x="name", y="values", data=df, capsize=.1, ci="sd")
        sns.swarmplot(x="name", y="values", data=df, color="0", alpha=.5)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.show()
    return avg_stability, error, real_solution, noisy_solutions



def quadri_stability_from_parcellation(parc_level, noise_amplitude, N, plot=True, save=True, **kwargs):
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = voxcell.VoxelData.load_nrrd(fm0_fn)  # Anatomical fm
    annotations = parc_level.region_volume
    results = pd.DataFrame(columns=['region', 'mean_stability', 'std_stability'])
    for region_name in parc_level.regions:
        r = region_map_at(parc_level.hierarchy_root, region_name)
        coords3d, coords2d = fm_analyses.flatmap_to_coordinates(annotations, fm0, r)
        gradient_dev = numpy.mean(fm_analyses.gradient_deviation_from_parcellation(parc_level, r, plot=False))
        reversal_idx = fm_analyses.reversal_index_from_parcellation(parc_level, r)
        if (reversal_idx > kwargs["thres_reversal_index"]) | (gradient_dev > kwargs["thres_gradient_deviation"]):
            split_is_required = True
        else: split_is_required = False
        if split_is_required:
            val, error = quadri_stability(parc_level, region_name, noise_amplitude, N, plot=plot, **kwargs)
            results = results.append({"region": region_name,
                                      "mean_stability": val,
                                      "std_stability": error}, ignore_index=True)
    if save is True:
        results.to_csv(parc_level.analysis_root+'/quadri_stability.csv', sep = ";", index=False)

    return results

# TODO: finish code
def quadri_stability(parc_level, region_name, noise_amplitude, N, plot=True, **kwargs):
    fm0_fn = parc_level._config["anatomical_flatmap"]
    fm0 = voxcell.VoxelData.load_nrrd(fm0_fn)
    annotations = parc_level.region_volume
    fm1 = parc_level.flatmap
    hierarchy_reg = region_map_at(parc_level.hierarchy_root, region_name)
    x1,y1,x2,y2 = fm_analyses.gradient_map(fm0, fm1, annotations, hierarchy_reg)  
    three_d_coords, two_d_coords = fm_analyses.flatmap_to_coordinates(annotations, fm0, hierarchy_reg)
    solution0 = splt.quadri_classification(x1, y1, x2, y2, two_d_coords)
    out_C1 = splt.split_with_SVM(solution0[0], kwargs['C'], kwargs['gamma'], thres_accuracy=0, show=False)
    out_C2 = splt.split_with_SVM(solution0[1], kwargs['C'], kwargs['gamma'], thres_accuracy=0, show=False)
    solution1 = numpy.column_stack((out_C1[:,:-1], out_C1[:, -1] + 2 * out_C2[:, -1]))
    solution2 = splt.unflattening(parc_level, region_name, solution1)
    restults_no_noise = splt.extract_subregions(solution2, t=kwargs['t'])
    lst_info_gain = []
    for i in range(N):
        da = numpy.vstack(
            [numpy.array([[x1[two_d_coords[i,0],two_d_coords[i,1]],
                           y1[two_d_coords[i,0],two_d_coords[i,1]]]]) for i in range(len(two_d_coords))])
        db = numpy.vstack(
            [numpy.array([[x2[two_d_coords[i,0],two_d_coords[i,1]],
                           y2[two_d_coords[i,0],two_d_coords[i,1]]]]) for i in range(len(two_d_coords))])
        X = two_d_coords[:,0]
        Y = two_d_coords[:,1]
        da_angle = numpy.arctan2(da[:, 0], da[:, 1])
        db_angle = numpy.arctan2(db[:, 0], db[:, 1])
        angle_difference = numpy.mod(db_angle - da_angle, 2*numpy.pi)
        angle_img = angle_difference.copy()
        noise = noise_amplitude * numpy.random.rand(len(angle_img)) - noise_amplitude / 2
        angle_img = angle_img + noise
        cls_sign = numpy.sign(numpy.sin(angle_img)) + 1
        cls_inversion = numpy.sign(numpy.cos(angle_img)) + 1
        out_C1 = splt.split_with_SVM(numpy.vstack([X, Y, cls_sign]).transpose(), kwargs['C'], kwargs['gamma'], thres_accuracy=0, show=False)
        out_C2 = splt.split_with_SVM(numpy.vstack([X, Y, cls_inversion]).transpose(), kwargs['C'], kwargs['gamma'], thres_accuracy=0, show=False)
        solution1 = numpy.column_stack((out_C1[:,:-1], out_C1[:, -1] + 2 * out_C2[:, -1]))
        solution2 = splt.unflattening(parc_level, region_name, solution1)
        noisy_results = splt.extract_subregions(solution2, t=kwargs['t'])
        distrib0, distrib1 = mutual_information_two_parcellations(restults_no_noise, noisy_results)
        info_gain, prior_entropy = _mi_implementation(distrib0, distrib1)
        lst_info_gain.append(info_gain)
    res = lst_info_gain / prior_entropy * 1 # Normalize between 0 and 1
    val = numpy.mean(res)
    error = numpy.std(res)
    if plot is True:
        x_pos = ['mean_information_gain']
        fig, ax = plt.subplots(figsize=(5,8))
        ax.bar(x_pos, val, yerr=error, ecolor='black', capsize=10)
        ax.set_ylabel('information gain in bits (normalized between [0:1])')
        ax.set_xticks(x_pos)
        ax.set_title(f"Stability to noise of {region_name}: mean = {val}, std = {error}")
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.show()
    return val, error