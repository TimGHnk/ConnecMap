"""
To generate some plots.
"""


import numpy, os
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from parcellation_project.analyses import flatmaps as fm_analyses
from parcellation_project.project import ParcellationLevel
from parcellation_project.analyses.parcellation import _mi_implementation
from parcellation_project.analyses.parcellation import mutual_information_two_parcellations
from voxel_maps import coordinates_to_image
import seaborn as sns

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
            
def viz_tree(root, df, stat, save, color='viridis', node_size=30, width=1):
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
    if save:
        plt.savefig(root+f"/tree_plot_{stat}.png")
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
    if save:
        results.to_csv(initial_root+f'/{analysis}.csv', sep = ";", index=False)
    return results


def build_tree_from_parcellation(project, statistic, save, color="viridis", node_size=30, width=1):
    """Build a hierarchical tree plot from the current parcellation.
    The color node corresponds to a given statistics
    Arguments:
    project (class): parcellation project object
    root (str): root directory of the parcellation to plot.
    statistic (str): statistic to plot, either "gradient_deviation" or "reversal_index".
    save (bool): if true, save the results in the root directory.
    color (str): color of the node. Default to "viridis".
    node_size (int): size of the node. Default to 30
    width (int): width of the branches. Default to 1.
    """
    results = final_analyses(project._root, statistic, save)
    hier_root = project.current_level.hierarchy_root.find("acronym", 'Module1')[0]
    viz_tree(hier_root, results, stat=statistic, save=save, color=color, node_size=node_size, width=width)

def connectivity_structure(region, fm0, fm1, annotations, hierarchy_root, show=False):
    '''Return a 2D image of the connectivity structure of a given region
    (Diffusion flatmap on Anatomical flatmap).
    '''
    mask = numpy.all((~numpy.isnan(fm1.raw)) & (fm1.raw > -1), axis=3)
    ann_vals = annotations.raw[mask]
    xy = fm0.raw[mask]
    ab = fm1.raw[mask]
    tgt_region_ids = list(hierarchy_root.collect('acronym', region, 'id'))
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
