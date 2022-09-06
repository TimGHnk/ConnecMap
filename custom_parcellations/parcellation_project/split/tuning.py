import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from parcellation_project.split import decide_split as splt
import hdbscan
from scipy.spatial.distance import pdist,squareform
from parcellation_project.analyses.flatmaps import flatmap_to_coordinates, gradient_map
from scipy.spatial.distance import cdist
from parcellation_project.analyses.parcellation import _mi_implementation
from parcellation_project.analyses.parcellation import mutual_information_two_parcellations
import seaborn as sns
from parcellation_project.analyses import flatmaps as fm_analyses


def SVM_tuner(parc_level, initial_solution, param_grid, **kwargs):
    regionsSplitted = []
    for i in range(len(param_grid['C'])):
        for j in range(len(param_grid['gamma'])):  
            results = {'C' : param_grid['C'][i],
                           'gamma': param_grid['gamma'][j],
                           'regions': []}
            for region_name, solution in initial_solution.items():
                is_valid = None
                if len(numpy.unique(solution[:,-1][~numpy.isnan(solution[:,-1])])) < 2:
                    is_valid = False
                else:
                    solution = splt.split_with_SVM(solution, param_grid['C'][i], param_grid['gamma'][j], kwargs['thres_accuracy'], show=False)
                    if len(numpy.unique(solution[:,-1])) == 1:
                        is_valid = False
                    else:
                        try:
                            solution = splt.extract_subregions(solution, kwargs['t'])
                        except: is_valid = False
                        else:
                            is_valid = splt.validate_split_lm(solution, **kwargs)
                if is_valid is True:
                    results['regions'].append(region_name)
            regionsSplitted.append(results)
    return regionsSplitted


def mi_gridSearch(fm0, fm1, annotations, hierarchy_reg, show=True):
    '''Computes the mutual information between the initial quadri-classification result
    and the SVM extrapolation for every sets of hyperparameters.    
    '''
    x1,y1,x2,y2 = fm_analyses.gradient_map(fm0, fm1, annotations, hierarchy_reg, show=False)  
    three_d_coords, two_d_coords = fm_analyses.flatmap_to_coordinates(annotations, fm0, hierarchy_reg)
    solution0 = splt.quadri_classification(x1, y1, x2, y2, two_d_coords)
    solution1 = numpy.column_stack((solution0[0][:,:-1], (solution0[0][:, -1] + 2 * solution0[1][:, -1])/2))
    paramC = numpy.around(numpy.arange(0.1,1,0.1),1)
    paramGamma = numpy.around(numpy.arange(0.01,0.1,0.01),2)
    df = pd.DataFrame(index=paramC, columns=paramGamma).astype('float')
    for c in paramC:
        for gamma in paramGamma:
            out_C1 = splt.split_with_SVM(solution0[0], c, gamma, thres_accuracy=0, show=False)
            out_C2 = splt.split_with_SVM(solution0[1], c, gamma, thres_accuracy=0, show=False)
            solution2 = numpy.column_stack((out_C1[:,:-1], (out_C1[:, -1] + 2 * out_C2[:, -1])/2))
            distrib0, distrib1 = mutual_information_two_parcellations(solution1[~numpy.isnan(solution1[:,-1])],solution2[~numpy.isnan(solution1[:,-1])])
            info_gain, prior_entropy = _mi_implementation(distrib0, distrib1)
            df.loc[c,gamma] = float(info_gain/prior_entropy * 1)
    if show is True:
        sns.heatmap(df, annot=True)
    return df


def stability_gridSearch(fm0, fm1, annotations, hierarchy_reg, N=10, noise_amplitude = 0.3, show=True):
    '''Compute the stability against noise (using mutual information) of the quadri-classification
    and SVM results for every sets of hyperparameters.
    '''
    x1,y1,x2,y2 = fm_analyses.gradient_map(fm0, fm1, annotations, hierarchy_reg, show=False)  
    three_d_coords, two_d_coords = fm_analyses.flatmap_to_coordinates(annotations, fm0, hierarchy_reg)
    solution0 = splt.quadri_classification(x1, y1, x2, y2, two_d_coords)
    paramC = numpy.around(numpy.arange(0.1,1,0.1),1)
    paramGamma = numpy.around(numpy.arange(0.01,0.1,0.01),2)
    df = pd.DataFrame(index=paramC, columns=paramGamma).astype('float')
    for c in paramC:
        for gamma in paramGamma:
            out_C1 = splt.split_with_SVM(solution0[0], c, gamma, thres_accuracy=0, show=False)
            out_C2 = splt.split_with_SVM(solution0[1], c, gamma, thres_accuracy=0, show=False)
            real_solution = numpy.column_stack((out_C1[:,:-1], (out_C1[:, -1] + 2 * out_C2[:, -1])/2))
            lst_info_gain = []
            noisy_solutions = []
            for j in range(N):
                solution_noise = noisy_quadri_classification(x1, y1, x2, y2, two_d_coords, noise_amplitude=noise_amplitude)
                out_C1 = splt.split_with_SVM(solution_noise[0], c, gamma, thres_accuracy=0, show=False)
                out_C2 = splt.split_with_SVM(solution_noise[1], c, gamma, thres_accuracy=0, show=False)
                noisy_results = numpy.column_stack((out_C1[:,:-1], (out_C1[:, -1] + 2 * out_C2[:, -1])/2))
                noisy_solutions.append(noisy_results)
                distrib0, distrib1 = mutual_information_two_parcellations(real_solution[~numpy.isnan(real_solution[:,-1])], noisy_results[~numpy.isnan(noisy_results[:,-1])])
                info_gain, prior_entropy = _mi_implementation(distrib0, distrib1)
                lst_info_gain.append(info_gain)
            df.loc[c,gamma] = float(numpy.mean(lst_info_gain/prior_entropy * 1))
    if show is True:
        sns.heatmap(df, annot=True)
    return df

def noisy_quadri_classification(x1, y1, x2, y2, two_d_coords, noise_amplitude=0.3):
    '''Compute a noisy quadri-classification. Useful to assess stability.
    '''
    da = numpy.column_stack((x1[two_d_coords[:,0], two_d_coords[:,1]], y1[two_d_coords[:,0], two_d_coords[:,1]]))
    db = numpy.column_stack((x2[two_d_coords[:,0], two_d_coords[:,1]], y2[two_d_coords[:,0], two_d_coords[:,1]]))
    da_angle = numpy.arctan2(da[:, 0], da[:, 1])
    db_angle = numpy.arctan2(db[:, 0], db[:, 1])
    angle_difference = numpy.mod(db_angle - da_angle, 2*numpy.pi)
    angle_img = angle_difference.copy()
    noise = noise_amplitude * numpy.random.rand(len(angle_img)) - noise_amplitude / 2
    angle_img = angle_img + noise
    cls_sign = numpy.sign(numpy.sin(angle_img)) + 1
    cls_inversion = numpy.sign(numpy.cos(angle_img)) + 1
    out_ssin = numpy.column_stack((two_d_coords, cls_sign))
    out_scos = numpy.column_stack((two_d_coords, cls_inversion))
    return [out_ssin, out_scos]


def tune_HDBSCAN_cluster_size(fm0, fm1, annotations, hierarchy_reg, show=False):
    results = pd.DataFrame(columns=['cluster_size', 'outliers_ratio', 'number_clusters',
                                    'mean_probability'])
    x1,y1,x2,y2 = gradient_map(fm0, fm1, annotations, hierarchy_reg, show=False)  
    three_d_coords, two_d_coords = flatmap_to_coordinates(annotations, fm0, hierarchy_reg)
    vecX = numpy.vstack(
    [numpy.array([[x1[two_d_coords[i,0],two_d_coords[i,1]],
                   y1[two_d_coords[i,0],two_d_coords[i,1]]]]) for i in range(len(two_d_coords))])
    vecY = numpy.vstack(
        [numpy.array([[x2[two_d_coords[i,0],two_d_coords[i,1]],
                       y2[two_d_coords[i,0],two_d_coords[i,1]]]]) for i in range(len(two_d_coords))])
    x_arr = vecX[numpy.unique(numpy.where(numpy.invert(numpy.isnan(vecX)))[0])]
    y_arr = vecY[numpy.unique(numpy.where(numpy.invert(numpy.isnan(vecY)))[0])]
    distX = squareform(pdist(x_arr, metric='cosine'))
    distY = squareform(pdist(y_arr, metric='cosine'))
    distGrad = distX + distY
    distGrad = distGrad / numpy.pi * 2 #Normalize between 0 and 2
    if len(distGrad) < 2000:
        parameter_grid = numpy.arange(40,401,20)
    else:
        parameter_grid = numpy.arange(100,1001,25)
    for param in parameter_grid:
        clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, metric='precomputed',
                                min_cluster_size=int(param), min_samples=10,
                                cluster_selection_method = "eom") 
        clusterer.fit(distGrad)
        outliers = numpy.count_nonzero(clusterer.labels_ == -1)/len(clusterer.labels_)
        n_clusters = numpy.unique(clusterer.labels_[clusterer.labels_ > -1])
        results = results.append({'cluster_size': param,
                        'outliers_ratio': outliers,
                        'number_clusters': int(len(n_clusters)),
                        'mean_probability': numpy.mean(clusterer.probabilities_)},
                                 ignore_index=True)
    if show is True:
        results.plot('cluster_size', 'outliers_ratio', grid=True)
        results.plot('cluster_size', 'number_clusters', kind='bar')
        results.plot('cluster_size', 'mean_probability', grid=True)
    selected = results[results['outliers_ratio'] == results['outliers_ratio'].min()]
    selected = selected[selected['mean_probability'] == selected['mean_probability'].max()]
    size_star = selected.loc[selected['number_clusters'].idxmin()]['cluster_size']
    return results, size_star

def similarity_clusters(gX, gY, clf):
    x_clust = numpy.column_stack((gX,clf))
    y_clust = numpy.column_stack((gY,clf))
    df = pd.DataFrame(index=numpy.unique(clf[clf >-1]),
                   columns=numpy.unique(clf[clf >-1]))
    for a in numpy.unique(clf[clf >-1]):
        for b in numpy.unique(clf[clf >-1]):  
            if a == b:
                continue
            x1 = x_clust[x_clust[:,-1] == a]
            x2 = x_clust[x_clust[:,-1] == b]
            dist_x = cdist(x1[:,:-1], x2[:,:-1], metric='cosine')
            dist_x = dist_x / numpy.pi * 2
            y1 = y_clust[y_clust[:,-1] == a]
            y2 = y_clust[y_clust[:,-1] == b]
            dist_y = cdist(y1[:,:-1], y2[:,:-1], metric='cosine')
            dist_y = dist_y / numpy.pi * 2
            mean_dist = numpy.mean(dist_x) + numpy.mean(dist_y)
            df.at[a,b] = mean_dist
    return df


