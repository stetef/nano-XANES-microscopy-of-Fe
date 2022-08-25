"""Useful python functions."""

import numpy as np
import itertools

import matplotlib.pyplot as plt
import mplcursors
from matplotlib.ticker import FormatStrFormatter
from PIL import Image, ImageSequence

from scipy.optimize import minimize

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

import umap


def parse_tiff(filename):
    img = Image.open(filename)
    data = np.array(img)
    N = 0
    for i, page in enumerate(ImageSequence.Iterator(img)):
        N += 1
    Data = np.zeros((N, data.shape[0], data.shape[1]))
    for i, page in enumerate(ImageSequence.Iterator(img)):
        Data[i] = np.array(page)
    return Data


def parse_nor(filename):
    data = []
    labels = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                if line.startswith('# Column.'):
                    label = line.replace(f'# Column.{len(labels) + 1}: ', '').replace('\n', '')
                    labels.append(label)
                else:
                    pass
            else:
                parsed = line.replace('\n', '').split(' ')
                while "" in parsed:
                    parsed.remove('')
                parsed = [float(e) for e in parsed]
                data.append(parsed)
    f.close()
    data = np.array(data).T
    print(data.shape)
    ref_energy = data[0, :]
    Refs = data[1:, :]
    Ref_dict = {labels[i + 1]: Refs[i] for i in range(Refs.shape[0])}
    return ref_energy, Ref_dict, Refs


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def add_point_label(pickable, data, ax):
    """Add point label."""
    def onselect(sel):
        x, y = sel.target.index[0], sel.target.index[1]
        annotation = f'({x}, {y})'
        sel.annotation.set_text(annotation)
        spectrum = data[:, x, y]
        ax.clear()
        ax.plot(np.arange(len(spectrum)), spectrum, linestyle='-', linewidth=2, c=plt.cm.tab10(7))
        remove_ticks(ax)
    mplcursors.cursor(pickable, highlight=True).connect("add", onselect)


def get_filtered_img(data, threshold=0.05, return_mask=False):
    mask = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    bool_arr = np.max(data, axis=0) < threshold
    mask[:, bool_arr] = 1
    filtered_img = np.ma.array(data, mask=mask)
    if return_mask:
        return filtered_img, mask
    else:
        return filtered_img


def evaluate_similarity(x, y, metric):
    if metric == 'cosine similarity':
        score = cosine_similarity([x], Y=[y])[0][0]
    elif metric == 'Pearson correlation':
        score, pval = pearsonr(x, y)
    elif metric == '1 - $\delta$':
        score = 1 - np.average(np.abs(x - y))
    elif metric == '1 - MSE':
        score = 1 - eval('mean_squared_error')(x, y)
    elif metric == '1 - IADR':
        score = 1 - np.sum(np.abs(x - y)) / max(np.sum(x), np.sum(y))
    return score


def plot_corr_matx(ax, Similarity_matrix, data_columns):
    img = ax.imshow(Similarity_matrix, cmap=plt.cm.RdPu,
                    interpolation='nearest', origin='lower')
    ax.tick_params(direction='out', width=2, length=6, labelsize=14)
    ax.set_title(f'{metric}', fontsize=20)

    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.tick_params(labelsize=14, width=2, length=3)

    N = len(Similarity_matrix)
    ax.set_yticks(np.arange(N))
    labels = [e.replace('_', ' ').replace('NP', '') for e in data_columns]
    ax.set_yticklabels(labels, fontsize=14)

    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(labels, fontsize=14, rotation=90)

    
def plot_MSE_hist(ax, tmp_X, Refs, bins=25,
                  colors=[plt.cm.tab20b(17), plt.cm.tab20b(13)]):
    exp_scores = get_least_squares_scores(tmp_X)

    kwargs = {'N': len(exp_scores), 'scale': 0.03, 'dropout': 0.85}
    x_data, coeffs = generate_linear_combos(Refs, **kwargs)

    fab_scores = get_least_squares_scores(x_data)

    labels = ['Experimental\ndata', 'True linear\ncombinations\n(3% noise)']
    ax.hist([exp_scores, fab_scores], bins=bins, density=True, edgecolor='w',
            color=colors, label=labels)

    ax.tick_params(direction='out', width=2, length=6, labelsize=14)
    ax.set_xlabel('MSE of Least Squares Solution', fontsize=16)
    ax.set_yticks([])
    ax.legend(fontsize=16)
    ax.set_xlim(.00025, 0.00190)


def plot_expected_results(expected_results, ax):
    labels = ['LFP', 'Pyr', 'SS', 'Hem']
    color_labels = [17, 13, 6, 19]

    for i, img in enumerate(expected_results):
        row = i // 2
        colm = i % 2 + row // 2

        threshold = 0.025
        mask = np.zeros((img.shape[0], img.shape[1]))
        bool_arr = img < threshold
        mask[bool_arr] = 1
        filtered_img = np.ma.array(img, mask=mask)

        filtered_img_dict = {}
        for x in range(filtered_img.shape[0]):
            for y in range(filtered_img.shape[1]):
                if mask[x, y] == False:
                    filtered_img_dict[(x, y)] = filtered_img[x, y]

        for j, key in enumerate(list(filtered_img_dict.keys())):
            x, y = key
            ax.plot(y, -x, color=plt.cm.tab20(color_labels[i]), marker='.', markersize=4.5)

        ax.set_xticks([])
        ax.set_yticks([])

    patches = [mpatches.Patch(color=plt.cm.tab20(color_labels[i]),
                              label=labels[i]) for i in range(len(labels))]
    leg = ax.legend(handles=patches, fontsize=18, ncol=2, framealpha=0, handlelength=1., loc=1,
                    handletextpad=0.25, columnspacing=0.7, bbox_to_anchor=(1.05, 1.03))


def make_scree_plot(data, n=5, threshold=0.95, show_first_PC=True, mod=0, c=17):
    fig, ax = plt.subplots(figsize=(8,6))
    pca = PCA()
    pca_components = pca.fit_transform(data)

    n_components = 0
    x = np.arange(n) + 1
    cdf = [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(n)]
    for i, val in enumerate(cdf):
        if val > threshold:
            print(f"It takes {i + 1} PCs to explain {int(threshold*100)}% variance.")
            n_components = i + 1
            break

    ax.plot(x, cdf, 's-', markersize=12, fillstyle='none',
            color=plt.cm.tab20b(c), linewidth=3)
    ax.plot(x, np.ones(len(x)) * threshold, 'k--', linewidth=3)

    if show_first_PC:
        PC1 = pca.components_[0]
        plt.plot(np.linspace(1, n, len(PC1)), -PC1*0.3 + min(cdf) + 0.05, 'k', linewidth=2)
        text = ax.text(n - 1, min(cdf) + 0.06, '$PC_1$', ha="right", va="bottom", size=20)

    if mod == 0:
        xticks = np.arange(n) + 1
    else:
        xticks = np.arange(0, n + 1, mod)
    plt.xticks(xticks, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(min(cdf) - 0.05, 1.02)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Number of Parameters', fontsize=22)
    plt.ylabel(f'Cumultative\nExplained Variance', fontsize=22)
    ax.tick_params(direction='in', width=2, length=8)
    
    return n_components


def normalize_spectrum(energy, spectrum, verbose=False):
    whiteline = np.argmax(np.gradient(spectrum))
    
    e_subset = energy[whiteline:].reshape(-1, 1)
    y_subset = spectrum[whiteline:].reshape(-1, 1)
    
    reg = LinearRegression().fit(e_subset, y_subset)
    y_fit = reg.predict(e_subset)
    
    y_norm = spectrum.copy()
    offset = y_fit.reshape(-1)
    y_norm[whiteline:] = y_norm[whiteline:] - offset + spectrum[whiteline + 1]
    y_norm = y_norm / (spectrum[whiteline + 1])
    #y_norm = y_norm - np.min(y_norm)
    
    if verbose:
        return whiteline, y_fit.reshape(-1), y_norm, reg
    else:
        return y_norm


def normalize_spectra(energy, spectra_list, spectra_dict):
    normalized_spectra = []
    for spectrum in spectra_list:
        y_norm = normalize_spectrum(energy, spectrum)
        normalized_spectra.append(y_norm)

    normalized_spectra_dict = {}
    for i, key in enumerate(list(spectra_dict.keys())):
        normalized_spectra_dict[key] = normalized_spectra[i]
        
    return normalized_spectra, normalized_spectra_dict


def show_normalization(energy, filtered_spectra, N=5, start_i=50, return_params=False,
                       plot=True):
    if plot:
        fig, axes = plt.subplots(figsize=(8, 2 * N), ncols=2, nrows=N)
        plt.subplots_adjust(wspace=0.2, hspace=0)

    coeffs = []
    intercepts = []
    whitelines = []
    for i, spectrum in enumerate(filtered_spectra[start_i:]):
        whiteline, y_fit, y_norm, regressor = normalize_spectrum(energy, spectrum, verbose=True)
        coeffs.append(regressor.coef_[0][0])
        intercepts.append(regressor.intercept_[0])
        whitelines.append(whiteline)

        if plot:
            ax = axes[i, 0]
            ax.plot(energy, spectrum)
            ax.plot(energy[whiteline], spectrum[whiteline],
                    's', c='k', markersize=10, fillstyle='none')
            ax.plot(energy[whiteline:], spectrum[whiteline:], '-', c=plt.cm.tab10(1))
            ax.plot(energy[whiteline:], y_fit, 'k-', linewidth=1)

            ax = axes[i, 1]
            ax.plot(energy, y_norm)

            for ax in axes[i]:
                ax.set_xticks([])
                ax.tick_params(direction='in', width=2, length=6, labelsize=14)
                ax.grid()

        if i == N - 1:
            break
            
    if return_params:
        return np.array(coeffs), np.array(intercepts), np.array(whitelines)


def make_PCA_traingle_plot(data, n_components):
    
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data)

    fig, axes = plt.subplots(figsize=(2 * n_components, 2 * n_components),
                             ncols=n_components, nrows=n_components)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in np.arange(n_components):
        for j in np.arange(n_components):
            ax = axes[i, j]
            if i == j:
                ax.hist(pca_components[:, i], color=plt.cm.tab20b(17), bins=23)
                ax.set_xticks([])
                ax.set_yticks([])
            elif j < i:
                #ax.scatter(pca_components[:, i], pca_components[:, j], marker='o', s=15, 
                #           color=plt.cm.tab10(1), edgecolor='w', linewidth=0.5)
                heatmap, xedges, yedges = np.histogram2d(pca_components[:, j],
                                                         pca_components[:, i],
                                                         bins=23)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto',
                          cmap=plt.cm.gnuplot)
                
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')

            if j == 0:
                ax.set_ylabel(f'$PC_{i + 1}$', fontsize=20, loc="center", rotation="horizontal")
                ax.yaxis.set_label_coords(-0.2, 0.37)
            if i == n_components - 1:
                ax.set_xlabel(f'$PC_{j + 1}$', fontsize=20)

    max_y = 0
    for i in np.arange(n_components):
        ax = axes[i, i]
        y_lim = ax.get_ylim()[1]
        if y_lim > max_y:
            max_y = y_lim
    for i in np.arange(n_components):
        ax = axes[i, i]
        ax.set_ylim(0, max_y)

    return pca, pca_components


def show_PCs(energy, pca, n=4):
    fig, axes = plt.subplots(figsize=(10, 4), ncols=2)
    plt.subplots_adjust(wspace=0)

    axes[0].plot(energy, pca.mean_, linewidth=4.5, label='mean', c=plt.cm.tab10(7))

    for i, pc in enumerate(pca.components_):
        axes[1].plot(energy, pc, linewidth=4.5, alpha=0.6, label=f"$PC_{i + 1}$")
        if i + 1 == n:
            break

    axes[0].legend(fontsize=16, loc='center right')
    axes[1].legend(fontsize=16, bbox_to_anchor=(1, 0.5), loc='center left')
    for ax in axes:
        ax.tick_params(direction='in', width=2, length=6, labelsize=14)
        ax.set_yticks([])
        ax.set_xlabel('Energy (eV)', fontsize=16)


def get_translated_colors(dbscan_clustering, filtered_spectra_dict, map_colors=True):
    points = list(filtered_spectra_dict.keys())
    point_index = {point: i for i, point in enumerate(points)}
    labels = dbscan_clustering.labels_.copy()
    
    color_codemap = {i: i for i in range(len(np.unique(labels)))}
    if map_colors:
        translation_map = {(60, 31): 13, (46, 69): 16, (54, 76): 17,
                           (90, 136): 18, (61, 124): 7, (142, 124): 19,
                           (98, 58): 6, (101, 115): 12}
    else:
        translation_map = {}
    
    for i, point in enumerate(points):
        if point in list(translation_map.keys()):
            translated_color = translation_map[point]
            original_label = labels[point_index[point]]
            print(f'{original_label} -> {translated_color}')
            color_codemap[original_label] = translated_color
            
    translated_colors = [color_codemap[label] for label in labels]
    return translated_colors, color_codemap


def make_UMAP_plot(pca_components, spectra_dict, n_neighbors=4.5, min_dist=0, dimension=4, eps=1):

    reducer = umap.UMAP(random_state=42, n_components=dimension,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    reduced_space = reducer.fit_transform(pca_components)

    dbscan_clustering = DBSCAN(eps=eps, min_samples=1).fit(reduced_space)
    cluster_dict = {loc: dbscan_clustering.labels_[i] for i, loc in enumerate(list(spectra_dict.keys()))}
    color_labels, codemap = get_translated_colors(dbscan_clustering, spectra_dict)
    colors = [plt.cm.tab20(c) for c in color_labels]

    fig, axes = plt.subplots(figsize=(2 * dimension, 2 * dimension),
                             ncols=dimension, nrows=dimension)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in np.arange(dimension):
        for j in np.arange(dimension):
            ax = axes[i, j]
            if i == j:
                ax.hist(reduced_space[:, i], color=plt.cm.tab20b(17), bins=35)
            elif j < i:
                #ax.scatter(reduced_space[:, j], reduced_space[:, i], marker='o', s=25, 
                #           color=plt.cm.tab10(1), edgecolor='w', linewidth=0.5)
                heatmap, xedges, yedges = np.histogram2d(reduced_space[:, j],
                                                         reduced_space[:, i],
                                                         bins=35)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto',
                          cmap=plt.cm.gnuplot) 
            else:
                ax.scatter(reduced_space[:, j], reduced_space[:, i], marker='o', s=15, 
                           color=colors)
            if j == 0:
                ax.set_ylabel(f'$x_{i + 1}$', fontsize=20, loc="center", rotation="horizontal")
                ax.yaxis.set_label_coords(-0.15, 0.37)
            if i == dimension - 1:
                ax.set_xlabel(f'$x_{j + 1}$', fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle("UMAP", fontsize=22, y=0.92)
    
    return color_labels, codemap, dbscan_clustering


def plot_color_code_map(plot, spectra_dict, color_labels, show_cluster='all'):
    fig, ax = plot
    for i, key in enumerate(list(spectra_dict.keys())):
        spectrum = spectra_dict[key]
        x, y = key
        if show_cluster == 'all':
            ax.plot(y, -x, color=plt.cm.tab20(color_labels[i]), marker='.', markersize=4.5)
        elif show_cluster == color_labels[i]:
            ax.plot(y, -x, color=plt.cm.tab20(color_labels[i]), marker='.', markersize=10)
        else:
            ax.plot(y, -x, color=plt.cm.tab20(15), marker='.', markersize=9, alpha=.3)
    remove_ticks(ax)


def get_cluster_avgs(spectra_dict, color_labels, dbscan_clustering):
    clusters = {i: [] for i in np.unique(dbscan_clustering.labels_)}

    for i, key in enumerate(list(spectra_dict.keys())):
        spectrum = spectra_dict[key]
        color = dbscan_clustering.labels_[i]
        clusters[color].append(spectrum)

    cluster_avgs = {key: np.average(clusters[key], axis=0) for key in list(clusters.keys())}
    return cluster_avgs


def plot_cluster_avgs(energy, cluster_avgs, codemap):
    fig, axes = plt.subplots(figsize=(20, 2), ncols=len(cluster_avgs))
    plt.subplots_adjust(wspace=0)

    for i, key in enumerate(list(cluster_avgs.keys())):
        axes[i].plot(energy, cluster_avgs[key] / np.sum(cluster_avgs[key]),
                     linewidth=4.5, alpha=0.6, c=plt.cm.tab20(codemap[i]))

    label_map = {0: 'I', 1: 'II', 2: 'III', 3: 'IV', 4: 'V', 5: 'VI', 6: 'VII', 7: 'VIII',
                 8: 'IX', 9: 'X', 10: 'XI', 11: 'XII', 12: 'XIII'}
    for i, ax in enumerate(axes):
        ax.tick_params(direction='in', width=2, length=6, labelsize=12)
        ax.set_yticks([])
        ax.set_xticks([7125, 7175])
        ax.set_xlabel('Energy (eV)', fontsize=12)
        ax.text(7175, 0.2 * ax.get_ylim()[1], label_map[i], fontsize=24, ha='center', va='center',
                c=plt.cm.tab20(codemap[i]))


def chi_square(data, fit, sigma_squared=1):
    return np.sum((data - fit)**2 / sigma_squared)


def reduced_chi_square(data, fit, m, sigma_squared=1):
    dof = len(data) - m - 1
    return chi_square(data, fit, sigma_squared=sigma_squared) / dof


def R_score(data, fit):
    return np.sum((data - fit)**2) / np.sum(data**2)


def scale_coeffs_to_add_to_one(coeff_mtx):
    sums = [np.sum(coeffs) for coeffs in coeff_mtx]
    normalized_coeffs = np.array([coeff_mtx[i] / sums[i] for i in range(len(sums))])
    return normalized_coeffs


def objective_function(x, Refs, target, lambda1, lambda2):
    scale = x[0]
    coeffs = x[1:]
    calc = Refs.T @ coeffs * scale
    calc = calc - np.min(calc)  # set min to zero
    return np.sum((calc - target)**2) \
           + lambda1 * np.sum(np.abs(coeffs)) \
           + lambda2 * (np.sum(coeffs) - 1)**2


def get_coeffs_from_spectra(spectra, Refs, lambda1=10, lambda2=1e8):
    m = Refs.shape[0]
    coeffs_0 = np.ones(m + 1) / (m + 1)
    bounds = np.zeros((m + 1, 2))
    bounds[:, 1] = 1
    bounds[0, 1] = 20
    results = [minimize(objective_function, coeffs_0,
               args=(Refs, spectrum, lambda1, lambda2),
               bounds=bounds) for spectrum in spectra]
    compiled_results = np.array([results[i].x for i in range(len(results))])
    coeffs = compiled_results[:, 1:]
    scales = compiled_results[:, 0]
    return scales, scale_coeffs_to_add_to_one(coeffs)


def get_sets_from_subset_indices(subset_indices, basis):
    subset = np.array([ele for i, ele in enumerate(basis) if i in subset_indices])
    non_subset_indices = np.array([i for i, ele in enumerate(basis) if i not in subset_indices])
    non_subset = np.array([ele for i, ele in enumerate(basis) if i not in subset_indices])
    return subset, non_subset_indices, non_subset


def get_goodness_of_fit_from_subset(subset, target, lambda1=10, lambda2=1e8):
    scales, coeffs_hat = get_coeffs_from_spectra([target], subset,
                                                 lambda1=lambda1, lambda2=lambda2)
    recon = coeffs_hat @ subset * scales
    recon = recon.reshape(-1)
    recon = recon - np.min(recon)
    score = chi_square(target, recon)
    return score


def LCF(target, basis, subset_size, eps=1e-6, lambda1=10, lambda2=1e8, verbose=True):
    
    indices = np.arange(basis.shape[0])     
    best_score = np.inf
    best_subset_indices = np.zeros(subset_size)
    
    i = 0
    for subset_indices in itertools.combinations(indices, subset_size):
        print(i + 1, end='\r')
        subset_indices = np.array(subset_indices)
        subset, non_subset_indices, non_subset = get_sets_from_subset_indices(subset_indices, basis) 
        set_tuple = (subset, subset_indices, non_subset, non_subset_indices)
        score = get_goodness_of_fit_from_subset(subset, target, lambda1=lambda1, lambda2=lambda2)
        if score < best_score:
            best_score = score.copy()
            best_subset_indices = subset_indices.copy()
            if best_score < eps:
                break
        i += 1            
        subset_indices = best_subset_indices
        subset, _, _ = get_sets_from_subset_indices(subset_indices, basis)
    
    if verbose:
        print(subset_indices, best_score)
    scales, coeffs_hat = get_coeffs_from_spectra([target], subset, lambda1=lambda1, lambda2=lambda2)
    return subset_indices, subset, scales, coeffs_hat, best_score


def label_ax_with_score(ax, target, pred, sub_idxs, conc, flag=False):
    R = R_score(target, pred)
    chi2 = chi_square(target, pred)
    xs = ax.get_xlim()
    ys = ax.get_ylim()
    label = '$C_{max} = ' + f'{sub_idxs}' + '_{(' + f'{int(conc * 100)}' + '\%)}$\n' \
            + '$\chi^2 = $' + f'{chi2:.03f}'
    if flag:
        c = 'red'
    else:
        c = 'k'
    ax.text(0.36 * (xs[1] - xs[0]) + xs[0], 0.12 * (ys[1] - ys[0]) + ys[0], label, 
            fontsize=16, color=c)

   
def plot_recon_grid(energy, targets, subset_indices, subsets, scales, coeffs, Ref_Data_dict,
                    confidence=0.8, ncols=5, flag_identity=True, verbose=False, c=3):
    m = len(targets)
    preds = []
    for i in range(m):
        pred = coeffs[i] @ subsets[i] * scales[i]
        pred = pred - np.min(pred)
        preds.append(pred.reshape(-1))
    preds = np.array(preds) 
    
    if m % ncols == 0:
        nrows = int(m // ncols)
    else:
        nrows = int(m // ncols) + 1
    fig, axes = plt.subplots(figsize=(3.3 * ncols, 2.3 * nrows), ncols=ncols, nrows=nrows)
    plt.subplots_adjust(wspace=0, hspace=0)

    if len(axes.shape) == 2:
        for i in range(axes.shape[0]):            
            for j in range(axes.shape[1]):
                ax = axes[i, j]
                if i * ncols + j < m:
                    ax.plot(energy, targets[i * ncols + j], '-', linewidth=3, label='target',
                            c=plt.cm.tab10(0))
                    ax.plot(energy, preds[i * ncols + j], '--', linewidth=3, label='fit',
                            c=plt.cm.tab10(c))
                    max_i = subset_indices[i * ncols + j][np.argmax(coeffs[i * ncols + j])]
                    max_conc = np.max(coeffs[i * ncols + j])
                    if (max_i != i * ncols + j or max_conc < confidence) and flag_identity:
                        flag = True
                        print(list(Ref_Data_dict.keys())[i * ncols + j])
                    else:
                        flag = False
                    label_ax_with_score(ax, targets[i * ncols + j], preds[i * ncols + j],
                                        max_i + 1, max_conc, flag=flag)
                    if (i * ncols + j + 1) % (ncols) == 0:
                        ax.legend(fontsize=16, bbox_to_anchor=(1, 0.5), loc='center left')
                    ax.set_yticks([])
                    ax.set_xticks([])
                else:
                    ax.axis('off')
    else:
        for i, ax in enumerate(axes):
            if i < m:
                ax.plot(energy, targets[i], '-', linewidth=3, label='target', c=plt.cm.tab10(7))
                ax.plot(energy, preds[i], '--', linewidth=3, label='fit', c=plt.cm.tab10(c))
                ax.set_yticks([])
                ax.set_xticks([])
                max_i = subset_indices[i][np.argmax(coeffs[i])]
                max_conc = np.max(coeffs[i])
                if (max_i != i or max_conc < confidence) and flag_identity:
                    flag = True
                    if verbose:
                        print(list(Ref_Data_dict.keys())[i])
                else:
                    flag = False
                label_ax_with_score(ax, targets[i], preds[i],
                                    max_i + 1, max_conc, flag=flag)
                if i == m - 1:
                    ax.legend(fontsize=16, bbox_to_anchor=(1, 0.5), loc='center left')
            else:
                ax.axis('off')


def plot_conc_from_subset(plot, coeffs, data_columns, subset_indices, color_codemap,
                          width=0.6):
    
    data_columns = [d.replace('_', ' ') for d in data_columns]
    num_refs = coeffs.shape[0]
    fig, ax = plot
    ax.grid(axis='y', alpha=0.7, linewidth=2, zorder=0)
    all_labels = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII']
    labels = [e for i, e in enumerate(all_labels) if i < len(coeffs)]

    for i in range(num_refs):
        conc_map = {subset_indices[i][num]: coeffs[i, num] for num in range(coeffs.shape[1])}
        sorted_conc_map = {idx: conc*100 for idx, conc in sorted(conc_map.items(),
                           key=lambda item: item[1], reverse=True)}
        bottoms = [np.sum(list(sorted_conc_map.values())[:tmp], axis=0)
                   for tmp in range(coeffs.shape[1])]
        keys = list(sorted_conc_map.keys())
        
        for k, conc in enumerate(list(sorted_conc_map.values())):
            key = keys[k]
            xlabel = labels[i]
            color = plt.cm.tab20(color_codemap[i])
            bottom = bottoms[k]
            rect = ax.bar(i, conc, width, zorder=2,
                          label=k, bottom=bottom,
                          fc=color, edgecolor='w', linewidth=2)
            names = [data_columns[key]]
            if k == len(keys) - 1 or conc < 60:
                rot = 25
                fontsize = 16
            else:
                rot = 90
                fontsize=18
            if conc > 15:
                ax.bar_label(rect, labels=names, label_type='center', c='k',
                             fontsize=fontsize, rotation=rot)
            
    
    ax.tick_params(direction='out', width=2, length=6, which='major', axis='both')
    ax.set_ylabel('Concentration (%)', fontsize=20)
    ax.set_xticks(np.arange(0, num_refs))
    ax.set_xticklabels(labels, fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=16)


def get_coeffs(n, dropout):
    """Randomly generate coeffs that add to one."""
    if dropout < 1:
        while True:
            coeffs = np.random.rand(n)
            proba = np.random.rand(n)
            set_to_zero = proba < dropout
            coeffs[set_to_zero] = 0
            if np.sum(coeffs) != 0:
                break
    else:
        coeffs = np.random.rand(n)
        indices = np.argsort(coeffs)[-dropout:]
        set_to_zero = [True for i in range(n)]
        for i in indices:
            set_to_zero[i] = False
        coeffs[set_to_zero] = 0
    scale = 1 / np.sum(coeffs)
    coeffs = coeffs * scale
    return coeffs


def generate_linear_combos(Refs, scale=0, N=10, dropout=0.5):
    """Create linear combo dataset from Refs."""
    n = len(Refs)
    Data = []
    Coeffs = []
    for i in range(N):
        coeffs = get_coeffs(n, dropout)
        if scale != 0:
            noise = np.random.normal(scale=scale,
                                     size=Refs.shape[1])
        else:
            noise = 0
        x = Refs.T @ coeffs
        x = x - np.min(x)
        Data.append(x + noise)
        Coeffs.append(coeffs)
    Data = np.array(Data)
    return Data, np.array(Coeffs)


def histogram_of_importance(plot, x, energy, Refs, bins=50, color=plt.cm.tab20b(17), fontsize=14,
                            label_map=None):
    
    fig, ax = plot
    n, bin_vals, patches = ax.hist(x, bins=bins, range=(0, bins),
                                   color=color, edgecolor='w')
    plt.xlim(-1, bins + 1)
    nticks = 6
    offset = 2
    xticks = np.linspace(offset, bins - offset, nticks)
    ax.set_xticks(xticks)
    dE = energy[1] - energy[0]
    energy = energy[:bins - 1]
    labels = [energy[int((i / nticks) * len(energy))] + offset * dE for i in range(nticks)]
    ax.set_xticklabels([f'{l:.0f}' for l in labels])
    
    ax.set_ylabel('Counts', fontsize=fontsize + 2)
    ax.set_xlabel('Energy(eV)', fontsize=fontsize + 2)
    ax.tick_params(direction='out', width=2, length=6, labelsize=16)

    if label_map is not None:
        for idx, label in label_map.items():
            rect = patches[idx]
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0, label,
                    ha='center', va='bottom', fontsize=fontsize + 1)
