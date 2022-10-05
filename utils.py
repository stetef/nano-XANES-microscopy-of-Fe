"""Useful python functions."""

import numpy as np
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
from matplotlib.offsetbox import HPacker, VPacker, AnnotationBbox
from PIL import Image, ImageSequence

from scipy.optimize import minimize
from scipy.stats import pearsonr

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
    elif metric == '$R^2$':
        score = r2_score(x, y, multioutput='variance_weighted')
    return score


def get_similarity_mtx(data, metric='cosine similarity'):
    N = len(data)
    sim_matrix = np.zeros((N, N))
    for i, j in itertools.product(range(N), range(N)):
        if i <= j:
            score = evaluate_similarity(data[i], data[j], metric)
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score
    return np.array(sim_matrix)


def plot_corr_matx(ax, Similarity_matrix, data_columns, metric, rot=90,
                   threshold=None, std=None):
    img = ax.imshow(Similarity_matrix, cmap=plt.cm.RdPu,
                    interpolation='nearest', origin='lower')
    ax.tick_params(direction='out', width=2, length=6, labelsize=14)
    if metric == 'cosine similarity':
        metric = 'cos. sim. \n$ \equiv \\frac{ \sum \; y_i \; \hat{y}_i }' +\
                 '{ \sqrt{ \sum y_i^2 \;} \sqrt{ \sum \; \hat{y}_i^2 } } $'
    elif metric == 'Pearson correlation':
        metric = 'Pearson \n$r_{y \hat{y}} \equiv \\frac{ \sum \; y_i \hat{y}_i - N \;' + \
                 '\overline{y} \; \hat{\overline{y}} }' + \
                 '{ \sqrt{ \sum \; y_i^2 - N \; \overline{y} \; ^2 } ' + \
                 '\sqrt{ \sum \; \hat{y}_i^2 - N \; \hat{\overline{y}} \; ^2 } }$'
    elif metric == '$R^2$':
        metric = metric + '\n$\equiv 1 - \\frac{ \sum (y_i - \hat{y}_i)^2}' + \
                 '{\sum (y_i - \overline{y})^2}$'
    elif metric == '1 - $\delta$':
        metric = metric + '\n$\equiv 1 - \\frac{ \sum \; |y_i - \hat{y}_i|}{N}$'
    elif metric == '1 - IADR':
        metric = '1 - I.A.D.R.\n$\equiv 1 - \\frac{ \sum \; |y_i - \hat{y}_i|}' + \
                 '{ max \\{ \sum \; y_{i}, \sum \; \hat{y}_{i} \\} }$'
    ax.set_title(f'{metric}', fontsize=20)

    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16, width=2, length=4)
    if threshold is not None:
        cbar.ax.axhline(y=threshold, xmin=0, xmax=1, linestyle='-', linewidth=4,
                        color='w')
        if std is not None:
            cbar.ax.annotate('', xy=(0.5, threshold + std),  xycoords='data',
                             xytext=(0.5, threshold), textcoords='data',
                             arrowprops=dict(facecolor='w', edgecolor='w', alpha=0.8),
                             ha='center', va='bottom')
        cbar.ax.text(1.1, threshold, ' random\nsampling', va='center', ha='left', fontsize=18)
        N = len(Similarity_matrix)
        for i, j in itertools.product(range(N), range(N)):
            if Similarity_matrix[i, j] > threshold:
                if i == j:
                    c, alpha = 'gray', 0.4
                else:
                    c, alpha = 'w', 0.7
                dot = plt.Circle((i, j), 0.1, color=c, alpha=alpha)
                ax.add_patch(dot)

    N = len(Similarity_matrix)
    ax.set_yticks(np.arange(N))
    labels = [e.replace('_', ' ').replace('NP', '') for e in data_columns]
    ax.set_yticklabels(labels, fontsize=16)

    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(labels, fontsize=16, rotation=rot)


def get_least_squares_scores(data, Refs):
    scores = []

    for i in range(len(data)):
        print(i, end='\r')

        # overdetermined solutions require least squares solutions
        # using the least squares analytical solution
        A = Refs.T
        y = data[i]
        x = np.linalg.pinv(A.T @ A) @ A.T @ y

        scores.append(mean_squared_error(A @ x, y))

    return np.array(scores)

    
def plot_MSE_hist(ax, tmp_X, Refs, bins=25,
                  colors=[plt.cm.tab20b(17), plt.cm.tab20b(13)]):
    exp_scores = get_least_squares_scores(tmp_X, Refs)

    kwargs = {'N': len(exp_scores), 'scale': 0.02, 'dropout': 0.85}
    x_data, coeffs = generate_linear_combos(Refs, **kwargs)

    fab_scores = get_least_squares_scores(x_data, Refs)

    labels = ['Experimental\ndata', 'True linear\ncombinations\n(2% noise)']
    ax.hist([exp_scores, fab_scores], bins=bins, density=True, edgecolor='w',
            color=colors, label=labels)
    ax.set_xlim(0.00005, 0.00085)
    ax.tick_params(direction='out', width=2, length=6, labelsize=13)
    ax.set_xlabel('MSE of Least Squares Solution', fontsize=16)
    ax.set_yticks([])
    ax.legend(fontsize=16)


def plot_expected_results(expected_results, ax):
    labels = ['LFP', 'Pyr', 'SS', 'Hem']
    color_labels = [12, 13, 6, 19]

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

        alphas = np.array(list(filtered_img_dict.values()))
        alphas = alphas - np.min(alphas)
        alphas = alphas / np.max(alphas)
        for j, key in enumerate(list(filtered_img_dict.keys())):
            x, y = key
            ax.plot(y, -x, color=plt.cm.tab20(color_labels[i]), marker='.', markersize=10,
                    alpha=alphas[j])

        ax.set_xticks([])
        ax.set_yticks([])

    patches = [mpatches.Patch(color=plt.cm.tab20(color_labels[i]),
                              label=labels[i]) for i in range(len(labels))]
    leg = ax.legend(handles=patches, fontsize=18, ncol=2, framealpha=0, handlelength=1., loc=1,
                    handletextpad=0.25, columnspacing=0.7, bbox_to_anchor=(1.05, 1.03))


def make_scree_plot(data, n=5, threshold=0.95, show_first_PC=True, mod=0, c=17,
                    xy=(0.7, 0.3)):
    fig, ax = plt.subplots(figsize=(8,6))
    pca = PCA()
    pca_components = pca.fit_transform(data)

    n_components = 0
    x = np.arange(n) + 1
    cdf = [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(n)]
    for i, val in enumerate(cdf):
        if val > threshold:
            text = f"It takes {i + 1} PCs to explain\n{int(threshold*100)}% variance."
            print(text)
            ax.text(xy[0], xy[1], text, va='center', ha='center', fontsize=20,
                    transform=ax.transAxes)
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


def normalize_spectrum(energy, spectrum, verbose=False, pre_edge_offset=10,
                       whiteline=None, y_fit_pre=None, y_fit_post=None):
    if whiteline is None:
        whiteline = np.argmax(np.gradient(spectrum))

    if y_fit_post is None:
        e_post = energy[whiteline:].reshape(-1, 1)
        y_post = spectrum[whiteline:].reshape(-1, 1)
        
        reg_post = LinearRegression().fit(e_post, y_post)
        post_edge = energy[whiteline:].reshape(-1, 1)
        y_fit_post = reg_post.predict(post_edge)

    y_norm = spectrum.copy()

    if y_fit_pre is None:
        if pre_edge_offset == 'none':
            y_fit_pre = y_norm[0]
        else:
            e_pre = energy[:whiteline - pre_edge_offset].reshape(-1, 1)
            y_pre = y_norm[:whiteline - pre_edge_offset].reshape(-1, 1)
            reg_pre = LinearRegression().fit(e_pre, y_pre)
            y_fit_pre = reg_pre.predict(energy.reshape(-1, 1)).reshape(-1)
    
    y_norm = y_norm - y_fit_pre

    line = y_fit_post.reshape(-1) 
    y_norm[whiteline:] = y_norm[whiteline:] - line + line[0]
    
    if y_fit_pre.shape == ():
        y_norm = y_norm / (line[0] - y_fit_pre)
    else:
        y_norm = y_norm / (line[0] - y_fit_pre[whiteline])

    if verbose:
        return whiteline, y_fit_pre.reshape(-1), y_fit_post.reshape(-1), y_norm
    else:
        return y_norm


def normalize_spectra(energy, spectra_list, spectra_dict,
                      pre_edge_offset=10):
    normalized_spectra = []
    for spectrum in spectra_list:
        y_norm = normalize_spectrum(energy, spectrum,
                                    pre_edge_offset=pre_edge_offset)
        normalized_spectra.append(y_norm)

    normalized_spectra_dict = {}
    for i, key in enumerate(list(spectra_dict.keys())):
        normalized_spectra_dict[key] = normalized_spectra[i]
        
    return normalized_spectra, normalized_spectra_dict


def show_normalization(energy, filtered_spectra, N=5, start_i=50, return_params=False,
                       plot=True, pre_edge_offset=10):
    if plot:
        fig, axes = plt.subplots(figsize=(8, 2 * N), ncols=2, nrows=N)
        plt.subplots_adjust(wspace=0.2, hspace=0)

    pre_edge_fits = []
    post_edge_fits = []
    whitelines = []
    for i, spectrum in enumerate(filtered_spectra[start_i:]):
        whiteline, y_fit_pre, y_fit_post, y_norm = normalize_spectrum(energy, spectrum,
                                                                      pre_edge_offset=pre_edge_offset,
                                                                      verbose=True)
        pre_edge_fits.append(y_fit_pre)
        post_edge_fits.append(y_fit_post)
        whitelines.append(whiteline)

        if plot:
            ax = axes[i, 0]
            ax.plot(energy, spectrum)
            ax.plot(energy[whiteline], spectrum[whiteline],
                    's', c='k', markersize=10, fillstyle='none')
            ax.plot(energy[whiteline:], spectrum[whiteline:], '-', c=plt.cm.tab10(1))
            ax.plot(energy[whiteline:], y_fit_post, 'k-', linewidth=1)
            if pre_edge_offset == 'none':
                ax.plot(energy, np.ones(len(energy)) * y_fit_pre, 'k--', linewidth=1)
            else:
                ax.plot(energy, y_fit_pre, 'k--', linewidth=1)

            ax = axes[i, 1]
            ax.plot(energy, y_norm)

            for ax in axes[i]:
                ax.set_xlabel('Energy (eV)', fontsize=16)
                ax.tick_params(direction='out', width=2, length=6, labelsize=14)
                ax.tick_params(rotation=30, axis='x')
                ax.grid(axis='x')
                ax.xaxis.set_major_locator(MultipleLocator(20))
                if i != N - 1:
                    ax.set_xlabel(None)
                    ax.xaxis.set_ticklabels([])
                    ax.xaxis.set_ticks_position('none')  

        if i == N - 1:
            break
            
    if return_params:
        return np.array(pre_edge_fits), post_edge_fits, np.array(whitelines)
    elif plot:
        return fig, axes

"""
def normalize_spectrum(energy, spectrum, verbose=False):
    whiteline = np.argmax(np.gradient(spectrum))
    
    e_subset = energy[whiteline:].reshape(-1, 1)
    y_subset = spectrum[whiteline:].reshape(-1, 1)
    
    reg = LinearRegression().fit(e_subset, y_subset)
    post_edge = energy[whiteline:].reshape(-1, 1)
    y_fit = reg.predict(post_edge)
    
    y_norm = spectrum.copy()
    line = y_fit.reshape(-1)
    y_norm[whiteline:] = y_norm[whiteline:] - line + line[0]
    y_norm = y_norm / line[0]
    
    if verbose:
        return whiteline, y_fit.reshape(-1), y_norm, reg
    else:
        return y_norm
"""

"""
def show_normalization(energy, filtered_spectra, N=5, start_i=50, return_params=False,
                       plot=True):
    if plot:
        fig, axes = plt.subplots(figsize=(8, 2 * N), ncols=2, nrows=N)
        plt.subplots_adjust(wspace=0.2, hspace=0)

    coeffs = []
    intercepts = []
    whitelines = []
    for i, spectrum in enumerate(filtered_spectra[start_i:]):
        whiteline, y_fit, y_norm, regressor = normalize_spectrum(energy, spectrum,
                                                                 verbose=True)
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
"""

def make_PCA_triangle_plot(data, n_components, cmap=plt.cm.gnuplot,
                           c=plt.cm.tab20b(17), bins=23):
    
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data)

    fig, axes = plt.subplots(figsize=(2 * n_components, 2 * n_components),
                             ncols=n_components, nrows=n_components)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in np.arange(n_components):
        for j in np.arange(n_components):
            ax = axes[i, j]
            if i == j:
                ax.hist(pca_components[:, i], color=c, bins=bins)
                ax.set_xticks([])
                ax.set_yticks([])
            elif j < i:
                heatmap, xedges, yedges = np.histogram2d(pca_components[:, j],
                                                         pca_components[:, i],
                                                         bins=bins)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(np.log(heatmap.T + 0.8), extent=extent, origin='lower', aspect='auto',
                          cmap=cmap)
                
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


def get_translated_colors(dbscan_clustering, filtered_spectra_dict, map_colors=True,
                          translation=1):
    points = list(filtered_spectra_dict.keys())
    point_index = {point: i for i, point in enumerate(points)}
    labels = dbscan_clustering.labels_.copy()
    
    color_codemap = {i: i for i in range(len(np.unique(labels)))}
    if map_colors:
        if translation == 1:
            translation_map = {(59, 49): 13, (64, 128): 6, (126, 114): 19,
                               (47, 69): 12}
        elif translation == 2:
            translation_map = {(59, 49): 13, (64, 128): 6, (126, 114): 19,
                               (47, 69): 12, (74, 46): 7}
        elif translation == 3:
            translation_map = {(59, 49): 13, (64, 128): 6, (126, 114): 19,
                               (47, 69): 12, (74, 46): 8, (73, 65): 0,
                               (18, 38): 7}
        elif translation == 4:
            translation_map = {(59, 49): 13, (64, 128): 6, (126, 114): 19,
                               (74, 46): 12, (73, 65): 0, (60, 124): 2}
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


def make_UMAP_plot(pca_components, spectra_dict, n_neighbors=4.5, min_dist=0,
                   dimension=4, eps=1, cmap=plt.cm.gnuplot, c=plt.cm.tab20(17),
                   bins=35, translation=1):

    reducer = umap.UMAP(random_state=42, n_components=dimension,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    reduced_space = reducer.fit_transform(pca_components)

    dbscan_clustering = DBSCAN(eps=eps, min_samples=1).fit(reduced_space)
    cluster_dict = {loc: dbscan_clustering.labels_[i] for i, loc in enumerate(list(spectra_dict.keys()))}
    color_labels, codemap = get_translated_colors(dbscan_clustering, spectra_dict,
                                                  translation=translation)
    colors = [plt.cm.tab20(c) for c in color_labels]

    fig, axes = plt.subplots(figsize=(2 * dimension, 2 * dimension),
                             ncols=dimension, nrows=dimension)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in np.arange(dimension):
        for j in np.arange(dimension):
            ax = axes[i, j]
            if i == j:
                ax.hist(reduced_space[:, i], color=c, bins=bins)
            elif j < i:
                heatmap, xedges, yedges = np.histogram2d(reduced_space[:, j],
                                                         reduced_space[:, i],
                                                         bins=bins)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(np.log(heatmap.T + 10), extent=extent, origin='lower', aspect='auto',
                          cmap=cmap) 
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


def plot_cluster_avgs(plot, energy, cluster_avgs, codemap, linewidth=4.5):
    fig, axes = plot
    ncols = len(cluster_avgs)

    for i, key in enumerate(list(cluster_avgs.keys())):
        axes[i].plot(energy, cluster_avgs[key] / np.sum(cluster_avgs[key]),
                     linewidth=linewidth, alpha=0.6, c=plt.cm.tab20(codemap[i]))

    label_map = {0: 'I', 1: 'II', 2: 'III', 3: 'IV', 4: 'V', 5: 'VI', 6: 'VII', 7: 'VIII',
                 8: 'IX', 9: 'X', 10: 'XI', 11: 'XII', 12: 'XIII'}
    for i, ax in enumerate(axes):
        ax.tick_params(direction='in', width=2, length=6, labelsize=12)
        ax.set_yticks([])
        ax.set_xticks([7125, 7175])
        ax.set_xlabel('Energy (eV)', fontsize=12)
        ax.text(7175, 0.2 * ax.get_ylim()[1], label_map[i], fontsize=24, ha='center', va='center',
                c=plt.cm.tab20(codemap[i]))


def plot_spaghetti_by_cluster(ax, energy, normalized_spectra, avg_spectra,
                              color_labels, labels, codemap):    
    n_clusters = len(avg_spectra)
    dys = np.arange(n_clusters)
    colors = np.unique(color_labels)
    color_to_cluster = {v: k for k, v in codemap.items()}

    for i, spectrum in enumerate(normalized_spectra):
        c = color_labels[i]
        cluster = color_to_cluster[c]
        ax.plot(energy, spectrum + cluster, linewidth=2, alpha=0.03,
                color=plt.cm.tab20(c))

    for color in colors:
        cluster = color_to_cluster[color]
        ax.text(energy[1], cluster + 0.1, labels[cluster], fontsize=24, 
                color=plt.cm.tab20(color))
        #ax.plot(energy, avg_spectra[cluster] + cluster, linewidth=3, alpha=1.,
        #        color=plt.cm.tab20(14))

    ax.set_yticks([])
    ax.tick_params(length=6, width=2, labelsize=16)
    ax.set_xlabel('Energy (eV)', fontsize=18)


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
    score = 1 - r2_score(target, recon, multioutput='variance_weighted')
    return score


def sort_by_x(x, y):
    sorted_indices = np.argsort(x)
    x = [x[i] for i in sorted_indices]
    y = [y[i] for i in sorted_indices]
    return x, y


def get_fit_params_from_indices(indices, basis, target, lambda1=10, lambda2=1e8):
    subset, _, _ = get_sets_from_subset_indices(indices, basis)
    scales, coeffs_hat = get_coeffs_from_spectra([target], subset,
                                                 lambda1=lambda1, lambda2=lambda2)
    return subset, scales, coeffs_hat


def LCF(target, basis, subset_size, eps=1e-16, lambda1=10, lambda2=1e8, verbose=False,
        reps=5):
    
    indices = np.arange(basis.shape[0])     
    best_subset_indices = list(np.tile(np.zeros(subset_size), reps).reshape(reps, subset_size))
    best_scores = list(np.zeros(reps))
    
    i = 0
    k = 0
    for subset_indices in itertools.combinations(indices, subset_size):
        print(i + 1, end='\r')
        subset_indices = np.array(subset_indices)
        subset, non_subset_indices, non_subset = get_sets_from_subset_indices(subset_indices, basis) 
        set_tuple = (subset, subset_indices, non_subset, non_subset_indices)
        score = get_goodness_of_fit_from_subset(subset, target, lambda1=lambda1, lambda2=lambda2)
        if i < reps:
            best_scores[k] = score.copy()
            best_subset_indices[k] = subset_indices.copy()
            k += 1
        if i == reps:
            best_scores, best_subset_indices = sort_by_x(best_scores, best_subset_indices)
        if i >= reps and score < best_scores[-1]:
            best_scores.append(score.copy())
            best_subset_indices.append(subset_indices.copy())
            best_scores, best_subset_indices = sort_by_x(best_scores, best_subset_indices)
            best_scores = best_scores[:-1]
            best_subset_indices = best_subset_indices[:-1]  
        
        if best_scores[0] < eps:
            print(f"Best score less than {eps}")
            break
        
        i += 1      
    
    print(best_subset_indices[0], best_scores[0])
    if verbose:
        return best_subset_indices, best_scores
    else:
        subset, scales, coeffs_hat = get_fit_params_from_indices(best_subset_indices[0], basis, target,
                                                                 lambda1=lambda1, lambda2=lambda2)
        return best_subset_indices[0], subset, scales, coeffs_hat, best_scores[0]


def format_bar_axes(ax, n, labels):
    ax.tick_params(labelsize=14, width=2, length=4)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_ylim(-0.5, n - 0.5)
    for s in ['top', 'bottom', 'right']:
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', zorder=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')


def add_multicolored_xlabel(ax, labels, colors, fontsize=16):
    boxes = []
    for i in range(len(labels)):
        box = TextArea(labels[i], textprops=dict(color=colors[i], size=fontsize,
                                                 ha='center', va='bottom'))
        boxes.append(box)
    box = VPacker(children=boxes, align="bottom", pad=0, sep=-12)
    anchored_box = AnchoredOffsetbox(loc='upper center', child=box, pad=0., frameon=False,
                                     bbox_to_anchor=(0.5, -0.05), 
                                     bbox_transform=ax.transAxes, borderpad=0)
    ax.add_artist(anchored_box)

    
def get_uniqueness_cost(coeffs, subset, best_contribs, subset_size):
    sorted_coeffs, sorted_subset = sort_by_x(coeffs, subset)
    contribs = np.array([sorted_coeffs[k] * sorted_subset[k]
                         for k in range(subset_size)])
    uniquesness_score = r2_score(best_contribs, contribs,
                                 multioutput='variance_weighted')
    return uniquesness_score


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
                            c=plt.cm.tab10(7))
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
        #x = x - np.min(x)
        Data.append(x + noise)
        Coeffs.append(coeffs)
    Data = np.array(Data)
    return Data, np.array(Coeffs)


def histogram_of_importance(plot, x, energy, Refs, bins=50, color=plt.cm.tab20b(17), fontsize=14,
                            label_map=None):
    
    fig, ax = plot
    if len(x.shape) == 1:
        n, bin_vals, patches = ax.hist(x, bins=bins, range=(0, bins),
                                       color=color, edgecolor='w')
    else:
        n, bin_vals, patches = ax.hist(x, bins=bins, range=(0, bins),
                                       edgecolor='w')
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


def plot_RFE_results(axes, x, basis, indices, Is, best_n, colors,
                     leg=True, loc=1, **kwargs):
    for i, b in enumerate(basis):
        axes[0].plot(x, b, linewidth=3, color=colors[i], label=f'$x_{i + 1}$')

    var = np.std(basis, axis=0)
    axes[0].plot(x, var, color='k', label='var.', linewidth=3)
    if leg:
        axes[0].legend(fontsize=18, handlelength=0.8, handleheight=1.3,
                       handletextpad=0.5)

    data, coeffs = generate_linear_combos(basis, **kwargs)

    for d in data:
        axes[1].plot(x, d, color='gray', alpha=0.1, linewidth=3)

    var = np.std(data, axis=0)
    axes[1].plot(x, var, color='k', linewidth=2, alpha=0.5)

    for ax in axes:
        ax.tick_params(width=2, length=6, labelsize=15, direction='in')
    
    labels = np.arange(1, len(indices) + 1)
    label_map = {idx: label for idx, label in zip(indices, labels)}
    
    bins = basis.shape[1]
    n_reps = len(Is)
    colors = [plt.cm.RdPu( (i + 1) / (n_reps + 1)) for i in range(n_reps)]
    n, bin_vals, patches = axes[2].hist(Is.T, bins=bins, range=(0, bins), edgecolor='w',
                                        linewidth=0.5, color=colors, stacked=True, alpha=1.)
    axes[2].plot(var / np.max(var) * np.max(n), color='k', linewidth=3)
    if loc == 1:
        axes[2].text(0.98, 0.98, f'n = {best_n}', fontsize=22, transform=ax.transAxes,
                     va='top', ha='right')
    elif loc == 2:
        axes[2].text(0.05, 0.98, f'n = {best_n}', fontsize=22, transform=ax.transAxes,
                     va='top', ha='left')
    else:
        axes[2].text(0.4, 0.98, f'n = {best_n}', fontsize=22, transform=ax.transAxes,
                     va='top', ha='left')

    for idx, label in label_map.items():
        height = 0
        for patch in patches:
            rect = patch[idx]
            h = rect.get_height()
            height += h
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0, label,
                ha='center', va='bottom', fontsize=14)


def get_RFE_results(base_estimator, x, basis, Ns, reps, n_estimators=1,
                    plot=False, select_n_for_me=False, verbose=True,
                    scoring='neg_root_mean_squared_error', **kwargs):

    if plot:
        fig = plt.figure(figsize=(16, 3 * len(Ns)))
        spec = fig.add_gridspec(ncols=3, nrows=len(Ns), width_ratios=[0.5, 0.5, 1.0])
        plt.subplots_adjust(wspace=0.15, hspace=0.2)

    Scores = []

    for row, best_n in enumerate(Ns):
        if select_n_for_me:
            best_n = None

        print(f'n = {best_n}')
        Is = []
        scores = []
        for rep in range(reps):
            print(rep, end='\r')
            data, coeffs = generate_linear_combos(basis, **kwargs)

            select = energy_point_selector.Selector(data, coeffs)
            if base_estimator == 'Random Forest':
                if verbose:
                    rfe, score = select.select_energy_points(estimator=base_estimator, n_points=best_n,
                                                             verbose=verbose, scoring=scoring,
                                                             n_estimators=n_estimators)
                    scores.append(score)
                else:
                    rfe = select.select_energy_points(estimator=base_estimator, n_points=best_n,
                                                      verbose=verbose, scoring=scoring,
                                                      n_estimators=n_estimators)
            else:
                if verbose:
                    rfe, score = select.select_energy_points(estimator=base_estimator, n_points=best_n,
                                                             verbose=verbose, scoring=scoring)
                    scores.append(score)
                else:
                    rfe = select.select_energy_points(estimator=base_estimator, n_points=best_n,
                                                      verbose=verbose, scoring=scoring)
            energy_measurements = x[rfe.support_]

            indices = [i for i, e in enumerate(x) if e in energy_measurements]
            Is.append(indices)

        if select_n_for_me:
            Is = np.array(Is, dtype=object)
        else:
            Is = np.array(Is)

        if plot:
            axes = [fig.add_subplot(spec[row, j]) for j in range(3)]

            if row == 0:
                axes[0].set_title('Basis set\n& variance (black)', fontsize=20)
                axes[1].set_title(f'Training dataset\nN = {N}', fontsize=20)
                title = '$N_{reps} = ' + f'{reps}$'
                if base_estimator == 'Random Forest':   
                    title = '$N_{estimators} = ' + f'{n_estimators}$\n' + title
                axes[2].set_title(title, fontsize=20)

        if select_n_for_me:
            best_n = f'{rfe.n_features_}*'

        indices = []
        if plot:
            plot_RFE_results(axes, x, basis, indices, Is, best_n, colors, leg=False, **kwargs)
        Scores.append(scores)

    model = base_estimator.replace(" ", "_")

    if plot:
        if model == 'Linear_Regression':
            plt.savefig(f'Figures/test_RFE_results_gaussian_{model}_{reps}_reps.png',
                        dpi=600, bbox_inches='tight', transparent=False)
        else:
            plt.savefig(f'Figures/test_RFE_results_gaussian_{model}_w_{n_estimators}_dts.png',
                        dpi=600, bbox_inches='tight', transparent=False)
            
    elif verbose: 
        return Scores


def plot_RFE_cv_scores(plot, Scores, base_estimator, n_estimators, N,
                       socring):
    fig, ax = plot

    title = base_estimator
    if base_estimator == 'Random Forest':
        title = title + '\n$n_{estimators} = ' + f'{n_estimators}$'
        c = 2
    else:
        c = 0
    title = title + '\nN = ' + f'{N}'

    ax.plot(Ns, np.average(Scores, axis=1), 'o-', color=plt.cm.tab20(c),
            linewidth=3, markersize=10)
    ax.fill_between(Ns, np.average(Scores, axis=1) - np.std(Scores, axis=1), 
                    np.average(Scores, axis=1) + np.std(Scores, axis=1),
                    color=plt.cm.tab20(c + 1))

    ax.set_xlabel('n', fontsize=18)
    
    if scoring == 'neg_root_mean_squared_error':
        ylabel = '-RMSE'
    elif scoring == 'r2':
        ylabel = '$R^2$'
    else:
        ylabel = scoring.replace('_', ' ')
    
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(width=1.5, length=8, direction='out', labelsize=16)
    ax.set_title(title, fontsize=18)
