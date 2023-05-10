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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import gridspec

from PIL import Image, ImageSequence

from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.stats import norm

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance, pearsonr

import sympy

from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import MultiTaskElasticNetCV
from scipy.optimize import nnls
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV

from joblib import dump, load

import umap

# from https://en.wikipedia.org/wiki/Help:Distinguishable_colors
discrete_cmap = [
    (240,163,255), 
    (0,117,220),
    (153,63,0),
    (76,0,92),
    (25,25,25),
    (0,92,49),
    (43,206,72),
    (255,204,153),
    (128,128,128),
    (148,255,181),
    (143,124,0),
    (157,204,0),
    (194,0,136),
    (0,51,128),
    (255,164,5),
    (255,168,187),
    (66,102,0),
    (255,0,16),
    (94,241,242),
    (0,153,143),
    (224,255,102),
    (116,10,255),
    (153,0,0),
    (255,255,128),
    (255,225,0),
    (255,80,5)
]
discrete_cmap = [tuple([c / 255. for c in color]) for color in discrete_cmap]

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
    d = len(data.shape)
    mask = np.zeros([data.shape[i] for i in range(d)])
    if d == 3:
        bool_arr = np.max(data, axis=0) < threshold
        mask[:, bool_arr] = 1
    else:
        bool_arr = data < threshold
        mask[bool_arr] = 1
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
                   threshold=None, std=None, vmin=None, vmax=None,
                   cmap=plt.cm.RdPu):
    if vmin is not None and vmax is not None:
        img = ax.imshow(Similarity_matrix, cmap=plt.cm.RdPu,
                        interpolation='nearest', origin='lower',
                        vmin=vmin, vmax=vmax)
    else:
        img = ax.imshow(Similarity_matrix, cmap=cmap,
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

    
def plot_MSE_hist(ax, tmp_X, Refs, bins=25, scale=0.06,
                  colors=[plt.cm.tab20b(17), plt.cm.tab20b(13)]):
    exp_scores = get_least_squares_scores(tmp_X, Refs)

    kwargs = {'N': len(exp_scores), 'scale': scale, 'dropout': 0.85}
    x_data, coeffs = generate_linear_combos(Refs, **kwargs)

    fab_scores = get_least_squares_scores(x_data, Refs)

    labels = ['Exp. data', f'Ref. lin. comb.\n$\sigma^2={scale}*I(E)$']
    ax.hist([exp_scores, fab_scores], bins=bins, density=True,
            color=colors, label=labels, rwidth=1)
    ax.tick_params(direction='out', width=2, length=6, labelsize=13)
    ax.set_xlabel('MSE of Least Squares Solution', fontsize=16)
    ax.set_yticks([])
    ax.legend(fontsize=16)


def plot_expected_results(expected_results, ax):
    labels = ['LFP', 'Pyr', 'SS', 'Hem']
    color_labels = [12, 13, 6, 19]
    colors = [plt.cm.tab20(c) for c in color_labels]

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
        
        keys = np.array(list(filtered_img_dict.keys()))
        xs, ys = keys.T
        ax.scatter(ys, -xs, color=colors[i], s=10, alpha=alphas)
        ax.set_xticks([])
        ax.set_yticks([])

    patches = [mpatches.Patch(color=plt.cm.tab20(color_labels[i]),
                              label=labels[i]) for i in range(len(labels))]
    leg = ax.legend(handles=patches, fontsize=18, ncol=2, framealpha=0, handlelength=1., loc=1,
                    handletextpad=0.25, columnspacing=0.7, bbox_to_anchor=(1.05, 1.03))


def make_scree_plot(data, n=5, threshold=0.95, show_first_PC=False, mod=0, c=17,
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


def normalize_spectrum(energy, spectrum, verbose=False, pre_edge_offset=20,
                       post_edge_offset=10, whiteline=None, y_fit_pre=None,
                       y_fit_post=None, whiteline_range=10):
    if whiteline is None:
        whiteline = np.argmax(np.gradient(spectrum[:whiteline_range]))
        #whiteline = np.argmax(spectrum[:whiteline_range])

    if y_fit_post is None:
        if post_edge_offset < 0:
            e_post = energy[post_edge_offset:].reshape(-1, 1)
            y_post = spectrum[post_edge_offset:].reshape(-1, 1)
        else:
            e_post = energy[whiteline + post_edge_offset:].reshape(-1, 1)
            y_post = spectrum[whiteline + post_edge_offset:].reshape(-1, 1)
        
        reg_post = LinearRegression().fit(e_post, y_post)

        post_edge = energy[whiteline:].reshape(-1, 1)
        y_fit_post = reg_post.predict(post_edge)

    y_norm = spectrum.copy()

    if y_fit_pre is None:
        if pre_edge_offset == 'none':
            y_fit_pre = y_norm[0]
        else:
            if pre_edge_offset > 0:
                e_pre = energy[:pre_edge_offset].reshape(-1, 1)
                y_pre = y_norm[:pre_edge_offset].reshape(-1, 1)
            else:
                e_pre = energy[:whiteline + pre_edge_offset].reshape(-1, 1)
                y_pre = y_norm[:whiteline + pre_edge_offset].reshape(-1, 1)
            
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


def normalize(energy, spectrum, pre_edge_offset=20, whiteline_mode='gradient',
              post_edge_offset=10, whiteline_range=10, verbose=False):
    
    if whiteline_mode == 'gradient':
        whiteline = np.argmax(np.gradient(spectrum[:whiteline_range]))
    else:
        whiteline = np.argmax(spectrum[:whiteline_range])

    y_norm = spectrum.copy()

    if post_edge_offset == 'none':
        y_fit_post = spectrum[-1]
    else:
        if post_edge_offset < 0:
            e_post = energy[post_edge_offset:].reshape(-1, 1)
            y_post = spectrum[post_edge_offset:].reshape(-1, 1)
        else:
            e_post = energy[whiteline + post_edge_offset:].reshape(-1, 1)
            y_post = spectrum[whiteline + post_edge_offset:].reshape(-1, 1)

        reg_post = LinearRegression().fit(e_post, y_post)

        post_edge = energy[whiteline:].reshape(-1, 1)
        y_fit_post = reg_post.predict(post_edge)

    if pre_edge_offset == 'none':
        y_fit_pre = y_norm[0]
    else:
        if pre_edge_offset > 0:
            e_pre = energy[:pre_edge_offset].reshape(-1, 1)
            y_pre = y_norm[:pre_edge_offset].reshape(-1, 1)
        else:
            e_pre = energy[:whiteline + pre_edge_offset].reshape(-1, 1)
            y_pre = y_norm[:whiteline + pre_edge_offset].reshape(-1, 1)
            
        reg_pre = LinearRegression().fit(e_pre, y_pre)
        y_fit_pre = reg_pre.predict(energy.reshape(-1, 1)).reshape(-1)
    
    y_norm = y_norm - y_fit_pre

    # post_edge_fit = reg_post.predict(energy.reshape(-1, 1)).reshape(-1)
    # y_norm = y_norm / post_edge_fit

    # if verbose:
    #     return whiteline, y_fit_pre.reshape(-1), post_edge_fit.reshape(-1), y_norm
    # else:
    #     return y_norm

    line = y_fit_post.reshape(-1) 
    y_norm[whiteline:] = y_norm[whiteline:] - line + line[0]
    
    if y_fit_pre.shape == ():
        y_norm = y_norm / (line[0] - y_fit_pre)
    else:
        y_norm = y_norm / (line[0] - y_fit_pre[whiteline])

    return y_norm


def normalize_spectra(energy, spectra_list, spectra_dict, whiteline_range=10,
                      pre_edge_offset=20, post_edge_offset=10, whiteline_mode='gradient'):
    normalized_spectra = []
    for i, spectrum in enumerate(spectra_list):
        y_norm = normalize(energy, spectrum, pre_edge_offset=pre_edge_offset,
                           post_edge_offset=pre_edge_offset,
                           whiteline_range=whiteline_range,
                           whiteline_mode=whiteline_mode)
        normalized_spectra.append(y_norm)
    normalized_spectra = np.array(normalized_spectra)

    normalized_spectra_dict = {}
    for i, key in enumerate(list(spectra_dict.keys())):
        normalized_spectra_dict[key] = normalized_spectra[i]
        
    return normalized_spectra, normalized_spectra_dict


def show_normalization(energy, filtered_spectra, N=5, start_i=0, return_params=False,
                       plot=True, pre_edge_offset=20, post_edge_offset=10,
                       whiteline_range=10, colors=[plt.cm.tab10(0), plt.cm.tab10(1)]):
    if plot:
        fig, axes = plt.subplots(figsize=(8, 2 * N), ncols=2, nrows=N)
        plt.subplots_adjust(wspace=0.2, hspace=0)

    pre_edge_fits = []
    post_edge_fits = []
    whitelines = []
    y_norms = []
    for i, spectrum in enumerate(filtered_spectra[start_i:]):
        if type(pre_edge_offset) is list:
            pre = pre_edge_offset[i]
        else:
            pre = pre_edge_offset
        if type(post_edge_offset) is list:
            post = int(post_edge_offset[i])
        else:
            post = post_edge_offset
        whiteline, y_fit_pre, y_fit_post, y_norm = normalize(energy, spectrum, verbose=True,
                                                             pre_edge_offset=pre,
                                                             post_edge_offset=post,
                                                             whiteline_range=whiteline_range)
        pre_edge_fits.append(y_fit_pre)
        post_edge_fits.append(y_fit_post)
        whitelines.append(whiteline)
        y_norms.append(y_norm)

        if plot:
            ax = axes[i, 0]
            ax.plot(energy, spectrum, color=colors[0])  # raw spectrum
            ax.plot(energy[whiteline], spectrum[whiteline],
                    's', c='k', markersize=10, fillstyle='none')  # whiteline
            if post_edge_offset < 0:
                ax.plot(energy[post_edge_offset:],  # post highlight
                        spectrum[post_edge_offset:], '-', c=colors[1])
            else:
                ax.plot(energy[whiteline + post_edge_offset:],  # post highlight
                        spectrum[whiteline + post_edge_offset:], '-', c=colors[1])
            #ax.plot(energy[whiteline:], y_fit_post, 'k-', linewidth=1)  # post edge fit line
            ax.plot(energy, y_fit_post, 'k-', linewidth=1)  # post edge fit line
            if pre_edge_offset == 'none':
                ax.plot(energy, np.ones(len(energy)) * y_fit_pre, 'k--', linewidth=1)
            else:
                ax.plot(energy, y_fit_pre, 'k--', linewidth=1)

            ax = axes[i, 1]
            ax.plot(energy, y_norm, color=colors[0])

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


def show_PCs(energy, pca, n=4, colors=None, alpha=0.6):
    fig, axes = plt.subplots(figsize=(10, 4), ncols=2)
    plt.subplots_adjust(wspace=0)
    if colors is None:
        colors = [plt.cm.Dark2(i + 1) for i in range(n)]

    axes[0].plot(energy, pca.mean_, linewidth=4.5, label='mean', c=plt.cm.tab10(7))

    for i, pc in enumerate(pca.components_):
        axes[1].plot(energy, pc, linewidth=3.5, alpha=alpha, label=f"$PC_{i + 1}$",
                     c=colors[i])
        if i + 1 == n:
            break

    axes[0].legend(fontsize=16, loc=1, frameon=False)
    axes[1].legend(fontsize=16, bbox_to_anchor=(1, 0.5), loc='center left')
    for ax in axes:
        ax.tick_params(direction='in', width=2, length=6, labelsize=14)
        ax.set_yticks([])
        ax.set_xlabel('Energy (eV)', fontsize=16)


def get_translated_colors(clustering, spectra_dict, map_colors=True,
                          translation=1):
    points = list(spectra_dict.keys())
    point_index = {point: i for i, point in enumerate(points)}
    labels = clustering.labels_.copy()
    
    color_codemap = {i: i for i in range(len(np.unique(labels)))}
    if map_colors:
        if translation == 1:
            translation_map = {(59, 49): 13, (64, 128): 6, (126, 114): 19,
                               (47, 69): 12, (16, 39): 7}
        elif translation == 2:
            translation_map = {(59, 49): 13, (64, 128): 6, (126, 114): 19,
                               (47, 69): 12}
        elif translation == 3:
            translation_map = {(59, 49): 13, (126, 114): 19,
                               (47, 69): 7, (74, 46): 12}
        elif translation == 4:
            translation_map = {(59, 49): 13, (64, 128): 6, (126, 114): 19,
                               (74, 46): 12, (73, 65): 7, (60, 124): 0}
        elif translation == 5:
            translation_map = {(59, 49): 13, (126, 114): 19, (18, 36): 6,
                               (47, 69): 7, (74, 46): 12, (26, 26): 3}                   
        else:
            translation_map = None
    else:
        translation_map = None
    
    if translation_map != None:
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
                ax.scatter(reduced_space[:, i], reduced_space[:, j], marker='o', s=15, 
                           color=colors)
                ax.text(0.05, 0.8, 'T', fontsize=18, transform=ax.transAxes)
            if j == 0:
                ax.set_ylabel(f'$x_{i + 1}$', fontsize=20, loc="center", rotation="horizontal")
                ax.yaxis.set_label_coords(-0.15, 0.37)
            if i == dimension - 1:
                ax.set_xlabel(f'$x_{j + 1}$', fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle("UMAP", fontsize=22, y=0.92)
    
    return color_labels, codemap, dbscan_clustering


def plot_color_code_map(plot, spectra_dict, colors):
    fig, ax = plot
    keys = np.array(list(spectra_dict.keys()))
    xs, ys = keys.T
    ax.scatter(ys, -xs, c=colors, s=4.5)
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
    coeff_mtx = np.array([[c if c > 0 else 0 for c in coeff] for coeff in coeff_mtx])
    sums = [np.sum(coeffs) for coeffs in coeff_mtx]
    normalized_coeffs = np.array([coeff_mtx[i] / sums[i] for i in range(len(sums))])
    return normalized_coeffs


def objective_function(x, basis, target, lambda1, lambda2):
    coeffs = x
    calc = basis.T @ coeffs
    calc = calc.reshape(-1)
    return 0.5 * np.sum((calc - target)**2) \
           + lambda1 * np.sum(np.abs(coeffs)) \
           + lambda2 * (np.sum(coeffs) - 1)**2


def get_coeffs_from_spectra(spectra, basis, lambda1=0.0006, lambda2=10, alpha=0.0006,
                            method='custom'):
    if method != 'lasso':
        m = basis.shape[0]
        coeffs_0 = np.ones((m)) / m  # uniform prior
        bounds = np.zeros((m, 2))
        bounds[:, 1] = 1
        results = [minimize(objective_function, coeffs_0,
                   args=(basis, target, lambda1, lambda2),
                   bounds=bounds, method='SLSQP') for target in spectra]
        coeffs = np.array([results[i]['x'] for i in range(len(results))])
        for r in results:
            if r['success'] == False:
                print('Did not converge.')
        coeffs = coeffs #scale_coeffs_to_add_to_one(coeffs)
    else:
        coeffs = []
        for target in spectra:
            lasso = Lasso(alpha=0.0005, max_iter=5000, positive=True)
            lasso.fit(basis.T, target)
            coeffs.append(lasso.coef_)
    return coeffs


def get_sets_from_subset_indices(subset_indices, basis):
    subset = np.array([ele for i, ele in enumerate(basis) if i in subset_indices])
    non_subset_indices = np.array([i for i, ele in enumerate(basis) if i not in subset_indices])
    non_subset = np.array([ele for i, ele in enumerate(basis) if i not in subset_indices])
    return subset, non_subset_indices, non_subset


def get_goodness_of_fit_from_subset(subset, target, lambda1=10, lambda2=1e8):
    coeffs_hat = get_coeffs_from_spectra([target], subset,
                                         lambda1=lambda1, lambda2=lambda2)
    recon = coeffs_hat @ subset
    recon = recon.reshape(-1)
    score = 1 - r2_score(target, recon, multioutput='variance_weighted')
    return score


def sort_by_x(x, y):
    sorted_indices = np.argsort(x)
    x = [x[i] for i in sorted_indices]
    y = [y[i] for i in sorted_indices]
    return x, y


def get_fit_params_from_indices(indices, basis, target, lambda1=0.0006, lambda2=10):
    subset, _, _ = get_sets_from_subset_indices(indices, basis)
    coeffs_hat = get_coeffs_from_spectra([target], subset,
                                         lambda1=lambda1, lambda2=lambda2)
    return subset, coeffs_hat


def LCF(target, basis, subset_size, eps=1e-16, lambda1=0.0006, lambda2=10, verbose=False,
        reps=5, print_best=True):
    
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
    
    if print_best:
        print(best_subset_indices[0], best_scores[0])
    if verbose:
        return best_subset_indices, best_scores
    else:
        subset, coeffs_hat = get_fit_params_from_indices(best_subset_indices[0], basis, target,
                                                         lambda1=lambda1, lambda2=lambda2)
        return best_subset_indices[0], subset, coeffs_hat, best_scores[0]


def make_LCF_bar_plot(basis, targets, colors, top_n, Results, keys, subset_size, figsize=(3.5, 0.7),
                      real_indices=None, real_coeffs=None, flag=True, show_avg=True, height=0.5,
                      labels=None, lind_thresh=0.03, unq_thresh=0.96, wspace=0.2, hspace=0.7):
    N = len(basis)
    if labels is None:
        labels = ['$R_{' + f'{N - c}' + '}$' for c in range(N)]
    n_targets = len(targets)

    if show_avg:
        ncols = top_n + 1
    else:
        ncols = top_n
    
    fig, axes_list = plt.subplots(figsize=(figsize[0] * (top_n + 1),
                                           figsize[1] * N * n_targets),
                                  ncols=ncols, nrows=n_targets)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    if n_targets == 1:
        axes_list = [axes_list]

    for i in range(n_targets):    
        axes = axes_list[i]
        top_picks = Results[i]

        best_coeffs = top_picks[1][keys[2]][0]
        best_subset = top_picks[1][keys[1]]
        best_sorted_coeffs, best_sorted_subset = sort_by_x(best_coeffs, best_subset)

        best_contribs = np.array([best_sorted_coeffs[j] * best_sorted_subset[j]
                                  for j in range(subset_size)])

        for j, ax in enumerate(axes):
            format_bar_axes(ax, N, labels)

        for j in range(top_n):
            indices = top_picks[j + 1][keys[0]]
            subset = top_picks[j + 1][keys[1]]
            coeffs = top_picks[j + 1][keys[2]][0]
            score = top_picks[j + 1][keys[3]]

            if flag:
                xlabel = '$R^2 = ' + f'{1 - score:.4f}$'
                text_colors = ['k' for k in range(3)]       
                unq = get_uniqueness_cost(coeffs, subset, best_contribs, subset_size)
                unq_label = f'\nUQS = {unq:.4f}'
                if unq < unq_thresh:
                    text_colors[1] = plt.cm.tab10(3)
                elif unq > .9999:
                    text_colors[1] = plt.cm.tab10(0)
                lind = 1 - np.max([r2_score(ri, rj) for ri, rj in itertools.combinations(subset, 2)])
                lind_label = f'\nLIS = {lind:.4f}'
                if lind < lind_thresh:
                    text_colors[2] = plt.cm.tab10(3)
                add_multicolored_xlabel(axes[j], [xlabel, unq_label, lind_label], text_colors)
            
            for k, tick in enumerate(axes[j].yaxis.get_ticklabels()):
                if k in N - 1 - indices:
                    tick.set_color(colors[i]) 

            if show_avg:
                idx_list = [j, top_n]
            else:
                idx_list = [j]
            for idx in idx_list:
                if idx == j:
                    alpha = 1
                else:
                    alpha = 0.15
                axes[idx].barh(N - 1 - indices, coeffs, height=height,
                               color=colors[i], alpha=alpha, zorder=2)
                if real_indices != None and real_coeffs != None:
                    axes[idx].barh(N - 1 - np.array(real_indices[i]), real_coeffs[i], height=height,
                                   edgecolor='k', fill=False, linewidth=1.5, linestyle='--', zorder=2)
        if show_avg:
            axes[top_n].set_xlabel('Avg. Contribs.', fontsize=16)
    return fig, axes_list


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
    label = '$C_{max} = ' + f'{sub_idxs}$' + f'\n({int(conc * 100)}%)'
    if flag:
        c = 'red'
    else:
        c = 'k'
    ax.text(0.5, 0.15, label, transform=ax.transAxes, fontsize=16, color=c)

   
def plot_recon_grid(energy, targets, subset_indices, subsets, coeffs, Ref_Data_dict,
                    confidence=0.8, ncols=5, flag_identity=True, verbose=False, c=3):
    m = len(targets)
    preds = []
    for i in range(m):
        pred = coeffs[i] @ subsets[i]
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
                    ax.plot(energy, targets[i * ncols + j] - preds[i * ncols + j], '-',
                            linewidth=2, label='difference', c='k')
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
            
    ax.set_yticks(np.arange(10, 100, 10))
    ax.tick_params(direction='out', width=2, length=6, which='major', axis='both',
                   labelsize=16)
    ax.set_ylabel('Concentration (%)', fontsize=20)
    ax.set_xticks(np.arange(0, num_refs))
    ax.set_xticklabels(labels, fontsize=18)


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


def generate_linear_combos(Refs, scale=0, N=10, dropout=0.5, percent=True):
    """Create linear combo dataset from Refs."""
    n = len(Refs)
    Data = []
    Coeffs = []
    for i in range(N):
        coeffs = get_coeffs(n, dropout)
        x = Refs.T @ coeffs
        x = x - min(x)
        if scale != 0:
            if percent:
                noise = np.random.normal(scale=scale * x,
                                         size=Refs.shape[1])
            else:
                noise = np.random.normal(scale=scale,
                                         size=Refs.shape[1])
        else:
            noise = 0
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
                     leg=False, loc=1, alpha=0.05, **kwargs):
    for i, b in enumerate(basis):
        axes[0].plot(x, b, linewidth=3, color=colors[i], label=f'$x_{i + 1}$')

    var = np.std(basis, axis=0)
    axes[0].plot(x, var, color='k', label='var.', linewidth=3)
    if leg:
        axes[2].text(0.97, 0.97, f'n = {best_n}', transform=axes[2].transAxes, fontsize=20,
                     va='top', ha='right')

    data, coeffs = generate_linear_combos(basis, **kwargs)

    for d in data:
        axes[1].plot(x, d, color='gray', alpha=alpha, linewidth=3)

    var = np.std(data, axis=0)
    axes[1].plot(x, var, color='k', linewidth=3, alpha=1)

    for ax in axes:
        ax.tick_params(width=2, length=6, labelsize=15, direction='out')
    for ax in [axes[0], axes[1]]:
        ax.text(0.5, 0.2, 'Variance', fontsize=20, ha='left', va='top',
                transform=ax.transAxes)
    
    labels = np.arange(1, len(indices) + 1)
    label_map = {idx: label for idx, label in zip(indices, labels)}
    
    bins = len(x)
    hist, bvals = np.histogram(Is.reshape(-1), bins=bins, range=(0, len(x)))
    res = x[1] - x[0]
    axes[2].bar(x, hist, width=res, edgecolor='w', linewidth=0.3, color=plt.cm.Dark2(2))
    axes[2].plot(x, var / np.max(var) * np.max(hist), color='k', linewidth=3, alpha=0.5)

    for idx, label in label_map.items():
        height = hist[idx]
        axes[2].text(x[idx] + res / 2, height + 0, label,
                     ha='center', va='bottom', fontsize=14)


def get_RFE_results(base_estimator, x_energy, basis, Ns, reps,
                    energy_point_selector, colors, n_estimators=1,
                    plot=False, select_n_for_me=False, verbose=True,
                    return_axes=False,
                    scoring='neg_root_mean_squared_error', **kwargs):

    if plot:
        fig = plt.figure(figsize=(16, 3 * len(Ns)))
        spec = fig.add_gridspec(ncols=3, nrows=len(Ns), width_ratios=[0.5, 0.5, 1.0])
        plt.subplots_adjust(wspace=0.15, hspace=0.2)
    N = kwargs['N']
    Scores = []
    Axes = []

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
            energy_measurements = x_energy[rfe.support_]

            indices = [i for i, e in enumerate(x_energy) if e in energy_measurements]
            Is.append(indices)

        if select_n_for_me:
            Is = np.array(Is, dtype=object)
        else:
            Is = np.array(Is)

        if plot:
            axes = [fig.add_subplot(spec[row, j]) for j in range(3)]
            if return_axes:
                Axes.append(axes)

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
            plot_RFE_results(axes, x_energy, basis, indices, Is, best_n, colors, leg=True,
                             **kwargs)
        Scores.append(scores)

    model = base_estimator.replace(" ", "_")

    if plot:
        if model == 'Linear_Regression':
            plt.savefig(f'Figures/test_RFE_results_gaussian_{model}_{reps}_reps.png',
                        dpi=600, bbox_inches='tight', transparent=False)
        else:
            plt.savefig(f'Figures/test_RFE_results_gaussian_{model}_w_{n_estimators}_dts.png',
                        dpi=600, bbox_inches='tight', transparent=False)
        if return_axes:
            return fig, Axes   
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


def find_diversity(avg_spectra, dbscan_clustering, normalized_spectra):
    total_avg = np.average(avg_spectra, axis=0)
    diversity = np.sum((avg_spectra - total_avg)**2) / np.sum(total_avg)
    print("Diversity of UMAP clusters:")
    print(diversity)

    cluster_sizes = []
    for label in np.unique(dbscan_clustering.labels_):
        count = np.sum([1 for l in dbscan_clustering.labels_ if l == label])
        cluster_sizes.append(count)
    cluster_sizes = np.array(cluster_sizes)

    random_diversities = []
    for k in range(100):
        print(k, end='\r')
        Is = []
        indices = list(np.arange(np.sum(cluster_sizes)))
        for cluster in cluster_sizes:
            selection = np.random.choice(indices, size=cluster, replace=False)
            Is.append(selection)
            indices = [i for i in indices if i not in selection]

        cluster_avgs_random = []
        for j, label in enumerate(np.unique(dbscan_clustering.labels_)):
            spectra = np.array([normalized_spectra[i] for i in Is[j]])
            cluster_avgs_random.append(np.average(spectra, axis=0))
        cluster_avgs_random = np.array(cluster_avgs_random)

        total_avg_random = np.average(cluster_avgs_random, axis=0)
        diversity_random = np.sum((cluster_avgs_random - total_avg_random)**2) / np.sum(total_avg_random)
        random_diversities.append(diversity_random)
    random_diversities = np.array(random_diversities)

    print("\nDiversity of random clusters:")
    xbar = np.average(random_diversities)
    s = np.std(random_diversities)
    print(f'{xbar} +/- {s}')  # sample mean and std

    """
    H0: Diversity from UMAP is not statistically significant compared to random groupings.

    H1: Diversity from UMAP is statistically significant compared to random groupings.

    We will be doing a z test with significance (alpha) = 0.01
    """

    z = (diversity - xbar) / s
    pval = norm.sf(z)
    print(f"p-val = {pval}")
    alpha = 0.01
    if pval < alpha:
        print("We reject the null hypothesis, so the diversity of the UMAP clusters is statisitcally significant.")
    else:
        print("We cannot reject the null hypothesis, so the diversity of the UMAP clusters " +
              "is not statisitcally significant.")
    return diversity, xbar, s

def get_reduced_space(normalized_spectra, data_dict, xrf_strength=0, xrf=None, spatial_strength=0,
                      method='PCA', perplexity=50, n_neighbors=80, early_exaggeration=12):
    
    pca = PCA(n_components=6)
    pca_components = pca.fit_transform(normalized_spectra)

    # spatial encoding
    if spatial_strength != 0:
        pts = np.array(list(data_dict.keys()))
        w, h = 155, 160
        input_space = np.zeros((pca_components.shape[0], pca_components.shape[1] + 2))
        input_space[:, :pca_components.shape[1]] = pca_components.copy()
        input_space[:, -2] = pts[:, 0] / w * spatial_strength
        input_space[:, -1] = pts[:, 1] / h * spatial_strength
    else:
        input_space = pca_components

    # xrf encoding
    if xrf_strength != 0:
        tmp = np.zeros((input_space.shape[0], input_space.shape[1] + 3))
        tmp[:, :input_space.shape[1]] = input_space.copy()
        if xrf is not None:
            tmp[:, -3:] = xrf_strength * xrf
        else:
            print("You specified an xrf stength but have not given xrf data.")
        input_space = tmp
    else:
        pass

    # dimensionality reduction
    if method == 'PCA':
        reduced_space = input_space
    elif method =='UMAP':
        reducer = umap.UMAP(random_state=42, n_components=2,
                            n_neighbors=n_neighbors, min_dist=0)
        reduced_space = reducer.fit_transform(input_space)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, perplexity=perplexity,
                       early_exaggeration=early_exaggeration, random_state=42)
        reduced_space = reducer.fit_transform(input_space)

    return reduced_space

def apply_clustering(reduced_space, clustering, data_dict, translation=-1, n_clusters=4, eps=1):
    # clustering
    if clustering == 'k-means':
        clusterizer = KMeans(n_clusters=n_clusters, random_state=42).fit(reduced_space)
    elif clustering == 'dbscan':
        clusterizer = DBSCAN(eps=eps, min_samples=1).fit(reduced_space)
        print(f"Couldn't cluster {np.sum(clusterizer.labels_ == -1)} points")

    # cluster color codes
    cluster_dict = {loc: clusterizer.labels_[i] for i, loc in enumerate(list(data_dict.keys()))}
    color_labels, codemap = get_translated_colors(clusterizer, data_dict, map_colors=True,
                                                  translation=translation)
    colors = [plt.cm.tab20(c) for c in color_labels]
    if translation not in np.arange(1, 6):
        colors = [discrete_cmap[c] for c in color_labels]

    return clusterizer, cluster_dict, color_labels, codemap, colors

def get_cluster_avgs(clusterizer, normalized_spectra):
    cluster_avgs = [[] for i in np.unique(clusterizer.labels_)]
    for i, s in enumerate(normalized_spectra):
        cluster_avgs[clusterizer.labels_[i]].append(s)
    for i in np.unique(clusterizer.labels_):
        cluster_avgs[i] = np.average(cluster_avgs[i], axis=0)
    return np.array(cluster_avgs)


def two_dimensional_clustering(plot, normalized_spectra, data_dict, Refs, xrf_strength=0, xrf=None,
                               method='PCA', clustering='k-means', spatial_strength=0,
                               translation=1, eps=1, perplexity=50, n_neighbors=80, n_clusters=4,
                               early_exaggeration=12, data_description='full_spectra', verbose=False):
    """Dimension reduction and clustering in 2D with different clustering and dim. red. methods."""
    true_contrib_indices = [0, 2, 3, 7]
    targets = normalized_spectra
    basis = Refs[true_contrib_indices]
    expected_coeffs = np.array([list(nnls(basis.T, target)[0]) for target in targets])

    reduced_space = get_reduced_space(normalized_spectra, data_dict, 
                                      xrf_strength=xrf_strength, xrf=xrf, 
                                      spatial_strength=spatial_strength,
                                      method=method, perplexity=perplexity, 
                                      n_neighbors=n_neighbors, 
                                      early_exaggeration=early_exaggeration)       
    clusterizer, cluster_dict, color_labels, codemap, colors = apply_clustering(reduced_space, 
        clustering, data_dict, translation=translation, n_clusters=n_clusters, eps=eps)

    cluster_avgs = get_cluster_avgs(clusterizer, normalized_spectra)

    fig, axes = plot

    # plot redcued space
    ax = axes[0]
    ax.scatter(reduced_space[:, 0], reduced_space[:, 1], marker='o', s=25, color=colors,
               edgecolor='w', linewidth=0.3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{clustering} on {method}', fontsize=19)

    # plot phase map
    plot_color_code_map((fig, axes[1]), data_dict, colors)
    axes[1].set_title('XANES map', fontsize=18)

    # plot LASSO resuts
    pts = np.array(list(data_dict.keys()))
    basis = Refs
    cluster_pred_coeffs = get_coeffs_from_spectra(cluster_avgs, basis)
    color_labels = [6, 3, 13, 12, 14, 14, 19, 19, 19, 19]
    
    cluster_colors = [plt.cm.tab20(color_labels[c]) for c in np.argmax(cluster_pred_coeffs, axis=1)] 
    pred_colors = [cluster_colors[clusterizer.labels_[i]] for i, s in enumerate(normalized_spectra)]
    pred_coeffs = np.array([cluster_pred_coeffs[clusterizer.labels_[i]] 
                            for i, s in enumerate(normalized_spectra)])
    alphas = np.sort(pred_coeffs, axis=1)
    alphas = alphas - np.min(alphas)
    alphas = alphas / np.max(alphas)

    if len(axes) != 3:
        axes[2].scatter(pts[:, 1], -pts[:, 0], c=pred_colors, s=2, alpha=alphas[:,  -1])
        axes[2].set_title('LASSO LCF', fontsize=19)

    #plot expected results
    color_labels = [6, 13, 12, 19]
    concentrations = np.argsort(expected_coeffs, axis=1)
    alphas = np.sort(expected_coeffs, axis=1)
    alphas = alphas - np.min(alphas)
    alphas = alphas / np.max(alphas)

    expected_colors = np.array([plt.cm.tab20(color_labels[c])
                                for c in concentrations[:, -1]])
    if len(axes) != 3:
        axes[3].scatter(pts[:, 1], -pts[:, 0], color=expected_colors, s=2, alpha=alphas[:, -1])
        axes[3].set_title('1st Expected Conc', fontsize=18)

        expected_colors = np.array([plt.cm.tab20(color_labels[c])
                                    for c in concentrations[:, -2]])
        axes[4].scatter(pts[:, 1], -pts[:, 0], color=expected_colors, s=2, alpha=alphas[:, -2])
        axes[4].set_title('2nd Expected Conc', fontsize=18)
    else:
        axes[2].scatter(pts[:, 1], -pts[:, 0], color=expected_colors, s=2, alpha=alphas[:, -1])
        axes[2].set_title('Expected Phases', fontsize=18)
    

    # add legends
    labels = ['LFP', 'Pyr', 'SS', 'Hem']
    color_labels = [12, 13, 6, 19]
    colors = [plt.cm.tab20(c) for c in color_labels]
    patches = [mpatches.Patch(color=plt.cm.tab20(color_labels[i]),
               label=labels[i]) for i in range(len(labels))]
    if len(axes) != 3:
        axlist = [axes[2], axes[3], axes[4]]
    else:
        axlist = [axes[2]]
    for ax in axlist:
        leg = ax.legend(handles=patches, fontsize=18, ncol=2, framealpha=0,
                        handlelength=.6, loc=1, bbox_to_anchor=(1.06, 1.04),
                        labelspacing=.1, handletextpad=0.12, columnspacing=0.25)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(f'Figures/{method}_{clustering}_{data_description}.png', dpi=600, bbox_inches='tight')

    if verbose:
        return cluster_dict, codemap

def plot_true_vs_pred_conc(plot, y_true, y_pred, data_columns, color, nrows=2, ncols=6):
    N = len(data_columns)
    MSEs = np.zeros(N)
    fig, axes = plot
    for i, j in itertools.product(range(nrows), range(ncols)):
        if nrows == 1:
            ax = axes[j]
        elif ncols == 1:
            ax = axes[i]
        else:
            ax = axes[i, j]
        idx = i * ncols + j
        if idx < N:
            ax.plot([0, 1], [0, 1], '--', c=plt.cm.tab10(7), alpha=0.5, linewidth=2)
            ax.plot(y_true[:, idx], y_pred[:, idx], 'o', c=color, markersize=5)
            mse = mean_squared_error(y_true[:, idx], y_pred[:, idx])
            MSEs[idx] = mse
            ax.tick_params(labelsize=20, width=2, length=6, direction='out')
            ax.tick_params(width=0, length=0, direction='out', which='minor')
            ax.set_xlim(-0.1, 1.1)
            ax.grid(axis='both', which='both')
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.set_xticks([0.1, 0.5, 0.9])
            if j != 0:
                ax.set_ylabel(None)
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticks_position('none') 
            else:
                ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
                ax.set_ylabel('Pred. Conc.', fontsize=22)

            ax.set_title(f'{data_columns[idx]}\nMSE={mse:.3f}', fontsize=21)
            ax.set_xlabel('True Conc.', fontsize=20)
        else:
            ax.axis('off')
    return MSEs


def generate_img(filtered_mask, spectra_dict, data):
    img = np.zeros(filtered_mask.shape)
    for idx, key in enumerate(list(spectra_dict.keys())):
        x, y = key
        img[x, y] = data[idx]
    return img
