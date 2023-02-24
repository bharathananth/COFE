"""Functions for plotting ordering results.

This module contains functions to ....
"""
import scipy
import biothings_client
import numpy as np
import matplotlib as mp
import seaborn as sns
import pandas as pd
from COFE.ellipse import *


def plot_circular_ordering(results, time = None, period = 24, filename=None, **kwargs):
    """Plot the ellipse producted by the projection, the two CPCs and comparison of the estimated and true phases

    Parameters
    ----------
    results : dict
        dictionary generated by the cross_validate function
    time : ndarray, optional
        the true/reference times of the samples, by default None
    """    
    sns.set_style("ticks")
    sns.set_context("notebook")
    fig = mp.pyplot.figure(**kwargs);
    gs = mp.gridspec.GridSpec(1, 3, width_ratios=[1.,1.,1.])
    ax = fig.add_subplot(gs[0, 0]);
    sns.scatterplot(x=results["CPCs"][:, 0], y=results["CPCs"][:, 1], ax=ax, palette='Set2', c=['0.3'] * results["CPCs"].shape[0])
    Y = results['CPCs']
    (axisB, axisA, x_center, y_center, tau) = direct_ellipse_est(Y[:, 0], Y[:, 1])
    ellipse = mp.patches.Ellipse((x_center, y_center), width=2*axisA, height=2*axisB, angle=tau*180/np.pi, alpha=0.5, ec='red', ls=(0, (5, 3)), fill=False, lw=2.0)
    ax.add_patch(ellipse)
    ax.set_aspect('equal')
    ax.set_xlabel("circularized principal component 1")
    ax.set_ylabel("circularized principal component 2")
    sns.despine()

    ax = fig.add_subplot(gs[0, 1]);
    df = pd.DataFrame({'t': results['phase'], 
                      'CPC1': results['CPCs'][:, 0],
                      'CPC2': results['CPCs'][:, 1]})
    df = df.melt(id_vars='t', var_name='var', value_name='val')
    sns.scatterplot(x='t', y='val', hue='var', ax=ax, data=df, palette="colorblind")
    ax.set_xlabel(r"predicted sample phase ($\times$ period)")
    ax.set_ylabel("circular principal components")
    ax.set_xlim([0,1])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    sns.despine()

    ax = fig.add_subplot(gs[0, 2]);
    ax.plot([0,1],[0,1], lw=2.0, ls=(0, (5, 8)), c='r')
    if time is not None:
        sns.scatterplot(x = (time % period)/period, y = results['phase'], ax=ax, edgecolor='black', c=['0.3'] * results['phase'].shape[0])
        ax.set_aspect(1)
        sns.despine()
        ax.set_xlabel(r"true sample phase ($\times$ period)")
        ax.set_ylabel(r"predicted sample phase ($\times$ period)")
    fig.tight_layout()

    if isinstance(filename, str):
        mp.pyplot.savefig(filename)


def plot_cv_run(results, cross_term_eps = 0.1, **kwargs):
    """Plot the error performance and number of features used for different choices of l1 threshold considered

    Parameters
    ----------
    results : dict
        dictionary generated by the cross_validate function
    """    
    sns.set_style("ticks");
    df1 = pd.DataFrame.from_dict({t: np.array(results['runs'][i]['test_se']).squeeze() for i, t in enumerate(results['t_choices'])})
    df1 = pd.melt(df1, var_name='t', value_name='se')
    def inner_prod(r):
        V = r['V']
        U = r['U']
        return (U.T @ U)[0,1] * (V.T @ V)[0,1] 
    
    df2 = pd.DataFrame({'t': results['t_choices'], 'cross_term': [inner_prod(r) for r in results['runs']]})
    df1 = df1.merge(df2, on="t")
    df1["large_cross_term"] = df1["cross_term"] > cross_term_eps

    fig = mp.pyplot.figure(**kwargs)
    gs = mp.gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, 0]);
    sns.pointplot(x="t", y="se", hue="large_cross_term", estimator=lambda x: np.percentile(x, 90), ax=ax, data=df1)
    ax.set_xlabel("Different choices of l1 constraint")
    ax.set_ylabel("Error of the fit")
    ax.set_xticklabels(["{:.2f}".format(float(x.get_text())) for x in ax.get_xticklabels()])
    ax.yaxis.set_major_formatter(mp.ticker.StrMethodFormatter('{x:,.2f}'))
    mp.pyplot.legend([],[], frameon=False)


    df3 = pd.DataFrame({'t': results['t_choices'], 
                        'nfeature': [np.unique(np.nonzero(r['V'])[0]).shape[0] for r in results['runs']],
                        'score': [r['score'] for r in results['runs']]})
    df3['score_per_f'] = df3['score']/df3['nfeature']
    ax = fig.add_subplot(gs[0, 1]);
    sns.barplot(x="t", y="nfeature", data=df3, ax=ax, color='0.3')
    ax.set_xlabel("Different choices of l1 constraint")
    ax.set_ylabel("No. of unique features")
    ax.annotate('Total number of features: {}'.format(len(results['features'])), (0.05, 0.95), xycoords='axes fraction')
    ax.set_xticklabels(["{:.2f}".format(float(x.get_text())) for x in ax.get_xticklabels()])
    ax.yaxis.set_major_formatter(mp.ticker.StrMethodFormatter('{x:,.0f}'))
    sns.despine()

    ax = fig.add_subplot(gs[0, 2]);
    sns.scatterplot(x='t', y='score', ax=ax, data=df3)
    ax.set_xlabel("Different choices of l1 constraint")
    ax.set_ylabel("Score")
    ax.xaxis.set_major_formatter(mp.ticker.StrMethodFormatter('{x:,.1f}'))
    ax.yaxis.set_major_formatter(mp.ticker.StrMethodFormatter('{x:,.1f}'))
    
    fig.tight_layout()

def plot_diagnostics(X, feature_dim = 'row', **kwargs):
    """Plot the comparison and distributions of the mean and standard deviation of each feature

    Parameters
    ----------
    X : ndarray
        2D array with the raw data
    feature_dim : str, optional
        specification of whether the features are along rows ('row') or columns ('col'), by default 'row'
    """    
    if feature_dim == 'row':
        axis = 1
    elif feature_dim == 'col':
        axis = 0

    sns.set_style("ticks");
    sns.set_context("notebook")
    fig = mp.pyplot.figure(**kwargs);

    gs = mp.gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, 0]);
    sns.regplot(x=X.mean(axis=axis), y=np.sqrt(X.var(axis=axis)), lowess=True, ax=ax, scatter_kws={'s':0.5})
    ax.set_xlabel("Mean value of the feature")
    ax.set_ylabel("Standard deviation of feature")

    ax = fig.add_subplot(gs[0, 1]);
    bin_edges = np.histogram_bin_edges(X.mean(axis=axis), bins=50)
    bin_edges = np.append(np.insert(bin_edges, 0, [-np.inf]), [np.inf])
    ax.hist(X.mean(axis=axis), bins=bin_edges, cumulative=-1, histtype='step', lw=2.0)
    ax.grid(True)
    ax.set_xlabel("Mean value of the feature")

    ax = fig.add_subplot(gs[0, 2]);
    bin_edges = np.histogram_bin_edges(1/X.std(axis=axis), bins=50)
    bin_edges = np.append(np.insert(bin_edges, 0, [-np.inf]), [np.inf])
    ax.hist(1/X.std(axis=axis), bins=bin_edges, cumulative=True, histtype='step', lw=2.0)
    ax.grid(True)
    ax.set_xlabel("Reciprocal standard deviation of the feature")
    fig.tight_layout()

def plot_markers(results, translate=False, **kwargs):
    """Plot the features and the weights assigned to them in each of the two sparse loading vectors

    Parameters
    ----------
    results : dict
        dictionary generated by the cross_validate function
    translate : bool, optional
        whether the feature names must be converted into gene symbols using standard annotation databases, by default False
    """    
    sns.set_style("ticks");
    sns.set_context("notebook")

    df = pd.DataFrame(results["SLs"], index=results["features"], columns=["X","Y"])
    df = df.loc[(df!=0).any(axis=1)].sort_index()

    if translate:
        mg = biothings_client.get_client('gene')
        symbols = mg.getgenes(df.index.values, as_dataframe=True, fields='symbol')
        df.set_index(symbols["symbol"].values, inplace=True)

    df.reset_index(level=0, inplace=True)
    df = pd.melt(df, id_vars="index", value_vars=['X','Y'], var_name="CPC")
    ax = sns.catplot(x="value", y="index", data=df, col="CPC", kind="bar", aspect=0.5, **kwargs)
    ax.set_xlabels("Loadings")
    ax.set_ylabels("Features")

def print_markers(results, translate=False):
    """Print the features in each of the two sparse loading vectors

    Parameters
    ----------
    results : dict
        dictionary generated by the cross_validate function
    translate : bool, optional
        whether the feature names must be converted into gene symbols using standard annotation databases, by default False
    """   
    sns.set_style("ticks");
    sns.set_context("notebook")

    df = pd.DataFrame(results["SLs"], index=results["features"], columns=["X","Y"])
    df = df.loc[(df!=0).any(axis=1)].sort_index()

    if translate:
        mg = biothings_client.get_client('gene')
        symbols = mg.getgenes(df.index.values, as_dataframe=True, fields='symbol')
        df.set_index(symbols["symbol"].values, inplace=True)

    df.reset_index(level=0, inplace=True)
    df = pd.melt(df, id_vars="index", value_vars=['X','Y'], var_name="CPC")

    return(df["index"].unique())