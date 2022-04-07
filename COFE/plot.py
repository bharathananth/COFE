"""Functions for plotting ordering results.

This module contains functions to ....
"""

import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
from biothings_client import get_client


def plot_circular_ordering(results, time = None, **kwargs):
    sns.set_style("ticks")
    fig = mp.figure(**kwargs);
    gs = gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, 2]);
    if time is not None:
        ax.scatter((time % 24)/24, results['phase'], s=1)
        ax.set_aspect(1)
        sns.despine()
        ax.set_xlabel("true sample phase (in fractions of the period)")
        ax.set_ylabel("predicted sample phase (in fractions of the period)")

    ax = fig.add_subplot(gs[0, 1]);
    df = pd.DataFrame({'t': results['phase'], 
                      'CPC1': results['CPCs'][:, 0],
                      'CPC2': results['CPCs'][:, 1]})
    df = df.melt(id_vars='t', var_name='var', value_name='val')
    sns.scatterplot(x='t', y='val', hue='var', ax=ax, data=df)
    ax.set_xlabel("predicted sample phase (in fractions of the period)")
    ax.set_ylabel("circular principal components")
    ax.set_xlim([0,1])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    sns.despine()

    # ax = fig.add_subplot(gs[1, 0]);
    # ax.scatter(results["transformed"]['x'], results["transformed"]['y'], s=1)
    # circ = mp.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
    # ax.add_patch(circ)
    # ax.set_aspect(1)
    # ax.set_xlabel("circularized principal component 1")
    # ax.set_ylabel("circularized principal component 2")
    # sns.despine()

    ax = fig.add_subplot(gs[0, 0]);
    ax.scatter(results["CPCs"][:, 0], results["CPCs"][:, 1], s=1)
    ax.set_aspect(1)
    ax.set_xlabel("circularized principal component 1")
    ax.set_ylabel("circularized principal component 2")
    sns.despine()
    fig.tight_layout()


def plot_cv_run(results, **kwargs):
    sns.set_style("ticks");
    df = pd.DataFrame.from_dict({t: np.array(results['runs'][i]['test_se']).squeeze() for i, t in enumerate(results['t_choices'])})
    df = pd.melt(df, var_name='t', value_name='se')

    fig = mp.figure(**kwargs)
    ax = sns.boxplot(x="t", y="se", data=df)
    ax.set_yscale("log")

    # Calculate the type of error to plot as the error bars
    # Make sure the order is the same as the points were looped over
    ax.set_xlabel("Different choices of l1 constraint")
    ax.set_ylabel("Cross validation error")
    sns.despine()

def plot_diagnostics(X, feature_dim = 'row', **kwargs):
    if feature_dim == 'row':
        axis = 1
    elif feature_dim == 'col':
        axis = 0

    sns.set_style("ticks");
    fig = mp.figure(**kwargs);

    gs = gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, 0]);
    ax.scatter(X.mean(axis=axis), X.std(axis=axis), s=0.5)
    ax.set_xlabel("Mean value of the feature")
    ax.set_ylabel("Standard deviation of feature")

    ax = fig.add_subplot(gs[0, 1]);
    ax.hist(X.mean(axis=axis), bins=50)
    ax.set_xlabel("Mean value of the feature")

    ax = fig.add_subplot(gs[0, 2]);
    ax.hist(1/X.std(axis=axis), bins=50)
    ax.set_xlabel("Reciprocal standard deviation of the feature")
    fig.tight_layout()

def plot_markers(results, translate=False, **kwargs):
    sns.set_style("ticks");
    fig = mp.figure(**kwargs);

    df = pd.DataFrame(results["SLs"], index=results["features"], columns=["X","Y"])
    df = df.loc[(df!=0).any(axis=1)].sort_index()

    if translate:
        mg = get_client('gene')
        symbols = mg.getgenes(df.index.values, as_dataframe=True, fields='symbol')
        df.set_index(symbols["symbol"].values, inplace=True)

    df.reset_index(level=0, inplace=True)
    df = pd.melt(df, id_vars="index", value_vars=['X','Y'], var_name="CPC")
    ax = sns.catplot(x="value", y="index", data=df, col="CPC", kind="bar", aspect=0.5)
    ax.set_xlabels("Loadings")
    ax.set_ylabels("Features")
    fig.tight_layout()

def print_markers(results, translate=False):
    sns.set_style("ticks");

    df = pd.DataFrame(results["SLs"], index=results["features"], columns=["X","Y"])
    df = df.loc[(df!=0).any(axis=1)].sort_index()

    if translate:
        mg = get_client('gene')
        symbols = mg.getgenes(df.index.values, as_dataframe=True, fields='symbol')
        df.set_index(symbols["symbol"].values, inplace=True)

    df.reset_index(level=0, inplace=True)
    df = pd.melt(df, id_vars="index", value_vars=['X','Y'], var_name="CPC")

    return(df["index"].unique())