"""Functions to analyse gene expression data sets using SPCA based methods.

This module contains functions to process data, analyse data using sparse 
loading vectors, and analyse data using SPCA principal components.
"""
import numpy as np
import pandas as pd
from COFE.spca import *
from COFE.ellipse import *
import concurrent.futures

def process_data(X, features, feature_dim='row', mean_threshold=None, scaling_threshold=None):
    """Function to process data prior to analysis.

    Applies centering, filtering and scaling operations to data. 

    Args:
        X (ndarray): Matrix where columns correspond to genes and rows 
            correspond to samples.
        feature_dim (string): 'row' if features along rows and 'col' if features along columns
    
        mean_threshold (float): Threshold that determines the detected features in the assay.
        scaling_threshold (float): Upper bound on the noise enhancement allowed in the standardization operation.

    Returns:

            2D ndarray containing processed data,
    """
    X_ = X.copy()
    features_ = features.copy()

    if feature_dim == 'row':
        axis = 1
    elif feature_dim == 'col':
        axis = 0
    else:
        raise ValueError("Invalid feature dimension.")

    if mean_threshold is not None:
        if axis == 1:
            keep = X_.mean(axis=1)>mean_threshold
            features_ =  features_[keep]
            X_ = X_[keep, :]
        else:
            keep = X_.mean(axis=0)>mean_threshold
            features_ =  features_[keep]
            X_ = X_[:, keep]

    if scaling_threshold is not None:
        if axis == 1:
            keep = X_.std(axis=1)>1/scaling_threshold
            features_ =  features_[keep]
            X_ = X_[keep, :]
        else:
            keep = X_.std(axis=0)>1/scaling_threshold
            features_ =  features_[keep]
            X_ = X_[:, keep]
        
    # Imputing extreme values
    upper_bound = np.percentile(X_, 97.5, axis=axis, keepdims=True)
    lower_bound = np.percentile(X_, 2.5, axis=axis, keepdims=True)
    X_ = np.maximum(np.minimum(X_, upper_bound), lower_bound)

    # Always center data to have zero mean
    X_ = X_ - np.mean(X_, axis=axis, keepdims=True)

    # Standardise every gene series to have variance of 1
    X_ = X_ / np.std(X_, axis=axis, keepdims=True)

    if axis == 1:
        X_ = X_.T
    
    return (X_, features_)

def train_test_split(num_samples, num_folds):
    """Defines how data is split into test and training sets.
    Method to create num_folds pairs of lists. Each pair has a list which 
    contains indices of samples to be included in test set, and another list
    which contains indices of samples to be included in training set.

    Parameters
    ----------
    num_samples : int
        Total number of samples.
    num_folds : int
        Number of test-training pairs to be found.

    Returns
    -------
    list of dicts
        each dict contains a pair of indices for training ('train') and testing ('test')

    Raises
    ------
    ValueError
        If num_samples is less than num_folds
    """
 
    if num_samples < num_folds:
        raise ValueError("Can't have fewer samples than folds.")

    rng = np.random.default_rng()
    fold_index = np.arange(num_samples) % num_folds
    samples = np.arange(num_samples)
    rng.shuffle(samples)
    test_train_pairs = [{'train': samples[fold_index!=i], 'test': samples[fold_index==i]} for i in np.arange(5)]
    return test_train_pairs

def multi_start(X, t=float('inf'), restarts=3, tol=1e-4, max_iter=500):
    runs = [coupled_spca(X, t=t, tol=tol, max_iter=max_iter) for i in range(restarts)]
    ind = np.argmax([r['score'] for r in runs])
    return runs[ind]

def circular_ordering(X, t, restarts=3, true_times=None):
    """Function to fit ellipse to SPCA principal components.

    Function to find the circular PCs for the chosen l1 sparsity constrain and computes MAPE for ordering if true_times is provided

    Args:
        X (ndarray): Matrix where columns correspond to genes and rows 
            correspond to samples.
        t (float): l1 norm constraint that sets SPCA sparsity level.
        true_times (array): The true time labels to compare results against. 
            Defaults to None. 

    Returns:

        Dictionary in which keys are different ellipse metrics names and values 
        are corresponding quantities.
    """

    decomp = multi_start(X, t=t, restarts=restarts)
    U = decomp['U']
    V = decomp['V']

    Y = X @ V

    try:
        theta = direct_ellipse_est(Y[:, 0], Y[:, 1])
        geo_param = algebraic_to_geometric(theta)
        transformed = transform(Y[:, 0], Y[:, 1], geo_param)
        mse = np.median(ellipse_metrics(Y[:, 0], Y[:, 1], theta, metric='geoSE'))
    except ValueError:
        return None
    
    mape_value = np.nan
    pred_phase = np.nan
    if true_times is not None:
        mape_value, pred_phase = calculate_mape(transformed['x'], transformed['y'], true_times)

    result = {"phase": pred_phase, "MAPE": mape_value, "CPCs": Y, "SLs": V, 
              "MSE": mse, "score": decomp['score'], "transformed": transformed}

    return result

def cross_validate(X, t_choices, num_folds = 5, restarts=3, metric = 'geoSE', tol=1e-4, max_iter=500):
    """Calculates mean and standard deviation of cross validation standard errors.

    Calculates squared errors on a test set after transforming it using geometric 
    parameters derived from training set. This process is repeated on num_folds 
    different test and training set pairs. Average and standard deviation
    of standard values are returned.

    Args:
        X (ndarray): Matrix where columns correspond to genes and rows 
            correspond to samples.
        t_choices (ndarray): array of l1 norm constraints (that sets SPCA sparsity level) to evaluate.
        num_folds (int): Number of random cross validation folds. Defaults to 5.

    Returns:

        Dict {'avg': average squared error, 'std': standard dev. of squared errors}

    Raises:
        ValueError: If x_vector and y_vector aren't the same length.
    """
    test_train_pairs = train_test_split(X.shape[0], num_folds)
    # Cross-validation
    se_values = [[_calculate_se(X, t, pair['test'], pair['train'], restarts=restarts, metric=metric, tol=tol, max_iter=max_iter) for pair in test_train_pairs] for t in t_choices]
    se_values = [np.concatenate(s) for s in se_values]
    mean_se = np.array([np.mean(s) for s in se_values])
    std_se = np.array([np.std(s) for s in se_values])
    ind = np.argmin(mean_se)
    if not np.isscalar(ind):
        print(ind)
    best_t = t_choices[ind]
    if np.any(np.logical_and(mean_se > mean_se[ind] + std_se[ind], t_choices<best_t)):
        best_t_1sd = t_choices[np.logical_and(mean_se > mean_se[ind] + std_se[ind], t_choices<best_t)][-1]
    else: 
        best_t_1sd = t_choices[0]

    return {'mean': mean_se, 'std': std_se, 'best_t': best_t, 'best_t_1sd': best_t_1sd, 't_choices': t_choices}

# INTERNAL FUNCTIONS

def _calculate_se(X, t, test, train, restarts, metric, tol, max_iter):
    decomp = multi_start(X[train, :], t, restarts=restarts, tol=tol, max_iter=max_iter)
    Y = X @ decomp['V']
    try:
        theta = direct_ellipse_est(Y[train, 0], Y[train, 1])
    except ValueError:
        return None
    
    test_se = ellipse_metrics(Y[test, 0], Y[test, 1], theta, metric=metric)
    return(test_se)
