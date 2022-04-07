"""Functions to analyse gene expression data sets using SPCA based methods.

This module contains functions to process data, analyse data using sparse 
loading vectors, and analyse data using SPCA principal components.
"""
import numpy as np
import pandas as pd
from COFE.spca import *
from COFE.ellipse import *
import concurrent.futures
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def process_data(X, features, feature_dim='row', mean_threshold=None, scaling_threshold=None, impute=True):
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
    if impute:
        upper_bound = np.percentile(X_, 97.5, axis=axis, keepdims=True)
        lower_bound = np.percentile(X_, 2.5, axis=axis, keepdims=True)
        X_ = np.maximum(np.minimum(X_, upper_bound), lower_bound)

    # Always center data to have zero mean
    X_ = X_ - np.mean(X_, axis=axis, keepdims=True)

    # Standardise every gene series to have variance of 1
    std_ = np.std(X_, axis=axis, keepdims=True)
    X_ = X_ / std_

    if axis == 1:
        X_ = X_.T
    else:
        std_ = std_.T
    
    return (X_, features_, std_)

def multi_start(X, t=float('inf'), feature_std=None, restarts=3, tol=1e-4, max_iter=200, ncores=None):
    if ncores is None:
        ncores = cpu_count()
    
    runs = Parallel(n_jobs=ncores)(delayed(coupled_spca)(X, t=t, tol=tol, max_iter=max_iter, feature_std=feature_std) for _ in range(restarts))

    ind = np.argmax([r['score'] for r in runs])
    return runs[ind]

def circular_ordering(X, t, feature_std=None, restarts=3, true_times=None, tol=1e-4):
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

    decomp = multi_start(X, t=t, feature_std=feature_std, restarts=restarts, tol=tol)
    U = decomp['U']
    V = decomp['V']

    Y = X @ V
    
    mape_value = np.nan
    pred_phase = np.nan
    mape_value, pred_phase = calculate_mape(Y, true_times)

    result = {'phase': pred_phase, 'MAPE': mape_value, 'CPCs': Y, 'SLs': V, 'true_times': true_times,
              'MSE': np.nan, 'score': decomp['score']}

    return result

def cross_validate(X, t_choices, choice='min', true_times = None, features=None, feature_std=None, restarts=3, tol=1e-4, max_iter=200):
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
    # Cross-validation
    run_t = [_calculate_se(X, t, feature_std, restarts, tol, max_iter) for t in t_choices]
    mean_se = np.array([np.quantile(s['test_se'], 0.75) for s in run_t])
    std_se = np.array([np.std(s['test_se']) for s in run_t])    
    ind_min = np.argmin(mean_se)
    
    if choice == '1sd':
        if np.any(np.logical_and(mean_se > mean_se[ind_min] + std_se[ind_min], t_choices<t_choices[ind_min])):
            ind  = np.where(np.logical_and(mean_se > mean_se[ind_min] + std_se[ind_min], t_choices<t_choices[ind_min]))[0][-1]
        else:
            ind = 0
    else:
        ind = ind_min

    decomp = run_t[ind]
    U = decomp['U']
    V = decomp['V']

    Y = X @ V

    mape_value = np.nan
    pred_phase = np.nan
    mape_value, pred_phase = calculate_mape(Y, true_times)

    return {'best_t': t_choices[ind], 't_choices': t_choices, 'runs': run_t,
            'MAPE': mape_value, 'phase': pred_phase, 'MSE': np.mean(decomp['test_se']), 'score': decomp['score'],
            'CPCs': Y, 'SLs': V, 'features': features}

# INTERNAL FUNCTIONS

def _calculate_se(X, t, feature_std, restarts, tol, max_iter):
    decomp = multi_start(X, t, restarts=restarts, tol=tol, max_iter=max_iter)
    Y = X @ decomp['V']
    try:
        theta = direct_ellipse_est(Y[:, 0], Y[:, 1])
    except ValueError:
        return None
    decomp['test_se'] = ellipse_metrics(Y[:, 0], Y[:, 1], theta)
    return(decomp)
