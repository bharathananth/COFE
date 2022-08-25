"""Functions to analyse gene expression data sets using SPCA based methods.

This module contains functions to process data, analyse data using sparse 
loading vectors, and analyse data using SPCA principal components.
"""
import numpy as np
import pandas as pd
from COFE.spca import *
from COFE.ellipse import *
import joblib


def process_data(X, features, feature_dim='row', mean_threshold=None, scaling_threshold=None, impute=None):
    """Function to preprocess data prior to analysis including centering, scaling and imputing.

    Parameters
    ----------
    X : ndarray
        Matrix of features across samples.
    features : str
        Names of the features
    feature_dim : str, optional
        Whether the features are the rows ('row') or columns ('col') of X, by default 'row'
    mean_threshold : float, optional
        minimum mean level of features that are retained for analysis, by default None
    scaling_threshold : (float, float), optional
        the range of reciprocal feature standard, by default None
    impute : float, optional
        features outside central percentile are truncated to the chosen level , by default None

    Returns
    -------
    (ndarray, array, array)
        Matrix of preprocessed data, names of retained features, standard deviation of retained features in raw data

    Raises
    ------
    ValueError
        if mean_threshold is not int or float
    TypeError
        if scaling_threshold is not a tuple
    ValueError
        if scaling_threshold has a length other than 2
    """    

    X_ = X.copy()
    features_ = features.copy()

    if feature_dim == 'row':
        axis = 1
    elif feature_dim == 'col':
        axis = 0
    else:
        raise ValueError("Invalid feature dimension.")

    # Imputing extreme values
    if impute is not None:
        if isinstance(impute, float) and impute < 100.0 and impute > 0.0:
            upper_bound = np.percentile(X_, 100-impute/2, axis=axis, keepdims=True)
            lower_bound = np.percentile(X_, impute/2, axis=axis, keepdims=True)
            X_ = np.maximum(np.minimum(X_, upper_bound), lower_bound)

    if mean_threshold is not None:
        if isinstance(mean_threshold, (int, float)):
            if axis == 1:
                keep = X_.mean(axis=1)>mean_threshold
                features_ =  features_[keep]
                X_ = X_[keep, :]
            else:
                keep = X_.mean(axis=0)>mean_threshold
                features_ =  features_[keep]
                X_ = X_[:, keep]
        else:
            raise ValueError("mean_threshold must be a float")

    if scaling_threshold is not None:
        if type(scaling_threshold) != tuple:
            raise TypeError("scaling_threshold must be a tuple if not None") 
        elif len(scaling_threshold) != 2:
            raise ValueError("Tuple must have only 2 numeric values")
        else:
            if axis == 1:
                keep = np.logical_and(X_.std(axis=1)>1/scaling_threshold[1], X_.std(axis=1)<1/scaling_threshold[0])
                features_ =  features_[keep]
                X_ = X_[keep, :]
            else:
                keep = np.logical_and(X_.std(axis=0)>1/scaling_threshold[1], X_.std(axis=0)<1/scaling_threshold[0])
                features_ =  features_[keep]
                X_ = X_[:, keep]

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

def circular_ordering(X, t, feature_std=None, restarts=3, true_times=None, features=None, tol=1e-4, max_iter=200):
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

    decomp = _multi_start(X, t=t, feature_std=feature_std, restarts=restarts, tol=tol, max_iter=max_iter)
    U = decomp['U']
    V = decomp['V']

    Y = X @ V

    try:
        geo_param = direct_ellipse_est(Y[:, 0], Y[:, 1])
    except ValueError:
        return None
    mse = np.quantile(ellipse_metrics(Y[:, 0], Y[:, 1], geo_param), 0.5)
    
    mape_value = np.nan
    pred_phase = np.nan
    mape_value, pred_phase, acw = calculate_mape(Y, true_times)

    result = {'phase': pred_phase, 'MAPE': mape_value, 'CPCs': Y, 'SLs': V, 'true_times': true_times,
              'MSE': np.nan, 'score': decomp['score'], 'features': features, 'MSE': mse, 'U': U, 'acw': acw}

    return result

def cross_validate(X, t_choices, choice='min', true_times=None, features=None, feature_std=None, outlier=10, hold=None, restarts=5, tol=1e-3, max_iter=100, ncores=None):
    """Calculates the median squared error and its standard deviation for different choices of sparsity threshold 't'

    Parameters
    ----------
    X : ndarray
        preprocessed data matrix 
    t_choices : array or list
        different values of l1 sparsity threshold to compare
    choice : str, optional
        criterion to pick the best sparsity threshold, either minimum or 1 SD above minimum, by default 'min'
    true_times : array, optional
        known reference times for the samples to compare the reconstruction against, by default None
    features : array, optional
        names of the features, by default None
    feature_std : array, optional
        weights for the different features that determine the st. dev. of the random initial conditions each restart, by default None
    hold : float, optional
        the fraction of samples to withhold as test set to estimate performance at each l1 threshold value, by default None
    restarts : int, optional
        the number of random initial conditions to begin alternating maximization from, by default 5
    tol : _type_, optional
        convergence threshold for the alternating maximization of circular sparse PCA, by default 1e-3
    max_iter : int, optional
        maximum number of iterations of the alternating maximization, by default 100
    ncores : int, optional
        number of cores to use for parallel computation of the multistarts, by default None, which computes serially. See joblib.Parallel for convention.

    Returns
    -------
    {
        'best_t': float
            best performing t
         t_choices': array-like float different values of l1 sparsity threshold to compare from the input
         'runs': list of dict
            the results structure from cross_validate for the different t choices tried
         'MAPE': float
            median absolute position error
         'pred_phase': ndarray
            reconstructed phases of the samples
         'true_times': ndarray 
            true/reference times of the samples from the input
         'MSE': float
            median squared error of the best performing t
         'score': float
            score of the best performing t
         'CPCs': ndarray
            2D array containing the 2 circular principal components for best performing t
         'SLs': ndarray
            2D array containing the 2 sparse loading vectors V_1 and V_2 for best performing t
         'features': ndarray
            names of the features from the input 
         'acw': float
            approximate MAPE calculation
    }
    """    
    
    test_indices = None
    train_indices = None
    if hold is not None:
        if isinstance(hold, float):
            rng = np.random.default_rng()
            shuffled_indices = rng.permutation(X.shape[0])
            test_indices = shuffled_indices[0:int(hold*X.shape[0])]
            train_indices = shuffled_indices[int(hold*X.shape[0]):]
        else:
            raise("hold must be a number between 0 and 1")

    # Cross-validation
    run_t = [_calculate_se(X, t, feature_std, restarts, tol, max_iter, train_indices, test_indices, ncores) for t in t_choices]
    med_se = np.array([np.percentile(s['test_se'], 100-outlier) for s in run_t])
    std_se = np.array([scipy.stats.iqr(s['test_se']) for s in run_t])    
    ind_min = np.argmin(med_se)
    
    if choice == '1sd':
        if np.any(np.logical_and(med_se > med_se[ind_min] + std_se[ind_min], t_choices<t_choices[ind_min])):
            ind  = np.where(np.logical_and(med_se > med_se[ind_min] + std_se[ind_min], t_choices<t_choices[ind_min]))[0][-1]
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
            'MAPE': mape_value, 'phase': pred_phase, 'true_times': true_times, 'MSE': med_se[ind_min], 'score': decomp['score'],
            'CPCs': Y, 'SLs': V, 'features': features}

# INTERNAL FUNCTIONS

def _calculate_se(X, t, feature_std, restarts, tol, max_iter, train_indices, test_indices, ncores):
    if train_indices is None or test_indices is None:
        decomp = _multi_start(X, t, feature_std, restarts=restarts, tol=tol, max_iter=max_iter, ncores=ncores)
        Y = X @ decomp['V']
        try:
            geo_param = direct_ellipse_est(Y[:, 0], Y[:, 1])
        except ValueError:
            return None
        decomp['test_se'] = ellipse_metrics(Y[:, 0], Y[:, 1], geo_param)
    else:
        decomp = _multi_start(X[train_indices, :], t, feature_std, restarts=restarts, tol=tol, max_iter=max_iter, ncores=ncores)
        Y = X @ decomp['V']
        try:
            geo_param = direct_ellipse_est(Y[train_indices, 0], Y[train_indices, 1])
        except ValueError:
            return None
        decomp['test_se'] = ellipse_metrics(Y[test_indices, 0], Y[test_indices, 1], geo_param)
    return(decomp)

def _multi_start(X, t, feature_std, restarts, tol, max_iter, ncores):
    if ncores is None:
        runs = [coupled_spca(X, t=t, tol=tol, max_iter=max_iter, feature_std=feature_std) for _ in range(restarts)]
    else:
        runs = joblib.Parallel(n_jobs=ncores)(joblib.delayed(coupled_spca)(X, t=t, tol=tol, max_iter=max_iter, feature_std=feature_std) for _ in range(restarts))

    ind = np.argmax([r['score'] for r in runs])
    return runs[ind]
