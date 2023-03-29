"""Functions to analyse gene expression data sets using SPCA based methods.

This module contains functions to process data, analyse data using sparse 
loading vectors, and analyse data using SPCA principal components.
"""
import numpy as np
from scipy.linalg import norm
import joblib
from COFE.spca import sparse_cyclic_pca_masked, sparse_cyclic_pca

def process_data(X, features, feature_dim='row', mean_threshold=None, scaling_threshold=None, impute=None, scale=True):
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
    if scale:
        X_ = X_ / std_
    else:
        X_ = X_ / np.max(std_)

    if axis == 1:
        X_ = X_.T
    else:
        std_ = std_.T
    
    return (X_, features_, std_)

def circular_ordering(X, lamb, feature_std=None, restarts=3, true_times=None, 
                      tol=1e-4, max_iter=200, period=24, ncores=None):
    """Function to fit ellipse to SPCA principal components.

    Function to find the circular PCs for the chosen l1 sparsity constrain and computes MAPE for ordering if true_times is provided

    Args:
        X (ndarray): Matrix where columns correspond to genes and rows 
            correspond to samples.
        lamb (float): l1 norm constraint that sets SPCA sparsity level.
        true_times (array): The true time labels to compare results against. 
            Defaults to None. 

    Returns:

        Dictionary in which keys are different ellipse metrics names and values 
        are corresponding quantities.
    """

    decomp = _multi_start(X, lamb, feature_std, restarts=restarts, tol=tol, 
                          max_iter=max_iter, ncores=ncores)
    V = decomp['V']

    Y = X @ V

    mape_value = np.nan
    pred_phase = np.nan
    mape_value, pred_phase = calculate_mape(Y, true_times, period=period)

    result = {'phase': pred_phase, 'MAPE': mape_value, 'CPCs': Y, 'SLs': V, 
              'd': decomp['d'], 'true_times': true_times, 'rss': decomp['rss']}

    return result

def cross_validate(X, lambda_choices, K=5, features=None, feature_std=None, 
                          restarts=10, tol=1e-3, max_iter=200, true_times=None, 
                          period=24, ncores=None):
    """Calculates the median squared error and its standard deviation for different choices of sparsity threshold 'lamb'

    Parameters
    ----------
    X : ndarray
        preprocessed data matrix 
    lambda_choices : array or list
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
        'best_lambda': float
            best performing lamb
         lambda_choices': array-like float different values of l1 sparsity threshold to compare from the input
         'runs': list of dict
            the results structure from cross_validate for the different lamb choices tried
         'MAPE': float
            median absolute position error
         'pred_phase': ndarray
            reconstructed phases of the samples
         'true_times': ndarray 
            true/reference times of the samples from the input
         'MSE': float
            median squared error of the best performing lamb
         'rss': float
            rss of the best performing lamb
         'CPCs': ndarray
            2D array containing the 2 circular principal components for best performing lamb
         'SLs': ndarray
            2D array containing the 2 sparse loading vectors V_1 and V_2 for best performing lamb
         'features': ndarray
            names of the features from the input 
         'acw': float
            approximate MAPE calculation
    }
    """
    indices = np.random.permutation(X.size)
    fold_size = np.ceil(X.size/K).astype(int)

    cv_indices = [indices[i*fold_size:(i+1)*fold_size] for i in range(K)]
    
    # Cross-validation
    cv_lambda = [_calculate_cv(X, lamb, feature_std, restarts, tol, max_iter, cv_indices, ncores) for lamb in lambda_choices]
    best_lambda = lambda_choices[np.argmin([cv_m for (cv_m, cv_s) in cv_lambda])]

    best_fit = circular_ordering(X, best_lambda, feature_std=feature_std, 
                                 restarts=restarts, true_times=true_times, 
                                 tol=tol, max_iter=max_iter, period=period, 
                                 ncores=ncores)

    return {'best_lambda': best_lambda, 'lambda_choices': lambda_choices, 
            'runs': cv_lambda, 'features': features} | best_fit

def calculate_mape(Y, true_times=None, period=24):
    """Calculate median absolute position error

    Parameters
    ----------
    Y : ndarray
        2D array consisting of the two circular sparse principal components, or
        1D array consisting of estimated sample times (normalized by period)
    true_times : ndarray, optional
        1D array of true/reference sample times for each sample in data, by default None
    period : int, optional
        _description_, by default 24

    Returns
    -------
    (float, ndarray, float)
        tuple containing the median absolute error, estimated phases of the samples and a fast median absolute error calculation
    """    
    Y = Y.squeeze()
    if Y.ndim == 2:
        # Scaled angular positions
        acw_angles, cw_angles = _scaled_angles(Y[:, 0], Y[:, 1])
    elif Y.ndim == 1:
        acw_angles, cw_angles = Y % 1.0, -Y % 1.0

    if true_times is not None:
        try: 
            # Scaled time values
            scaled_time = true_times/period - true_times//period
            # Choosing direction (bias variation)
            acw_bias = _delta(acw_angles, scaled_time)
            cw_bias = _delta(cw_angles, scaled_time)
            # Optimal direction is direction with lowest bias value std dev
            if _angular_spread(np.cos(2*np.pi*acw_bias), np.sin(2*np.pi*acw_bias)) < \
                _angular_spread(np.cos(2*np.pi*cw_bias), np.sin(2*np.pi*cw_bias)):
                adjusted_opt_angles = (acw_angles - _angular_mean(np.cos(2*np.pi*acw_bias), np.sin(2*np.pi*acw_bias))) % 1
            else:
                adjusted_opt_angles = (cw_angles - _angular_mean(np.cos(2*np.pi*cw_bias), np.sin(2*np.pi*cw_bias))) % 1
            mape_value = np.min([np.median(np.abs(_delta(scaled_time - adjusted_opt_angles, d))) for d in np.arange(-0.5, 0.5, 0.005)]) if true_times is not None else np.nan
        except RuntimeWarning:
            mape_value = np.nan
    else:
        mape_value = np.nan
        adjusted_opt_angles = acw_angles
    return (mape_value, adjusted_opt_angles)

# INTERNAL FUNCTIONS

def _calculate_cv(X, lamb, feature_std, restarts, tol, max_iter, cv_indices, ncores):
    rss_cv = list()
    mask_size = list()
    for cv_ind in cv_indices:
        mask = np.zeros(X.size, dtype=bool)
        mask[cv_ind] = True
        X_ = np.ma.array(X, copy=True, mask=np.reshape(mask, X.shape))
        decomp = _multi_start(X_, lamb, feature_std, restarts=restarts, tol=tol, max_iter=max_iter, ncores=ncores)
        rss_cv.append(decomp['cv_err']) 
        mask_size.append(np.sum(mask))
        mean_rss = np.array(rss_cv)/np.array(mask_size)
    return((np.sum(rss_cv)/X.size, np.std(mean_rss)/np.sqrt(len(cv_indices))))

def _multi_start(X, lamb, feature_std, restarts, tol, max_iter, ncores):
    if ncores is None:
        if isinstance(X, np.ma.MaskedArray):
            runs = [sparse_cyclic_pca_masked(X, lamb=lamb, tol=tol, max_iter=max_iter, feature_std=feature_std) for _ in range(restarts)]
        else:
            runs = [sparse_cyclic_pca(X, lamb=lamb, tol=tol, max_iter=max_iter, feature_std=feature_std) for _ in range(restarts)]
    else:
        if isinstance(X, np.ma.MaskedArray):
            runs = joblib.Parallel(n_jobs=ncores)(joblib.delayed(sparse_cyclic_pca_masked)(X, lamb=lamb, tol=tol, max_iter=max_iter, feature_std=feature_std) for _ in range(restarts))
        else:
            runs = joblib.Parallel(n_jobs=ncores)(joblib.delayed(sparse_cyclic_pca)(X, lamb=lamb, tol=tol, max_iter=max_iter, feature_std=feature_std) for _ in range(restarts))
    ind = np.argmin([r['rss'] for r in runs])
    return runs[ind]

def _scaled_angles(x, y):
    return (np.arctan2(y, x)/(2*np.pi)) % 1,  (np.arctan2(-y, x)/(2*np.pi)) % 1

def _delta(x, y):
    return ((x - y + 1/2) % 1) - 1/2

def _angular_mean(x, y):
    return (np.arctan2(np.sum(y), np.sum(x))/(2*np.pi)) % 1

def _angular_spread(x, y):
    return x.shape[0] - np.sqrt(np.sum(x) ** 2 + np.sum(y) ** 2)
