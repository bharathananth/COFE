"""Functions to analyse gene expression data sets using SPCA based methods.

This module contains functions to process data, analyse data using sparse 
loading vectors, and analyse data using SPCA principal components.
"""
import numpy as np
from scipy.linalg import norm
from joblib import delayed, Parallel
from COFE.spca import sparse_cyclic_pca_masked, sparse_cyclic_pca

def process_data(X, features, feature_dim='row', mean_threshold=None, 
                 scaling_threshold=None, impute=None, scale=True):
    """Function to preprocess data prior to analysis including 
    centering, scaling and imputing.

    Parameters
    ----------
    X : ndarray
        Matrix of features across samples.
    features : str
        Names of the features
    feature_dim : str, optional
        Whether the features are the rows ('row') or columns ('col') 
        of X, by default 'row'
    mean_threshold : float, optional
        minimum mean level of features that are retained for analysis, 
        by default None
    scaling_threshold : (float, float), optional
        the range of reciprocal feature standard, by default None
    impute : float, optional
        features outside central percentile are truncated to the chosen 
        level , by default None

    Returns
    -------
    (ndarray, array, array)
        Matrix of preprocessed data, names of retained features, 
        standard deviation of retained features in raw data

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
            upper_bound = np.percentile(X_, 100-impute/2, axis=axis, 
                                        keepdims=True)
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
                keep = np.logical_and(X_.std(axis=1)>1/scaling_threshold[1], 
                                      X_.std(axis=1)<1/scaling_threshold[0])
                features_ =  features_[keep]
                X_ = X_[keep, :]
            else:
                keep = np.logical_and(X_.std(axis=0)>1/scaling_threshold[1], 
                                      X_.std(axis=0)<1/scaling_threshold[0])
                features_ =  features_[keep]
                X_ = X_[:, keep]

    # Always center data to have zero mean
    X_ = X_ - np.mean(X_, axis=axis, keepdims=True)

    # Standardise every gene series to have variance of 1
    std_ = np.std(X_, axis=axis, keepdims=True)
    if scale:
        X_ = X_ / std_

    if axis == 1:
        X_ = X_.T
    else:
        std_ = std_.T
    
    return (X_, features_, std_)

def cyclic_ordering(X, s, feature_std=None, restarts=5, tol=1e-4, max_iter=100, 
                    true_times=None, period=24.0, ncores=None):
    """Find the CPCs and SLs for this chosem sparsity 's'

    Parameters
    ----------
    X : ndarray
        preprocessed data matrix
    s : double
        chosen l1 sparsity threshold for the loading vectors
    feature_std : array, optional
        weights for the different features that determine the st. dev. 
        of the random initial conditions each restart, by default None
    restarts : int, optional
        the number of random initial conditions to begin alternating 
        maximization from, by default 5
    tol : _type_, optional
        _description_, by default 1e-4
    max_iter : int, optional
        _description_, by default 200
    true_times : _type_, optional
        _description_, by default None
    period : double, optional
        period of the underlying rhythm, by default 24.0
    ncores : int, optional
        number of cores to use for parallel computation of the 
        multistarts, by default None, which computes serially. See 
        joblib.Parallel for convention, by default None

    Returns
    -------
        {
         'phase': ndarray
            reconstructed phases of the samples,
         'MAPE': float
            median absolute position error,
         'CPCs': ndarray
            2D array containing the 2 cyclic principal components for 
            chosen 's',
         'SLs': ndarray
            2D array containing the 2 sparse loading vectors for chosen 
            's',
         'd' : double
            scale factor for the outer product approximation,
         'true_times': ndarray 
            known reference times for the samples to compare the 
            reconstruction against,
         'rss': float
            rss of the chosen 's'
    }
    """    
    decomp = _multi_start(X, s, feature_std, restarts=restarts, tol=tol, 
                          tol_z=1e-3, max_iter=max_iter, ncores=ncores)
    V = decomp['V']
    U = decomp['U']

    mape_value = np.nan
    pred_phase = np.nan
    mape_value, pred_phase = calculate_mape(U, true_times, period=period)

    result = {'phase': pred_phase, 
              'MAPE': mape_value, 
              'CPCs': U, 
              'SLs': V, 
              'd': decomp['d'], 
              'true_times': true_times, 
              'rss': decomp['rss']}

    return result

def cross_validate(X, s_choices, features=None, feature_std=None, K=5, 
                          restarts=5, tol=1e-3, tol_z=1e-4, max_iter=200, 
                          true_times=None, period=24.0, ncores=None):
    """Calculate the optimal choice of sparsity threshold 's' and the 
    cyclic ordering for the best 's'
    
    Parameters
    ----------
    X : ndarray
        preprocessed data matrix 
    s_choices : array or list
        different values of l1 sparsity threshold to compare
    features : array, optional
        names of the features, by default None
    feature_std : array, optional
        weights for the different features that determine the st. dev. 
        of the random initial conditions each restart, by default None
    K : int, optional
        number of folds used for cross-validation
    restarts : int, optional
        the number of random initial conditions to begin alternating 
        maximization from, by default 5
    tol : double, optional
        convergence threshold for the alternating maximization of 
        cyclic PCA, by default 1e-3
    tol_z : double, optional
        convergence threshold for imputation of missing and cross 
        validation, by default 1e-5
    max_iter : int, optional
        maximum number of iterations of the alternating maximization, 
        by default 200
    true_times : array, optional
        known reference times for the samples to compare the 
        reconstruction against, by default None
    period : double, optional
        period of the underlying rhythm, by default 24.0
    ncores : int, optional
        number of cores to use for parallel computation of the 
        multistarts, by default None, which computes serially. See 
        joblib.Parallel for convention.

    Returns
    -------
    {
        'best_s': float
            best performing lamb
         s_choices': array-like float different values of l1 sparsity 
            threshold to compare from the input
         'runs': list of dict
            the results structure from cross_validate for the different 
            lamb choices tried
         'MAPE': float
            median absolute position error
         'pred_phase': ndarray
            reconstructed phases of the samples
         'true_times': ndarray 
            true/reference times of the samples from the input
         'rss': float
            rss of the best performing lamb
         'CPCs': ndarray
            2D array containing the 2 cyclic principal components for 
            best_s
         'SLs': ndarray
            2D array containing the 2 sparse loading vectors for best_s
         'features': ndarray
            names of the features from the input 
    }
    """
    indices = np.random.permutation(X.size)
    fold_size = np.ceil(X.size/K).astype(int)

    cv_indices = [indices[i*fold_size:(i+1)*fold_size] for i in range(K)]
    
    # Cross-validation
    cv_s = [_calculate_cv(X, lamb, feature_std, restarts, tol, tol_z, max_iter, 
                          cv_indices, ncores) for lamb in s_choices]
    best_s = s_choices[np.argmin([cv_m for (cv_m, _) in cv_s])]

    best_fit = cyclic_ordering(X, best_s, feature_std=feature_std, 
                               restarts=restarts, true_times=true_times, 
                               tol=tol, max_iter=max_iter, period=period, 
                               ncores=ncores)

    return {'best_s': best_s, 
            's_choices': s_choices, 
            'runs': cv_s, 
            'features': features} | best_fit

def calculate_mape(Y, true_times=None, period=24.0):
    """Calculate median absolute position error

    Parameters
    ----------
    Y : ndarray
        2D array consisting of the two sparse cyclic principal 
        components, or 1D array consisting of estimated sample times 
        (normalized by period)
    true_times : ndarray, optional
        1D array of true/reference sample times for each sample in data, 
        by default None
    period : double, optional
        period of the underlying rhythm, by default 24.0

    Returns
    -------
    (float, ndarray)
        tuple containing the median absolute error, estimated phases of 
        the samples
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
            acw_bias = _delta(acw_angles, scaled_time) * 2*np.pi
            cw_bias = _delta(cw_angles, scaled_time) * 2*np.pi
            # Optimal direction is direction with lowest bias value std dev
            if _angular_spread(np.cos(acw_bias), np.sin(acw_bias)) < \
                _angular_spread(np.cos(cw_bias), np.sin(cw_bias)):
                adjusted_opt_angles = (acw_angles \
                                       - _angular_mean(np.cos(acw_bias), 
                                                       np.sin(acw_bias))) % 1
            else:
                adjusted_opt_angles = (cw_angles \
                                       - _angular_mean(np.cos(cw_bias), 
                                                       np.sin(cw_bias))) % 1
            mape_value = np.min([np.median(np.abs(_delta(scaled_time 
                                                         - adjusted_opt_angles, 
                                                         d))) 
                                        for d in np.arange(-0.5, 0.5, 0.005)])
        except RuntimeWarning:
            mape_value = np.nan
    else:
        mape_value = np.nan
        adjusted_opt_angles = acw_angles
    return (mape_value, adjusted_opt_angles)

# INTERNAL FUNCTIONS

def _calculate_cv(X, s, feature_std, restarts, tol, tol_z, max_iter, cv_indices, 
                  ncores):
    rss_cv = list()
    mask_size = list()
    for cv_ind in cv_indices:
        mask = np.zeros(X.size, dtype=bool)
        mask[cv_ind] = True
        X_ = np.ma.array(X, copy=True, mask=np.reshape(mask, X.shape))
        decomp = _multi_start(X_, s, feature_std, restarts, tol, tol_z,
                              max_iter, ncores)
        rss_cv.append(decomp['cv_err']) 
        mask_size.append(np.sum(mask))
        mean_rss = np.array(rss_cv)/np.array(mask_size)
    return((np.sum(rss_cv)/X.size, np.std(mean_rss)/np.sqrt(len(cv_indices))))

def _multi_start(X, s, feature_std, restarts, tol, tol_z, max_iter, ncores):
    if ncores is None:
        if isinstance(X, np.ma.MaskedArray):
            runs = [sparse_cyclic_pca_masked(X, s=s, tol=tol, tol_z=tol_z, 
                                             max_iter=max_iter, 
                                             feature_std=feature_std) 
                                             for _ in range(restarts)]
        else:
            runs = [sparse_cyclic_pca(X, s=s, tol=tol, max_iter=max_iter, 
                                      feature_std=feature_std) 
                                      for _ in range(restarts)]
    else:
        if isinstance(X, np.ma.MaskedArray):
            runs = Parallel(n_jobs=ncores)(
                delayed(sparse_cyclic_pca_masked)(X, s=s, tol=tol, tol_z=tol_z,
                                                  max_iter=max_iter, 
                                                  feature_std=feature_std) 
                                                  for _ in range(restarts))
        else:
            runs = Parallel(n_jobs=ncores)(
                delayed(sparse_cyclic_pca)(X, s=s, tol=tol, max_iter=max_iter, 
                                           feature_std=feature_std) 
                                           for _ in range(restarts))
    
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
