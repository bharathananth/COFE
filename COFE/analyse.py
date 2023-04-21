"""Functions to analyse gene expression data sets using SPCA based methods.

This module contains functions to process data, analyse data using sparse 
loading vectors, and analyse data using SPCA principal components.
"""
import numpy as np
from scipy.linalg import norm
from warnings import warn
from scipy.interpolate import interp1d
from joblib import delayed, Parallel
# from statsmodels.nonparametric.smoothers_lowess import lowess
from COFE.spca import sparse_cyclic_pca_masked, sparse_cyclic_pca

def preprocess_data(X_train, X_test, features, feature_dim='row', 
                 mean_threshold=None, scaling_threshold=None, 
                 impute=None, scale=True):
    """Function to preprocess data prior to analysis including 
    centering, scaling and imputing.

    Parameters
    ----------
    X : ndarray
        matrix of features across samples.
    features : str
        names of the features
    feature_dim : str, optional
        whether the features are the rows ('row') or columns ('col') 
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
    (ndarray, ndarray, array, array)
        Matrix of preprocessed training and test data, names of retained
          features, standard deviation of retained features in raw data

    Raises
    ------
    ValueError
        if mean_threshold is not int or float
    TypeError
        if scaling_threshold is not a tuple
    ValueError
        if scaling_threshold has a length other than 2
    """    
    X_train_ = X_train.copy()
    X_test_ = X_test.copy() if X_test is not None else None
    features_ = features.copy()

    if feature_dim == 'row':
        axis = 1
        if X_test_ is not None and X_train_.shape[0]!=X_test_.shape[0]:
            raise ValueError("Different number of features between "
                             "train and test data")
    elif feature_dim == 'col':
        axis = 0
        if X_test_ is not None and X_train_.shape[1]!=X_test_.shape[1]:
            raise ValueError("Different number of features between "
                             "train and test data")
    else:
        raise ValueError("Invalid feature dimension.")

    # Imputing extreme values
    if impute is not None:
        if isinstance(impute, float) and impute < 100.0 and impute > 0.0:
            upper_bound = np.percentile(X_train_, 100-impute/2, axis=axis, 
                                        keepdims=True)
            lower_bound = np.percentile(X_train_, impute/2, axis=axis, 
                                        keepdims=True)
            X_train_ = np.maximum(np.minimum(X_train_, upper_bound), lower_bound)

    if mean_threshold is not None:
        if isinstance(mean_threshold, (int, float)):
            if axis == 1:
                keep = X_train_.mean(axis=1)>mean_threshold
                features_ =  features_[keep]
                X_train_ = X_train_[keep, :]
                if X_test_ is not None:
                    X_test_ = X_test_[keep, :]
            else:
                keep = X_train_.mean(axis=0)>mean_threshold
                features_ =  features_[keep]
                X_train_ = X_train_[:, keep]
                if X_test_ is not None:
                    X_test_ = X_test_[:, keep]
        else:
            raise ValueError("mean_threshold must be a float")

    if scaling_threshold is not None:
        if type(scaling_threshold) != tuple:
            raise TypeError("scaling_threshold must be a tuple if not None") 
        elif len(scaling_threshold) != 2:
            raise ValueError("Tuple must have only 2 numeric values")
        else:
            if axis == 1:
                # mean_sd = lowess(X_train_.std(axis=1), X_train_.mean(axis=1), 
                #                  frac=0.2)
                # f = interp1d(mean_sd[:, 0], mean_sd[:, 1], bounds_error=False)
                # keep = X_train_.std(axis=1) > f(mean_sd[:, 0])
                keep = np.logical_and(X_train_.std(axis=1)>1/scaling_threshold[1], 
                                      X_train_.std(axis=1)<1/scaling_threshold[0])
                features_ =  features_[keep]
                X_train_ = X_train_[keep, :]
                if X_test_ is not None:
                    X_test_ = X_test_[keep, :]
            else:
                # mean_sd = lowess(X_train_.std(axis=0), X_train_.mean(axis=0), 
                #                  frac=0.2)
                # f = interp1d(mean_sd[:, 0], mean_sd[:, 1], bounds_error=False)
                # keep = X_train_.std(axis=0) > f(mean_sd[:, 0])
                keep = np.logical_and(X_train_.std(axis=0)>1/scaling_threshold[1], 
                                      X_train_.std(axis=0)<1/scaling_threshold[0])
                features_ =  features_[keep]
                X_train_ = X_train_[:, keep]
                if X_test_ is not None:
                    X_test_ = X_test_[:, keep]

    # Always center data to have zero mean
    mean_ = np.mean(X_train_, axis=axis, keepdims=True)
    X_train_ = X_train_ - mean_
    if X_test_ is not None:
        X_test_ = X_test_ - mean_

    # Standardise every gene series to have variance of 1
    std_ = np.std(X_train_, axis=axis, keepdims=True)
    if scale:
        X_train_ = X_train_ / std_
        if X_test is not None:
            X_test_ = X_test_ / std_

    if axis == 1:
        X_train_ = X_train_.T
        if X_test_ is not None:
            X_test_ = X_test_.T
    else:
        std_ = std_.T
    
    return (X_train_, X_test_, features_, std_)

def cross_validate(X_train, s_choices, features, feature_std=None, K=5, 
                   restarts=5, tol=1e-3, tol_z=1e-6, max_iter=200, ncores=None):
    """Calculate the optimal choice of sparsity threshold 's' and the 
    cyclic ordering for the best 's'
    
    Parameters
    ----------
    X_train : ndarray
        preprocessed training data matrix 
    s_choices : array or list
        different values of l1 sparsity threshold to compare
    features: array
        names of the features
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
         'CPCs': ndarray
            2D array containing the 2 cyclic principal components for 
            best_s
         'SLs': ndarray
            2D array containing the 2 sparse loading vectors for best_s
         'd' : double
            scale factor for the outer product approximation
         'rss': float
            rss of the chosen 's'
    }
    """
    indices = np.random.permutation(X_train.size)
    fold_size = np.ceil(X_train.size/K).astype(int)

    cv_indices = [indices[i*fold_size:(i+1)*fold_size] for i in range(K)]
    
    # Cross-validation
    cv_stats = [_calculate_cv(X_train, lamb, feature_std, restarts, tol, tol_z, 
                              max_iter, cv_indices, ncores) 
                              for lamb in s_choices]
    best_s = s_choices[np.argmin([cv_m for (cv_m, _) in cv_stats])]

    best_fit = _multi_start(X_train, best_s, feature_std, restarts=restarts, tol=tol, 
                            tol_z=1e-3, max_iter=max_iter, ncores=ncores)

    return {'best_s': best_s, 
            's_choices': s_choices, 
            'runs': cv_stats, 
            'CPCs': best_fit['U'], 
            'SLs': best_fit['V'], 
            'd': best_fit['d'],
            'rss': best_fit['rss'],
            'features': features
            }

def predict_time(X_test, cv_results, true_times=None, period=24.0):
    """_summary_

    Parameters
    ----------
    X_test : ndarray
        preprocessed test data matrix 
    cv_results : dict
        output of the cross-validation on the test data
    true_times : array, optional
        known reference times for the samples to compare the 
        reconstruction against, by default None
    period : double, optional
        period of the underlying rhythm, by default 24.0

    Returns
    -------
    dict
        _description_
    """    
    V = cv_results['SLs']
    pred_phase, mape_value = calculate_mape(X_test @ V, true_times, 
                                            period=period)

    return {'phase': pred_phase, 
            'MAPE': mape_value, 
            'true_times': true_times} | cv_results

def calculate_mape(Y, true_times=None, period=24.0):
    """Calculate sample phase and median absolute position error, if 
    true sample time provided.

    Parameters
    ----------
    Y : ndarray
        2D array consisting of the two sparse cyclic principal 
        components to estimate sample times or, 1D array consisting of 
        estimated sample times (normalized by period)
    true_times : ndarray, optional
        1D array of true/reference sample times for each sample in data, 
        by default None
    period : double, optional
        period of the underlying rhythm, by default 24.0

    Returns
    -------
    (ndarray, float)
        tuple containing the estimated phases of the samples and the 
        median absolute error if true_times and period provided
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
            diff_offsets = [(np.median(np.abs(_delta(scaled_time 
                                                     - adjusted_opt_angles, 
                                                     d))), d) 
                                        for d in np.arange(-0.5, 0.5, 0.005)]
            best_offset_ind = np.argmin([offset 
                                            for (offset, _) in diff_offsets])
            mape_value = diff_offsets[best_offset_ind][0]
            adjusted_opt_angles = adjusted_opt_angles \
                                    + diff_offsets[best_offset_ind][1]
        except RuntimeWarning:
            mape_value = np.nan
    else:
        mape_value = np.nan
        adjusted_opt_angles = acw_angles
    return (adjusted_opt_angles, mape_value)

# INTERNAL FUNCTIONS

def _calculate_cv(X, s, feature_std, restarts, tol, tol_z, max_iter, cv_indices, 
                  ncores):
    rss_cv = list()
    mask_size = list()
    count_nan = 0
    for cv_ind in cv_indices:
        mask = np.zeros(X.size, dtype=bool)
        mask[cv_ind] = True
        X_ = np.ma.array(X, copy=True, mask=np.reshape(mask, X.shape))
        decomp = _multi_start(X_, s, feature_std, restarts, tol, tol_z,
                              max_iter, ncores)
        rss_cv.append(decomp['cv_err']) 
        count_nan = count_nan + np.isnan(decomp['cv_err'])
        mask_size.append(np.sum(mask))
    if count_nan > len(cv_indices)/2:
        warn("Too many runs did not converge. CV results might be unreliable."
        "Try increasing max_iter.")
    rss_cv = np.ma.array(rss_cv, mask = np.isnan(rss_cv))
    mask_size = np.ma.array(mask_size, mask = np.isnan(rss_cv))
    mean_rss = rss_cv/mask_size
    return(rss_cv.sum()/mask_size.sum(), 
           mean_rss.std()/np.sqrt(len(cv_indices)))

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
