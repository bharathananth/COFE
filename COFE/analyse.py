"""Functions to apply sparse cyclic PCA to high-dimensional data sets.

This module contains functions to process data, analyse data using sparse 
loading vectors, and analyse data using SPCA principal components.
"""
import numpy as np
from scipy.linalg import norm
from warnings import warn
from scipy.interpolate import interp1d
from joblib import delayed, Parallel
from COFE.scpca import sparse_cyclic_pca_masked, sparse_cyclic_pca
import anndata as ad

def preprocess_data(adata, mean_threshold=None, scaling_threshold=None, 
                 impute=None, scale=True):
    """Function to preprocess data prior to analysis including 
    centering, scaling and imputing.

    Parameters
    ----------
    adata : AnnData object
        matrix of features across samples.
    mean_threshold : float, optional
        minimum mean level of features that are retained for analysis, 
        by default None
    scaling_threshold : float, optional
        reciprocal of minimum standard deviation of features that are 
        retained for analysis, by default None
    impute : float, optional
        features outside central percentile are truncated to the chosen 
        level , by default None
    scale : boolean
        should the features be scaled in addition to centering, default True

    Returns
    -------
    AnnData object
        with preprocessed training and test data, names of retained
          features, standard deviation of retained features in raw data

    Raises
    ------
    ValueError
        if mean_threshold is not int or float
    ValueError
        if scaling_threshold is not int or float
    """    
    
    adata = adata[np.argsort(~adata.obs["train"]),:] 
    adata.raw = adata
    X_train_ = adata.X[adata.obs["train"], :].copy()
    X_test_ = adata.X[~adata.obs["train"], :].copy() if any(adata.obs["train"]) else None

    # Imputing extreme values
    if impute is not None:
        if isinstance(impute, float) and impute < 100.0 and impute > 0.0:
            upper_bound = np.percentile(X_train_, 100-impute/2, axis=0, 
                                        keepdims=True)
            lower_bound = np.percentile(X_train_, impute/2, axis=0, 
                                        keepdims=True)
            X_train_ = np.maximum(np.minimum(X_train_, upper_bound), 
                                  lower_bound)
            if X_test_ is not None:
                X_test_ = np.maximum(np.minimum(X_test_, upper_bound), 
                                  lower_bound)
                adata.X = np.vstack((X_train_, X_test_))
            else:
                adata.X = X_train_
    
    adata.var["mean"] = X_train_.mean(axis=0)
    adata.var["std"] = X_train_.std(axis=0)

    if mean_threshold is not None:
        if isinstance(mean_threshold, (int, float)):
            adata = adata[:, adata.var["mean"]>=mean_threshold].copy()
        else:
            raise ValueError("mean_threshold must be a float")

    if scaling_threshold is not None:
        if isinstance(mean_threshold, (int, float)):
            adata = adata[:, adata.var["std"]>=1/scaling_threshold].copy()
        else:
            raise ValueError("scaling_threshold must be a float")

    # Always center data to have zero mean
    adata.X = adata.X - adata.var["mean"].to_numpy()[None, :]

    # Standardise every gene series to have variance of 1
    if scale:
        adata.X = adata.X/adata.var["std"].to_numpy()[None, :]
    
    return adata

def cross_validate(adata, s_choices, scale_by_features=False, K=5, repeats=3, 
                    restarts=5, tol=1e-3, tol_z=1e-6, max_iter=400, ncores=None):
    """Calculate the optimal choice of sparsity threshold 's' and the 
    cyclic ordering for the best 's'
    
    Parameters
    ----------
    adata : AnnData object
        with preprocessed training data matrix 
    s_choices : array or list or None
        different values of l1 sparsity threshold to compare. If None 
        then directly computes non-sparse solution.
    scale_by_features: boolean
        whether to weigh features by their the st. dev. for the random initial 
        conditions each restart, by default False
    K : int, optional
        number of folds used for cross-validation
    repeats : int, optional
        the number of different random repetition (of the splits) in 
        K-fold cross-validation, by default 3
    restarts : int, optional
        the number of random initial conditions to begin alternating 
        maximization from, by default 5
    tol : double, optional
        convergence threshold for the alternating maximization of 
        cyclic PCA, by default 1e-3
    tol_z : double, optional
        convergence threshold for imputation of missing and cross 
        validation, by default 1e-6
    max_iter : int, optional
        maximum number of iterations of the alternating maximization, 
        by default 400
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
    if s_choices is None:
        best_s = None
        cv_stats = None
    else:
        adata_train = adata[adata.obs["train"], :].copy()
        cv_indices = _shuffled_checkerboard(adata_train.shape, K, repeats)
        
        # Cross-validation
        if ncores is None:
            runs = [_calculate_cv(adata=adata_train, 
                                  s=lamb, 
                                  tol=tol, 
                                  tol_z=tol_z, 
                                  max_iter=max_iter, 
                                  cv_ind=cv_ind,
                                  scale_by_features=scale_by_features
                                )
                                        for lamb in s_choices
                                        for cv_ind in cv_indices 
                                        for _ in range(restarts)]
        else:
            runs = Parallel(n_jobs=ncores, backend="loky", 
                            backend_kwargs=dict(inner_max_num_threads=1))(
                delayed(_calculate_cv)(
                    adata=adata, 
                    s=lamb, 
                    tol=tol, 
                    tol_z=tol_z, 
                    max_iter=max_iter, 
                    cv_ind=cv_ind,
                    scale_by_features=scale_by_features
                    ) 
                    for lamb in s_choices 
                    for cv_ind in cv_indices 
                    for _ in range(restarts))

        rss = np.array([r.uns["scpca"]["rss"] for r in runs]).reshape(len(runs)//restarts, 
                                                            restarts)
        
        ind  = np.arange(len(runs)//restarts)*restarts + np.argmin(rss, axis=1)

        best_runs = [runs[i] for i in ind]
        
        rss_cv = [r.uns["scpca"]['cv_err'] for r in best_runs]

        rss_cv = np.ma.array(rss_cv, 
                            mask = np.isnan(rss_cv)).reshape((len(rss_cv)//(repeats*K), repeats, K))

        count_nan = np.sum(np.isnan(rss_cv), axis=(1,2))

        if any(count_nan > len(cv_indices)/4):
            warn("Too many runs did not converge for s={}. CV results might be "
        "unreliable. Try increasing max_iter or reducing the tolerances."
        .format(s_choices[count_nan > len(cv_indices)/4]))

        mask_size = np.array([m.shape[0] for m in cv_indices]).reshape(repeats, K)
        
        mask_size = np.ma.array(np.repeat(mask_size[None, :, :], 
                                    s_choices.shape[0], axis=0), 
                                mask = np.isnan(rss_cv))
                        
        mean_rss = rss_cv.sum(axis=2)/mask_size.sum(axis=2)

        cv_stats = list(zip(mean_rss.mean(axis=1), mean_rss.std(axis=1)))
        
        best_s = s_choices[np.argmin(mean_rss.mean(axis=1))]

        best_fit = _multi_start(adata=adata_train, 
                                s=best_s, 
                                restarts=restarts,
                                tol=tol, 
                                tol_z=tol_z, 
                                max_iter=max_iter, 
                                ncores=ncores,
                                scale_by_features=scale_by_features)
        best_fit.uns["scpca"].update({'best_s': best_s, 
            's_choices': s_choices, 
            'runs': [list(stats) for stats in cv_stats]})
        
        X_test = adata[~adata.obs["train"], :].X
        PCs_test = X_test @ best_fit.varm["PCs"]
        y_circ = np.sqrt((PCs_test ** 2).sum(axis=1, keepdims=True))
        PCs_test = PCs_test/y_circ
        
        adata.varm["PCs"] = best_fit.varm["PCs"]
        adata.obsm["X_scpca"] = np.vstack((best_fit.obsm["X_scpca"], PCs_test))
        adata.uns["scpca"] = best_fit.uns["scpca"]

    return adata

def predict_time(X_test, cv_results, true_times=None, period=24.0):
    """Predict the phase for test data using the training results from the 
    cross validation run on the training data

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
    AnnData object with the cross-validation results augmented with following information
    {
        'phase': ndarray
            the estimated phases of the samples modulo 1
         'MAPE': float
            median absolute error of prediction if true_times are provided
         'FCC': float
            Fisher (circular) correlation coefficient if true_times are provided
         'true_times': ndarray
            1D array of true/reference sample times for each sample in data
    }
    """    
    V = cv_results['SLs']
    pred_phase, mape_value, circ_corr = calculate_mape(X_test @ V, true_times, 
                                            period=period)

    return {'phase': pred_phase, 
            'MAPE': mape_value, 
            'FCC': circ_corr,
            'true_times': true_times} | cv_results

def calculate_mape(adata, train=False):
    """Calculate sample phase and median absolute position error, if 
    true sample time provided.

    Parameters
    ----------
    adata : AnnData object
        with the results of COFE (SCPCA) run.
    train : boolean, optional
        whether the MAPE and angles must be computed for the train or test data, 
        default False

    Returns
    -------
    AnnData object
        tuple containing the estimated phases of the samples and the 
        median absolute error if true_times and period provided
    """    
    if "X_scpca" in adata.obsm.keys():
        # Scaled angular positions
        Y = adata.obsm["X_scpca"]
        ind = adata.obs["train"] if train else ~adata.obs["train"]
        angles = _scaled_angles(Y[:, 0], Y[:, 1])
    
        mape_value = np.nan

        if "time" in adata.obs_keys():
            # Scaled time values
            scaled_time = adata.obs["time"]/adata.uns["period"] % 1

            diff_offsets_cw = [(np.median(np.abs(_delta(scaled_time[ind] 
                                                        - angles[ind], 
                                                        d))), d) 
                                        for d in np.arange(-0.5, 0.5, 0.005)]

            diff_offsets_acw = [(np.median(np.abs(_delta(scaled_time[ind] 
                                                        + angles[ind], 
                                                        d))), d) 
                                        for d in np.arange(-0.5, 0.5, 0.005)]
            
            best_offset_ind_cw = np.argmin(list(zip(*diff_offsets_cw))[0]) 
            
            best_offset_ind_acw = np.argmin(list(zip(*diff_offsets_acw))[0])

            if diff_offsets_cw[best_offset_ind_cw][0]< \
                                diff_offsets_acw[best_offset_ind_acw][0]:
                adjusted_opt_angles = (angles +
                                    diff_offsets_cw[best_offset_ind_cw][1]) % 1
                mape_value = diff_offsets_cw[best_offset_ind_cw][0]
            else: 
                adjusted_opt_angles = (-angles +
                                diff_offsets_acw[best_offset_ind_acw][1]) % 1
                mape_value = diff_offsets_acw[best_offset_ind_acw][0]
        else:
            adjusted_opt_angles = angles
            
        adata.uns["scpca"].update({"MAPE": mape_value})
        adata.obs["phase"] = adjusted_opt_angles * adata.uns["period"]

    return adata

# INTERNAL FUNCTIONS

def _shuffled_checkerboard(size, K, repeats):
    nrow, ncol = size[0], size[1]
    rng = np.random.default_rng()
    return_list = list()
    for r in range(repeats):
        row_permute = rng.permutation(nrow)
        col_permute = rng.permutation(ncol)
        for i in range(K):
            row_one = np.arange(i, nrow, step=K)
            row_ind = np.array([(row_one + nrow*i + i) % nrow 
                                for i in range(ncol)], dtype='int').flatten()
            col_ind = np.array([i*np.ones(row_one.shape) 
                                for i in range(ncol)], dtype='int').flatten()
            return_list.append(np.ravel_multi_index((row_permute[row_ind], 
                                                     col_permute[col_ind]), 
                                                     (nrow, ncol)))
    return(return_list)
    
def _fischer_circ_corr(phi_1, phi_2):
    phi_1 = phi_1 % 1
    phi_2 = phi_2 % 1
    ph_1 = 2*np.pi*phi_1
    ph_2 = 2*np.pi*phi_2
    mean_ph_1 = _angular_mean(np.cos(ph_1), np.sin(ph_1))
    mean_ph_2 = _angular_mean(np.cos(ph_2), np.sin(ph_2))
    num = np.sum(np.sin(ph_1 - mean_ph_1)*np.sin(ph_2 - mean_ph_2))
    den = np.sqrt(np.sum(np.sin(ph_1 - mean_ph_1)**2) * \
                                    np.sum(np.sin(ph_2 - mean_ph_2)**2))
    rho = num/den
    return(rho)

def _calculate_cv(adata, s, tol, tol_z, max_iter, cv_ind, scale_by_features):
    mask = np.zeros(adata.n_obs*adata.n_vars, dtype=bool)
    mask[cv_ind] = True
    if np.any(np.reshape(mask, adata.n_obs*adata.n_vars).sum(axis=0) == adata.n_obs):
        print("all masked column")
    mask=np.reshape(mask, adata.shape)
    adata = sparse_cyclic_pca_masked(adata=adata, 
                                      mask=mask,
                                      s=s, 
                                      tol=tol, 
                                      tol_z=tol_z, 
                                      max_iter=max_iter, 
                                      scale_by_features=scale_by_features)
    return(adata)

def _multi_start(adata, s, scale_by_features, restarts, tol, tol_z, max_iter, ncores):
    if ncores is None:
        runs = [sparse_cyclic_pca(adata=adata, 
                                s=s, 
                                tol=tol, 
                                max_iter=max_iter, 
                                scale_by_features=scale_by_features) 
                                for _ in range(restarts)]
    else:
        runs = Parallel(n_jobs=ncores)(
            delayed(sparse_cyclic_pca)(adata=adata, 
                                        s=s, 
                                        tol=tol, 
                                        max_iter=max_iter, 
                                        scale_by_features=scale_by_features) 
                                        for _ in range(restarts))
    
    ind = np.argmin([r.uns["scpca"]["rss"] for r in runs])
    return runs[ind]

def _scaled_angles(x, y):
    return (np.arctan2(y, x)/(2*np.pi)) % 1

def _delta(x, y):
    return ((x - y + 1/2) % 1) - 1/2

def _angular_mean(x, y):
    return np.arctan2(np.sum(y), np.sum(x))

def _angular_spread(x, y):
    return x.shape[0] - np.sqrt(np.sum(x) ** 2 + np.sum(y) ** 2)