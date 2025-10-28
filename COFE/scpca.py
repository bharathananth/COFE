"""Implementation of standard and sparse cyclic PCA.

This module contains functions to find sparse loading vectors and SPCA
principal components for a particular data set.
"""

import numpy as np
from scipy.linalg import norm
from warnings import warn


def sparse_cyclic_pca(
    adata, s=None, tol=1e-6, max_iter=300, scale_by_features=False, seed=None
):
    """Generates a pair of loading vectors and cyclic principal
    components that satisfy specific sparsity constraint.

    Parameters
    ----------
    adata : AnnData object
        data matrix
    s : float, optional
        l1 norm constraint that determines level of sparsity, by default
        None, i.e., no sparsity
    tol : float, optional
        convergence criterion for the iterative algorithm, by default 1e-3
    max_iter : int, optional
        maximum number of iterations to look for convergence, by default 300
    scale_by_features : boolean, optional
        whether to weigh features according to their st. dev. for their random
        initial conditions, by default False
    seed : integer, optional
        seed to make the randomized algorithm deterministic, by default None

    Returns
    -------
    AnnData object augmented with sparse right eigenvectors, circular left
    eigenvectors, scale factor for the outer product approximation along with
    finall rss after convergence and final score

    Raises
    ------
    ValueError
        when feature weights for initialization are not the same size as
         feature
    """
    N = adata.n_obs
    p = adata.n_vars

    sparsify = False if s is None else True

    rng = np.random.default_rng(seed=seed)
    v_1 = rng.laplace(size=(p, 1))
    if scale_by_features:
        v_1 = v_1 * adata.var["std"].to_numpy()
    Sv_1 = _opt_thresh(v_1, s) if sparsify else v_1.copy()
    v_1 = Sv_1 / norm(Sv_1, ord=2)
    v_2 = rng.laplace(size=(p, 1))
    if scale_by_features:
        v_2 = v_2 * adata.var["std"].to_numpy()
    Sv_2 = _opt_thresh(v_2, s) if sparsify else v_2.copy()
    v_2 = Sv_2 / norm(Sv_2, ord=2)

    phi = rng.uniform(low=0.0, high=2 * np.pi, size=(N, 1))
    u_1 = np.cos(phi)
    u_2 = np.sin(phi)

    X = adata.X
    X_T = X.T
    d = (u_1.T @ X @ v_1 + u_2.T @ X @ v_2).item() / N

    count = 0
    score = (2 * d * u_1.T @ X @ v_1 + 2 * d * u_2.T @ X @ v_2 - N * d**2).item()
    rss = norm(X, ord="fro") ** 2

    while count < max_iter:
        y_1 = X @ v_1
        y_2 = X @ v_2

        y_circ = np.sqrt(y_1**2 + y_2**2)
        u_1 = y_1 / y_circ
        u_2 = y_2 / y_circ

        XTu_1 = X_T @ u_1
        S_XTu_1 = _opt_thresh(XTu_1, s) if sparsify else XTu_1.copy()
        v_1 = S_XTu_1 / norm(S_XTu_1, ord=2)
        XTu_2 = X_T @ u_2
        S_XTu_2 = _opt_thresh(XTu_2, s) if sparsify else XTu_2.copy()
        v_2 = S_XTu_2 / norm(S_XTu_2, ord=2)

        d = (u_1.T @ X @ v_1 + u_2.T @ X @ v_2).item() / N

        score_new = (
            2 * d * u_1.T @ X @ v_1 + 2 * d * u_2.T @ X @ v_2 - N * d**2
        ).item()
        err = np.abs(score_new - score) / score
        score = score_new

        if err <= tol:
            rss = norm(X - d * (u_1 @ v_1.T + u_2 @ v_2.T), ord="fro") ** 2
            break

        count += 1

    adata.obsm["X_scpca"] = np.hstack((u_1, u_2))
    adata.varm["PCs"] = np.hstack((v_1, v_2))
    adata.uns["scpca"] = {
        "d": d,
        "converged": (count < max_iter),
        "rss": rss,
        "score": score,
    }

    return adata


def sparse_cyclic_pca_masked(
    adata,
    mask,
    s=None,
    tol=1e-3,
    tol_z=1e-6,
    max_iter=300,
    scale_by_features=False,
    seed=None,
):
    """Generates a pair of loading vectors and cyclic principal
    components that satisfy specific sparsity constraint for data with
    missing values. The code then imputes these missing values.

    Parameters
    ----------
    adata : AnnData object
       data matrix
    s : float, optional
        l1 norm constraint that determines level of sparsity, by default
         float('inf')
    tol : float, optional
        convergence criterion for the iterative algorithm, by default
        1e-6
    tol_z : float, optional
        convergence criterion for the iterative algorithm, by default
        1e-7
    max_iter : int, optional
        maximum number of iterations to look for convergence, by default
        300
    scale_by_features : boolean, optional
        whether to weigh features according to their st. dev. for their random
        initial conditions, by default False
    seed : integer, optional
        seed to make the randomized algorithm deterministic, by default None

    Returns
    -------
    AnnData object augmented with
    {
        'V': ndarray
            sparse right eigenvectors as columns,
        'U': ndarray
            circular left eigenvectors as columns,
        'd': double
            scale factor for the outer product approximation,
        'converged': bool
            if the iterations converged
        'rss': float
            final rss after convergence. With no convergence, rss is
                set to -1.
        'cv_err': float
            cross validation error when input is data with test values
            masked
        'X_imputed': ndarray
            the input data with missing values imputed
    }

    Raises
    ------
    TypeError
        input data matrix does not have missing values.
    ValueError
        when feature weights for initialization are not the same size as
         feature
    """

    N = adata.n_obs
    p = adata.n_vars

    sparsify = False if s is None else True
    
    X_zeroed = np.where(mask, 0.0, adata.X)
    masked_mean = X_zeroed.sum(axis=0)/np.sum(~mask, axis=0)

    Z = np.where(mask, masked_mean[None,:], adata.X)

    rng = np.random.default_rng(seed=seed)
    v_1 = rng.laplace(size=(p, 1))
    if scale_by_features:
        v_1 = v_1 * adata.var["std"].to_numpy()
    Sv_1 = _opt_thresh(v_1, s) if sparsify else v_1.copy()
    v_1 = Sv_1 / norm(Sv_1, ord=2)
    v_2 = rng.laplace(size=(p, 1))
    if scale_by_features:
        v_2 = v_2 * adata.var["std"].to_numpy()
    Sv_2 = _opt_thresh(v_2, s) if sparsify else v_2.copy()
    v_2 = Sv_2 / norm(Sv_2, ord=2)

    phi = rng.uniform(low=0.0, high=2 * np.pi, size=(N, 1))
    u_1 = np.cos(phi)
    u_2 = np.sin(phi)

    d = (u_1.T @ Z @ v_1 + u_2.T @ Z @ v_2).item() / N

    score = (2 * d * u_1.T @ Z @ v_1 + 2 * d * u_2.T @ Z @ v_2 - N * d**2).item()

    rss = norm(Z, ord="fro") ** 2
    count = 0
    Z_imputed = Z.copy()

    while count < max_iter:
        Z_T = Z.T
        y_1 = Z @ v_1
        y_2 = Z @ v_2

        y_circ = np.sqrt(y_1**2 + y_2**2)
        u_1 = y_1 / y_circ
        u_2 = y_2 / y_circ

        XTu_1 = Z_T @ u_1
        S_XTu_1 = _opt_thresh(XTu_1, s) if sparsify else XTu_1.copy()
        v_1 = S_XTu_1 / norm(S_XTu_1, ord=2)
        XTu_2 = Z_T @ u_2
        S_XTu_2 = _opt_thresh(XTu_2, s) if sparsify else XTu_2.copy()
        v_2 = S_XTu_2 / norm(S_XTu_2, ord=2)

        d = (u_1.T @ Z @ v_1 + u_2.T @ Z @ v_2).item() / N

        score_new = (
            2 * d * u_1.T @ Z @ v_1 + 2 * d * u_2.T @ Z @ v_2 - N * d**2
        ).item()
        err = np.abs(score_new - score) / score
        score = score_new

        if err <= tol:
            Z_imputed = d * (u_1 @ v_1.T + u_2 @ v_2.T)
            Z = np.where(mask, Z_imputed, Z)
            rss_new = norm(Z - Z_imputed, ord="fro") ** 2
            err_z = np.abs(rss_new - rss) / rss
            rss = rss_new
            if err_z <= tol_z:
                E = adata.X - Z
                cv_err = norm(E, ord="fro") ** 2
                break
        count += 1

    if count == max_iter:
        cv_err = np.nan

    adata.obsm["X_scpca"] = np.hstack((u_1, u_2))
    adata.varm["PCs"] = np.hstack((v_1, v_2))
    adata.uns["scpca"] = {
        "d": d,
        "converged": (count < max_iter),
        "rss": rss,
        "cv_err": cv_err,
    }

    return adata


def _opt_thresh(x, s):
    """Finds the optimal soft-thresholding to satisfy both l1 and l2 contraints

    Parameters
    ----------
    x : ndarray
        1D numpy array to be soft thresholded
    s : float
        the desired l1 constraint

    Returns
    -------
    ndarray
        input array that has been soft-thresholded
        (Note: the array needs to be l2 normalized to satisfy desired l1)

    Reference
    ---------
        Guillemot et al. (2019) A constrained singular value
        decomposition method that integrates sparsity and orthogonality
    """

    x_tilde = np.sort(np.fabs(x), axis=None)[::-1]
    if norm(x / norm(x, ord=2), ord=1) <= s:
        return x
    else:

        def psi(c):
            x_thresholded = _soft_thresh(x_tilde, c)
            return norm(x_thresholded, ord=1) / norm(x_thresholded, ord=2)

        low = 1
        high = x_tilde.shape[0] - 1
        while low < high - 1:
            ind = low + (high - low) // 2
            if psi(x_tilde[ind]) < s:
                low = ind
            else:
                high = ind
        psi_low = psi(x_tilde[low])
        delta = (
            norm(_soft_thresh(x_tilde, x_tilde[low]), ord=2, check_finite=False)
            / (low + 1)
            * ((s * np.sqrt((low + 1 - psi_low**2) / (low + 1 - s**2))) - psi_low)
        )
        l = max(x_tilde[low] - delta, 0)
        return _soft_thresh(x, l)


def _soft_thresh(x, l):
    """Soft-thresholding function.
    Method to reduce absolute values of vector entries by a specifc
    quantity.

    Parameters
    ----------
    x : ndarray
        1D vectors of values
    l : float
        positive quantity by which to reduce absolute value of each
        entry of x

    Returns
    -------
    ndarray
        adjusted input vector

    Raises
    ------
    ValueError
        if thresholding quantity is not positive
    """

    if l < 0:
        raise ValueError("Thresholding quantity must be non-negative.")
    x_tilde = np.abs(x)
    return np.sign(x) * np.maximum(x_tilde - l, 0)
