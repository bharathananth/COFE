"""Functions to carry out the standard and sparse Cyclic PCA.

This module contains functions to find sparse loading vectors and SPCA
principal components for a particular data set.
"""
import numpy as np
from scipy.linalg import norm
from warnings import warn

def sparse_cyclic_pca(X, lamb=None, tol=1e-3, max_iter=100, feature_std=None):
    """Generates a pair of sparse loading vectors that satisfy the circular constraints.

    Parameters
    ----------
    X : ndarray
       data matrix with features along columns and samples along rows.
    lamb : float, optional
        l1 norm constraint that determines level of sparsity, by default float('inf')
    tol : float, optional
        convergence criterion for the iterative algorithm, by default 1e-3
    max_iter : int, optional
        maximum number of iterations to look for convergence, by default 100
    feature_std : array-like, optional
        weights for the different features that determine the st. dev. of the random initial conditions, by default None
    verbose : bool, optional
        whether to print convergence information, by default False

    Returns
    -------
    dict
        {
            'V': ndarray 
                sparse right eigenvectors as columns,
            'U': ndarray
                circular left eigenvectors as columns,
            'converged': bool
                if the iterations converged
            'rss': float
                final rss after convergence. With no convergence, rss is set to -1.
        }

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """        
    N = X.shape[0]

    sparsify = False if lamb is None else True

    rng = np.random.default_rng()    
    v_1 = rng.laplace(size=(X.shape[1],1))
    if feature_std is not None:
        if feature_std.shape[0] != X.shape[1]:
            raise ValueError("Feature standard deviations not the same size as feature.")
        v_1 = v_1 * feature_std
    Sv_1 = _opt_thresh(v_1, lamb) if sparsify else v_1.copy()
    v_1 = Sv_1/norm(Sv_1, ord=2)
    v_2 = rng.laplace(size = (X.shape[1],1))
    if feature_std is not None:
        if feature_std.shape[0] != X.shape[1]:
            raise ValueError("Feature standard deviations not the same size as feature.")
        v_2 = v_2 * feature_std
    Sv_2 = _opt_thresh(v_2, lamb) if sparsify else v_2.copy()
    v_2 = Sv_2/norm(Sv_2, ord=2)

    phi = rng.uniform(low=0.0, high=2*np.pi, size=(X.shape[0],1))
    u_1 = np.cos(phi)
    u_2 = np.sin(phi)

    d = (u_1.T @ X @ v_1 + u_2.T @ X @ v_2).item()/N
    X_T = X.T
    
    count = 0
    rss = -1.0
    
    while count < max_iter:
        y_1 = X @ v_1
        y_2 = X @ v_2

        y_circ = np.sqrt(y_1 **2 + y_2 ** 2)
        u_1_n = y_1/y_circ
        u_2_n = y_2/y_circ

        u_1_diff = norm(u_1_n - u_1, ord=2) \
                    /norm(u_1, ord=2)
        u_2_diff = norm(u_2_n - u_2, ord=2) \
                    /norm(u_2, ord=2)

        u_1 = u_1_n.copy()
        u_2 = u_2_n.copy()

        XTu_1 = X_T @ u_1
        S_XTu_1 = _opt_thresh(XTu_1, lamb) if sparsify else XTu_1.copy()
        v_1_n = S_XTu_1/norm(S_XTu_1, ord=2)
        XTu_2 = X_T @ u_2
        S_XTu_2 = _opt_thresh(XTu_2, lamb) if sparsify else XTu_2.copy()
        v_2_n = S_XTu_2/norm(S_XTu_2, ord=2)

        v_1_diff = norm(v_1_n - v_1, ord=2)
        v_2_diff = norm(v_2_n - v_2, ord=2)

        v_1 = v_1_n.copy()
        v_2 = v_2_n.copy()

        d_n = (u_1.T @ X @ v_1 + u_2.T @ X @ v_2).item()/N
        d_diff = np.abs(d_n - d)/np.abs(d)
        d = d_n

        if u_1_diff < tol and u_2_diff < tol \
            and v_1_diff < tol and v_2_diff < tol and d_diff<tol:
            rss = norm(X - d * (u_1 @ v_1.T + u_2 @ v_2.T), ord='fro') ** 2
            break
        
        count += 1

    return {'V': np.hstack((v_1, v_2)), 'U': np.hstack((u_1, u_2)), 
            'd': d, 'converged': (count<max_iter), 'rss': rss}

def sparse_cyclic_pca_masked(X, lamb=None, tol=1e-3, tol_z=1e-5, max_iter=100, 
                             feature_std=None):
    """Generates a pair of sparse loading vectors that satisfy the circular constraints.

    Parameters
    ----------
    X : ndarray
       data matrix with features along columns and samples along rows.
    lamb : float, optional
        l1 norm constraint that determines level of sparsity, by default float('inf')
    tol : float, optional
        convergence criterion for the iterative algorithm, by default 1e-3
    max_iter : int, optional
        maximum number of iterations to look for convergence, by default 100
    feature_std : array-like, optional
        weights for the different features that determine the st. dev. of the random initial conditions, by default None
    verbose : bool, optional
        whether to print convergence information, by default False

    Returns
    -------
    dict
        {
            'V': ndarray 
                sparse right eigenvectors as columns,
            'U': ndarray
                circular left eigenvectors as columns,
            'converged': bool
                if the iterations converged
            'rss': float
                final rss after convergence. With no convergence, rss is set to -1.
        }

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    N = X.shape[0]

    sparsify = False if lamb is None else True

    Z = X.data.copy()
    rng = np.random.default_rng() 
    Z[X.mask] = rng.standard_normal(size=np.ma.count_masked(X))
    v_1 = rng.laplace(size=(Z.shape[1],1))
    if feature_std is not None:
        if feature_std.shape[0] != Z.shape[1]:
            raise ValueError("Feature standard deviations not the same size as feature.")
        v_1 = v_1 * feature_std
    Sv_1 = _opt_thresh(v_1, lamb) if sparsify else v_1.copy()
    v_1 = Sv_1/norm(Sv_1, ord=2, check_finite=False)
    v_2 = rng.laplace(size=(Z.shape[1],1))
    if feature_std is not None:
        if feature_std.shape[0] != Z.shape[1]:
            raise ValueError("Feature standard deviations not the same size as feature.")
        v_2 = v_2 * feature_std
    Sv_2 = _opt_thresh(v_2, lamb) if sparsify else v_2.copy()
    v_2 = Sv_2/norm(Sv_2, ord=2, check_finite=False)

    phi = rng.uniform(low=0.0, high=2*np.pi, size=(Z.shape[0],1))
    u_1 = np.cos(phi)
    u_2 = np.sin(phi)

    d = (u_1.T @ Z @ v_1 + u_2.T @ Z @ v_2).item()/N

    rss = norm(Z - d*(u_1 @ v_1.T + u_2 @ v_2.T), ord='fro') ** 2
    err = 1.0
    
    while err > tol_z:
        Z_T = Z.T
        count = 0
        converged = False
        while count < max_iter:
            y_1 = Z @ v_1
            y_2 = Z @ v_2

            y_circ = np.sqrt(y_1 ** 2 + y_2 ** 2)
            u_1_n = y_1/y_circ
            u_2_n = y_2/y_circ

            u_1_diff = norm(u_1_n - u_1, ord=2) \
                        /norm(u_1, ord=2)
            u_2_diff = norm(u_2_n - u_2, ord=2) \
                        /norm(u_2, ord=2)

            u_1 = u_1_n.copy()
            u_2 = u_2_n.copy()

            XTu_1 = Z_T @ u_1
            S_XTu_1 = _opt_thresh(XTu_1, lamb) if sparsify else XTu_1.copy()
            v_1_n = S_XTu_1/norm(S_XTu_1, ord=2)
            XTu_2 = Z_T @ u_2
            S_XTu_2 = _opt_thresh(XTu_2, lamb) if sparsify else XTu_2.copy()
            v_2_n = S_XTu_2/norm(S_XTu_2, ord=2)

            v_1_diff = norm(v_1_n - v_1, ord=2)
            v_2_diff = norm(v_2_n - v_2, ord=2)

            v_1 = v_1_n.copy()
            v_2 = v_2_n.copy()

            d_n = (u_1.T @ Z @ v_1 + u_2.T @ Z @ v_2).item()/N
            d_diff = np.abs(d_n - d)/d
            d = d_n

            if u_1_diff < tol and u_2_diff < tol \
                and v_1_diff < tol and v_2_diff < tol and d_diff < tol:
                Z_imputed = d * (u_1 @ v_1.T + u_2 @ v_2.T)
                Z[X.mask] = Z_imputed[X.mask]
                converged = True
                break
            count += 1
        if converged:
            rss_new = norm(Z - Z_imputed, ord='fro') ** 2
            err = np.abs(rss_new - rss)/rss
            rss = rss_new
            E = (X.data - Z)
            cv_err = norm(E, ord='fro') ** 2
        else:
            rss = np.inf
            cv_err = np.inf
            break

    return {'V': np.hstack((v_1, v_2)), 'U': np.hstack((u_1, u_2)), 
            'd': d, 'converged': converged, 'rss': rss, 'cv_err': cv_err}

def _opt_thresh(x, lamb):
    """Finds the optimal soft-thresholding to satisfy both l1 and l2 contraints

    Args: 
        x (ndarray): 1D numpy array to be soft thresholded
        lamb (float): the desired l1 constraint

    Returns:

        ndarray that has been soft-thresholded 
        (Note: the array needs to be l2 normalized to satisfy desired l1)

    Reference:
        Guillemot et al. (2019) A constrained singular value decomposition method 
        that integrates sparsity and orthogonality
    """
    x_tilde = np.sort(np.fabs(x), axis=None)[::-1]
    if norm(x/norm(x, ord=2), 
                         ord = 1) <= lamb:
        return x
    else:
        def psi(c):
            x_thresholded = _soft_thresh(x_tilde, c)
            return norm(x_thresholded, ord=1) \
                    /norm(x_thresholded, ord=2)
        
        low = 1
        high = x_tilde.shape[0] - 1
        while (low < high - 1):
            ind = low + (high - low) // 2
            if (psi(x_tilde[ind])<lamb):
                low = ind
            else:
                high = ind
        psi_low = psi(x_tilde[low])
        delta = norm(_soft_thresh(x_tilde, x_tilde[low]), ord = 2, 
                                  check_finite=False)/(low + 1) \
                * ((lamb * np.sqrt((low + 1 - psi_low**2)/(low + 1 - lamb**2))) 
                    - psi_low)
        l = max(x_tilde[low] - delta, 0)
        return _soft_thresh(x, l)

def _soft_thresh(x, l):
    """Soft-thresholding function.
    Method to reduce absolute values of vector entries by a specifc quantity.

    Parameters
    ----------
    x : ndarray
        1D vectors of values
    l : float
        positive quantity by which to reduce absolute value of each entry of x

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
    x_tilde = np.fabs(x)
    return np.sign(x) * np.maximum(x_tilde - l, 0)