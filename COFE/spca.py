"""Functions to carry out SPCA.

This module contains functions to find sparse loading vectors and SPCA
principal components for a particular data set.
"""
import numpy as np
from numpy.linalg.linalg import norm
from warnings import warn

def coupled_spca(X, t=float('inf'), tol=1e-3, max_iter=200, feature_std=None, verbose=False):
    """Generates a pair of sparse loading vectors that satisfy the circular constraints.

    Args:
        X (ndarray): 2D numpy array with features along columns and samples
            along rows. All columns should be centered to have zero mean.
        t (float): l1 norm constraint that determines level of sparsity. Defaults to 1e6 (no constraint).
        tol (float): Measure of SPCA algorithm convergence. Defaults to 1e-3.
        max_iter (int): The maximum of iterations to run in the biconvex optimization. Defaults to 250.
        
    Returns:

        Dict with sparse loading vector matrix and orthogonal SPCA principal 
        components matrix::

            {
                'V1': 2D numpy array with sparse right eigenvectors as columns,
                'U2': 2D numpy array with circular left eigenvectors as columns
            }
    """
    rng = np.random.default_rng()    
    v_1 = rng.standard_normal(size = (X.shape[1],1))
    if feature_std is not None:
        if feature_std.shape[0] != X.shape[1]:
            raise ValueError("Feature standard deviations not the same size as feature.")
        v_1 = v_1 * feature_std
    v_1 = v_1/np.linalg.norm(v_1, ord=2)
    v_2 = rng.standard_normal(size = (X.shape[1],1))
    if feature_std is not None:
        if feature_std.shape[0] != X.shape[1]:
            raise ValueError("Feature standard deviations not the same size as feature.")
        v_2 = v_2 * feature_std
    v_2 = v_2/np.linalg.norm(v_2, ord=2)

    phi = 2*np.pi*rng.random(size = (X.shape[0],1))
    u_1 = np.cos(phi)
    u_2 = np.sin(phi)
    
    count = 0
    score = -1.0
    
    while count < max_iter:
        y_1 = X @ v_1
        y_2 = X @ v_2

        u_1_n = y_1/np.sqrt(y_1 **2 + y_2 ** 2)
        u_2_n = y_2/np.sqrt(y_1 **2 + y_2 ** 2)

        u_1_diff = np.linalg.norm(u_1_n - u_1, ord=np.inf)
        u_2_diff = np.linalg.norm(u_2_n - u_2, ord=np.inf)

        u_1 = u_1_n
        u_2 = u_2_n

        XTu_1 = X.T @ u_1
        S_XTu_1 = _opt_thresh(XTu_1, t)
        v_1_n = S_XTu_1/np.linalg.norm(S_XTu_1, ord=2)
        XTu_2 = X.T @ u_2
        S_XTu_2 = _opt_thresh(XTu_2, t)
        v_2_n = S_XTu_2/np.linalg.norm(S_XTu_2, ord=2)

        v_1_diff = np.linalg.norm(v_1_n - v_1, ord=np.inf)
        v_2_diff = np.linalg.norm(v_2_n - v_2, ord=np.inf)

        v_1 = v_1_n
        v_2 = v_2_n

        if u_1_diff < tol and u_2_diff < tol and v_1_diff < tol and v_2_diff < tol:
            score = np.asscalar(u_1.T @ X @ v_1 + u_2.T @ X @ v_2)
            break
        
        count += 1
    
    if count == max_iter and verbose:
        warn("Iterations did not converge to within the tolerance.")

    return {'V': np.hstack((v_1, v_2)), 'U': np.hstack((u_1, u_2)), 'converged': (count<max_iter), 'score': score}

def _opt_thresh(x, t):
    """Finds the optimal soft-thresholding to satisfy both l1 and l2 contraints

    Args: 
        x (ndarray): 1D numpy array to be soft thresholded
        t (float): the desired l1 constraint

    Returns:

        ndarray that has been soft-thresholded 
        (Note: the array needs to be l2 normalized to satisfy desired l1)

    Reference:
        Guillemot et al. (2019) A constrained singular value decomposition method 
        that integrates sparsity and orthogonality
    """
    x_tilde = np.sort(np.abs(x.flatten()))[::-1]
    if np.linalg.norm(x/np.linalg.norm(x, ord=2), ord = 1) <= t:
        lamb = 0
    else:
        psi = lambda c: np.linalg.norm(_soft_thresh(x_tilde, c), ord=1)/np.linalg.norm(_soft_thresh(x_tilde, c), ord=2)
        low = 1
        high = x_tilde.shape[0] - 1
        while (low < high - 1):
            ind = low + (high - low) // 2
            if (psi(x_tilde[ind])<t):
                low = ind
            else:
                high = ind

        delta = np.linalg.norm(_soft_thresh(x_tilde, x_tilde[low]), ord = 2)/(low + 1) * ((t * np.sqrt((low + 1 - psi(x_tilde[low])**2)/(low + 1 - t**2))) - psi(x_tilde[low]))
        lamb = max(x_tilde[low] - delta, 0)
    
    return _soft_thresh(x, lamb)

def _soft_thresh(x, l):
    """Soft-thresholding function.

    Method to reduce absolute values of vector entries by a specifc quantity.

    Args:
        x (ndarray): 1D vector
        l (float): Positive quantity by which to reduce absolute value of
            each entry in x.

    Returns:

        ndarray of adjusted input vector.

    Raises:
        ValueError: If thresholding quantity is not positive.
    """
    if l < 0:
        raise ValueError("Thresholding quantity must be non-negative.")
    x_tilde = np.abs(x)
    return np.sign(x) * np.maximum(x_tilde - l, 0)