"""Functions for ellipse evaluation.

This module contains functions to calculate algebraic & geometric parameters 
for best elliptical fit, unsupervised ellipse evaluation metrics, and quantity 
that evaluates ellipse reordering potential.
"""

from matplotlib.pyplot import axis
import numpy as np
from sklearn import linear_model
from sklearn.metrics import explained_variance_score

def ellipse_metrics(x_vector, y_vector, theta):
    """Returns different error metrics on the samples for a particular elliptical fit.

    Parameters
    ----------
    x_vector : ndarray
        1D array of sample x-coordinates
    y_vector : ndarray
        1D array of sample y-coordinates
    theta : ndarray
        1D array of algebraic ellipse equation parameters [a, b, c, d, e, f]
    metric : str, optional
        metric of choice, by default 'focidistSE'

    Returns
    -------
    ndarray
        1D array of error metric for each sample or fold

    Raises
    ------
    ValueError
        if x_vector and y_vector are unequal in length
    ValueError
        if length(theta) is not 6
    """
    if x_vector.size != y_vector.size:
        raise ValueError("x and y vectors are different lengths.")
    if theta.size != 6:
        raise ValueError("Algebraic parameters vector must only have 6 values")

    x = x_vector.reshape((x_vector.shape[0], 1))
    y = y_vector.reshape((y_vector.shape[0], 1))

    geo_param = algebraic_to_geometric(theta)
    c = np.sqrt(geo_param[1]**2 - geo_param[0]**2)
    focus1 = np.array([geo_param[2] + c * np.cos(geo_param[4]), geo_param[3] + c * np.sin(geo_param[4])])
    focus2 = np.array([geo_param[2] - c * np.cos(geo_param[4]), geo_param[3] - c * np.sin(geo_param[4])])
    p = np.sqrt(np.square(x - focus1[0]) + np.square(y - focus1[1]))
    q = np.sqrt(np.square(x - focus2[0]) + np.square(y - focus2[1]))
    se = (p + q - 2 * geo_param[1]) ** 2 / (geo_param[1] * geo_param[0]) * 2 * p * q /( (p + q) ** 2 - 4* c**2)
    return se

def algebraic_to_geometric(theta):
    """Calculates geometric characteristics of the ellipse from algebraic ellipse parameters.

    Parameters
    ----------
    theta : ndarray
        1D numpy array that stores algebraic ellipse equation parameters [a, b, c, d, e, f]

    Returns
    -------
    tuple
        geometric parameters (minor, major, x_center, y_center, tilt)

    Raises
    ------
    ValueError
        if algebraic parameters consist of other than 6 values
    """
    if theta.size != 6:
        raise ValueError("Algebraic parameters vector must only have 6 values")

    a = theta[0]
    b = theta[1]
    c = theta[2]
    d = theta[3]
    e = theta[4]
    f = theta[5]

    delta = b ** 2 - 4 * a * c
    
    lambdaPlus = 0.5 * ( a + c - (b ** 2 + (a - c) ** 2) ** 0.5)
    lambdaMinus = 0.5 * ( a + c + (b ** 2 + (a - c) ** 2) ** 0.5)

    psi = b*d*e - a*(e**2) - (b**2)*f + c*(4*a*f - d**2)
    VPlus = (psi/(lambdaPlus*delta)) ** 0.5
    VMinus = (psi/(lambdaMinus*delta)) ** 0.5

    # Major semi-axis
    axisA = max(VPlus, VMinus)
    # Minor semi-axis
    axisB = min(VPlus, VMinus)
    # center
    x_center = (2*c*d - b*e)/delta
    y_center = (2*a*e - b*d)/delta
    # Determine tilt (angle between x-axis and major axis)
    tau = 0
    if VPlus >= VMinus:
        if b == 0 and a < c:
            tau = 0
        elif b == 0 and a >= c:
            tau = 0.5*np.pi
        elif b < 0 and a < c:
            tau = 0.5*np.arctan(b/(a - c))
        elif b < 0 and a == c:
            tau = np.pi/4
        elif b < 0 and a > c:
            tau = 0.5*np.arctan(b/(a - c)) + np.pi/2
        elif b > 0 and a < c:
            tau = 0.5*np.arctan(b/(a - c)) + np.pi
        elif b > 0 and a == c:
            tau = np.pi*(3/4)
        elif b > 0 and a > c:
            tau = 0.5*np.arctan(b/(a - c)) + np.pi/2
    elif VPlus < VMinus:
        if b == 0 and a < c:
            tau = np.pi/2
        elif b == 0 and a >= c:
            tau = 0
        elif b < 0 and a < c:
            tau = 0.5*np.arctan(b/(a - c)) + np.pi/2
        elif b < 0 and a == c:
            tau = np.pi*(3/4)
        elif b < 0 and a > c:
            tau = 0.5*np.arctan(b/(a - c)) + np.pi
        elif b > 0 and a < c:
            tau = 0.5*np.arctan(b/(a - c)) + np.pi/2
        elif b > 0 and a == c:
            tau = np.pi/4
        elif b > 0 and a > c:
            tau = 0.5*np.arctan(b/(a - c))

    return (axisB, axisA, x_center, y_center, tau)

def direct_ellipse_est(x_vector, y_vector):
    """Finds algebraic parameters of best elliptical fit.
    
    Finds parameters corresponding to ellipse equation 
    ax^2 + bxy + cy^2 + dx + ey + f = 0 subject to
    b^2 - 4ac < 0. Translated from `this link <https://github.com/zygmuntszpak/guaranteed-ellipse-fitting-with-a-confidence-region-and-an-uncertainty-measure/blob/master/Guaranteed%20Ellipse%20Fitting%20with%20a%20Confidence%20Region/Direct%20Ellipse%20Fit/compute_directellipse_estimates.m>`_.

    Args:
        x_vector (ndarray): 1D numpy array that is plotted as x-coordinates.
        y_vector (ndarray): 1D numpy array that is plottes as y-coordinates.

    Returns:
        
        1D numpy array that stores algebraic ellipse equation parameters 
        [a, b, c, d, e, f].

    Raises:
        ValueError: If x_vector and y_vector aren't the same length or if
        x_vector and y_vector are similar.
    """
    if x_vector.size != y_vector.size:
        raise ValueError("x and y vectors are different lengths.")
    if np.allclose(x_vector, y_vector):
        raise ValueError("vectors are the same.")
    x = x_vector.reshape((x_vector.size, 1))
    y = y_vector.reshape((y_vector.size, 1))
    nPts = x.shape[0]
    normalisedPts, T = _normalize_iso(x, y)
    normalisedPts = np.block([[normalisedPts, np.ones(x.shape)]])
    theta = _direct_ellipse_fit(normalisedPts.T)

    a = theta[0]
    b = theta[1]
    c = theta[2]
    d = theta[3]
    e = theta[4]
    f = theta[5]

    # denormalise
    C = np.block([[a, b/2, d/2], 
                  [b/2, c, e/2],
                  [d/2, e/2, f]])
    C = T.transpose() @ C @ T
    # a, b, c, d, e, f
    theta = np.block([C[0, 0], C[0, 1] * 2, C[1, 1], 
                        C[0, 2] * 2, C[1, 2] * 2, C[2, 2]])
    theta = theta/np.linalg.norm(theta, ord=2)

    return theta

def transform(x_vector, y_vector, geo_param):
    """Transforms data so that its best elliptical fit is a unit circle.

    Args:
        x_vector (ndarray): 1D numpy array that is plotted as x-coordinates.
        y_vector (ndarray): 1D numpy array that is plottes as y-coordinates.
        geo_param (tuple): Geometric parameters (minor, major, x_center, 
            y_center, tilt)


    Returns:
        
        Dict with transformed x_vector and y_vector ndarrays::

            {
                'x': x_transformed,
                'y': y_transformed
            }
    
    Raises:
        ValueError: If x_vector and y_vector aren't the same length, or 
            geo_param does not have 5 values.
    """
    if x_vector.size != y_vector.size:
        raise ValueError("x and y vectors are different lengths.")
    if len(geo_param) != 5:
        raise ValueError("There should be 5 geometric parameters.")
    # Find elliptical fit geometric parameters from algebraic parameters
    x = x_vector.reshape((x_vector.size, 1))
    y = y_vector.reshape((y_vector.size, 1))
    minor, major, center_x, center_y, tilt = geo_param
    # Center ellipse
    x_centered = x - center_x
    y_centered = y - center_y
    # Clockwise rotation matrix
    c, s = np.cos(tilt), np.sin(tilt)
    R = np.array([[c, s], [-s, c]])
    # Rotate ellipse, align major axis and x-axis
    P = np.hstack((x_centered, y_centered)) @ R.T
    x_rotated = P[:, [0]]
    y_rotated = P[:, [1]] 
    # Scale to make circular
    x_transformed = x_rotated/major
    y_transformed = y_rotated/minor
    return {'x': x_transformed.flatten(), 'y': y_transformed.flatten()}

def calculate_mape(Y, true_times=None, period=24):
    """Function to calculate mean absolute position error.

    Args:
        x_transformed (ndarray): 1D vector transformed to fit unit circle when 
            paired with another transformed vector.
        y_transformed (ndarray): 1D vector transformed to fit unit circle when 
            paired with another transformed vector.
        true_times (ndarray): 1D array in which each value corresponds to 
            collection time of value with same index in x_transformed and 
            y_transformed.
    Returns:
        Tuple with a float representing mean absolute position error and an array of predicted phases.
    """
    try:
        theta = direct_ellipse_est(Y[:, 0], Y[:, 1])
        minor, major, center_x, center_y, tilt = algebraic_to_geometric(theta)
    except ValueError:
        return None

    # Scaled angular positions
    acw_angles, cw_angles = _scaled_angles((Y[:, 0] - center_x)/major, (Y[:, 1] - center_y)/minor)
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
            mape_value = np.mean(np.abs(_delta(scaled_time, adjusted_opt_angles))) if true_times is not None else np.nan
        except RuntimeWarning:
            mape_value = np.nan
    else:
        mape_value = np.nan
        adjusted_opt_angles = acw_angles
    return (mape_value, adjusted_opt_angles)

# INTERNAL FUNCTIONS

def _scaled_angles(x, y):
    return (np.arctan2(y, x)/(2*np.pi)) % 1,  (np.arctan2(-y, x)/(2*np.pi)) % 1

def _zeitDiff(x, y):
    return np.abs(x - y +((x - y) < -1/2) - ((x - y) > 1/2))

def _delta(x, y):
    return ((x - y + 1/2) % 1) - 1/2

def _angular_mean(x, y):
    return (np.arctan2(np.sum(y), np.sum(x))/(2*np.pi)) % 1

def _angular_spread(x, y):
    return x.shape[0] - np.sqrt(np.sum(x) ** 2 + np.sum(y) ** 2)

def _normalize_iso(X, Y):
    """Normalise points isotropically prior to analysis.

    Translated from `here <https://github.com/zygmuntszpak/guaranteed-
    ellipse-fitting-with-a-confidence-region-and-an-uncertainty-measure/blob/master/Guaranteed%20Ellipse%20Fitting%20with%20a%20Confidence%20Region/Helper%20Functions/normalize_data_isotropically.m>`_.

    Args:
        X (ndarray): First vector.
        Y (ndarray): Second vector.

    Returns:
    
        Normalised points and Isotropic scaling matrix.
    """
    nPoints = X.shape[0]
    points = np.block([[X, Y, np.ones(X.shape)]]).T
    meanX = np.mean(points[0])
    meanY = np.mean(points[1])
    # Isotropic Scaling Factor
    meandist = np.sqrt(np.mean(np.square(points[0]- meanX) + np.square(points[1]-meanY)))
    s = meandist/np.sqrt(2)
    # Isotropic Scaling Matrix
    T = np.array([[1/s, 0, -1/s * meanX],
                  [0, 1/s, -1/s * meanY],
                  [0, 0, 1]])
    normalisedPts = T @ points
    normalisedPts = normalisedPts.T[:, :-1]

    return normalisedPts, T

def _direct_ellipse_fit(data):
    """Implementing method described by R. Halif and J. Flusser 98.

    Translated from `here <https://github.com/zygmuntszpak/guaranteed-ellipse-fitting-with-a-confidence-region-and-an-uncertainty-measure/blob/master/Guaranteed%20Ellipse%20Fitting%20with%20a%20Confidence%20Region/Direct%20Ellipse%20Fit/direct_ellipse_fit.m>`_.

    Args:
        data: Normalised points with homogenous coordinates

    Returns:
        
        Transformed ellipse equation coefficients.
    """
    x = data[[0], :].T
    y = data[[1], :].T

    # Quadratic part of design matrix
    D1 = np.block([[x ** 2, x * y, y ** 2]])
    # Linear part of design matrix
    D2 = np.block([[x, y, np.ones(x.shape)]])
    # Quad part of scatter mat
    S1 = D1.T @ D1
    # Combined part of scatter mat
    S2 = D1.T @ D2
    # Linear part of scatter mat
    S3 = D2.T @ D2
    # getting a2 from a1
    T = -np.linalg.solve(S3, S2.T)
    # Reduce scatter
    M = S1 + S2 @ T
    # Premultiply by inv(C1)
    M = np.block([[M[2, :]/2], [-M[1, :]], [M[0, :]/2]])
    # solve eigensystem
    eVal, eVec = np.linalg.eig(M)
    # evaluate a.TCa
    cond = 4 * eVec[0, :] * eVec[2, :] - eVec[1, :] ** 2
    # evec for min pos eval
    al = eVec[:, np.argwhere(cond > 0).flatten()]
    # Ellipse coefficients
    a = np.block([[al], [T @ al]]).flatten()
    a = a/np.linalg.norm(a, ord=2)

    return a

def gradient(x, y, s, beta):
    """From Yu, Kulkarni & Poor(2012)

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    s : _type_
        _description_
    beta : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    c1x, c1y, c2x, c2y, a= s[0], s[1], s[2], s[3], s[4]
    p = np.sqrt((x - c1x) ** 2 + (y - c1y) ** 2)
    q = np.sqrt((x - c2x) ** 2 + (y - c2y) ** 2)
    r = np.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)
    g = np.zeros((5,))
    t1 = p + q - 2*a
    t2 = 2 * p * q + beta * (p**2 + q**2 - r**2)
    g[0] = np.mean((2 * q * t1 ** 2 * (c1x - x))/(p * t2) - 4 * q * t1 ** 2 * (q * (c1x - x) + beta * p * (c2x - x)) / t2**2 + 4 * q * t1 * (c1x - x)/t2)
    g[1] = np.mean((2 * q * t1 ** 2 * (c1y - y))/(p * t2) - 4 * q * t1 ** 2 * (q * (c1y - y) + beta * p * (c2y - y)) / t2**2 + 4 * q * t1 * (c1y - y)/t2)
    g[2] = np.mean((2 * p * t1 ** 2 * (c2x - x))/(q * t2) - 4 * p * t1 ** 2 * (p * (c2x - x) + beta * q * (c1x - x)) / t2**2 + 4 * p * t1 * (c2x - x)/t2)
    g[3] = np.mean((2 * p * t1 ** 2 * (c2y - y))/(q * t2) - 4 * p * t1 ** 2 * (p * (c2y - y) + beta * q * (c1y - y)) / t2**2 + 4 * p * t1 * (c2y - y)/t2)
    g[4] = np.mean( -8 * p * q * t1 / t2)
    return g

def cost(x, y, s, beta):
    """From Yu, Kulkarni & Poor(2012)

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    s : _type_
        _description_
    beta : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    c1x, c1y, c2x, c2y, a = s[0], s[1], s[2], s[3], s[4]
    p = np.sqrt((x - c1x) ** 2 + (y - c1y) ** 2)
    q = np.sqrt((x - c2x) ** 2 + (y - c2y) ** 2)
    r = np.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)
    return np.mean((p + q - 2*a) ** 2 * 2 * p * q/ ((p ** 2 + q ** 2 - r ** 2) * beta + 2 * p * q))

def gradient_descent(x, y, learning_rate = 0.01, iterations = 200, convergence=1e-4):
    # theta = COFE.ellipse.direct_ellipse_est(x,y)
    # geo_param = COFE.ellipse.algebraic_to_geometric(theta)
    # c = np.sqrt(geo_param[1]**2 - geo_param[0]**2)
    # focus1 = np.array([geo_param[2] + c * np.cos(geo_param[4]), geo_param[3] + c * np.sin(geo_param[4])])
    # focus2 = np.array([geo_param[2] - c * np.cos(geo_param[4]), geo_param[3] - c * np.sin(geo_param[4])])
    # s = np.array([focus1[0], focus1[1], focus2[0], focus2[1], geo_params[1]])
    s = np.array([np.mean(x) - np.std(x), np.mean(y) - np.std(y), np.mean(x) + np.std(x), np.mean(y) + np.std(y), 3*np.std(x) + 3*np.std(y)])
    beta = 0
    last_cost = cost(x, y, s, beta)
    for it in range(iterations):
        s = s -  learning_rate * gradient(x, y, s, beta)
        delta_cost = cost(x, y, s, beta) - last_cost
        betahat = 0.6 * beta  + 0.4 * 2/ np.pi * np.arctan(0.001/np.abs(delta_cost))
        last_cost = delta_cost + last_cost
        if betahat >= beta:
            beta = betahat
        print([delta_cost, beta])
        if np.abs(delta_cost) < convergence:
            break;
    return s