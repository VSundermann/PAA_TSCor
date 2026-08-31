# Administracao de diretorios e arquivos
import os
import sys
import glob

# Funcionalidades principais, matematica, classificadores, grafos...
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.stats
import numpy as np
import pandas as pd
import networkx as nx

# Visualização e plotagem dos graficos
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Log e metricas dos experimentos
import time
import random
import tracemalloc

# Aceleração de codigos, converte funcoes direto para codigo de maquina
from numba import njit

# Interface grafica
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Slot


#########################################
#                                       #
#           Variaveis Globais           #
#          Funcoes Utilitarias          #
#                                       #
#########################################

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}

# Adaptado do TSLearn, retirava NaNs que estavam somente no final das series
# agora qualquer NaN e retirado, reduzindo o tamanho total da serie
def preprocess_timeseries(ts, remove_nans=False):
    if ts.ndim <= 1:
        ts = ts.reshape(-1, 1)
    if remove_nans:
        ts = ts[~np.isnan(ts).any(axis=1)]
    return ts

# Codigos de mascaras do TSLearn, utilizados pelo CDTW
def make_sakoe_chiba_mask():

    def _sakoe_chiba_mask_generic(sz1, sz2, radius=1):
        mask = np.full((sz1, sz2), False)
        if sz1 > sz2:
            width = sz1 - sz2 + radius
            for i in range(sz2):
                lower = max(0, i - radius)
                upper = min(sz1, i + width) + 1
                mask[lower:upper, i] = True
        else:
            width = sz2 - sz1 + radius
            for i in range(sz1):
                lower = max(0, i - radius)
                upper = min(sz2, i + width) + 1
                mask[i, lower:upper] = True
        return mask
    
    return njit(nogil=True, cache=True, fastmath=True)(_sakoe_chiba_mask_generic)

def make_itakura_mask():
    def _itakura_mask_generic(sz1, sz2, max_slope=2.0):
        min_slope = 1 / float(max_slope)
        max_slope *= float(sz1) / float(sz2)
        min_slope *= float(sz1) / float(sz2)

        lower_bound = np.empty((2, sz2))
        lower_bound[0] = min_slope * np.arange(sz2)
        lower_bound[1] = (sz1 - 1) - max_slope * (sz2 - 1) + max_slope * np.arange(sz2)
        lower_bound_ = np.empty(sz2)
        for i in range(sz2):
            lower_bound_[i] = max(
                np.round(lower_bound[0, i], decimals=2),
                np.round(lower_bound[1, i], decimals=2)
            )
        lower_bound_ = np.ceil(lower_bound_)

        upper_bound = np.empty((2, sz2))
        upper_bound[0] = max_slope * np.arange(sz2)
        upper_bound[1] = (sz1 - 1) - min_slope * (sz2 - 1) + min_slope * np.arange(sz2)
        upper_bound_ = np.empty(sz2)
        for i in range(sz2):
            upper_bound_[i] = min(
                np.round(upper_bound[0, i], decimals=2),
                np.round(upper_bound[1, i], decimals=2)
            )
        upper_bound_ = np.floor(upper_bound_ + 1)

        mask = np.full((sz1, sz2), False)
        for i in range(sz2):
            mask[int(lower_bound_[i]): int(upper_bound_[i]), i] = True

        # Post-check
        raise_warning = False
        for i in range(sz1):
            if not np.any(mask[i]):
                raise_warning = True
                break
        if not raise_warning:
            for j in range(sz2):
                if not np.any(mask[:, j]):
                    raise_warning = True
                    break
        if raise_warning:
            raise RuntimeWarning(
                "'itakura_max_slope' constraint is unfeasible "
                "(ie. leads to no admissible path) for the "
                "provided time series sizes",
            )

        return mask
    return njit(nogil=True, cache=True, fastmath=True)(_itakura_mask_generic)

# Codigos de mascaras do AeonTK, utilizados pelo resto
@njit(nogil=True, cache=True)
def _itakura_parallelogram(x_size: int, y_size: int, max_slope: float):
    min_slope = 1 / float(max_slope)
    max_slope *= float(y_size - 1) / float(x_size - 2)
    max_slope = max(max_slope, 1.0)

    min_slope *= float(y_size - 2) / float(x_size - 1)
    min_slope = min(min_slope, 1.0)

    centered_scale = np.arange(x_size) - x_size + 1

    lower_bound = np.empty(x_size, dtype=np.float64)
    upper_bound = np.empty(x_size, dtype=np.float64)

    for i in range(x_size):
        lb0 = min_slope * i
        lb1 = max_slope * centered_scale[i] + y_size - 1
        lower_bound[i] = math.ceil(max(round(lb0, 2), round(lb1, 2)))

        ub0 = max_slope * i + 1
        ub1 = min_slope * centered_scale[i] + y_size
        upper_bound[i] = math.floor(min(round(ub0, 2), round(ub1, 2)))

    if max_slope == 1.0:
        if y_size > x_size:
            for i in range(x_size - 1):
                upper_bound[i] = lower_bound[i + 1]
        else:
            for i in range(x_size):
                upper_bound[i] = lower_bound[i] + 1

    for i in range(x_size):
        if lower_bound[i] < 0:
            lower_bound[i] = 0
        if lower_bound[i] > y_size:
            lower_bound[i] = y_size
        if upper_bound[i] < 0:
            upper_bound[i] = 0
        if upper_bound[i] > y_size:
            upper_bound[i] = y_size

    bounding_matrix = np.empty((x_size, y_size), dtype=np.bool_)
    for i in range(x_size):
        for j in range(y_size):
            bounding_matrix[i, j] = False

    for i in range(x_size):
        start = int(lower_bound[i])
        end = int(upper_bound[i])
        for j in range(start, end):
            bounding_matrix[i, j] = True

    return bounding_matrix

@njit(nogil=True, cache=True)
def _sakoe_chiba_bounding(
    x_size: int, y_size: int, radius_percent: float
) -> np.ndarray:

    # Garante que a TS maior esteja 'por cima'
    if x_size > y_size:
        return _sakoe_chiba_bounding(y_size, x_size, radius_percent).T

    matrix = np.full((x_size, y_size), False)  # Create a matrix filled with False

    max_size = max(x_size, y_size) + 1

    shortest_dimension = min(x_size, y_size)
    thickness = int(radius_percent * shortest_dimension)
    for step in range(max_size):
        x_index = math.floor((step / max_size) * x_size)
        y_index = math.floor((step / max_size) * y_size)

        upper = max(0, (x_index - thickness))
        lower = min(x_size, (x_index + thickness + 1))

        matrix[upper:lower, y_index] = True

    return matrix

@njit(nogil=True, cache=True)
def create_bounding_matrix(
    x_size: int,
    y_size: int,
    window: float | None = None,
    itakura_max_slope: float | None = None,
):
    """Create a bounding matrix for an elastic distance.

    Parameters
    ----------
    x_size : int
        Size of the first time series.
    y_size : int
        Size of the second time series.
    window : float, default=None
        Window size as a percentage of the smallest time series.
        If None, the bounding matrix will be full.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
        Itakura parallelogram does not support unequal length time series.

    Returns
    -------
    np.ndarray of shape (x_size, y_size)
        Bounding matrix where values in bound are True and values out of bounds are
        False.
    """
    if itakura_max_slope is not None:
        return _itakura_parallelogram(x_size, y_size, itakura_max_slope)
    if window is not None and window != 1.0:
        if window < 0 or window > 1:
            raise ValueError("window must be between 0 and 1")
        return _sakoe_chiba_bounding(x_size, y_size, window)
    return np.full((x_size, y_size), True)

# Codigo para calculo do caminho otimo, a partir da matriz de custos
@njit(nogil=True, cache=True, fastmath=True)
def compute_min_return_path(cost_matrix: np.ndarray) -> list[tuple]:
    """Compute the minimum return path through a cost matrix.

    Parameters
    ----------
    cost_matrix : np.ndarray, of shape (n_timepoints_x, n_timepoints_y)
        Cost matrix.

    Returns
    -------
    List[Tuple]
        List of indices that make up the minimum return path.
    """
    x_size, y_size = cost_matrix.shape
    i, j = x_size - 1, y_size - 1
    alignment = []

    while i > 0 or j > 0:
        alignment.append((i, j))

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_index = np.argmin(
                np.array(
                    [
                        cost_matrix[i - 1, j - 1],
                        cost_matrix[i - 1, j],
                        cost_matrix[i, j - 1],
                    ]
                )
            )

            if min_index == 0:
                i, j = i - 1, j - 1
            elif min_index == 1:
                i -= 1
            else:
                j -= 1

    alignment.append((0, 0))
    return alignment[::-1]

# Calculo de distancias base para os metodos
@njit(cache=True, fastmath=True)
def _univariate_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(_univariate_squared_distance(x, y))

@njit(nogil=True, cache=True, fastmath=True)
def _univariate_squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        difference = x[i] - y[i]
        distance += difference * difference
    return distance

#########################################
#                                       #
#         Dynamic Time Warping          #
#          Adaptado do AeonTK           #
#                                       #
#########################################

@njit(nogil=True, cache=True, fastmath=True)
def _dtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[:, i], y[:, j]
                ) + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]

@njit(nogil=True, cache=True, fastmath=True)
def dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> np.ndarray:
    r"""Compute the DTW cost matrix between two time series.

    The cost matrix is the pairwise Euclidean distance between all points
    :math:`M_{i,j}=(x_i-x_j)^2`. It is used in the DTW path calculations.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1``,
        10% of the series length is the max warping allowed.
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        dtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(_x, _y, bounding_matrix)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _dtw_cost_matrix(x, y, bounding_matrix)
    raise ValueError("x and y must be 1D or 2D")

#########################################
#                                       #
#    Continuous Dynamic Time Warping    #
#                                       #
#########################################

@njit(nogil=True, cache=True, fastmath=True)
def _cdtw_distance_matrix_njit(r, t, mask):
    M = len(r)
    N = len(t)
    d = np.full((2*M-1, 2*N-1), np.inf)

    for i in range(0, 2*M-1, 2):
        m = i // 2
        for j in range(0, 2*N-1, 2):
            n = j // 2

            if mask[m, n]:
                d[i, j] = (r[m] - t[n])**2

                if m < M - 1 and n < N - 1:
                    if (t[n] <= r[m] <= t[n+1]) or (t[n+1] <= r[m] <= t[n]):
                        d[i, j+1] = 0
                    else:
                        d[i, j+1] = min((r[m] - t[n])**2, (r[m] - t[n+1])**2)
                    if (r[m] <= t[n] <= r[m+1]) or (r[m+1] <= t[n] <= r[m]):
                        d[i+1, j] = 0
                    else:
                        d[i+1, j] = min((t[n] - r[m])**2, (t[n] - r[m+1])**2)
    
    return d

def cdtw_distance_matrix(r, t, radius):
    if r.ndim > 1: r = r.flatten()
    if t.ndim > 1: t = t.flatten()

    M = len(r)
    N = len(t)

    if radius is None:
        mask = np.full((M, N), True)
    else:
        mask = make_sakoe_chiba_mask()(M, N, radius=int(radius))

    return _cdtw_distance_matrix_njit(r, t, mask)

@njit(nogil=True, cache=True, fastmath=True)
def cdtw_accumulated_dist_matrix(r, t, d):
    M = len(r)
    N = len(t)
    D = np.zeros_like(d)
    D[0, 0] = d[0, 0]

    for i in range(2, 2*M-1, 2):
        D[i-1, 0] = d[i-1, 0] + D[i-2, 0]
        D[i, 0] = d[i, 0] + D[i-1, 0]
    
    for j in range(2, 2*N-1, 2):
        D[0, j-1] = d[0, j-1] + D[0, j-2]
        D[0, j] = d[0, j] + D[0, j-1]

    for i in range(2, 2*M-1, 2):
        for j in range(2, 2*N-1, 2):
            D[i-1, j] = d[i-1, j] + D[i-2, j]
            D[i, j-1] = d[i, j-1] + D[i, j-2]
            D[i, j] = d[i, j] + min(D[i, j-1], D[i-1, j], D[i-2, j-2])
    
    return D

@njit(nogil=True, cache=True, fastmath=True)
def _cdtw_return_path_njit(r, t, acc_cost_mat):
    M = len(r)
    N = len(t)
    i = 2*M - 2
    j = 2*N - 2

    w = [(M-1, N-1)]
    rw = [r[-1]]
    tw = [t[-1]]

    while i + j != 0:
        m = i // 2
        n = j // 2

        if i - 2 < 0:
            w.append((m, n-1))
            rw.append(r[m])
            tw.append(t[n-1])
            j -= 2
        elif j - 2 < 0:
            w.append((m-1, n))
            rw.append(r[m-1])
            tw.append(t[n])
            i -= 2
        else:
            opts = np.array([acc_cost_mat[i, j-1], acc_cost_mat[i-1, j], acc_cost_mat[i-2, j-2]])
            num = np.argmin(opts)

            if num == 0:
                if (t[n-1] <= r[m] <= t[n]) or (t[n] <= r[m] <= t[n-1]):
                    x = (r[m] - t[n-1]) / (t[n] - t[n-1]) if t[n] != t[n-1] else 0.5
                elif (r[m] - t[n-1])**2 <= (r[m] - t[n])**2:
                    x = 0.0
                else:
                    x = 1.0

                w.append((m, n-1+x))
                rw.append(r[m])
                tw.append(x*(t[n] - t[n-1]) + t[n-1])
                j -= 2
            elif num == 1:
                if (r[m-1] <= t[n] <= r[m]) or (r[m] <= t[n] <= r[m-1]):
                    x = (t[n] - r[m-1]) / (r[m] - r[m-1]) if r[m] != r[m-1] else 0.5
                elif (t[n] - r[m-1])**2 <= (t[n] - r[m])**2:
                    x = 0.0
                else:
                    x = 1.0

                w.append((m-1+x, n))
                rw.append(x*(r[m] - r[m-1]) + r[m-1])
                tw.append(t[n])
                i -= 2
            elif num == 2:
                w.append((m-1, n-1))
                rw.append(r[m-1])
                tw.append(t[n-1])
                i -= 2
                j -= 2

    w.reverse()
    return w

def cdtw_return_path(r, t, acc_cost_mat):
    if r.ndim > 1: r = r.flatten()
    if t.ndim > 1: t = t.flatten()
    return _cdtw_return_path_njit(r, t, acc_cost_mat)

#########################################
#                                       #
#       Longest Common Subsequence      #
#          Adaptado do AeonTK           #
#########################################

@njit(nogil=True, cache=True, fastmath=True)
def _lcss_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, epsilon
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.zeros((x_size + 1, y_size + 1))

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1] and _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]) <= epsilon:
                cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
            else:
                cost_matrix[i, j] = max(
                    cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                )
    return cost_matrix

@njit(nogil=True, cache=True, fastmath=True)
def lcss_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    epsilon: float = 1.0,
    itakura_max_slope: float | None = None,
) -> np.ndarray:
    r"""Return the LCSS cost matrix between x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=1.
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'. The default is 1.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray
        The LCSS cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _lcss_cost_matrix(_x, _y, bounding_matrix, epsilon)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _lcss_cost_matrix(x, y, bounding_matrix, epsilon)
    raise ValueError("x and y must be 1D or 2D")

@njit(nogil=True, cache=True, fastmath=True)
def compute_lcss_return_path(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    bounding_matrix: np.ndarray,
    cost_matrix: np.ndarray,
) -> list[tuple]:
    """Compute the return path through a cost matrix for the LCSS algorithm.

    Parameters
    ----------
    x : np.ndarray, of shape (n_channels, n_timepoints)
        First time series.
    y : np.ndarray (m_channels, m_timepoints)
        Second time series.
    epsilon : float
        Threshold for the LCSS algorithm.
    bounding_matrix : np.ndarray (n_timepoints_x, n_timepoints_y)
        Bounding matrix for the LCSS algorithm.
    cost_matrix : np.ndarray (n_timepoints_x, n_timepoints_y)
        Cost matrix for the LCSS algorithm.

    Returns
    -------
    List[Tuple]
        List of indices that make up the return path.
    """
    x_size = x.shape[1]
    y_size = y.shape[1]

    i, j = (x_size, y_size)
    path = []

    while i > 0 and j > 0:
        # Uniao do if externo com interno, em certos casos gerava loop infinito
        if bounding_matrix[i - 1, j - 1] and _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]) <= epsilon:
            path.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif cost_matrix[i - 1, j] > cost_matrix[i, j - 1]:
            i = i - 1
        else:
            j = j - 1
    return path[::-1]

#########################################
#                                       #
#       Soft Dynamic Time Warping       #
#          Adaptado do AeonTK           #
#                                       #
#########################################

@njit(nogil=True, cache=True, fastmath=True)
def _soft_dtw_cost_matrix(x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, gamma: float) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost = _univariate_squared_distance(x[:, i], y[:, j])
                
                a = cost_matrix[i, j]
                b = cost_matrix[i, j + 1]
                c = cost_matrix[i + 1, j]
                
                a_gamma = -a / gamma
                b_gamma = -b / gamma
                c_gamma = -c / gamma
                
                max_val = max(a_gamma, b_gamma, c_gamma)
                
                soft_min = -gamma * (max_val + np.log(
                    np.exp(a_gamma - max_val) + 
                    np.exp(b_gamma - max_val) + 
                    np.exp(c_gamma - max_val)
                ))
                
                cost_matrix[i + 1, j + 1] = cost + soft_min

    return cost_matrix[1:, 1:]

def soft_dtw_cost_matrix(
    s1: np.ndarray,
    s2: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> np.ndarray:
    r"""Compute the soft-DTW cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    gamma : float, default=1.0
        Controls the smoothness of the warping. A value of 0.0 is equivalent to DTW.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1``,
        10% of the series length is the max warping allowed.
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        soft-DTW cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if s1.ndim == 1 and s2.ndim == 1:
        _x = s1.reshape((1, s1.shape[0]))
        _y = s2.reshape((1, s2.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _soft_dtw_cost_matrix(_x, _y, bounding_matrix, gamma)
    if s1.ndim == 2 and s2.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            s1.shape[1], s2.shape[1], window, itakura_max_slope
        )
        return _soft_dtw_cost_matrix(s1, s2, bounding_matrix, gamma)
    raise ValueError("x and y must be 1D or 2D")

#########################################
#                                       #
#     Amerced Dynamic Time Warping      #
#          Adaptado do AeonTK           #
#                                       #
#########################################

@njit(nogil=True, cache=True, fastmath=True)
def _adtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, warp_penalty: float
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[:, i], y[:, j]
                ) + min(
                    cost_matrix[i, j + 1] + warp_penalty,
                    cost_matrix[i + 1, j] + warp_penalty,
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]

@njit(nogil=True, cache=True, fastmath=True)
def adtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    warp_penalty: float = 1.0,
) -> np.ndarray:
    r"""Compute the ADTW cost matrix between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1``,
        10% of the series length is the max warping allowed.
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    warp_penalty: float, default=1.0
        Penalty for warping. A high value will mean less warping.

    Returns
    -------
    np.ndarray (n_timepoints, m_timepoints)
        adtw cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _adtw_cost_matrix(_x, _y, bounding_matrix, warp_penalty)
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _adtw_cost_matrix(x, y, bounding_matrix, warp_penalty)
    raise ValueError("x and y must be 1D or 2D")

#########################################
#                                       #
#  Central Time Series Processing Unit  #
#                                       #
#########################################

class CTSPU:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.score = 0.0
        self.computing_time = 0.0
        self.space_usage = 0.0
        self.path = None
        self.cost_matrix = None

    # Funcao principal para calculo de distancia entre series temporais
    def _compute_distance(self, ts1, ts2):
        ts1 = preprocess_timeseries(ts1, remove_nans=True)
        ts2 = preprocess_timeseries(ts2, remove_nans=True)
        
        alg = self.config.get("algorithm", "DTW")
        constraint = self.config.get("constraint", "Sakoe Chiba")
        
        # Raio da janela de constraint, definido como percentagem por boas praticas
        radius_percent = float(self.config.get("radius", 10.0)) / 100.0
        
        slope = float(self.config.get("slope", 2.0))
        eps = float(self.config.get("eps", 1.0))
        gamma = float(self.config.get("gamma", 1.0))
        warp_penalty = float(self.config.get("warp_penalty", 1.0))
        
        global_constraint = GLOBAL_CONSTRAINT_CODE.get(constraint.lower().replace(" ", "_"), 0)
        
        sz1 = len(ts1)
        sz2 = len(ts2)
        
        # Calculo do raio de forma dinamica, a partir do tamanho da maior serie temporal
        dynamic_radius = int(max(sz1, sz2) * radius_percent)
        
        if alg == "DTW":
            window = radius_percent if global_constraint == 2 else None
            itakura_slope = slope if global_constraint == 1 else None

            cost_matrix = dtw_cost_matrix(ts1.T, ts2.T, window, itakura_slope)
            path = compute_min_return_path(cost_matrix)
            # Acessa ultimo valor da matriz, que contem o custo total
            score = cost_matrix[-1, -1] if cost_matrix.size > 0 else float('inf')
        elif alg == "CDTW":
            window = dynamic_radius if global_constraint == 2 else None
            itakura_slope = slope if global_constraint == 1 else None

            # Como CDTW foi adaptado seguindo TSLearn as series nao precisam ser transpostas
            cdtw_dist = cdtw_distance_matrix(ts1, ts2, window)
            cost_matrix = cdtw_accumulated_dist_matrix(ts1, ts2, cdtw_dist)
            path = cdtw_return_path(ts1, ts2, cost_matrix)
            score = np.sqrt(cost_matrix[-1, -1]) if cost_matrix.size > 0 else float('inf')
            
            # Reduz matriz de custo para suas dimensoes originais, para nao dar problema no plot
            cost_matrix = cost_matrix[::2, ::2]
        elif alg == "LCSS":
            window = radius_percent if global_constraint == 2 else None
            itakura_slope = slope if global_constraint == 1 else None

            cost_matrix = lcss_cost_matrix(ts1.T, ts2.T, window, eps, itakura_slope)
            bounding_matrix = create_bounding_matrix(sz1, sz2, window, itakura_slope)
            path = compute_lcss_return_path(ts1.T, ts2.T, eps, bounding_matrix, cost_matrix)
            # Score e dado por similaridade(quantidade de pontos em comum), onde 0 e identico e 1 e totalmente diferente
            score = 1.0 - (float(cost_matrix[-1, -1]) / min([sz1, sz2])) if cost_matrix.size > 0 else 1.0
        elif alg == "SDTW":
            window = radius_percent if global_constraint == 2 else None
            itakura_slope = slope if global_constraint == 1 else None

            cost_matrix = soft_dtw_cost_matrix(ts1.T, ts2.T, gamma, window, itakura_slope)
            path = compute_min_return_path(cost_matrix)
            score = abs(cost_matrix[-1, -1]) if cost_matrix.size > 0 else float('inf')
        elif alg == "ADTW":
            window = radius_percent if global_constraint == 2 else None
            itakura_slope = slope if global_constraint == 1 else None

            cost_matrix = adtw_cost_matrix(ts1.T, ts2.T, window, itakura_slope, warp_penalty)
            path = compute_min_return_path(cost_matrix)
            score = cost_matrix[-1, -1] if cost_matrix.size > 0 else float('inf')
        else:
            return False, None, None, None
            
        return True, score, path, cost_matrix

    # Execucao unica dos algoritmos de casamento entre 2 series temporais
    def run_single(self):
        if not self.data or len(self.data) != 2:
            return False
        
        # Inicio das medicoes de tempo de execucao e espaco utilizado
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            success, score, path, cost_matrix = self._compute_distance(self.data[0], self.data[1])
            if not success:
                tracemalloc.stop()
                return False
            
            # Atualiza variaveis, para serem acessadas pela GUI
            self.score = score
            self.path = path
            self.cost_matrix = cost_matrix
        except Exception as e:
            print("Computation error:", e)
            tracemalloc.stop()
            return False
            
        self.computing_time = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        self.space_usage = peak / (1024 * 1024)     # MB
        tracemalloc.stop()
        
        return True

    # Execucao em lote dos algoritmos para todos os pares de TS em teste e treino
    def run_dataset(self, train_data, test_data, dataset_name):
        run_all = self.config.get("run_all_algorithms", False)
        
        if run_all:
            algorithms = ["DTW", "CDTW", "LCSS", "SDTW", "ADTW"]
        else:
            algorithms = [self.config.get("algorithm", "DTW")]
            
        all_results = {alg: [] for alg in algorithms}
        total_times = {alg: 0.0 for alg in algorithms}
        
        # Itera sobre series de teste e treino, realizando casamento 1 a 1 buscando encontrar o par que minimize o score
        # ou seja, que tenha maior similaridade
        for test_idx, test_class, test_ts in test_data:
            best_scores = {alg: float('inf') for alg in algorithms}
            best_train_idxs = {alg: -1 for alg in algorithms}
            best_train_classes = {alg: "" for alg in algorithms}
            best_times = {alg: 0.0 for alg in algorithms}
            best_spaces = {alg: 0.0 for alg in algorithms}
            
            for train_idx, train_class, train_ts in train_data:
                for alg in algorithms:
                    self.config["algorithm"] = alg
                    
                    tracemalloc.start()
                    start_time = time.perf_counter()

                    try:
                        success, score, _, _ = self._compute_distance(test_ts, train_ts)
                    except Exception as e:
                        success = False
                        score = 0.0
                    
                    comp_time = time.perf_counter() - start_time
                    total_times[alg] += comp_time
                    _, peak = tracemalloc.get_traced_memory()
                    space_usage = peak / (1024 * 1024) # MB
                    tracemalloc.stop()
                    
                    if success:
                        if score < best_scores[alg]: # Algoritmos retornam a medida de distancia, entao o melhor score e o menor
                            best_scores[alg] = score
                            best_train_idxs[alg] = train_idx
                            best_train_classes[alg] = train_class
                            best_times[alg] = comp_time
                            best_spaces[alg] = space_usage

            for alg in algorithms:
                same_class = (test_class == best_train_classes[alg])    # Para calculo de acuracia
                all_results[alg].append({
                    "Test Index": test_idx,
                    "Train Index": best_train_idxs[alg],
                    "Score": best_scores[alg],
                    "Computing Time (s)": best_times[alg],
                    "Space Usage (MB)": best_spaces[alg],
                    "Same Class": same_class
                })
                
        # Restaura config original
        if run_all:
            self.config["algorithm"] = "DTW"
            
        for alg in algorithms:
            self.save_results(all_results[alg], dataset_name, alg, total_time=total_times[alg])
            
        return True

    # Execucao de um classificador para um dataset
    def run_classifier(self, train_data, test_data, dataset_name, clf_algo):
        if not train_data or not test_data:
            return False, 0.0, 0.0, 0.0

        # Avalia tamanho das series, aplicando padding quando necessario    
        max_len = 0
        for _, _, ts in train_data + test_data:
            if len(ts) > max_len:
                max_len = len(ts)
                
        def pad_ts(ts, max_l):
            if len(ts) < max_l:
                return np.pad(ts, (0, max_l - len(ts)), 'constant')
            return ts
        
        # Divisao de treino e teste para classificador baseado no sampling
        X_train = np.array([pad_ts(ts, max_len) for _, _, ts in train_data])
        y_train = np.array([ts_class for _, ts_class, _ in train_data])
        
        X_test = np.array([pad_ts(ts, max_len) for _, _, ts in test_data])
        y_test = np.array([ts_class for _, ts_class, _ in test_data])
        
        if clf_algo == "1-NN":
            model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        elif clf_algo == "Naive Bayes":
            model = GaussianNB()
        elif clf_algo == "Linear SVM":
            model = SVC(kernel='linear')
        else:
            return False, 0.0, 0.0, 0.0

            
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
        except Exception as e:
            print("Classifier error:", e)
            tracemalloc.stop()
            return False, 0.0, 0.0, 0.0
            
        comp_time = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        space_usage = peak / (1024 * 1024) # Em MB
        tracemalloc.stop()
        
        # Formatacao do log
        results = [{
            "Dataset": dataset_name,
            "Classifier": clf_algo,
            "Accuracy": accuracy,
            "Computing Time (s)": comp_time,
            "Space Usage (MB)": space_usage
        }]
        self.save_results(results, dataset_name, clf_algo.replace(" ", "_"))
        
        return True, accuracy, comp_time, space_usage

    def compute_spearman_correlation():
        algorithms = ["DTW", "CDTW", "LCSS", "SDTW", "ADTW", "1-NN", "Linear_SVM", "Naive_Bayes"]
        dataset_accuracies = {}
        
        # Cria um vetor de acuracias para cada dataset, baseado no que foi encontrado no diretorio de logs
        files = glob.glob("experimental_results/*_results.csv")
        for f in files:
            filename = os.path.basename(f)
            if "spearman" in filename.lower():
                continue
                
            base_part = filename.replace("_results.csv", "")
            if base_part.endswith("s"):
                parts = base_part.rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        float(parts[1][:-1])
                        base_part = parts[0]
                    except ValueError:
                        pass
                        
            parts = base_part.rsplit("_", 1)
            if len(parts) != 2: continue
            ds_name, alg = parts
            
            if alg not in algorithms:
                continue
                
            try:
                df = pd.read_csv(f)
                if "Same Class" in df.columns:
                    acc = df["Same Class"].mean()
                elif "Accuracy" in df.columns:
                    acc = df["Accuracy"].iloc[0]
                else:
                    acc = 0.0   # Se nao for encontrado log para Algoritmo X do dataset
            except Exception:
                acc = 0.0
                
            if ds_name not in dataset_accuracies:
                dataset_accuracies[ds_name] = {a: 0.0 for a in algorithms}
                
            dataset_accuracies[ds_name][alg] = acc
            
        datasets = sorted(list(dataset_accuracies.keys()))
        if not datasets:
            return False, None, None
            
        n = len(datasets)
        spearman_matrix = np.zeros((n, n))
        
        # Calcula correlacao de spearman entre datasets i e j com base no vetor de acuracias
        for i in range(n):
            vec_i = [dataset_accuracies[datasets[i]][a] for a in algorithms]
            for j in range(n):
                vec_j = [dataset_accuracies[datasets[j]][a] for a in algorithms]
                corr, _ = scipy.stats.spearmanr(vec_i, vec_j)
                if np.isnan(corr):
                    corr = 0.0
                spearman_matrix[i, j] = corr
        
        # Salva matriz em formato predeterminado para uso da funcao de Clique
        df_matrix = pd.DataFrame(spearman_matrix, index=datasets, columns=datasets)
        os.makedirs("experimental_results", exist_ok=True)
        df_matrix.to_csv("experimental_results/spearman_correlation_matrix.csv")
        
        return True, spearman_matrix, datasets

    def compute_max_clique_groups(matrix, labels, threshold):
        G = nx.Graph()
        G.add_nodes_from(labels)
        n = len(labels)
        
        # Itera sobre a matriz de Spearman adicionando arestas entre vertices que possuem correlacao maior que o threshold
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] >= threshold:
                    G.add_edge(labels[i], labels[j], weight=matrix[i, j])

        # Vetor para guardar grupos de vertices do clique maximo            
        groups = []
        remaining_nodes = set(G.nodes())
        
        while remaining_nodes:
            subgraph = G.subgraph(remaining_nodes)
            cliques = list(nx.find_cliques(subgraph))
            if not cliques:
                max_cl = [list(remaining_nodes)[0]]     # Se nao houver clique, pega o primeiro vertice restante
            else:
                max_cl = max(cliques, key=len)  # Pega o maior clique encontrado
            
            groups.append(max_cl)

            # Remocao dos vertices do Clique do grafo
            remaining_nodes -= set(max_cl)
            
        return G, groups

    # Salva resultados de experimentos em um diretorio predeterminado
    def save_results(self, results_list, dataset_name, algorithm_name, total_time=None):
        if not results_list:
            return
        df = pd.DataFrame(results_list)
        os.makedirs("experimental_results", exist_ok=True)
        if total_time is not None:
            filename = os.path.join("experimental_results", f"{dataset_name}_{algorithm_name}_{total_time:.2f}s_results.csv")
        else:
            filename = os.path.join("experimental_results", f"{dataset_name}_{algorithm_name}_results.csv")
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

# Retorna um sample do dataset, garantindo representatividade de todas as classes atraves do Largest Remainder Method
# Amostra e dividida em: classe, indice da serie temporal, serie temporal
def stratified_sample(data, percentage):
    if percentage >= 100.0 or percentage <= 0.0:
        return data
    
    # Coleta as classes das series temporais, agrupando os indices
    class_groups = {}
    for item in data:
        ts_class = item[1]
        if ts_class not in class_groups:
            class_groups[ts_class] = []
        class_groups[ts_class].append(item)
        
    target_total = int(round(len(data) * percentage / 100.0))
    if target_total < len(class_groups):
        target_total = len(class_groups)
    
    # Calculo da quantidade exata de amostras por classe
    exact_counts = {c: target_total * (len(items) / len(data)) for c, items in class_groups.items()}
    
    # Vetores para acompanhar quantas series foram escolhidas, e quantas sobraram
    assignments = {}
    remainders = {}
    for c, exact in exact_counts.items():
        assignments[c] = int(exact)
        remainders[c] = exact - int(exact)
        
    for c in class_groups:
        if assignments[c] == 0:
            assignments[c] = 1
            remainders[c] = -1.0 
            
    allocated = sum(assignments.values())
    
    # Adiciona series por classe, atualizando sua representatividade na amostra
    if allocated < target_total:
        shortfall = target_total - allocated
        sorted_classes = sorted([c for c in class_groups if remainders[c] >= 0], key=lambda x: remainders[x], reverse=True)
        if not sorted_classes:
            sorted_classes = list(class_groups.keys())
            
        for i in range(shortfall):
            assignments[sorted_classes[i % len(sorted_classes)]] += 1
    
    # Remove series de grupos mais representativos, se numero selecionado exce
    elif allocated > target_total:
        overage = allocated - target_total
        for _ in range(overage):
            candidates = [c for c in class_groups if assignments[c] > 1]
            if not candidates:
                break 
            best_candidate = max(candidates, key=lambda c: assignments[c] - exact_counts[c])
            assignments[best_candidate] -= 1

    # Cria o vetor final de amostras, randomizando a distribuicao das series temporais
    sampled_data = []
    for ts_class, items in class_groups.items():
        k = min(len(items), assignments[ts_class])
        sampled_data.extend(random.sample(items, k))
        
    sampled_data.sort(key=lambda x: x[0])
    return sampled_data

#########################################
#                                       #
#       Graphical User Interface        #
#                                       #
#########################################

class DataSideBar(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.dataset_dir = ""
        self.selected_dataset = ""
        self.layout = QtWidgets.QVBoxLayout(self)

        # Tabela superior para listagem dos datasets
        self.dataset_table = QtWidgets.QTableWidget()
        self.dataset_table.setColumnCount(1)
        self.dataset_table.setHorizontalHeaderLabels(["Dataset Folders"])
        self.dataset_table.horizontalHeader().setStretchLastSection(True)
        self.dataset_table.cellDoubleClicked.connect(self.on_dataset_double_clicked)
        self.layout.addWidget(self.dataset_table)

        # Tabela inferior para listagem das series temporais
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Tab especifica para series de treino
        self.train_table = QtWidgets.QTableWidget()
        self.train_table.setSortingEnabled(False)
        self.train_table.setColumnCount(3)
        # Tabela mostra o index da serie no arquivo, sua classe, e uma checkbox para selecionar
        self.train_table.setHorizontalHeaderLabels(["Select", "TS Index", "Class"])
        self.train_table.verticalHeader().setVisible(False) # Opcao de indice, seria confuso com o item acima
        self.train_table.horizontalHeader().setStretchLastSection(True)
        self.train_table.setSortingEnabled(True)
        self.tab_widget.addTab(self.train_table, "Train")
        
        # Tab especifica para series de treino
        self.test_table = QtWidgets.QTableWidget()
        self.test_table.setSortingEnabled(False)
        self.test_table.setColumnCount(3)
        # Tabela mostra o index da serie no arquivo, sua classe, e uma checkbox para selecionar
        self.test_table.setHorizontalHeaderLabels(["Select", "TS Index", "Class"])
        self.test_table.verticalHeader().setVisible(False)  # Opcao de indice, seria confuso com o item acima
        self.test_table.horizontalHeader().setStretchLastSection(True)
        self.test_table.setSortingEnabled(True)
        self.tab_widget.addTab(self.test_table, "Test")

        self.layout.addWidget(self.tab_widget)
        
        # Logica para garantir que apenas 2 series temporais possam ser selecionadas, independente do conjunto
        self.checked_items = []
        self.train_table.itemChanged.connect(self.on_item_changed)
        self.test_table.itemChanged.connect(self.on_item_changed)
        
    def on_item_changed(self, item):
        if item.column() == 0:
            if item.checkState() == QtCore.Qt.Checked:
                if item not in self.checked_items:
                    self.checked_items.append(item)

                    # Verifica se um terceiro item foi selecionado, coloca ele no lugar do mais antigo
                    if len(self.checked_items) > 2:
                        oldest = self.checked_items.pop(0)
                        oldest.tableWidget().blockSignals(True)
                        oldest.setCheckState(QtCore.Qt.Unchecked)
                        oldest.tableWidget().blockSignals(False)
            else:
                if item in self.checked_items:
                    self.checked_items.remove(item)

    # Adiciona os datasets no diretorio para a primeira tabela
    def populate_datasets(self, dir_name):
        self.dataset_dir = dir_name
        self.dataset_table.setRowCount(0)
        try:
            folders = sorted([f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))])
        except Exception:
            return
        
        self.dataset_table.setRowCount(len(folders))
        for row, folder in enumerate(folders):
            item = QtWidgets.QTableWidgetItem(folder)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.dataset_table.setItem(row, 0, item)

    # Selecao de dataset, coleta suas series temporais e guarda o dataset selecionado
    def on_dataset_double_clicked(self, row, column):
        dataset_name = self.dataset_table.item(row, column).text()
        dataset_path = os.path.join(self.dataset_dir, dataset_name)
        self.selected_dataset = dataset_path
        self.populate_timeseries(dataset_path)

    # Adiciona series temporais de treino e teste do dataset selecionado na segunda tabela
    def populate_timeseries(self, dataset_path):
        self.checked_items.clear()
        self.train_table.setRowCount(0)
        self.test_table.setRowCount(0)
        
        try:
            files = sorted([f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))])
        except Exception:
            return
            
        train_file = next((f for f in files if "TRAIN" in f.upper()), None)
        test_file = next((f for f in files if "TEST" in f.upper()), None)
        
        if train_file:
            self.current_train_file = os.path.join(dataset_path, train_file)
            self._load_file_into_table(self.current_train_file, self.train_table)
        else:
            self.current_train_file = None
            
        if test_file:
            self.current_test_file = os.path.join(dataset_path, test_file)
            self._load_file_into_table(self.current_test_file, self.test_table)
        else:
            self.current_test_file = None

    # Carrega as series temporais, realizando um preprocessamento das classes e indices
    def _load_file_into_table(self, filepath, table):
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            return
            
        table.blockSignals(True)
        table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            if '\t' in line:
                parts = line.split('\t')
            elif ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
                
            ts_class = parts[0] if len(parts) > 0 else ""
            
            # Checkbox para selecao das series
            item_check = QtWidgets.QTableWidgetItem()
            item_check.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item_check.setCheckState(QtCore.Qt.Unchecked)
            table.setItem(row, 0, item_check)

            # Indice da serie no arquivo
            item_index = QtWidgets.QTableWidgetItem(str(row))
            item_index.setFlags(item_index.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, 1, item_index)
            
            # Classe da serie no arquivo
            item_class = QtWidgets.QTableWidgetItem(ts_class)
            item_class.setFlags(item_class.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, 2, item_class)
            
        table.blockSignals(False)

    # Carrega as series selecionadas pelo usuario para execucao individual
    def get_selected_data(self):
        data = []
        for item in self.checked_items:
            table = item.tableWidget()
            row = item.row()
            filepath = self.current_train_file if table == self.train_table else self.current_test_file
            if not filepath:
                continue
            try:
                with open(filepath, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    if row < len(lines):
                        line = lines[row]
                        if '\t' in line:
                            parts = line.split('\t')
                        elif ',' in line:
                            parts = line.split(',')
                        else:
                            parts = line.split()
                        
                        if len(parts) > 1:
                            ts_data = np.array([float(x) for x in parts[1:] if x], dtype=np.float64)
                            data.append(ts_data)
            except Exception as e:
                print("Error reading data:", e)
        return data

    # Coleta o path do dataset selecionado
    def get_selected_dataset(self):
        return os.path.basename(self.selected_dataset) if hasattr(self, 'selected_dataset') and self.selected_dataset else "dataset"

    # Coleta todas as series de treino e teste que serao usadas para execucao de um algoritmo para o dataset
    def get_all_data(self):
        def read_file(filepath):
            data = []
            if not filepath:
                return data
            try:
                with open(filepath, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    for row, line in enumerate(lines):
                        if '\t' in line:
                            parts = line.split('\t')
                        elif ',' in line:
                            parts = line.split(',')
                        else:
                            parts = line.split()
                        
                        ts_class = parts[0] if len(parts) > 0 else ""
                        if len(parts) > 1:
                            ts_data = np.array([float(x) for x in parts[1:] if x], dtype=np.float64)
                            data.append((row, ts_class, ts_data))
            except Exception as e:
                print("Error reading data:", e)
            return data
            
        train_data = read_file(self.current_train_file)
        test_data = read_file(self.current_test_file)
        return train_data, test_data

class VisualizationFrame(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Define quatro tabs para visualizacao
        tabs = ["TS Correlation", "TS Heatmap", "Spearman Matrix", "Correlation Graph"]
        self.canvases = {}

        # Cria um widget para cada tab definida
        for tab_name in tabs:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            
            fig = Figure()
            canvas = FigureCanvas(fig)
            self.canvases[tab_name] = (fig, canvas)
            
            tab_layout.addWidget(canvas)
            self.tab_widget.addTab(tab, tab_name)
            
        self.layout.addWidget(self.tab_widget)

    def plot_correlation(self, ts1, ts2, path):
        fig, canvas = self.canvases["TS Correlation"]
        fig.clear()
        ax = fig.add_subplot(111)

        # Series com NaN dao erro, e devem ser pre-processadas aqui tambem
        ts1 = preprocess_timeseries(ts1, True)
        ts2 = preprocess_timeseries(ts2, True)
        
        if ts1.ndim > 1:
            ts1 = ts1.flatten()
        if ts2.ndim > 1:
            ts2 = ts2.flatten()
        
        # Calcula um offset baseado nos valores maximo e minimo das series, para evitar sobreposicoes
        offset = np.max(ts1) - np.min(ts2) + (np.max(ts1) - np.min(ts1)) * 0.5
        ax.plot(ts1, label="TS 1", color="blue")
        ax.plot(ts2 + offset, label="TS 2", color="orange")
        
        # Desenha uma linha demonstrando quais pontos da TS1 foram casados com a TS2
        for i, j in path:
            ax.plot([i, j], [ts1[int(round(i))], ts2[int(round(j))] + offset], color='gray', alpha=0.3, linewidth=1.0)
            
        #ax.set_title("Time Series Alignment")
        ax.legend()
        fig.tight_layout()
        canvas.draw()

    def plot_heatmap(self, cost_matrix, path):
        fig, canvas = self.canvases["TS Heatmap"]
        fig.clear()
        ax = fig.add_subplot(111)
        
        matrix_to_plot = np.copy(cost_matrix).astype(float)
        matrix_to_plot[np.isinf(matrix_to_plot)] = np.nan
        
        im = ax.imshow(matrix_to_plot, origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')
        fig.colorbar(im, ax=ax)
        
        # Plota o melhor caminho em vermelho
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, color='red', linewidth=2, label="Optimal Path")
        
        #ax.set_title("Accumulated Cost Matrix & Path")
        ax.set_xlabel("TS 2 Index")
        ax.set_ylabel("TS 1 Index")
        ax.legend()
        fig.tight_layout()
        canvas.draw()

    def plot_spearman_matrix(self, spearman_matrix, labels):
        fig, canvas = self.canvases["Spearman Matrix"]
        fig.clear()
        ax = fig.add_subplot(111)
        
        im = ax.imshow(spearman_matrix, origin='lower', cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        
        # Adiciona nomes dos datasets nos eixos
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        if len(labels) <= 15:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{spearman_matrix[i, j]:.2f}", ha="center", va="center", color="black" if abs(spearman_matrix[i, j]) < 0.5 else "white")
        
        #ax.set_title("Spearman Correlation Between Datasets")
        fig.tight_layout()
        canvas.draw()

    def plot_clique_graph(self, G, groups):
        fig, canvas = self.canvases["Correlation Graph"]
        fig.clear()
        ax = fig.add_subplot(111)
        
        pos = nx.spring_layout(G, seed=42)
        # Mapa de cores para representar os cliques
        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']
        
        # Itera sobre os grupos de vertices que representam os cliques, colorindo eles
        for idx, group in enumerate(groups):
            color = colors[idx % len(colors)]
            nx.draw_networkx_nodes(G, pos, nodelist=group, node_color=color, ax=ax, label=f"Group {idx+1}")
            
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        #ax.set_title("Dataset Correlation Graph (Max Clique Grouping)")
        ax.legend()
        fig.tight_layout()
        canvas.draw()

class ControlPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QVBoxLayout(self)

        # Campos para definir amostragem de Treino e Teste
        self.train_sampling_spinbox = QtWidgets.QDoubleSpinBox()
        self.train_sampling_spinbox.setRange(1.0, 100.0)
        self.train_sampling_spinbox.setValue(20.0)
        
        self.test_sampling_spinbox = QtWidgets.QDoubleSpinBox()
        self.test_sampling_spinbox.setRange(1.0, 100.0)
        self.test_sampling_spinbox.setValue(20.0)
        
        sampling_layout = QtWidgets.QFormLayout()
        sampling_layout.addRow("Train Sampling (%):", self.train_sampling_spinbox)
        sampling_layout.addRow("Test Sampling (%):", self.test_sampling_spinbox)
        self.layout.addLayout(sampling_layout)

        # Tabs: Casamento de Series // Outros
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Tab para casamento de series temporais - Execucao Unica ou por Dataset
        self.single_exec_tab = QtWidgets.QWidget()
        self.single_exec_layout = QtWidgets.QFormLayout(self.single_exec_tab)
        
        # Parametros dos algoritmos
        self.radius_spinbox = QtWidgets.QDoubleSpinBox()
        self.radius_spinbox.setValue(10.0)
        self.slope_spinbox = QtWidgets.QDoubleSpinBox()
        self.slope_spinbox.setValue(2.0)
        self.eps_spinbox = QtWidgets.QDoubleSpinBox()
        self.eps_spinbox.setValue(1.0)
        self.gamma_spinbox = QtWidgets.QDoubleSpinBox()
        self.gamma_spinbox.setValue(1.0)
        self.warpPenalty = QtWidgets.QDoubleSpinBox()
        self.warpPenalty.setValue(1.0)

        self.single_exec_layout.setVerticalSpacing(15)

        # Tipo da janela de restricao
        self.constraint_combo = QtWidgets.QComboBox()
        self.constraint_combo.addItems(["Sakoe Chiba", "Itakura"])
        
        # Algoritmos implememtados
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItems(["DTW", "CDTW", "LCSS", "SDTW", "ADTW"])

        self.single_exec_layout.addRow("Algorithm:", self.algorithm_combo)
        self.single_exec_layout.addRow("Constraint (All except CDTW):", self.constraint_combo)
        self.single_exec_layout.addRow("Radius (%) (Sakoe Chiba):", self.radius_spinbox)
        self.single_exec_layout.addRow("Slope (Itakura):", self.slope_spinbox)
        self.single_exec_layout.addRow("Eps (LCSS):", self.eps_spinbox)
        self.single_exec_layout.addRow("Gamma (SDTW):", self.gamma_spinbox)
        self.single_exec_layout.addRow("Warp Penalty (ADTW):", self.warpPenalty)
        
        # Botao para execucao de teste individual
        self.run_single_button = QtWidgets.QPushButton("Run TS Correlation")
        self.single_exec_layout.addRow(self.run_single_button)

        # Checkbox para rodar todos os algoritmos com as configuracoes especificadas para o dataset
        self.run_all_algorithms_checkbox = QtWidgets.QCheckBox("Run All Algorithms")
        self.single_exec_layout.addRow(self.run_all_algorithms_checkbox)

        # Botao para execucao de dataset
        self.run_dataset_button = QtWidgets.QPushButton("Run Dataset Correlation")
        self.single_exec_layout.addRow(self.run_dataset_button)
        
        self.tab_widget.addTab(self.single_exec_tab, "Correlator")
        
        # Tab de Classificadores, Spearman e Grafo
        self.batch_exec_tab = QtWidgets.QWidget()
        self.batch_exec_layout = QtWidgets.QFormLayout(self.batch_exec_tab)
        
        # Classificadores implementados
        self.classifier_combo = QtWidgets.QComboBox()
        self.classifier_combo.addItems(["1-NN", "Naive Bayes", "Linear SVM"])
        self.batch_exec_layout.addRow("Classifier Algorithm:", self.classifier_combo)

        self.batch_exec_layout.setVerticalSpacing(15)
        
        # Execucao do classificador
        self.run_classifier_button = QtWidgets.QPushButton("Run Classifier")
        self.batch_exec_layout.addRow(self.run_classifier_button)
        
        # Execucao da Matriz de Spearman, com base nos logs
        self.compute_spearman_btn = QtWidgets.QPushButton("Compute Spearman Matrix")
        self.batch_exec_layout.addRow(self.compute_spearman_btn)
        
        # Execucao do clique maximo, com threshold escolhido
        self.graph_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.graph_threshold_spinbox.setRange(0.0, 1.0)
        self.graph_threshold_spinbox.setSingleStep(0.05)
        self.graph_threshold_spinbox.setValue(0.8)
        self.batch_exec_layout.addRow("Graph Edge Threshold:", self.graph_threshold_spinbox)

        self.generate_clique_btn = QtWidgets.QPushButton("Generate Clique Graph")
        self.batch_exec_layout.addRow(self.generate_clique_btn)
        
        self.tab_widget.addTab(self.batch_exec_tab, "Others")

        self.layout.addWidget(self.tab_widget)

        # Caixa de metricas da execucao
        self.execution_group = QtWidgets.QGroupBox("Execution Data")
        self.execution_layout = QtWidgets.QFormLayout(self.execution_group)
        
        self.score_label = QtWidgets.QLabel("N/A")
        self.time_label = QtWidgets.QLabel("N/A")
        self.space_label = QtWidgets.QLabel("N/A")
        
        self.execution_layout.addRow("Score:", self.score_label)
        self.execution_layout.addRow("Computing Time:", self.time_label)
        self.execution_layout.addRow("Space Usage:", self.space_label)

        self.layout.addWidget(self.execution_group)

    # Coleta os parametros para execucao no CTSPU
    def get_configuration(self):
        return {
            "algorithm": self.algorithm_combo.currentText(),
            "run_all_algorithms": self.run_all_algorithms_checkbox.isChecked(),
            "constraint": self.constraint_combo.currentText(),
            "radius": self.radius_spinbox.value(),
            "slope": self.slope_spinbox.value(),
            "eps": self.eps_spinbox.value(),
            "gamma": self.gamma_spinbox.value(),
            "warp_penalty": self.warpPenalty.value(),
            "classifier_algorithm": getattr(self, "classifier_combo", None) and self.classifier_combo.currentText() or "1-NN",
            "train_sampling_percentage": getattr(self, "train_sampling_spinbox", None) and self.train_sampling_spinbox.value() or 100.0,
            "test_sampling_percentage": getattr(self, "test_sampling_spinbox", None) and self.test_sampling_spinbox.value() or 100.0
        }

    # Atualiza resultados e metricas apos execucao
    def set_execution_data(self, score, computing_time, space_usage):
        self.score_label.setText(f"{score:.4f}")
        self.time_label.setText(f"{computing_time:.4f} s")
        self.space_label.setText(f"{space_usage:.4f} MB")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Time Series Correlator")

        # Inicia VisualizationFrame como Widget central
        self.visualization_frame = VisualizationFrame()
        self.setCentralWidget(self.visualization_frame)

        # Cria DataSideBar e ControlPanel como docks, mas sem features para fechar ou mover
        self.data_sidebar = DataSideBar()
        self.sidebar_dock = QtWidgets.QDockWidget("Data", self, features=QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.sidebar_dock.setWidget(self.data_sidebar)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.sidebar_dock)

        self.control_panel = ControlPanel()
        self.control_dock = QtWidgets.QDockWidget("Control Panel", self, features=QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.control_dock.setWidget(self.control_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.control_dock)

        self.control_panel.run_single_button.clicked.connect(self.run_single_computation)
        self.control_panel.run_dataset_button.clicked.connect(self.run_dataset_computation)
        self.control_panel.run_classifier_button.clicked.connect(self.run_classification_task)
        self.control_panel.compute_spearman_btn.clicked.connect(self.create_spearmann_correlation)
        self.control_panel.generate_clique_btn.clicked.connect(self.generate_clique_graph)

        # Cria a barrinha para abrir dataset
        self.create_actions()
        self.create_toolbars()

    @Slot()
    def open_dataset_folder(self):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if dir_name:
            self.data_sidebar.populate_datasets(dir_name)

    @Slot()
    def run_single_computation(self):
        data = self.data_sidebar.get_selected_data()
        if len(data) != 2:
            QtWidgets.QMessageBox.warning(self, "Selection Error", "Please select exactly two time series.")
            return
            
        # Coleta os parametros da GUI e inicializa o CTSPU
        config = self.control_panel.get_configuration()
        processor = CTSPU(data, config)

        # Transforma o cursor para mostrar execucao em andamento
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        success = processor.run_single()
        
        if success:
            # Se a execucao foi bem sucedida, envia os resultados para plotagem e display
            self.control_panel.set_execution_data(processor.score, processor.computing_time, processor.space_usage)
            self.visualization_frame.plot_correlation(processor.data[0], processor.data[1], processor.path)
            self.visualization_frame.plot_heatmap(processor.cost_matrix, processor.path)
        else:
            QtWidgets.QMessageBox.critical(self, "Computation Error", "Failed to compute distance.")
        
        # Reinicia o cursor
        QtWidgets.QApplication.restoreOverrideCursor()

    @Slot()
    def run_dataset_computation(self):
        train_data, test_data = self.data_sidebar.get_all_data()
        dataset_name = self.data_sidebar.get_selected_dataset()
        
        if not train_data or not test_data:
            QtWidgets.QMessageBox.warning(self, "Data Error", "Could not load train or test data for the selected dataset.")
            return
        
        # Realiza divisao dos dados baseado no percentual de amostragem escolhido
        config = self.control_panel.get_configuration()
        train_sampling = config.get("train_sampling_percentage", 100.0)
        test_sampling = config.get("test_sampling_percentage", 100.0)
        train_data = stratified_sample(train_data, train_sampling)
        test_data = stratified_sample(test_data, test_sampling)
        
        processor = CTSPU(None, config)
        
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            # Neste caso os resultados sao salvos direto para logs, retornando somente o estado
            success = processor.run_dataset(train_data, test_data, dataset_name)
            if success:
                if config.get("run_all_algorithms", False):
                    QtWidgets.QMessageBox.information(self, "Batch Complete", "All algorithms ran successfully on the dataset.\nResults saved to experimental_results folder.")
                else:
                    QtWidgets.QMessageBox.information(self, "Batch Complete", "Dataset correlation ran successfully.\nResults saved to experimental_results folder.")
            else:
                QtWidgets.QMessageBox.warning(self, "Batch Failed", "An error occurred during dataset correlation.")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            
    @Slot()
    def run_classification_task(self):
        train_data, test_data = self.data_sidebar.get_all_data()
        dataset_name = self.data_sidebar.get_selected_dataset()
        
        if not train_data or not test_data:
            QtWidgets.QMessageBox.warning(self, "Data Error", "Could not load train or test data for the selected dataset.")
            return
        
        # Realiza divisao dos dados baseado no percentual de amostragem escolhido
        config = self.control_panel.get_configuration()
        train_sampling = config.get("train_sampling_percentage", 100.0)
        test_sampling = config.get("test_sampling_percentage", 100.0)
        train_data = stratified_sample(train_data, train_sampling)
        test_data = stratified_sample(test_data, test_sampling)
        
        clf_algo = config.get("classifier_algorithm", "1-NN")
        processor = CTSPU(None, config)
        
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        try:
            success, acc, comp_time, mem_usage = processor.run_classifier(train_data, test_data, dataset_name, clf_algo)
            if success:
                msg = f"Algorithm: {clf_algo}\nAccuracy: {acc*100:.2f}%\nTime: {comp_time:.4f} s\nMemory: {mem_usage:.4f} MB"
                QtWidgets.QMessageBox.information(self, "Classification Complete", msg)
            else:
                QtWidgets.QMessageBox.warning(self, "Classification Failed", "An error occurred during classification.")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    @Slot()
    def create_spearmann_correlation(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        try:
            # Retornam matrix e datasets utilizados, porem a matriz tbm e salva em log
            success, matrix, labels = CTSPU.compute_spearman_correlation()
            if success:
                self.visualization_frame.plot_spearman_matrix(matrix, labels)
                self.visualization_frame.tab_widget.setCurrentIndex(2) # Muda para tab de Spearman em VisFrame
                QtWidgets.QMessageBox.information(self, "Success", "Spearman Correlation Matrix generated successfully and saved to experimental_results/spearman_correlation_matrix.csv.")
            else:
                QtWidgets.QMessageBox.warning(self, "No Data", "Could not find sufficient dataset results in experimental_results/ to compute the matrix. Make sure you run datasets first.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute Spearman Correlation: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    @Slot()
    def generate_clique_graph(self):
        # Recupera o log da matrix de spearman, que deve ser computada antes
        try:
            df = pd.read_csv("experimental_results/spearman_correlation_matrix.csv", index_col=0)
            matrix = df.values
            labels = df.index.tolist()
        except Exception:
            QtWidgets.QMessageBox.warning(self, "No Matrix", "Spearman matrix not found. Please compute it first.")
            return
        
        # Coleta parametro de threshold para adicao de arestas
        threshold = self.control_panel.graph_threshold_spinbox.value()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            # Retorna o grafo original e os grupos de vertices dos cliques obtidos
            G, groups = CTSPU.compute_max_clique_groups(matrix, labels, threshold)
            self.visualization_frame.plot_clique_graph(G, groups)
            self.visualization_frame.tab_widget.setCurrentIndex(3) # Muda para tab do grafo em VisFrame
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate graph: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # Acoes para a barrinha de abrir diretorio
    def create_actions(self):
        icon = QtGui.QIcon.fromTheme(QtGui.QIcon.ThemeIcon.DocumentOpen, QtGui.QIcon(':/images/open.png'))
        self._open_dataset_act = QtGui.QAction(icon, "&Open Dataset Folder...", self,
                                 shortcut=QtGui.QKeySequence.StandardKey.Open,
                                 statusTip="Open database folder",
                                 triggered=self.open_dataset_folder)

    def create_toolbars(self):
        file_toolbar = self.addToolBar("Open")
        file_toolbar.addAction(self._open_dataset_act)
        
if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    main_window = MainWindow()

    # Usar .show para abrir a janela fora do fullscreen
    main_window.showMaximized()

    sys.exit(app.exec())