# Directory and files handling
import os
import sys

# Main funcionalities, graph, classifier, etc...
import math
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats
import glob

# Image visualization and plotting

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Logs and metrics
import time
import random
import tracemalloc

# This thing aparently accelerates code?? IDK
from numba import njit

# Graphical User Interface
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Slot

# Global variables
GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}

# Utility code for handling data, also taken from TSLearn

def preprocess_timeseries(ts, remove_nans=False):
    if ts.ndim <= 1:
        ts = ts.reshape(-1, 1)
    if remove_nans:
        sz = len(ts)
        while sz > 0 and np.all(np.isnan(ts[sz - 1])):
            sz -= 1
        ts = ts[:sz]
    return ts

#########################################
#                                       #
#         Dynamic Time Warping          #
#            Pego do TSLearn            #
#########################################

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

def make_compute_mask():
    sakoe_chiba_mask_ = make_sakoe_chiba_mask()
    itakura_mask_ = make_itakura_mask()

    def _compute_mask_generic(
            sz1,
            sz2,
            global_constraint=0,
            sakoe_chiba_radius=None,
            itakura_max_slope=None,
    ):
        if (
                global_constraint == 0
                and sakoe_chiba_radius is not None
                and itakura_max_slope is not None
        ):
            raise RuntimeWarning(
                "global_constraint is not set for DTW, but both "
                "sakoe_chiba_radius and itakura_max_slope are "
                "set, hence global_constraint cannot be inferred "
                "and no global constraint will be used."
            )
        if global_constraint == 2 or (
                global_constraint == 0 and sakoe_chiba_radius is not None
        ):
            if sakoe_chiba_radius is None:
                sakoe_chiba_radius = 1
            mask = sakoe_chiba_mask_(sz1, sz2, radius=sakoe_chiba_radius)

        elif global_constraint == 1 or (
                global_constraint == 0 and itakura_max_slope is not None
        ):
            if itakura_max_slope is None:
                itakura_max_slope = 2.0
            mask = itakura_mask_(sz1, sz2, max_slope=itakura_max_slope)
        else:
            mask = np.full((sz1, sz2), True)
        return mask

    return njit(nogil=True, cache=True, fastmath=True)(_compute_mask_generic)

def compute_mask(
    s1,
    s2,
    global_constraint=0,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
):
    r"""Compute the mask (region constraint).

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,) or int
        A time series or integer.
        If shape is (sz1,), the time series is assumed to be univariate.
        If int, size sz1 used to dimension the mask.
    s2 : array-like, shape=(sz2, d) or (sz2,) or int
        Another time series or integer.
        If shape is (sz2,), the time series is assumed to be univariate.
        If int, size sz2 used to dimension the mask.
    global_constraint : {0, 1, 2} (default: 0)
        Global constraint to restrict admissible paths for DTW:
        - "itakura" if 1
        - "sakoe_chiba" if 2
        - no constraint otherwise
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to 2 (sakoe-chiba), a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to 1 (itakura), a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    be : np object or string or None
        np. If `be` is an instance of the class `NumPynp` or the string `"numpy"`,
        the NumPy np is used.
        If `be` is an instance of the class `PyTorchnp` or the string `"pytorch"`,
        the PyTorch np is used.
        If `be` is `None`, the np is determined by the input arrays.
        See our :ref:`dedicated user-guide page <np>` for more information.

    Returns
    -------
    mask : array-like, shape=(sz1, sz2)
        Constraint region.

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    # The output mask will be of shape (sz1, sz2)
    if isinstance(s1, int) and isinstance(s2, int):
        sz1, sz2 = s1, s2
    else:
        sz1 = len(s1)
        sz2 = len(s2)


    compute_mask_fn = make_compute_mask()
    x = compute_mask_fn(
        sz1,
        sz2,
        global_constraint,
        sakoe_chiba_radius,
        itakura_max_slope,
    )
    return x

@njit(nogil=True, cache=True, fastmath=True)
def njit_local_squared_dist(x, y):
    """Compute the squared distance between two vectors.

    Parameters
    ----------
    x : array-like, shape=(d,)
        A vector.
    y : array-like, shape=(d,)
        Another vector.

    Returns
    -------
    dist : float
        Squared distance between x and y.
    """
    dist = 0.0
    for di in range(x.shape[0]):
        diff = x[di] - y[di]
        dist += diff * diff
    return dist

@njit(nogil=True, cache=True, fastmath=True)
def njit_accumulated_matrix(s1, s2, mask):
    """Compute the accumulated cost matrix score between two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        First time series.
    s2 : array-like, shape=(sz2, d)
        Second time series.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.0

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                cum_sum[i + 1, j + 1] = njit_local_squared_dist(s1[i], s2[j])
                cum_sum[i + 1, j + 1] += min(
                    cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]
                )
    return cum_sum[1:, 1:]

@njit(nogil=True, cache=True, fastmath=True)
def njit_return_path(acc_cost_mat):
    """Return path from accumulated cost matrix.

    Parameters
    ----------
    acc_cost_mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.

    Returns
    -------
    path : list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to a first time series s1 and the second one
        corresponds to a second time series s2.
    """
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = np.array(
                [
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1],
                ]
            )
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]

def dtw_path(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return both the path and the
    similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} (X_{i} - Y_{j})^2}

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.

    float
        Similarity score

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """

    s1 = preprocess_timeseries(s1, remove_nans=True)
    s2 = preprocess_timeseries(s2, remove_nans=True)

    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if np.shape(s1)[1] != np.shape(s2)[1]:
        raise ValueError("All input time series must have the same feature size.")

    mask = compute_mask(
        s1,
        s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius,
        itakura_max_slope,
    )
    
    acc_cost_mat = njit_accumulated_matrix(s1, s2, mask=mask)
    path = njit_return_path(acc_cost_mat)

    return path, np.sqrt(acc_cost_mat[-1, -1])

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
def cdtw_accumulated_dits_matrix(r, t, d):
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

def cdtw_path(
    s1,
    s2,
):
    s1 = preprocess_timeseries(s1, remove_nans=True)
    s2 = preprocess_timeseries(s2, remove_nans=True)

    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )
    
    if np.shape(s1)[1] != np.shape(s2)[1]:
        raise ValueError("All input time series must have the same feature size.")
    
    mask = cdtw_distance_matrix(s1, s2)

    acc_cost_mat = cdtw_accumulated_dits_matrix(s1, s2, mask)

    path = cdtw_return_path(s1, s2, acc_cost_mat)

    return path, np.sqrt(acc_cost_mat[-1, -1])

#########################################
#                                       #
#       Longest Common Subsequence      #
#             Pego do TSLearn           #
#########################################

@njit(nogil=True, cache=True, fastmath=True)
def njit_lcss_accumulated_matrix(s1, s2, eps, mask):
    """Compute the longest common subsequence similarity score between
    two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        First time series.
    s2 : array-like, shape=(sz2, d)
        Second time series.
    eps : float
        Matching threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
        
    Returns
    -------
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    acc_cost_mat = np.full((l1 + 1, l2 + 1), 0)

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if mask[i - 1, j - 1]:
                if np.sqrt(njit_local_squared_dist(s1[i - 1], s2[j - 1])) <= eps:
                    acc_cost_mat[i][j] = 1 + acc_cost_mat[i - 1][j - 1]
                else:
                    acc_cost_mat[i][j] = max(
                        acc_cost_mat[i][j - 1], acc_cost_mat[i - 1][j]
                    )

    return acc_cost_mat

@njit(nogil=True, cache=True, fastmath=True)
def _return_lcss_path_njit(s1, s2, eps, mask, acc_cost_mat, sz1, sz2):
    i = sz1
    j = sz2
    path = [(0, 0)] # type inference
    path.pop()

    while i > 0 and j > 0:
        if mask[i - 1, j - 1]:
            squared_dist = njit_local_squared_dist(s1[i - 1], s2[j - 1])

            if np.sqrt(squared_dist) <= eps:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
                continue
                
        if acc_cost_mat[i - 1][j] > acc_cost_mat[i][j - 1]:
            i = i - 1
        else:
            j = j - 1
            
    path.reverse()
    return path

def return_lcss_path(s1, s2, eps, mask, acc_cost_mat, sz1, sz2):
    """Return the Longest Common Subsequence (LCSS) path.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        A time series.
    s2 : array-like, shape=(sz2, d)
        Another time series.
    eps : float
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    sz1 : int
        Length of the first time series.
    sz2 : int
        Length of the second time series.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    path : list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.
    """
    s1 = np.array(s1)
    s2 = np.array(s2)
    acc_cost_mat = np.array(acc_cost_mat)
    return _return_lcss_path_njit(s1, s2, eps, mask, acc_cost_mat, sz1, sz2)

def lcss_path(
    s1,
    s2,
    eps=1,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
):
    r"""Compute the Longest Common Subsequence (LCSS) similarity measure
    between (possibly multidimensional) time series and return both the
    path and the similarity.

    LCSS is computed by matching indexes that are met up until the eps
    threshold, so it leaves some points unmatched and focuses on the
    similar parts of two sequences. The matching can occur even if the
    time indexes are different. One can set additional constraints to
    the set of acceptable paths: the Sakoe-Chiba band which is parametrized
    by a radius or the Itakura parallelogram which is parametrized by a
    maximum slope. Both these constraints consists in forcing paths to lie
    close to the diagonal.

    To retrieve a meaningful similarity value from the length of the
    longest common subsequence, the percentage of that value regarding
    the length of the shortest time series is returned.

    According to this definition, the values returned by LCSS range from
    0 to 1, the highest value taken when two time series fully match,
    and vice-versa. It is not required that both time series share the
    same size, but they must be the same dimension. LCSS was originally
    presented in [1]_ and is discussed in more details in our
    :ref:`dedicated user-guide page <lcss>`.

    Notes
    -----
    Contrary to Dynamic Time Warping and variants, an LCSS path does not need to be contiguous.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    eps : float (default: 1.)
        Maximum matching distance threshold.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for LCSS.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:/delta mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    be : np object or string or None
        np. If `be` is an instance of the class `NumPynp` or the string `"numpy"`,
        the NumPy np is used.
        If `be` is an instance of the class `PyTorchnp` or the string `"pytorch"`,
        the PyTorch np is used.
        If `be` is `None`, the np is determined by the input arrays.
        See our :ref:`dedicated user-guide page <np>` for more information.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2

    float
        Similarity score

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
            Similar Multidimensional Trajectories", In Proceedings of the
            18th International Conference on Data Engineering (ICDE '02).
            IEEE Computer Society, USA, 673.

    """
    s1 = preprocess_timeseries(s1, remove_nans=True)
    s2 = preprocess_timeseries(s2, remove_nans=True)

    mask = compute_mask(
        s1,
        s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius,
        itakura_max_slope,
    )

    l1 = s1.shape[0]
    l2 = s2.shape[0]

    acc_cost_mat = njit_lcss_accumulated_matrix(s1, s2, eps, mask)

    path = return_lcss_path(s1, s2, eps, mask, acc_cost_mat, l1, l2)

    return path, float(acc_cost_mat[-1][-1]) / min([l1, l2])

#########################################
#                                       #
#       Soft Dynamic Time Warping       #
#         Pego da Aeon Toolkit          #
#    Adaptado para o modelo TSlearn     #
#                                       #
#########################################

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

def sdtw_path(
    s1: np.ndarray,
    s2: np.ndarray,
    gamma: float = 1.0,
    window: float | None = None,
    itakura_max_slope: float | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the soft-DTW alignment path between two time series.

    Parameters
    ----------
    s1 : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    s2 : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    gamma : float, default=1.0
        Controls the smoothness of the warping. A value of 0.0 is equivalent to DTW.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The soft-DTW distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    """
    cost_matrix = soft_dtw_cost_matrix(s1, s2, gamma, window, itakura_max_slope)
    return (
        compute_min_return_path(cost_matrix),
        abs(cost_matrix[s1.shape[-1] - 1, s2.shape[-1] - 1]),
    )

#########################################
#                                       #
#     Amerced Dynamic Time Warping      #
#         Pego da Aeon Toolkit          #
#    Adaptado para o modelo TSlearn     #
#                                       #
#########################################

@njit(nogil=True, cache=True, fastmath=True)
def _univariate_squared_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        difference = x[i] - y[i]
        distance += difference * difference
    return distance

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

@njit(nogil=True, cache=True, fastmath=True)
def adtw_path(
    s1: np.ndarray,
    s2: np.ndarray,
    window: float | None = None,
    itakura_max_slope: float | None = None,
    warp_penalty: float = 1.0,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the ADTW alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    warp_penalty: float, default=1.0
        Penalty for warping. A high value will mean less warping.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The ADTW distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import adtw_alignment_path
    >>> x = np.array([[1, 2, 3, 6]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> adtw_alignment_path(x, y)
    ([(0, 0), (1, 1), (2, 2), (3, 3)], 4.0)
    """
    cost_matrix = adtw_cost_matrix(s1, s2, window, itakura_max_slope, warp_penalty)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[s1.shape[-1] - 1, s2.shape[-1] - 1],
    )

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
        self.mask = None

    def _compute_distance(self, ts1, ts2):
        ts1 = preprocess_timeseries(ts1, remove_nans=True)
        ts2 = preprocess_timeseries(ts2, remove_nans=True)
        
        alg = self.config.get("algorithm", "DTW")
        constraint = self.config.get("constraint", "Sakoe Chiba")
        
        # Radius is now treated as a percentage (e.g. 10.0 for 10%)
        radius_percent = float(self.config.get("radius", 10.0)) / 100.0
        
        slope = float(self.config.get("slope", 2.0))
        eps = float(self.config.get("eps", 1.0))
        gamma = float(self.config.get("gamma", 1.0))
        warp_penalty = float(self.config.get("warp_penalty", 1.0))
        
        global_constraint = GLOBAL_CONSTRAINT_CODE.get(constraint.lower().replace(" ", "_"), 0)
        
        sz1 = len(ts1)
        sz2 = len(ts2)
        
        # Calculate dynamic absolute radius based on the series length
        dynamic_radius = int(max(sz1, sz2) * radius_percent)
        
        if alg == "DTW":
            compute_mask_fn = make_compute_mask()
            mask = compute_mask_fn(sz1, sz2, global_constraint, dynamic_radius, slope)
            cost_matrix = njit_accumulated_matrix(ts1, ts2, mask)
            path = njit_return_path(cost_matrix)
            score = cost_matrix[-1, -1] if cost_matrix.size > 0 else float('inf')
        elif alg == "CDTW":
            window = dynamic_radius if global_constraint == 2 else None
            itakura_slope = slope if global_constraint == 1 else None

            cdtw_dist = cdtw_distance_matrix(ts1, ts2, window)
            cost_matrix = cdtw_accumulated_dits_matrix(ts1, ts2, cdtw_dist)
            path = cdtw_return_path(ts1, ts2, cost_matrix)
            score = np.sqrt(cost_matrix[-1, -1]) if cost_matrix.size > 0 else float('inf')
            
            # Reduce cost matrix to original dimensions for plotting (M x N)
            cost_matrix = cost_matrix[::2, ::2]
            mask = np.full((sz1, sz2), True)
        elif alg == "LCSS":
            compute_mask_fn = make_compute_mask()
            mask = compute_mask_fn(sz1, sz2, global_constraint, dynamic_radius, slope)
            cost_matrix = njit_lcss_accumulated_matrix(ts1, ts2, eps, mask)
            path = return_lcss_path(ts1, ts2, eps, mask, cost_matrix, sz1, sz2)
            score = float(cost_matrix[-1][-1]) / min([sz1, sz2]) if cost_matrix.size > 0 else 0.0
        elif alg == "SDTW":
            window = radius_percent if global_constraint == 2 else None
            itakura_slope = slope if global_constraint == 1 else None

            # Need to transpose time series due to a difference between Aeon and TSLearn
            cost_matrix = soft_dtw_cost_matrix(ts1.T, ts2.T, gamma, window, itakura_slope)
            path = compute_min_return_path(cost_matrix)
            score = abs(cost_matrix[-1, -1]) if cost_matrix.size > 0 else float('inf')
            mask = np.full((sz1, sz2), True)
        elif alg == "ADTW":
            window = radius_percent if global_constraint == 2 else None
            itakura_slope = slope if global_constraint == 1 else None

            # Need to transpose time series due to a difference between Aeon and TSLearn
            cost_matrix = adtw_cost_matrix(ts1.T, ts2.T, window, itakura_slope, warp_penalty)
            path = compute_min_return_path(cost_matrix)
            score = cost_matrix[-1, -1] if cost_matrix.size > 0 else float('inf')
            mask = np.full((sz1, sz2), True)
        else:
            return False, None, None, None, None
            
        return True, score, path, cost_matrix, mask

    def run_single(self):
        if not self.data or len(self.data) != 2:
            return False
            
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            success, score, path, cost_matrix, mask = self._compute_distance(self.data[0], self.data[1])
            if not success:
                tracemalloc.stop()
                return False
                
            self.score = score
            self.path = path
            self.cost_matrix = cost_matrix
            self.mask = mask
        except Exception as e:
            print("Computation error:", e)
            tracemalloc.stop()
            return False
            
        self.computing_time = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        self.space_usage = peak / (1024 * 1024)
        tracemalloc.stop()
        
        return True

    def run_dataset(self, train_data, test_data, dataset_name):
        run_all = self.config.get("run_all_algorithms", False)
        
        if run_all:
            algorithms = ["DTW", "CDTW", "LCSS", "SDTW", "ADTW"]
        else:
            algorithms = [self.config.get("algorithm", "DTW")]
            
        all_results = {alg: [] for alg in algorithms}
        total_times = {alg: 0.0 for alg in algorithms}
        
        for test_idx, test_class, test_ts in test_data:
            best_scores = {alg: -1.0 if alg == "LCSS" else float('inf') for alg in algorithms}
            best_train_idxs = {alg: -1 for alg in algorithms}
            best_train_classes = {alg: "" for alg in algorithms}
            best_times = {alg: 0.0 for alg in algorithms}
            best_spaces = {alg: 0.0 for alg in algorithms}
            
            for train_idx, train_class, train_ts in train_data:
                for alg in algorithms:
                    self.config["algorithm"] = alg
                    is_similarity = (alg == "LCSS")
                    
                    tracemalloc.start()
                    start_time = time.perf_counter()
                    try:
                        success, score, _, _, _ = self._compute_distance(test_ts, train_ts)
                    except Exception as e:
                        success = False
                        score = 0.0
                    
                    comp_time = time.perf_counter() - start_time
                    total_times[alg] += comp_time
                    _, peak = tracemalloc.get_traced_memory()
                    space_usage = peak / (1024 * 1024)
                    tracemalloc.stop()
                    
                    if success:
                        if (is_similarity and score > best_scores[alg]) or (not is_similarity and score < best_scores[alg]):
                            best_scores[alg] = score
                            best_train_idxs[alg] = train_idx
                            best_train_classes[alg] = train_class
                            best_times[alg] = comp_time
                            best_spaces[alg] = space_usage
                            
            for alg in algorithms:
                same_class = (test_class == best_train_classes[alg])
                all_results[alg].append({
                    "Test Index": test_idx,
                    "Train Index": best_train_idxs[alg],
                    "Score": best_scores[alg],
                    "Computing Time (s)": best_times[alg],
                    "Space Usage (MB)": best_spaces[alg],
                    "Same Class": same_class
                })
                
        # Restore the original algorithm setting
        if run_all:
            self.config["algorithm"] = "DTW"
            
        for alg in algorithms:
            self.save_results(all_results[alg], dataset_name, alg, total_time=total_times[alg])
            
        return True

    def run_classifier(self, train_data, test_data, dataset_name, clf_algo):
        if not train_data or not test_data:
            return False, 0.0, 0.0, 0.0
            
        max_len = 0
        for _, _, ts in train_data + test_data:
            if len(ts) > max_len:
                max_len = len(ts)
                
        def pad_ts(ts, max_l):
            if len(ts) < max_l:
                return np.pad(ts, (0, max_l - len(ts)), 'constant')
            return ts
            
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
        space_usage = peak / (1024 * 1024)
        tracemalloc.stop()
        
        results = [{
            "Dataset": dataset_name,
            "Classifier": clf_algo,
            "Accuracy": accuracy,
            "Computing Time (s)": comp_time,
            "Space Usage (MB)": space_usage
        }]
        self.save_results(results, dataset_name, clf_algo.replace(" ", "_"))
        
        return True, accuracy, comp_time, space_usage

    @staticmethod
    def compute_spearman_correlation():
        algorithms = ["DTW", "CDTW", "LCSS", "SDTW", "ADTW"]
        dataset_accuracies = {}
        
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
                    acc = 0.0
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
        
        for i in range(n):
            vec_i = [dataset_accuracies[datasets[i]][a] for a in algorithms]
            for j in range(n):
                vec_j = [dataset_accuracies[datasets[j]][a] for a in algorithms]
                corr, _ = scipy.stats.spearmanr(vec_i, vec_j)
                if np.isnan(corr):
                    corr = 0.0
                spearman_matrix[i, j] = corr
                
        df_matrix = pd.DataFrame(spearman_matrix, index=datasets, columns=datasets)
        os.makedirs("experimental_results", exist_ok=True)
        df_matrix.to_csv("experimental_results/spearman_correlation_matrix.csv")
        
        return True, spearman_matrix, datasets

    @staticmethod
    def compute_max_clique_groups(matrix, labels, threshold):
        G = nx.Graph()
        G.add_nodes_from(labels)
        n = len(labels)
        
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] >= threshold:
                    G.add_edge(labels[i], labels[j], weight=matrix[i, j])
                    
        groups = []
        remaining_nodes = set(G.nodes())
        
        while remaining_nodes:
            subgraph = G.subgraph(remaining_nodes)
            cliques = list(nx.find_cliques(subgraph))
            if not cliques:
                max_cl = [list(remaining_nodes)[0]]
            else:
                max_cl = max(cliques, key=len)
                
            groups.append(max_cl)
            remaining_nodes -= set(max_cl)
            
        return G, groups

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

# Largest Remainder Method (also known as the Hamilton apportionment method)
def stratified_sample(data, percentage):
    if percentage >= 100.0 or percentage <= 0.0:
        return data
        
    class_groups = {}
    for item in data:
        ts_class = item[1]
        if ts_class not in class_groups:
            class_groups[ts_class] = []
        class_groups[ts_class].append(item)
        
    target_total = int(round(len(data) * percentage / 100.0))
    if target_total < len(class_groups):
        target_total = len(class_groups)
        
    exact_counts = {c: target_total * (len(items) / len(data)) for c, items in class_groups.items()}
    
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
    
    if allocated < target_total:
        shortfall = target_total - allocated
        sorted_classes = sorted([c for c in class_groups if remainders[c] >= 0], key=lambda x: remainders[x], reverse=True)
        if not sorted_classes:
            sorted_classes = list(class_groups.keys())
            
        for i in range(shortfall):
            assignments[sorted_classes[i % len(sorted_classes)]] += 1
            
    elif allocated > target_total:
        overage = allocated - target_total
        for _ in range(overage):
            candidates = [c for c in class_groups if assignments[c] > 1]
            if not candidates:
                break 
            best_candidate = max(candidates, key=lambda c: assignments[c] - exact_counts[c])
            assignments[best_candidate] -= 1

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

        # Upper table for dataset folders
        self.dataset_table = QtWidgets.QTableWidget()
        self.dataset_table.setColumnCount(1)
        self.dataset_table.setHorizontalHeaderLabels(["Dataset Folders"])
        self.dataset_table.horizontalHeader().setStretchLastSection(True)
        self.dataset_table.cellDoubleClicked.connect(self.on_dataset_double_clicked)
        self.layout.addWidget(self.dataset_table)

        # Bottom tab view for Train and Test files
        self.tab_widget = QtWidgets.QTabWidget()
        
        self.train_table = QtWidgets.QTableWidget()
        self.train_table.setSortingEnabled(False)
        self.train_table.setColumnCount(3)
        self.train_table.setHorizontalHeaderLabels(["Select", "TS Index", "Class"])
        self.train_table.verticalHeader().setVisible(False)
        self.train_table.horizontalHeader().setStretchLastSection(True)
        self.train_table.setSortingEnabled(True)
        self.tab_widget.addTab(self.train_table, "Train")
        
        self.test_table = QtWidgets.QTableWidget()
        self.test_table.setSortingEnabled(False)
        self.test_table.setColumnCount(3)
        self.test_table.setHorizontalHeaderLabels(["Select", "TS Index", "Class"])
        self.test_table.verticalHeader().setVisible(False)
        self.test_table.horizontalHeader().setStretchLastSection(True)
        self.test_table.setSortingEnabled(True)
        self.tab_widget.addTab(self.test_table, "Test")

        self.layout.addWidget(self.tab_widget)
        
        self.checked_items = []
        self.train_table.itemChanged.connect(self.on_item_changed)
        self.test_table.itemChanged.connect(self.on_item_changed)
        
    def on_item_changed(self, item):
        if item.column() == 0:
            if item.checkState() == QtCore.Qt.Checked:
                if item not in self.checked_items:
                    self.checked_items.append(item)
                    if len(self.checked_items) > 2:
                        oldest = self.checked_items.pop(0)
                        oldest.tableWidget().blockSignals(True)
                        oldest.setCheckState(QtCore.Qt.Unchecked)
                        oldest.tableWidget().blockSignals(False)
            else:
                if item in self.checked_items:
                    self.checked_items.remove(item)

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

    def on_dataset_double_clicked(self, row, column):
        dataset_name = self.dataset_table.item(row, column).text()
        dataset_path = os.path.join(self.dataset_dir, dataset_name)
        self.selected_dataset = dataset_path
        self.populate_timeseries(dataset_path)

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
                
            item_check = QtWidgets.QTableWidgetItem()
            item_check.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item_check.setCheckState(QtCore.Qt.Unchecked)
            table.setItem(row, 0, item_check)
            
            item_index = QtWidgets.QTableWidgetItem(str(row))
            item_index.setFlags(item_index.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, 1, item_index)
            
            item_class = QtWidgets.QTableWidgetItem(ts_class)
            item_class.setFlags(item_class.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, 2, item_class)
            
        table.blockSignals(False)

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

    def get_selected_dataset(self):
        return os.path.basename(self.selected_dataset) if hasattr(self, 'selected_dataset') and self.selected_dataset else "dataset"

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
        
        # Define tabs
        tabs = ["TS Correlation", "TS Heatmap", "Dataset Classifier", "Spearman Matrix", "Correlation Graph"]
        self.canvases = {}

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
        
        if ts1.ndim > 1:
            ts1 = ts1.flatten()
        if ts2.ndim > 1:
            ts2 = ts2.flatten()
            
        offset = np.max(ts1) - np.min(ts2) + (np.max(ts1) - np.min(ts1)) * 0.5
        ax.plot(ts1, label="TS 1", color="blue")
        ax.plot(ts2 + offset, label="TS 2", color="orange")
        
        for i, j in path:
            ax.plot([i, j], [ts1[int(round(i))], ts2[int(round(j))] + offset], color='gray', alpha=0.3, linewidth=1.0)
            
        ax.set_title("Time Series Alignment")
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
        
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, color='red', linewidth=2, label="Optimal Path")
        
        ax.set_title("Accumulated Cost Matrix & Path")
        ax.set_xlabel("TS 2 Index")
        ax.set_ylabel("TS 1 Index")
        ax.legend()
        fig.tight_layout()
        canvas.draw()

    def plot_spearman_matrix(self, spearman_matrix, labels):
        fig, canvas = self.canvases["Correlation Graph"]
        fig.clear()
        ax = fig.add_subplot(111)
        
        im = ax.imshow(spearman_matrix, origin='lower', cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        if len(labels) <= 15:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{spearman_matrix[i, j]:.2f}", ha="center", va="center", color="black" if abs(spearman_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title("Spearman Correlation Between Datasets")
        fig.tight_layout()
        canvas.draw()

    def plot_clique_graph(self, G, groups):
        fig, canvas = self.canvases["Clique Graph"]
        fig.clear()
        ax = fig.add_subplot(111)
        
        pos = nx.spring_layout(G, seed=42)
        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']
        
        for idx, group in enumerate(groups):
            color = colors[idx % len(colors)]
            nx.draw_networkx_nodes(G, pos, nodelist=group, node_color=color, ax=ax, label=f"Group {idx+1}")
            
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        ax.set_title("Dataset Correlation Graph (Max Clique Grouping)")
        ax.legend()
        fig.tight_layout()
        canvas.draw()

class ControlPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QVBoxLayout(self)

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

        # Upper box: Tabs
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Single Execution Tab
        self.single_exec_tab = QtWidgets.QWidget()
        self.single_exec_layout = QtWidgets.QFormLayout(self.single_exec_tab)
        
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
        
        self.constraint_combo = QtWidgets.QComboBox()
        self.constraint_combo.addItems(["Sakoe Chiba", "Itakura"])
        
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItems(["DTW", "CDTW", "LCSS", "SDTW", "ADTW"])

        self.single_exec_layout.setVerticalSpacing(15)

        self.single_exec_layout.addRow("Algorithm:", self.algorithm_combo)
        self.single_exec_layout.addRow("Constraint (All except CDTW):", self.constraint_combo)
        self.single_exec_layout.addRow("Radius (%) (Sakoe Chiba):", self.radius_spinbox)
        self.single_exec_layout.addRow("Slope (Itakura):", self.slope_spinbox)
        self.single_exec_layout.addRow("Eps (LCSS):", self.eps_spinbox)
        self.single_exec_layout.addRow("Gamma (SDTW):", self.gamma_spinbox)
        self.single_exec_layout.addRow("Warp Penalty (ADTW):", self.warpPenalty)
        
        self.run_single_button = QtWidgets.QPushButton("Run TS Correlation")
        self.single_exec_layout.addRow(self.run_single_button)

        self.run_all_algorithms_checkbox = QtWidgets.QCheckBox("Run All Algorithms")
        self.single_exec_layout.addRow(self.run_all_algorithms_checkbox)

        self.run_dataset_button = QtWidgets.QPushButton("Run Dataset Correlation")
        self.single_exec_layout.addRow(self.run_dataset_button)
        
        self.tab_widget.addTab(self.single_exec_tab, "Correlator")
        
        # Batch Experiments Tab
        self.batch_exec_tab = QtWidgets.QWidget()
        self.batch_exec_layout = QtWidgets.QFormLayout(self.batch_exec_tab)
        
        self.classifier_combo = QtWidgets.QComboBox()
        self.classifier_combo.addItems(["1-NN", "Naive Bayes", "Linear SVM"])
        self.batch_exec_layout.addRow("Classifier Algorithm:", self.classifier_combo)

        self.batch_exec_layout.setVerticalSpacing(15)
        
        self.run_classifier_button = QtWidgets.QPushButton("Run Classifier")
        self.batch_exec_layout.addRow(self.run_classifier_button)
        
        self.compute_spearman_btn = QtWidgets.QPushButton("Compute Spearman Matrix")
        self.batch_exec_layout.addRow(self.compute_spearman_btn)
        
        self.graph_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.graph_threshold_spinbox.setRange(0.0, 1.0)
        self.graph_threshold_spinbox.setSingleStep(0.05)
        self.graph_threshold_spinbox.setValue(0.8)
        self.batch_exec_layout.addRow("Graph Edge Threshold:", self.graph_threshold_spinbox)

        self.generate_clique_btn = QtWidgets.QPushButton("Generate Clique Graph")
        self.batch_exec_layout.addRow(self.generate_clique_btn)
        
        self.tab_widget.addTab(self.batch_exec_tab, "Classifier")

        self.layout.addWidget(self.tab_widget)

        # Bottom box: Execution Data
        self.execution_group = QtWidgets.QGroupBox("Execution Data")
        self.execution_layout = QtWidgets.QFormLayout(self.execution_group)
        
        self.score_label = QtWidgets.QLabel("N/A")
        self.time_label = QtWidgets.QLabel("N/A")
        self.space_label = QtWidgets.QLabel("N/A")
        
        self.execution_layout.addRow("Score:", self.score_label)
        self.execution_layout.addRow("Computing Time:", self.time_label)
        self.execution_layout.addRow("Space Usage:", self.space_label)

        self.layout.addWidget(self.execution_group)

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

    def set_execution_data(self, score, computing_time, space_usage):
        self.score_label.setText(f"{score:.4f}")
        self.time_label.setText(f"{computing_time:.4f} s")
        self.space_label.setText(f"{space_usage:.4f} MB")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Time Series Correlator")

        # Set VisualizationFrame as the central widget
        self.visualization_frame = VisualizationFrame()
        self.setCentralWidget(self.visualization_frame)

        # Create and add the DataSideBar as a dock widget
        self.data_sidebar = DataSideBar()
        self.sidebar_dock = QtWidgets.QDockWidget("Data", self, features=QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.sidebar_dock.setWidget(self.data_sidebar)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.sidebar_dock)

        # Create and add the ControlPanel as a dock widget
        self.control_panel = ControlPanel()
        self.control_dock = QtWidgets.QDockWidget("Control Panel", self, features=QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.control_dock.setWidget(self.control_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.control_dock)

        self.control_panel.run_single_button.clicked.connect(self.run_single_computation)
        self.control_panel.run_dataset_button.clicked.connect(self.run_dataset_computation)
        self.control_panel.run_classifier_button.clicked.connect(self.run_classification_task)
        self.control_panel.compute_spearman_btn.clicked.connect(self.create_spearmann_correlation)
        self.control_panel.generate_clique_btn.clicked.connect(self.generate_clique_graph)

        self.create_actions()

        self.create_toolbars()

    @Slot()
    def open_TS_file(self):
        pass

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
            
        config = self.control_panel.get_configuration()
        processor = CTSPU(data, config)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        success = processor.run_single()
        
        if success:
            self.control_panel.set_execution_data(processor.score, processor.computing_time, processor.space_usage)
            self.visualization_frame.plot_correlation(processor.data[0], processor.data[1], processor.path)
            self.visualization_frame.plot_heatmap(processor.cost_matrix, processor.path)
        else:
            QtWidgets.QMessageBox.critical(self, "Computation Error", "Failed to compute distance.")
        
        QtWidgets.QApplication.restoreOverrideCursor()

    @Slot()
    def run_dataset_computation(self):
        train_data, test_data = self.data_sidebar.get_all_data()
        dataset_name = self.data_sidebar.get_selected_dataset()
        
        if not train_data or not test_data:
            QtWidgets.QMessageBox.warning(self, "Data Error", "Could not load train or test data for the selected dataset.")
            return
            
        config = self.control_panel.get_configuration()
        train_sampling = config.get("train_sampling_percentage", 100.0)
        test_sampling = config.get("test_sampling_percentage", 100.0)
        train_data = stratified_sample(train_data, train_sampling)
        test_data = stratified_sample(test_data, test_sampling)
        
        processor = CTSPU(None, config)
        
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
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
            success, matrix, labels = CTSPU.compute_spearman_correlation()
            if success:
                self.visualization_frame.plot_spearman_matrix(matrix, labels)
                self.visualization_frame.tab_widget.setCurrentIndex(3) # Switch to Correlation Graph
                QtWidgets.QMessageBox.information(self, "Success", "Spearman Correlation Matrix generated successfully and saved to experimental_results/spearman_correlation_matrix.csv.")
            else:
                QtWidgets.QMessageBox.warning(self, "No Data", "Could not find sufficient dataset results in experimental_results/ to compute the matrix. Make sure you run datasets first.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute Spearman Correlation: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    @Slot()
    def generate_clique_graph(self):
        try:
            df = pd.read_csv("experimental_results/spearman_correlation_matrix.csv", index_col=0)
            matrix = df.values
            labels = df.index.tolist()
        except Exception:
            QtWidgets.QMessageBox.warning(self, "No Matrix", "Spearman matrix not found. Please compute it first.")
            return
            
        threshold = self.control_panel.graph_threshold_spinbox.value()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            G, groups = CTSPU.compute_max_clique_groups(matrix, labels, threshold)
            self.visualization_frame.plot_clique_graph(G, groups)
            self.visualization_frame.tab_widget.setCurrentIndex(4) # Switch to Clique Graph
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate graph: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def create_actions(self):
        icon = QtGui.QIcon.fromTheme(QtGui.QIcon.ThemeIcon.DocumentOpen, QtGui.QIcon(':/images/open.png'))
        self._open_dataset_act = QtGui.QAction(icon, "&Open Dataset Folder...", self,
                                 shortcut=QtGui.QKeySequence.StandardKey.Open,
                                 statusTip="Open database folder",
                                 triggered=self.open_dataset_folder)

    def create_toolbars(self):
        # Create a Toolbar and add the actions as buttons
        file_toolbar = self.addToolBar("File")
        file_toolbar.addAction(self._open_dataset_act)
        
if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    main_window = MainWindow()

    # Usar .show para abrir a janela fora do fullscreen
    main_window.showMaximized()

    sys.exit(app.exec())