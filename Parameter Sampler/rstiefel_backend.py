"""
Optional R rstiefel backend for matrix von Mises-Fisher Gibbs updates.

The default sampler path is pure Python. This module is imported only when the
user requests mv_sampler="rstiefel", so rpy2, R, and the R package rstiefel are
optional runtime dependencies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _to_r_matrix(array: np.ndarray):
    """Convert a 2D numpy array to an R matrix without activating global converters."""
    import rpy2.robjects as ro  # type: ignore[import]

    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"rstiefel input must be a 2D matrix, got shape {array.shape}.")
    return ro.r.matrix(
        ro.FloatVector(array.ravel(order="F")),
        nrow=array.shape[0],
        ncol=array.shape[1],
    )


def rmf_matrix_gibbs_rstiefel(
    M: np.ndarray,
    X: np.ndarray,
    rscol: Optional[int] = None,
) -> np.ndarray:
    """Call rstiefel::rmf.matrix.gibbs(M, X[, rscol]) and return a numpy matrix.

    Args:
        M: Matrix parameter of the matrix-variate von Mises-Fisher distribution.
        X: Current orthonormal matrix value on the Stiefel manifold.
        rscol: Optional number of columns to update simultaneously.

    Returns:
        New orthonormal matrix sample with the same shape as X.
    """
    try:
        from rpy2.robjects.packages import importr  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "mv_sampler='rstiefel' requires rpy2 and R. Install rpy2 in Python "
            "and install the R package with install.packages('rstiefel')."
        ) from exc

    M = np.asarray(M, dtype=float)
    X = np.asarray(X, dtype=float)
    if M.shape != X.shape:
        raise ValueError(f"rstiefel requires M and X to have the same shape, got {M.shape} and {X.shape}.")

    try:
        rstiefel = importr("rstiefel")
    except Exception as exc:
        raise RuntimeError(
            "mv_sampler='rstiefel' requires the R package rstiefel. "
            "Install it in R with install.packages('rstiefel')."
        ) from exc

    # rpy2 converts R names with dots to underscores by default, so
    # rstiefel::rmf.matrix.gibbs is usually exposed as rmf_matrix_gibbs.
    rmf_gibbs = rstiefel.__dict__.get("rmf.matrix.gibbs", rstiefel.__dict__.get("rmf_matrix_gibbs"))
    if rmf_gibbs is None:
        raise RuntimeError("Could not find rstiefel::rmf.matrix.gibbs in the loaded rstiefel package.")
    rmf_direct = rstiefel.__dict__.get("rmf.matrix", rstiefel.__dict__.get("rmf_matrix"))
    if rmf_direct is None:
        raise RuntimeError("Could not find rstiefel::rmf.matrix in the loaded rstiefel package.")
    M_r = _to_r_matrix(M)
    X_r = _to_r_matrix(X)

    if X.shape[1] == 1:
        # rstiefel::rmf.matrix.gibbs in rstiefel 1.0.1 fails for one-column
        # matrices because diag(sM$d) is treated as an identity size, not a
        # 1x1 diagonal matrix. With one column the Gibbs update degenerates to
        # a direct matrix-vMF draw, so use rstiefel::rmf.matrix.
        result = rmf_direct(M_r)
    elif rscol is None:
        result = rmf_gibbs(M_r, X_r)
    else:
        result = rmf_gibbs(M_r, X_r, rscol=int(rscol))

    sample = np.asarray(result, dtype=float)
    if sample.shape != X.shape:
        sample = sample.reshape(X.shape, order="F")
    return sample
