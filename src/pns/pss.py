"""Detect the principal subsphere."""

__all__ = [
    "pss",
]


def pss(x, tol=1e-3, maxiter=None):
    r"""Find the principal subsphere from data on a hypersphere.

    Parameters
    ----------
    x : (N, d+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        embedded in a ``d+1``-dimensional space.
    tol : float, default=1e-3
        Convergence tolerance in radian.
    maxiter : int, optional
        Maximum number of iterations for the optimization.
        If None, the number of iterations is not checked.

    Returns
    -------
    v : (d+1,) real array
        Estimated principal axis of the subsphere in extrinsic coordinates.
    r : scalar in [0, pi]
        Geodesic distance from the pole by *v* to the estimated principal subsphere.

    Notes
    -----
    This function determines the best fitting subsphere
    :math:`\hat{A}_{d-k} = A_{d-k}(\hat{v}_k, \hat{r}_k) \subset S^{d-k+1}` for
    :math:`k = 1, 2, \ldots, d`.

    The Fréchet mean :math:`\hat{A}_0` of the lowest level best fitting subsphere
    :math:`\hat{A}_1` is also determined by this function.
    """
    raise NotImplementedError
