"""Principal nested spheres (PNS) analysis [1]_.

.. [1] Jung, Sungkyu, Ian L. Dryden, and James Stephen Marron.
       "Analysis of principal nested spheres." Biometrika 99.3 (2012): 551-568.
"""

import numpy as np

from .pss import pss
from .transform import embed, project, reconstruct

__all__ = [
    "pns",
    "extrinsic_pns",
    "inverse_extrinsic_pns",
    "intrinsic_pns",
]


def pns(x, n_components, tol=1e-3, maxiter=None, lm_kwargs=None):
    r"""Principal nested spheres analysis.

    Parameters
    ----------
    x : real array of shape (N, d+1)
        Data on a d-sphere.
    n_components : int
        Target dimension.
    tol : float, default=1e-3
        Convergence tolerance in radians.
    maxiter : int, optional
        Maximum number of iterations for the optimization.
        If None, the number of iterations is not checked.
    lm_kwargs : dict, optional
        Additional keyword arguments to be passed for Levenberg-Marquardt optimization.
        Passed to :func:`pns.pss`.

    Returns
    -------
    vs : list of array
        Principal axes.
    rs : 1-D array of length (d+1-n_components)
        Principal geodesic distances.
    xis : 2-D array of shape (N, d+1-n_components)
        Unscaled residuals.
    x_transform : real array of shape (N, n_components)
        Data transformed onto low-dimensional sphere.

    Notes
    -----
    The input data is :math:`x \in S^d \subset \mathbb{R}^{d+1}`.

    The :math:`k`-th element of *vs*, *rs* and *xis* are:

    1. The principal axis :math:`\hat{v}_{k} \in S^{d-k+1} \subset \mathbb{R}^{d-k+2}`,
    2. The principal geodesic distance :math:`\hat{r}_k \in \mathbb{R}`, and
    3. Unscaled residual :math:`\xi_{d-k}`.

    Examples
    --------
    >>> from pns import pns
    >>> from pns.util import unit_sphere, circular_data, circle_3d
    >>> x = circular_data([0, -1, 0])
    >>> vs, rs, _, x_transform = pns(x.reshape(-1, x.shape[-1]), 2)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*x.T)
    ... ax1.scatter(*vs[0])
    ... ax1.plot(*circle_3d(vs[0], rs[0]), color="tab:orange", zorder=10)
    ... ax2 = fig.add_subplot(122)
    ... ax2.scatter(*x_transform.T)
    ... ax2.set_aspect("equal")
    """
    N, d_plus_one = x.shape
    M = d_plus_one - n_components

    vs = []
    rs = np.empty((M,))
    xis = np.empty((N, M))
    for i in range(M):
        v, r = pss(x, tol, maxiter, lm_kwargs)  # v_k, r_k
        P, xi = project(x, v, r)
        if len(v) > 2:
            x = embed(P, v, r)
        else:
            x = np.full((len(x), 1), 0, dtype=x.dtype)

        vs.append(v)
        rs[i] = r
        xis[:, [i]] = xi
    return vs, rs, xis, x


def extrinsic_pns(X, vs, rs):
    r"""Transform data to low-dimensional hypersphere in extrinsic coordinates.

    Parameters
    ----------
    X : (N, d+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        embedded in a ``d+1``-dimensional space.
    vs : list of k real arrays
        Subsphere axes.
    rs : list of k scalars
        Subsphere geodesic distances.

    Returns
    -------
    (N, d-k+1) real array
        Extrinsic coordinates of data on a low-dimensional unit hypersphere.

    Examples
    --------
    >>> from pns import extrinsic_pns
    >>> from pns.pss import pss
    >>> from pns.util import circular_data
    >>> x = circular_data([0, -1, 0]).reshape(-1, 3)
    >>> v, r = pss(x)
    >>> x_transformed = extrinsic_pns(x, [v], [r])
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.scatter(*x_transformed.T)
    ... plt.gca().set_aspect("equal")
    """
    for v, r in zip(vs, rs):
        P, _ = project(X, v, r)
        X = embed(P, v, r)
    return X


def inverse_extrinsic_pns(x, vs, rs):
    """Inverse of :func:`extrinsic_pns`.

    Parameters
    ----------
    x : (N, d-k+1) real array
        Extrinsic coordinates of data on a low-dimensional unit hypersphere.
    vs : list of (m+1,) real arrays
        Subsphere axes.
    rs : list of scalars
        Subsphere geodesic distances.

    Returns
    -------
    (N, d+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        embedded in a ``d+1``-dimensional space.

    Examples
    --------
    >>> from pns import inverse_extrinsic_pns, extrinsic_pns
    >>> from pns.pss import pss
    >>> from pns.util import circular_data, unit_sphere
    >>> x = circular_data([0, -1, 0]).reshape(-1, 3)
    >>> v, r = pss(x)
    >>> x_transformed = extrinsic_pns(x, [v], [r])
    >>> x_reconstructed = inverse_extrinsic_pns(x_transformed, [v], [r])
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*x.T, marker=".", zorder=10)
    ... ax.scatter(*x_reconstructed.T, marker="x", zorder=10)
    """
    for v, r in zip(reversed(vs), reversed(rs)):
        x = reconstruct(x, v, r)
    return x


def intrinsic_pns(X, vs, rs):
    r"""Transform data to low-dimensional hypersphere in intrinsic coordinates.

    Parameters
    ----------
    X : (N, d+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        embedded in a ``d+1``-dimensional space.
    vs : list of d real arrays
        Subsphere axes.
    rs : list of d scalars
        Subsphere geodesic distances.

    Returns
    -------
    Xi : (N, d) real array
        Intrinsic coordinates of data on a low-dimensional unit hypersphere.

    Examples
    --------
    >>> import numpy as np
    >>> from pns import pns, intrinsic_pns
    >>> from pns.util import unit_sphere, circular_data
    >>> X = circular_data([0, -1, 0]).reshape(-1, 3)
    >>> vs, rs, _, _ = pns(X, 1)
    >>> Xi = intrinsic_pns(X, vs, rs)[:, :2]  # Get the first two components
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', edgecolor='gray')
    ... ax1.scatter(*X.T, c=Xi[:, 0])
    ... ax2 = fig.add_subplot(122)
    ... ax2.scatter(*Xi.T, c=Xi[:, 0])
    ... ax2.set_xlim(-np.pi, np.pi)
    ... ax2.set_ylim(-np.pi/2, np.pi/2)
    """
    d = X.shape[1] - 1
    residuals = []

    sin_r = 1
    for k in range(1, d):
        v, r = vs[k - 1], rs[k - 1]
        P, xi = project(X, v, r)
        X = embed(P, v, r)
        Xi = sin_r * xi
        residuals.append(Xi)
        sin_r *= np.sin(r)

    v, r = vs[d - 1], rs[d - 1]
    _, xi = project(X, v, r)
    Xi = sin_r * xi
    residuals.append(Xi)

    ret = np.flip(np.concatenate(residuals, axis=-1), axis=-1)
    return ret
