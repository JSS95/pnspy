"""Classes to transform data."""

import numpy as np

from .base import rotation_matrix

__all__ = [
    "project",
    "embed",
    "reconstruct",
]


def project(x, v, r):
    r"""Minimum-geodesic projection of points to a subsphere.

    Parameters
    ----------
    x : (N, m+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        embedded in a ``d+1``-dimensional space.
    v : (m+1,) real array
        Subsphere axis.
    r : scalar
        Subsphere geodesic distance.

    Returns
    -------
    xP : (N, m+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        projected onto the found principal subsphere.
    res : (N, 1) real array
        Projection residuals.

    Notes
    -----
    This is the function
    :math:`P: S^{d-k+1} \to A_{d-k}(v_k, r_k ) \subset S^{d-k+1}` for
    :math:`k = 1, 2, \ldots, d` in the original paper.
    Here, :math:`A_{d-k}(v_k, r_k)` is a subsphere of the hypersphere :math:`S^{d-k+1}`.
    The input and output data dimension are :math:`m+1`, where :math:`m = d-k+1`.

    The resulting points have same number of components but their rank is reduced
    by one in the manifold.

    Examples
    --------
    >>> from pns.pss import pss
    >>> from pns.transform import project
    >>> from pns.util import unit_sphere, circular_data
    >>> x = circular_data([0, -1, 0]).reshape(-1, 3)
    >>> v, r = pss(x)
    >>> A, _ = project(x, v, r)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*x.T, marker="x")
    ... ax.scatter(*A.T, marker=".")
    """
    if x.shape[1] > 2:
        rho = np.arccos(x @ v)[..., np.newaxis]
    elif x.shape[1] == 2:
        rho = np.arctan2(x @ (v @ [[0, 1], [-1, 0]]), x @ v)[..., np.newaxis]
    return (np.sin(r) * x + np.sin(rho - r) * v) / np.sin(rho), rho - r


def embed(x, v, r):
    r"""Embed data on a sub-hypersphere to a low-dimensional unit hypersphere.

    Parameters
    ----------
    x : (N, m+1) real array
        Data :math:`x \in A_{m-1} \subset S^m \subset \mathbb{R}^{m+1}`,
        on a subsphere :math:`A_{m-1}` of a unit hypersphere :math:`S^m`.
    v : (m+1,) real array
        Sub-hypersphere axis.
    r : scalar
        Sub-hypersphere geodesic distance.

    Returns
    -------
    (N, m) real array
        Data :math:`x^\dagger` on a low-dimensional unit hypersphere :math:`S^{m-1}`.

    Notes
    -----
    This is the function
    :math:`f_k: A_{d-k}(v_k, r_k) \subset S^{d-k+1} \to S^{d-k}` for
    :math:`k = 1, 2, \ldots, d-1` in the original paper.
    Here, :math:`A_{d-k}(v_k, r_k)` is a subsphere of the hypersphere :math:`S^{d-k+1}`.
    The input is :math:`x \in S^m \subset \mathbb{R}^{m+1}`
    and the output is :math:`x^\dagger \in S^{m-1} \subset \mathbb{R}^{m}`,
    where :math:`m = d-k+1`.

    Examples
    --------
    >>> from pns.pss import pss
    >>> from pns.transform import embed
    >>> from pns.util import unit_sphere, circular_data
    >>> x = circular_data([0, -1, 0]).reshape(-1, 3)
    >>> v, r = pss(x)
    >>> x_embed = embed(x, v, r)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*x.T, marker=".", zorder=10)
    ... ax2 = fig.add_subplot(122)
    ... ax2.scatter(*x_embed.T, marker=".", zorder=10)
    ... ax2.set_aspect("equal")
    """
    R = rotation_matrix(v)
    return x @ (1 / np.sin(r) * R[:-1:, :]).T


def reconstruct(x, v, r):
    r"""Reconstruct data on a low-dimensional unit hypersphere. to a sub-hypersphere.

    Parameters
    ----------
    x : (N, m) real array
        Data :math:`x^\dagger` on a low-dimensional unit hypersphere :math:`S^{m-1}`.
    v : (m+1,) real array
        Sub-hypersphere axis.
    r : scalar
        Sub-hypersphere geodesic distance.

    Returns
    -------
    (N, m+1) real array
        Data :math:`x \in A_{m-1} \subset S^m \subset \mathbb{R}^{m+1}`,
        on a subsphere :math:`A_{m-1}` of a unit hypersphere :math:`S^m`.

    Notes
    -----
    This is the function
    :math:`f^{-1}_k: S^{d-k} \to A_{d-k}(v_k, r_k) \subset S^{d-k+1}` for
    :math:`k = 1, 2, \ldots, d-1` in the original paper.
    Here, :math:`A_{d-k}(v_k, r_k)` is a subsphere of the hypersphere :math:`S^{d-k+1}`.
    The input is :math:`x^\dagger \in S^{m-1} \subset \mathbb{R}^{m}`
    and the output is :math:`x \in S^m \subset \mathbb{R}^{m+1}`,
    where :math:`m = d-k+1`.

    Examples
    --------
    >>> import numpy as np
    >>> from pns.transform import reconstruct
    >>> from pns.util import unit_sphere
    >>> t = np.linspace(0, 2 * np.pi, 100)
    >>> x = np.vstack((np.cos(t), np.sin(t))).T
    >>> x_rec = reconstruct(x, np.array([0, -1, 0]), 0.5)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121)
    ... ax1.scatter(*x.T)
    ... ax1.set_aspect("equal")
    ... ax2 = fig.add_subplot(122, projection='3d', computed_zorder=False)
    ... ax2.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax2.scatter(*x_rec.T)
    """
    R = rotation_matrix(v)
    vec = np.hstack([np.sin(r) * x, np.full(len(x), np.cos(r)).reshape(-1, 1)])
    return vec @ R
