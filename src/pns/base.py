"""Basic functions to handle data on a hypersphere."""

import numpy as np

__all__ = [
    "rotation_matrix",
    "exp_map",
    "log_map",
    "embed",
]


def rotation_matrix(v):
    r"""Rotation matrix.

    Moves :math:`v` to the north pole.

    Parameters
    ----------
    v : (m+1) real array
        Unit-norm direction vector.

    Returns
    -------
    (m+1, m+1) array of float64
        Rotational matrix.

    Notes
    -----
    This is the function :math:`R(v_k)` in the original paper.

    Examples
    --------
    Moving ``[0, 1, 0]`` to the north pole, in other words,
    moving the north pole to ``[0, -1, 0]``.

    >>> from pns.base import rotation_matrix
    >>> from pns.util import unit_sphere, circular_data
    >>> X = circular_data()
    >>> R = rotation_matrix([0, 1, 0])
    >>> X_rotated = X @ R.T
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*X.T)
    ... ax.scatter(*X_rotated.T)
    """
    a = np.zeros_like(v, dtype=np.float64)
    a[-1] = 1.0
    b = v
    c = b - a * (a @ b)
    c_norm = np.linalg.norm(c)
    if c_norm == 0:
        R = np.eye(len(c))
    else:
        c /= np.linalg.norm(c)

        A = np.outer(a, c) - np.outer(c, a)
        theta = np.arccos(v[-1])
        Id = np.eye(len(A))
        R = (
            Id
            + np.sin(theta) * A
            + (np.cos(theta) - 1) * (np.outer(a, a) + np.outer(c, c))
        )
    return R.astype(np.float64)


def exp_map(z):
    """Exponential map of hypersphere at (0, 0, ..., 0, 1).

    Parameters
    ----------
    z : (N, d) real array
        Vectors on tangent space.

    Returns
    -------
    (N, d+1) real array
        Points on d-sphere.
    """
    norm = np.linalg.norm(z, axis=1)[..., np.newaxis]
    return np.hstack([np.sin(norm) / norm * z, np.cos(norm)])


def log_map(x):
    """Log map of hypersphere at (0, 0, ..., 0, 1).

    Parameters
    ----------
    x : (N, d+1) real array
        Points on d-sphere.

    Returns
    -------
    (N, d) real array
        Vectors on tangent space.
    """
    thetas = np.arccos(x[:, -1:])
    return thetas / np.sin(thetas) * x[:, :-1]


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
    >>> from pns.base import embed
    >>> from pns.util import unit_sphere, circular_data
    >>> x = circular_data()
    >>> x_embed = embed(x, [0, 0, 1], 0.5)
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
