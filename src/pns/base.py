"""Basic functions to handle data on a hypersphere."""

import numpy as np

__all__ = [
    "rotation_matrix",
    "exp_map",
    "log_map",
    "circle_mean",
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


def circle_mean(X):
    """Frechet mean of data on a circle.

    Parameters
    ----------
    X : real array of shape (N, 2)

    Returns
    -------
    mean : array of shape (2,)

    Notes
    -----
    Uses the algorithm [1]_ implemented by the Geomstats [2]_ package.

    Copyright (c) 2019-2024 geomstats developers

    References
    ----------
    .. [1] Hotz, T. and S. F. Huckemann (2015), "Intrinsic means on the
        circle: Uniqueness, locus and asymptotics", Annals of the Institute of
        Statistical Mathematics 67 (1), 177-193.
        https://arxiv.org/abs/1108.2141
    .. [2] https://github.com/geomstats/geomstats

    Examples
    --------
    >>> import numpy as np
    >>> from pns.base import circle_mean
    >>> t = np.linspace(-np.pi, np.pi / 2, 10)
    >>> X = np.stack([np.cos(t), np.sin(t)]).T
    >>> mean = circle_mean(X)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.scatter(*X.T, label="data")
    ... plt.scatter(*mean, label="mean")
    ... plt.legend()
    ... plt.gca().set_aspect("equal")
    """
    # Convert X to angles in [-pi, pi)
    X = np.arctan2(X[..., 1], X[..., 0])[..., np.newaxis]

    # CircleMean._circle_mean from geomstats
    mean0 = np.mean(X)
    var0 = np.sum((X - mean0) ** 2)
    N, _ = X.shape
    X = np.sort(X, axis=0)

    # CircleMean._circle_variances from geomstats
    means = (mean0 + np.linspace(0.0, 2 * np.pi, N + 1)[:-1]) % (2 * np.pi)
    means = np.where(means >= np.pi, means - 2 * np.pi, means)
    parts = np.array([np.sum(X) / N if means[0] < 0 else 0])
    m_plus = means >= 0
    left_sums = np.cumsum(X)
    right_sums = left_sums[-1] - left_sums
    i = np.arange(N, dtype=right_sums.dtype)
    j = i[1:]
    parts2 = right_sums[:-1] / (N - j)
    first_term = parts2[:1]
    parts2 = np.where(m_plus[1:], left_sums[:-1] / j, parts2)
    parts = np.concatenate([parts, first_term, parts2[1:]])
    plus_vec = (4 * np.pi * i / N) * (np.pi + parts - mean0) - (2 * np.pi * i / N) ** 2
    minus_vec = (4 * np.pi * (N - i) / N) * (np.pi - parts + mean0) - (
        2 * np.pi * (N - i) / N
    ) ** 2
    minus_vec = np.where(m_plus, plus_vec, minus_vec)
    means = np.transpose(np.vstack([means, var0 + minus_vec]))

    frechet_mean = means[np.argmin(means[:, 1]), 0]
    cos = np.cos(frechet_mean)
    sin = np.sin(frechet_mean)
    return np.hstack([cos, sin])
