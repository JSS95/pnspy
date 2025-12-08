"""Python implementation of PNS."""

import numpy as np

__all__ = [
    "rotation_matrix",
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
    c /= np.linalg.norm(c)

    A = np.outer(a, c) - np.outer(c, a)
    theta = np.arccos(v[-1])
    Id = np.eye(len(A))
    R = Id + np.sin(theta) * A + (np.cos(theta) - 1) * (np.outer(a, a) + np.outer(c, c))
    return R.astype(np.float64)
