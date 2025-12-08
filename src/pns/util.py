"""Utility functions to generate sample data."""

import numpy as np

__all__ = [
    "unit_sphere",
    "circular_data",
]


def unit_sphere():
    """Helper function to plot a unit sphere.

    Returns
    -------
    x, y, z : array
        Coordinates for unit sphere.

    Examples
    --------
    >>> from pns.util import unit_sphere
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d')
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def circular_data():
    """Circular data on a 3D unit sphere.

    Returns
    -------
    ndarray of shape (100, 3)
        Data coordinates.

    Examples
    --------
    >>> from pns.util import unit_sphere, circular_data
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*circular_data().T)
    """
    t = np.random.uniform(0.1 * np.pi, 0.2 * np.pi, 100)
    p = np.random.uniform(-np.pi, np.pi / 2, 100)
    x = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]).T
    return x
