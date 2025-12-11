"""Classes to transform data."""

import numpy as np

__all__ = [
    "Project",
    "project",
]


class Project:
    r"""Minimum-geodesic projection of points to a subsphere.

    Parameters
    ----------
    [op]_func : callable
        Operator function.
    dtype : type
        Data type.

    Notes
    -----
    The purpose of this class is to facilitate defining computational graphs,
    without writing the same function multiple times for different frameworks.

    The instance of this class is the function
    :math:`P: S^{d-k+1} \to A_{d-k}(v_k, r_k ) \subset S^{d-k+1}` for
    :math:`k = 1, 2, \ldots, d` in the original paper.
    Here, :math:`A_{d-k}(v_k, r_k)` is a subsphere of the hypersphere :math:`S^{d-k+1}`.
    The input and output data dimension are :math:`m+1`, where :math:`m = d-k+1`.

    The resulting points have same number of components but their rank is reduced
    by one in the manifold.

    Examples
    --------
    >>> import numpy as np
    >>> from pns.transform import Project
    >>> from pns.util import circular_data
    >>> x = circular_data()
    >>> project = Project(np.add, np.subtract, np.multiply, np.divide, np.sin,
    ... np.acos, np.atan, np.matmul)
    >>> xP, res = project(x, np.array([0, 0, 1]), 0.5)
    """

    def __init__(
        self,
        add_func,
        sub_func,
        mul_func,
        div_func,
        sin_func,
        acos_func,
        atan2_func,
        matmul_func,
        dtype=np.float64,
    ):
        self.add = add_func
        self.sub = sub_func
        self.mul = mul_func
        self.div = div_func
        self.sin = sin_func
        self.acos = acos_func
        self.atan2 = atan2_func
        self.matmul = matmul_func
        self.dtype = dtype

    def __call__(self, X, v, r):
        if v.shape[0] > 2:
            rho = self.acos(self.matmul(X, v.reshape(-1, 1)))  # (N, 1)
        else:
            # For 2D case (circle), use arctan2 to preserve sign
            rotated_v = v @ np.array([[0, 1], [-1, 0]], dtype=self.dtype)
            y = self.matmul(X, rotated_v.reshape(-1, 1))  # (N, 1)
            x = self.matmul(X, v.reshape(-1, 1))  # (N, 1)
            rho = self.atan2(y, x)  # (N, 1)

        P = self.div(
            self.add(
                self.mul(np.sin(r).astype(self.dtype), X),  # (N, d+1)
                self.mul(
                    self.sin(self.sub(rho, r)),  # (N, 1)
                    v,  # (d+1,)
                ),  # (N, d+1)
            ),  # (N, d+1)
            self.sin(rho),  # (N, 1)
        )  # (N, d+1)
        return P, self.sub(rho, r)


_project = Project(
    np.add,
    np.subtract,
    np.multiply,
    np.divide,
    np.sin,
    np.acos,
    np.atan,
    np.matmul,
)


def project(X, v, r):
    """Numpy-compatible instance of :class:`Project`.

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

    Examples
    --------
    >>> import numpy as np
    >>> from pns.transform import project
    >>> from pns.util import unit_sphere, circular_data
    >>> x = circular_data()
    >>> A, _ = project(x, np.array([0, 0, 1]), 0.5)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*x.T, marker="x")
    ... ax.scatter(*A.T, marker=".")
    """
    return _project(X, v, r)
