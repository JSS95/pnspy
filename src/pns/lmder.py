"""Interface for minpack lmder, copied and adapted from SciPy."""

__all__ = [
    "least_squares",
]


def least_squares(
    fun, x0, jac, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=None, args=(), kwargs=None
):
    """Solve an unbounded nonlinear least-squares problem.

    This is a scipy-like function, restricted to `method="lm"`
    with analytical Jacobian matrix.
    Most of the function is directly copied and adapted from
    :func:`scipy.optimize.least_squares`.

    Parameters
    ----------
    fun : callable
        Function which computes the vector of residuals, with the signature
        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
        respect to its first argument. The argument ``x`` passed to this
        function is an ndarray of shape (n,) (never a scalar, even for n=1).
        It must allocate and return a 1-D array_like of shape (m,) or a scalar.
        If the argument ``x`` is complex or the function ``fun`` returns
        complex residuals, it must be wrapped in a real function of real
        arguments, as shown at the end of the Examples section.
    x0 : array_like with shape (n,) or float
        Initial guess on independent variables. If float, it will be treated
        as a 1-D array with one element. When `method` is 'trf', the initial
        guess might be slightly adjusted to lie sufficiently within the given
        `bounds`.
    jac : callable
        Function which computes the Jacobian matrix (an m-by-n matrix, where
        element (i, j) is the partial derivative of f[i] with respect to
        x[j]). Its signature must be ``jac(x, *args, **kwargs)`` and should
        return a good approximation (or the exact value) for the Jacobian as
        an numpy array.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step.
        This tolerance must be higher than machine epsilon.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8. The exact condition is``Delta < xtol * norm(jac)``,
        where ``Delta`` is a trust-region radius.
        This tolerance must be higher than machine epsilon.
    gtol : float or None, optional
        Tolerance for termination by the norm of the gradient. Default is 1e-8.
        The exact condition is the maximum absolute value of the cosine of angles
        between columns of the Jacobian and the residual vector is less
        than `gtol`, or the residual vector is zero.
        This tolerance must be higher than machine epsilon.
    max_nfev : None or int, optional
        Controls the maximum number of function evaluations used by each method.
        If None (default), the value is chosen automatically as 100 * n.
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun` and `jac`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)`` and the same for
        `jac`.
    """
    raise NotImplementedError
