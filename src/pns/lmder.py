"""Interface for minpack lmder."""

__all__ = [
    "lmder",
]


def lmder(res_func, jac_func, init_guess, func_args):
    raise NotImplementedError
