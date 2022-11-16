import numpy as np
import qutip as qu
from numba import jit


def get_traces(rho):
    traces = []
    for i in range(rho.shape[0]):
        traces.append(np.trace(np.linalg.matrix_power(rho, i + 1)))
    return np.array(traces)


@jit(nopython=True)
def c_n_m(n, m, vec_of_traces):
    """Auxiliary function used in the recursive part of the Faddeev LeVerrier algorithm.
    Ref: https://en.wikipedia.org/wiki/Faddeev%E2%80%93LeVerrier_algorithm
    """
    suma = 0
    if m == 0:
        return 1.0

    for k in range(1, m + 1):
        suma = suma + c_n_m(n, m - k, vec_of_traces) * vec_of_traces[k - 1]
    return suma * (-1 / m)


@jit(nopython=True)
def compute_coefs(n, vec_of_traces):
    """Compute coefficients of characteristic polynomial using
    the Faddeev LeVerrier algorithm.

    Refs: https://en.wikipedia.org/wiki/Newton%27s_identities
          https://en.wikipedia.org/wiki/Faddeev%E2%80%93LeVerrier_algorithm


    Args:
        n (int): Degree of the polynomial. In this case is the dimension n  of the density matrix of shape (nxn)
        vec_of_traces (np.array[float]): Array with the traces of A^i for i=1,..., n

    Returns:
        np.array[float]: Array with the coefficients of the characteristic polynomial.
    """
    c_list = [1.0]
    for m in range(1, n + 1):
        c_list.append(c_n_m(n, m, vec_of_traces))

    return np.array(c_list)


def mse(a, b):
    return np.sum((a - b) ** 2, axis=0)


def compute_eig_newton_identities(d, rho, scale=10000):
    # scale = 1000
    rhop = scale * rho
    coefs = compute_coefs(d, get_traces(rhop))
    poly = np.polynomial.polynomial.Polynomial(coefs[::-1])
    roots_with_imaginary = poly.roots() / scale
    if (np.imag(roots_with_imaginary) > 1e-8).any():
        print(f"WARNING, dimension {d}")
        return np.NaN
    else:
        newton_roots = np.sort(np.real(roots_with_imaginary))
        numpy_direct_roots = np.sort(np.real(np.linalg.eig(rho)[0]))
        return mse(newton_roots, numpy_direct_roots)
