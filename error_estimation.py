import numpy as np
import qutip as qu
from aux import *
from numba import jit

seed = 1
rng = np.random.default_rng(seed)


# @jit(nopython=True)
def generate_mse(
    density_matrices_list, epsilon, dimension, noisy_samples=1000
):

    mse_list = []
    d = dimension
    eps = epsilon
    mse_mean_list_rhos = []
    mse_std_list_rhos = []
    for rho_i in density_matrices_list:

        rho = rho_i  # density_matrices_list[i]#qu.rand_dm(d)[:]

        exact_traces = get_traces(rho)

        noisy_samples = noisy_samples

        noise = [rng.uniform(-eps, eps, d) for _ in range(noisy_samples)]

        noisy_traces = [exact_traces + noisy_case for noisy_case in noise]

        noisy_coefs = [
            compute_coefs(d, noisy_traces_i) for noisy_traces_i in noisy_traces
        ]

        get_poly = lambda x: np.polynomial.polynomial.Polynomial(x[::-1])

        noisy_roots = [
            np.sort((get_poly(noisy_coef_i).roots()))
            for noisy_coef_i in noisy_coefs
        ]

        clean_roots = np.sort((np.linalg.eig(rho)[0]))

        mse = np.sum(
            [
                np.power(np.abs(noisy_roots_i - clean_roots), 2)
                for noisy_roots_i in noisy_roots
            ]
        )
        # print(mse)
        mse_list.append(mse)

        # return mse_mean, mse_std
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    return mse_mean, mse_std


# @jit(nopython=True)
def single_generate_mse(rho, epsilon, dimension, noisy_samples=1000):

    mse_list = []
    d = dimension
    eps = epsilon
    mse_mean_list_rhos = []
    mse_std_list_rhos = []

    # rho = rho_i  # density_matrices_list[i]#qu.rand_dm(d)[:]

    exact_traces = get_traces(rho)

    noisy_samples = noisy_samples

    noise = [rng.uniform(-eps, eps, d) for _ in range(noisy_samples)]

    noisy_traces = [exact_traces + noisy_case for noisy_case in noise]

    noisy_coefs = [
        compute_coefs(d, noisy_traces_i) for noisy_traces_i in noisy_traces
    ]

    get_poly = lambda x: np.polynomial.polynomial.Polynomial(x[::-1])

    noisy_roots = [
        np.sort((get_poly(noisy_coef_i).roots()))
        for noisy_coef_i in noisy_coefs
    ]

    clean_roots = np.sort((np.linalg.eig(rho)[0]))

    mse = (
        np.sum(
            [
                np.sum(np.power(np.abs(noisy_roots_i - clean_roots), 2))
                for noisy_roots_i in noisy_roots
            ]
        )
        / noisy_samples
    )
    return mse
