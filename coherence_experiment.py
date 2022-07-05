"""
coherence_experiment



Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2022, Aaron Berk, all rights reserved.
Created: 28 June 2022
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import dct
import torch
from torch import nn, optim
from tqdm import trange, tqdm


def dct_matrix(n):
    """nxn (orthogonal) discrete cosine transform matrix"""
    U = dct(np.eye(n), axis=0, norm="ortho")
    return U


def subsampled_dct_matrix(m, n):
    """subsampled DCT matrix - rows chosen uniform at random without replacement."""
    p = m / n
    assert p <= 1
    m_new = np.random.binomial(n, p)
    indices = np.random.choice(n, m_new, replace=False).astype(int)
    D = dct_matrix(n)[indices] / p**0.5
    return D


class GenerativeNetwork(nn.Module):
    """
    GenerativeNetwork

    Using Kaiming Normal initialization.

    Parameters
    ----------
    k : int
        input dimension
    m : int
        width of hidden layer
    n : int
        output dimension
    """

    def __init__(self, k, m, n):
        super().__init__()
        self.input_dimension = k
        self.width = m
        self.output_dimension = n
        self.relu = nn.ReLU()

        self.first_layer = nn.Linear(k, m, bias=False)
        self.second_layer = nn.Linear(m, n, bias=False)
        # nn.init.kaiming_normal_(self.first_layer.weight)
        nn.init.kaiming_normal_(self.second_layer.weight)
        self.eval()
        self.requires_grad_(False)

    def forward(self, x):
        out = self.relu(self.first_layer(x))
        out = self.second_layer(out)
        return out

    def interp(self, M, t):
        """Sets second_layer weight matrix W2 to
            Mt := (1 - t)*W2 + t*M.T

        Parameters
        ----------
        M: torch.Tensor
            Matrix whose shape is the transpose of self.second_layer.weight
        t: float
            interpolation parameter in [0, 1]
        """
        if not isinstance(M, torch.Tensor):
            M = torch.from_numpy(M)

        assert t >= 0 and t <= 1
        Mt = (1 - t) * self.second_layer.weight.detach() + t * M.T
        self.second_layer.weight.data = Mt


def coherence(network, U):
    """Upper bound on the coherence parameter between network and the orthogonal
    matrix U.

    If W2 = network.second_layer.weight, and U_i is the ith row of U viewed as a
    column vector, then this function returns:

    max_i { norm(W2.T @ U_i)^2 / norm(W2 @ W2.T @ U_i) }
    """
    W2 = network.second_layer.weight.detach()
    numerators = torch.matmul(W2.T, U.T)
    denominators = torch.matmul(W2, numerators)
    return torch.max(
        numerators.abs().pow(2).sum(dim=0) / torch.norm(denominators, dim=0)
    )


def gcs_problem_setup(k, m, n, t, noise_level=0.1, seed=None):
    """Set up a generative compressed sensing problem

    Parameters
    ----------
    k: int
        latent code dimension
    m: int
        average number of measurements
    n: int
        ambient dimension of observed signal
    t: float
        interpolation parameter (see GenerativeNetwork.interp)
    noise_level: float
        standard deviation of measurement noise
    seed: int
        Optional seed for randomness

    Returns
    -------
    network: GenerativeNetwork
    A: torch.Tensor
        measurement operator, size m_tilde x n with E m_tilde = m
    z0: torch.Tensor
        ground truth latent code
    x0: torch.Tensor
        ground truth observed signal
    b: torch.Tensor
        noisy measurements A @ x0 + noise_level * normal_random_noise
    """
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    z0 = torch.randn(1, k)  # ground truth latent code
    A = torch.from_numpy(subsampled_dct_matrix(m, n)).float()
    m_tilde = A.shape[0]  # actual number of rows
    network = GenerativeNetwork(k, m_tilde, n).eval().requires_grad_(False)
    network.interp(A, t)  # set second_layer to interpolant
    x0 = network(z0).view(-1)
    b = torch.mv(A, x0)
    b = b + noise_level * torch.randn(m_tilde)
    return network, A, z0, x0, b


def gcs_recover(b, A, network, opt_tol=None, max_iter=None, verbose=False):
    """Approximate ground truth signal using measurement pair (b, A) and structural prior network.

    Parameters
    ----------
    b: torch.Tensor
        measurements
    A: torch.Tensor
        measurement operator, size m_tilde x n
    network: GenerativeNetwork
    opt_tol: float
        optimization tolerance (for measuring norm of gradient). default: 1e-6
    max_iter: int
        max number of iterations. default: 12000
    verbose: bool
        default: False.

    Returns
    -------
    z_hat: torch.Tensor
        recovered latent code
    x_hat: torch.Tensor
        recovered signal
    b_hat: torch.Tensor
        recovered measurements
    history: dict
        With 'loss' key.
    """
    if max_iter is None:
        max_iter = 12000
    if opt_tol is None:
        opt_tol = 1e-6
    k = network.first_layer.weight.shape[1]
    z_hat = nn.Parameter(torch.randn(1, k), requires_grad=True)
    # criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam([z_hat], lr=5e-3)

    history = {"loss": []}

    for i in range(max_iter):
        # if i % 5000 == 0:
        #     optimizer.param_groups[0]["lr"] /= 10
        optimizer.zero_grad()
        x_hat = network(z_hat).view(-1)
        b_guess = torch.mv(A, x_hat)
        loss = (b - b_guess).abs().pow(2).mean()
        history["loss"].append(loss.item())
        loss.backward()
        optimizer.step()
        grad_norm = torch.norm(z_hat.grad).item()
        if verbose and (i % 1000 == 0):
            print(
                f"{i//1000:d} loss: {loss.item():.3e}\t"
                f"grad_norm: {grad_norm:.3e}\t"
                # f"lr: {optimizer.param_groups[0]['lr']:.3e}"
            )
        if grad_norm < opt_tol:
            break
    with torch.no_grad():
        x_hat = network(z_hat).view(-1)
        b_hat = torch.mv(A, x_hat)
    if verbose:
        print()
    return z_hat, x_hat, b_hat, history


def default_setup():
    """Not a real function. Exists just to have this code written somewhere"""
    k = 25
    m = 200
    n = 1000
    t = 0.33
    noise_level = 0.1
    seed = 2022
    network, A, z0, x0, b = gcs_problem_setup(k, m, n, t, noise_level, seed)


def run_experiment(
    k=10, m=50, n=200, t=0.0, noise_level=None, seed=None, verbose=False
):
    """Run a generative compressed sensing experiment with the given parameters.
    Parameters
    ----------
    k: int
    m: int
    n: int
    t: float
    noise_level: float
    seed: int
    verbose: bool

    Returns
    -------
    results: dict
        with keys: k, m, n, t, noise_level, alpha, recovery_error, rel_error, loss
    """
    if verbose:
        print(f"t {t:.3e}")
    if noise_level is None:
        noise_level = 0.1
    U = torch.from_numpy(dct_matrix(n)).float()
    network, A, z0, x0, b = gcs_problem_setup(
        k, m, n, t, noise_level, seed=seed
    )
    z_hat, x_hat, b_hat, history = gcs_recover(b, A, network, verbose=verbose)
    alpha = coherence(network, U)
    recovery_error = torch.norm(x0.view(-1) - x_hat.view(-1))
    rel_error = recovery_error / torch.norm(x0)
    loss = torch.norm(b.view(-1) - b_hat.view(-1))

    results = {
        "k": k,
        "m": m,
        "n": n,
        "t": t,
        "noise_level": noise_level,
        "alpha": alpha,
        "error": recovery_error,
        "rel_error": rel_error,
        "loss": loss,
        # "history": history,
    }

    return results


if __name__ == "__main__":
    parms = {"k": 32, "m": 150, "n": 512}
    n_reps = 21
    t_vec = np.logspace(-0.2, 0, 31)
    all_results = [
        [
            run_experiment(**parms, t=t, noise_level=0.1, verbose=False)
            for t in t_vec
        ]
        for _ in trange(n_reps)
    ]
    all_results_df = [pd.DataFrame(results) for results in all_results]
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for results_df in all_results_df:
        ax[0].plot(results_df.alpha, results_df.rel_error)
        ax[1].plot(results_df.t, results_df.rel_error)
    ax[0].set_xlabel("alpha upper bound")
    ax[0].set_ylabel("relative error")
    ax[1].set_xlabel("interp. param")
    ax[1].set_ylabel("relative error")
    plt.show()
    plt.close("all")
    del fig, ax