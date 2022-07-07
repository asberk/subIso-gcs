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


def subsampled_dct_matrix(m, n, rescale=None):
    """subsampled DCT matrix - rows chosen uniform at random without replacement."""
    p = m / n
    assert p <= 1

    if rescale is None:
        rescale = p**0.5
    m_new = np.random.binomial(n, p)
    indices = np.random.choice(n, m_new, replace=False).astype(int)
    D = dct_matrix(n)[indices] / rescale
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

    def __init__(self, k, m, n, A, mnist_layer=False):
        super().__init__()
        self.input_dimension = k
        self.width = m
        self.output_dimension = n
        self.relu = nn.ReLU()

        self.first_layer = nn.Linear(k, m, bias=False)
        self.second_layer = nn.Linear(m, n, bias=False)

        self.A = A.T
        self.W = torch.randn((n, m)) / n**0.5
        if mnist_layer:
            W1 = torch.load("data/first_layer_mnist.pt").requires_grad_(False)
            W1_m = W1[:m]
            self.first_layer.weight.data = W1_m / torch.linalg.svdvals(W1_m)[0]
        else:
            self.first_layer.weight.data = torch.randn((m, k)) / m**0.5
        self.eval()
        self.requires_grad_(False)

    def forward(self, x):
        out = self.relu(self.first_layer(x))
        out = self.second_layer(out)
        return out

    def interp(self, beta):
        """Sets second_layer weight matrix W2 to
            Mt := (1 - beta) * self.W + beta * self.A

        Parameters
        ----------
        beta: float
            interpolation parameter in [0, 1]
        """
        assert beta >= 0 and beta <= 1
        Mt = (1 - beta) * self.W + beta * self.A
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
        numerators.pow(2).sum(dim=0) / torch.norm(denominators, dim=0)
    )


def gcs_problem_setup(k, m, n, noise_level=0.1, seed=None):
    """Set up a generative compressed sensing problem

    Parameters
    ----------
    k: int
        latent code dimension
    m: int
        average number of measurements
    n: int
        ambient dimension of observed signal
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
    W_dct = torch.from_numpy(subsampled_dct_matrix(m, n, 1.0)).float()
    k_tilde = W_dct.shape[0]  # actual number of rows
    network = (
        GenerativeNetwork(k, k_tilde, n, A=W_dct, mnist_layer=False)
        .eval()
        .requires_grad_(False)
    )
    return z0, network


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
    k=10,
    m=50,
    n=200,
    beta_min=0.5,
    num_betas=31,
    num_reps=25,
    noise_level=None,
    seed=None,
    verbose=False,
):
    """Run a generative compressed sensing experiment with the given parameters.
    Parameters
    ----------
    k: int
    m: int
    n: int
    beta_min : float
    noise_level: float
    seed: int
    verbose: bool

    Returns
    -------
    results: dict
        with keys: k, m, n, t, noise_level, alpha, recovery_error, rel_error, loss
    """
    if noise_level is None:
        noise_level = 0.0
    U = torch.from_numpy(dct_matrix(n)).float()
    z0, network = gcs_problem_setup(k, m, n, noise_level, seed=seed)
    results = []
    for ctr in trange(num_reps):
        A = torch.from_numpy(subsampled_dct_matrix(m, n)).float()
        for beta in np.linspace(beta_min, 1.0, num_betas):
            network.interp(beta)
            x0 = network(z0).view(-1)
            b = torch.mv(A, x0) + noise_level * torch.randn(A.shape[0])
            z_hat, x_hat, b_hat, history = gcs_recover(
                b, A, network, verbose=verbose
            )
            alpha = coherence(network, U)
            recovery_error = torch.norm(x0.view(-1) - x_hat.view(-1))
            rel_error = recovery_error / torch.norm(x0)
            loss = torch.norm(b.view(-1) - b_hat.view(-1))
            results.append(
                {
                    "rep": ctr,
                    "k": k,
                    "m": m,
                    "n": n,
                    "noise_level": noise_level,
                    "beta": beta,
                    "alpha": alpha,
                    "error": recovery_error,
                    "rel_error": rel_error,
                    "loss": loss,
                }
            )

    return results


def make_plots(dframe, agg_alpha, agg_beta, parms_string=None, savefig=False):
    plt.style.use("/Users/aberk/code/theme_bw.mplstyle")
    plt.rcParams["font.size"] = 14
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.labelsize"] = 14

    # First plot(s): t vs. alpha and t vs. relative error
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].fill_between(
        agg_beta.index.values,
        agg_beta[("alpha", "mean")].values - agg_beta[("alpha", "std")].values,
        agg_beta[("alpha", "mean")].values + agg_beta[("alpha", "std")].values,
        alpha=0.5,
    )
    ax[0].plot(agg_beta.index.values, agg_beta[("alpha", "mean")].values)
    ax[1].fill_between(
        agg_beta.index.values,
        agg_beta[("rel_error", "mean")].values
        - agg_beta[("rel_error", "std")].values,
        agg_beta[("rel_error", "mean")].values
        + agg_beta[("rel_error", "std")].values,
        alpha=0.5,
    )
    ax[1].plot(agg_beta.index.values, agg_beta[("rel_error", "mean")].values)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("alpha")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("relative error")
    fig.tight_layout()
    if savefig and isinstance(parms_string, str):
        fig.savefig(
            f"recovery_agg_{parms_string}.pdf",
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close("all")
    del fig, ax

    # Second plot: alpha vs. relative error
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(dframe.alpha, dframe.rel_error, alpha=0.3)
    ax.set_xlabel("alpha")
    ax.set_ylabel("relative error")
    ax.set_yscale("log")
    fig.tight_layout()
    if savefig and isinstance(parms_string, str):
        fig.savefig(
            f"recovery_scatter_{parms_string}.pdf",
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close("all")
    del fig, ax

    # Third plot: alpha vs. relative error (fill_between)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.fill_between(
        agg_alpha.index.values,
        agg_alpha[("rel_error", "P25")].values,
        agg_alpha[("rel_error", "P50")].values,
        agg_alpha[("rel_error", "P75")].values,
        alpha=0.5,
    )
    ax.plot(agg_alpha.index.values, agg_alpha[("rel_error", "P50")].values)
    ax.set_xlabel("coherence upper bound")
    ax.set_ylabel("relative error")
    ax.set_yscale("log")
    fig.tight_layout()
    if savefig and isinstance(parms_string, str):
        fig.savefig(
            f"recovery_quartiles_{parms_string}.pdf",
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close("all")
    del fig, ax


def pctl(q):
    return lambda x: np.percentile(x, q=q)


if __name__ == "__main__":
    parms = {"k": 20, "m": 128, "n": 1024}
    n_reps, n_betas = 20, 31
    results = run_experiment(
        **parms,
        beta_min=0.5,
        num_betas=n_betas,
        num_reps=n_reps,
        noise_level=0.0,
        verbose=False,
    )
    results_df = pd.DataFrame(results)
    df_beta = results_df.groupby(["beta"]).agg(["mean", "std"])
    df_alpha = (
        results_df.groupby(results_df.alpha.apply(lambda x: x.item()))[
            ["rel_error"]
        ]
        .agg([pctl(25), pctl(50), pctl(75)])
        .rename(
            columns={
                "<lambda_0>": "P25",
                "<lambda_1>": "P50",
                "<lambda_2>": "P75",
            }
        )
    )

    parms_string = "new_k{k}_m{m}_n{n}_rep{n_reps}_T{n_pts}".format(
        **parms, n_reps=n_reps, n_pts=n_betas
    )
    make_plots(results_df, df_alpha, df_beta, parms_string, savefig=True)
