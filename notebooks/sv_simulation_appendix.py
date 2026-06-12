"""
Simulation appendix for Bayesian stochastic volatility models.

This script produces a simulation-based appendix for a thesis on
Bayesian inference for stochastic volatility models. The goal is to
simulate data from a simple SV model, recover the latent volatility
path using MCMC under fixed parameters, and produce figures showing
posterior credible bands and basic MCMC diagnostics.

The exercise is intentionally controlled: the parameters are fixed at
their data-generating values, and only the latent log-volatility path
is sampled.
"""

import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figures"
OUTPUT_DIR = SCRIPT_DIR / "output"
MPL_CACHE_DIR = SCRIPT_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

N_ITER = 12000
BURN_IN = 4000
PROPOSAL_SD = 0.15

import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None


def simulate_sv(T=1000, mu=-10.0, phi=0.98, sigma_eta=0.20, seed=123):
    """
    Simulate returns and latent log-volatility from a basic stochastic
    volatility model.

    Model:
        r_t = exp(h_t / 2) * epsilon_t
        h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t
    """
    rng = np.random.default_rng(seed)
    h = np.empty(T)

    # Start the latent AR(1) process from its stationary distribution.
    stationary_sd = sigma_eta / np.sqrt(1.0 - phi**2)
    h[0] = rng.normal(mu, stationary_sd)

    for t in range(1, T):
        innovation = rng.normal()
        h[t] = mu + phi * (h[t - 1] - mu) + sigma_eta * innovation

    eps = rng.normal(size=T)
    returns = np.exp(h / 2.0) * eps

    data_dict = {
        "time": np.arange(1, T + 1),
        "h_true": h,
        "vol_true": np.exp(h / 2.0),
        "return": returns,
        "abs_return": np.abs(returns),
    }
    data = pd.DataFrame(data_dict) if pd is not None else data_dict
    params = {"T": T, "mu": mu, "phi": phi, "sigma_eta": sigma_eta, "seed": seed}
    return data, params


def get_column(data, name):
    """Return a named data column as a NumPy array."""
    values = data[name]
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def local_log_posterior(h_value, t, h_path, returns_sq, mu, phi, sigma2, stationary_var):
    """
    Local log-posterior contribution for one latent state h_t.

    Only terms that depend on h_t are included. Constants are omitted
    because they cancel in the Metropolis-Hastings ratio.
    """
    log_density = -0.5 * (h_value + returns_sq[t] * np.exp(-h_value))

    if t == 0:
        log_density += -0.5 * (h_value - mu) ** 2 / stationary_var
    else:
        mean_t = mu + phi * (h_path[t - 1] - mu)
        log_density += -0.5 * (h_value - mean_t) ** 2 / sigma2

    if t < len(h_path) - 1:
        mean_next = mu + phi * (h_value - mu)
        log_density += -0.5 * (h_path[t + 1] - mean_next) ** 2 / sigma2

    return log_density


def mcmc_latent_sv(
    returns,
    mu,
    phi,
    sigma_eta,
    n_iter=12000,
    burn_in=4000,
    proposal_sd=0.15,
    seed=456,
    selected_indices=None,
):
    """
    Run a component-wise random-walk Metropolis-Hastings sampler for the
    latent log-volatility path, conditional on fixed parameters.
    """
    if burn_in >= n_iter:
        raise ValueError("burn_in must be smaller than n_iter.")

    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    n_saved = n_iter - burn_in

    if selected_indices is None:
        selected_indices = [249, 499, 749]
    selected_indices = [idx for idx in selected_indices if 0 <= idx < T]

    returns_sq = returns**2
    sigma2 = sigma_eta**2
    stationary_var = sigma2 / (1.0 - phi**2)

    # A simple deterministic starting value based on squared returns.
    h_current = np.log(returns_sq + 1e-6)

    draws = np.empty((n_saved, T))
    n_accept = 0
    n_proposals = n_iter * T

    for iteration in range(n_iter):
        for t in range(T):
            current_value = h_current[t]
            proposed_value = current_value + rng.normal(0.0, proposal_sd)

            log_current = local_log_posterior(
                current_value,
                t,
                h_current,
                returns_sq,
                mu,
                phi,
                sigma2,
                stationary_var,
            )
            log_proposed = local_log_posterior(
                proposed_value,
                t,
                h_current,
                returns_sq,
                mu,
                phi,
                sigma2,
                stationary_var,
            )

            log_acceptance = log_proposed - log_current
            if np.log(rng.random()) < log_acceptance:
                h_current[t] = proposed_value
                n_accept += 1

        if iteration >= burn_in:
            draws[iteration - burn_in] = h_current

        if (iteration + 1) % 2000 == 0:
            print(f"MCMC iteration {iteration + 1:,} of {n_iter:,} completed.")

    posterior_mean = draws.mean(axis=0)
    posterior_lower = np.quantile(draws, 0.025, axis=0)
    posterior_upper = np.quantile(draws, 0.975, axis=0)
    selected_traces = {f"h_{idx + 1}": draws[:, idx].copy() for idx in selected_indices}

    return {
        "draws": draws,
        "posterior_mean": posterior_mean,
        "posterior_lower": posterior_lower,
        "posterior_upper": posterior_upper,
        "acceptance_rate": n_accept / n_proposals,
        "selected_indices": selected_indices,
        "selected_traces": selected_traces,
    }


def autocorrelation(x, max_lag=60):
    """Compute sample autocorrelations from lag 0 through max_lag."""
    x = np.asarray(x, dtype=float)
    x_centered = x - x.mean()
    denominator = np.dot(x_centered, x_centered)

    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x_centered[:-lag], x_centered[lag:]) / denominator
    return acf


def configure_plots():
    """Set a clean plotting style suitable for thesis figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 4.2),
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.2,
        }
    )


def print_parameter_summary(params, n_iter, burn_in, proposal_sd):
    """Print the fixed SV parameters and MCMC settings."""
    rows = [
        ("T", params["T"]),
        ("mu", params["mu"]),
        ("phi", params["phi"]),
        ("sigma_eta", params["sigma_eta"]),
        ("MCMC iterations", n_iter),
        ("burn-in", burn_in),
        ("proposal standard deviation", proposal_sd),
    ]

    print("\nFixed SV parameters and MCMC settings:")
    print(f"{'Quantity':<30}Value")
    print("-" * 42)
    for name, value in rows:
        print(f"{name:<30}{value}")


def save_figure(fig, stem):
    """Save one figure as both PDF and high-resolution PNG."""
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def plot_latent_log_volatility(data):
    fig, ax = plt.subplots()
    ax.plot(get_column(data, "time"), get_column(data, "h_true"), color="black")
    ax.set_title("Simulated Latent Volatility")
    ax.set_xlabel("Time index t")
    ax.set_ylabel("Latent log-volatility h_t")
    return save_figure(fig, "fig_A1_latent_volatility")


def plot_returns(data):
    fig, ax = plt.subplots()
    ax.plot(get_column(data, "time"), get_column(data, "return"), color="black", linewidth=0.9)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("Simulated Returns from the SV Model")
    ax.set_xlabel("Time index t")
    ax.set_ylabel("Simulated return r_t")
    return save_figure(fig, "fig_A2_simulated_returns")


def plot_absolute_returns(data):
    fig, ax = plt.subplots()
    ax.plot(get_column(data, "time"), get_column(data, "abs_return"), color="black", linewidth=0.9)
    ax.set_title("Absolute Returns and Volatility Clustering")
    ax.set_xlabel("Time index t")
    ax.set_ylabel("Absolute return |r_t|")
    return save_figure(fig, "fig_A3_absolute_returns")


def plot_posterior_credible_bands(data, posterior_mean, posterior_lower, posterior_upper):
    time = get_column(data, "time")
    h_true = get_column(data, "h_true")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.fill_between(
        time,
        posterior_lower,
        posterior_upper,
        color="lightgray",
        alpha=0.8,
        label="95% credible band",
    )
    ax.plot(time, h_true, color="black", linewidth=1.0, label="True h_t")
    ax.plot(time, posterior_mean, color="#1f77b4", linewidth=1.2, label="Posterior mean")
    ax.set_title("Posterior Recovery of Latent Log-Volatility")
    ax.set_xlabel("Time index t")
    ax.set_ylabel("Latent log-volatility h_t")
    ax.legend(loc="best")
    return save_figure(fig, "fig_A4_posterior_credible_bands")


def plot_trace_plots(selected_traces):
    n_traces = len(selected_traces)
    fig, axes = plt.subplots(n_traces, 1, figsize=(8.0, 2.3 * n_traces), sharex=True)
    if n_traces == 1:
        axes = [axes]

    x = None
    for ax, (label, trace) in zip(axes, selected_traces.items()):
        x = np.arange(1, len(trace) + 1)
        ax.plot(x, trace, color="black", linewidth=0.7)
        ax.set_ylabel(label)

    axes[0].set_title("Trace Plots for Selected Latent States")
    axes[-1].set_xlabel("MCMC iteration after burn-in")
    return save_figure(fig, "fig_A5_trace_plots")


def plot_autocorrelation(trace, label, max_lag=60):
    acf = autocorrelation(trace, max_lag=max_lag)
    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots()
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.vlines(lags, 0.0, acf, color="black", linewidth=1.0)
    ax.plot(lags, acf, "o", color="black", markersize=3)
    ax.set_title("Autocorrelation of MCMC Draws for a Selected Latent State")
    ax.set_xlabel("Lag")
    ax.set_ylabel(f"Autocorrelation of {label}")
    return save_figure(fig, "fig_A6_autocorrelation")


def make_selected_summary(draws, selected_indices):
    rows = []
    for idx in selected_indices:
        trace = draws[:, idx]
        rows.append(
            {
                "state": f"h_{idx + 1}",
                "posterior_mean": trace.mean(),
                "posterior_sd": trace.std(ddof=1),
                "q_2_5": np.quantile(trace, 0.025),
                "q_97_5": np.quantile(trace, 0.975),
            }
        )
    return pd.DataFrame(rows) if pd is not None else rows


def save_selected_summary(summary, path):
    """Save the selected-state posterior summary as a CSV file."""
    if hasattr(summary, "to_csv"):
        summary.to_csv(path, index=False)
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plots()

    data, params = simulate_sv(T=1000, mu=-10.0, phi=0.98, sigma_eta=0.20, seed=123)
    print_parameter_summary(params, N_ITER, BURN_IN, PROPOSAL_SD)

    # Parameters are fixed at their data-generating values; only h_{1:T} is sampled.
    mcmc = mcmc_latent_sv(
        returns=get_column(data, "return"),
        mu=params["mu"],
        phi=params["phi"],
        sigma_eta=params["sigma_eta"],
        n_iter=N_ITER,
        burn_in=BURN_IN,
        proposal_sd=PROPOSAL_SD,
        seed=456,
        selected_indices=[249, 499, 749],
    )

    figure_files = []
    figure_files.extend(plot_latent_log_volatility(data))
    figure_files.extend(plot_returns(data))
    figure_files.extend(plot_absolute_returns(data))
    figure_files.extend(
        plot_posterior_credible_bands(
            data,
            mcmc["posterior_mean"],
            mcmc["posterior_lower"],
            mcmc["posterior_upper"],
        )
    )
    figure_files.extend(plot_trace_plots(mcmc["selected_traces"]))

    acf_label = "h_500" if "h_500" in mcmc["selected_traces"] else next(iter(mcmc["selected_traces"]))
    figure_files.extend(plot_autocorrelation(mcmc["selected_traces"][acf_label], acf_label))

    selected_summary = make_selected_summary(mcmc["draws"], mcmc["selected_indices"])
    selected_summary_path = OUTPUT_DIR / "selected_posterior_summary.csv"
    save_selected_summary(selected_summary, selected_summary_path)

    selected_idx = 499 if len(get_column(data, "time")) >= 500 else mcmc["selected_indices"][0]
    selected_label = f"h_{selected_idx + 1}"
    selected_trace = mcmc["draws"][:, selected_idx]
    average_ci_width = np.mean(mcmc["posterior_upper"] - mcmc["posterior_lower"])

    acceptance_path = OUTPUT_DIR / "acceptance_rate.txt"
    acceptance_path.write_text(
        "\n".join(
            [
                f"Overall acceptance rate: {mcmc['acceptance_rate']:.4f}",
                f"Posterior mean of {selected_label}: {selected_trace.mean():.4f}",
                f"Posterior standard deviation of {selected_label}: {selected_trace.std(ddof=1):.4f}",
                f"Average 95% credible interval width: {average_ci_width:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nAppendix SV simulation completed.")
    print(f"MCMC acceptance rate: {mcmc['acceptance_rate']:.4f}")
    print(f"Posterior mean of {selected_label}: {selected_trace.mean():.4f}")
    print(f"Posterior standard deviation of {selected_label}: {selected_trace.std(ddof=1):.4f}")
    print(f"Average 95% credible interval width: {average_ci_width:.4f}")
    print(f"Selected posterior summary saved to: {selected_summary_path}")
    print(f"Acceptance-rate summary saved to: {acceptance_path}")
    print(
        "\nInterpretation note: this exercise is illustrative. "
        "A full Bayesian estimation would also sample the model parameters "
        "jointly with the latent states."
    )
    print("\nGenerated figure files:")
    for path in figure_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
