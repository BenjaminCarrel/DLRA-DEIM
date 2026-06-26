"""Adaptive RK45 PRK-DEIM experiment for the nonlinear Schrodinger problem.

This script mirrors the schrodinger_global_error experiment, but uses QDEIM
and adds a Dormand-Prince 5(4) step-size controller.  It logs accepted steps,
rejected steps, global errors, and changes in the QDEIM row/column selections.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse as sps

from low_rank_toolbox import QDEIM, SVD
from matrix_ode_toolbox import NonLinearSchrodingerOde
from matrix_ode_toolbox.dlra_deim import ProjectedRungeKuttaDeim, solve_dlra_deim
from matrix_ode_toolbox.integrate import solve_matrix_ivp


@dataclass
class StepRecord:
    t0: float
    t1: float
    h: float
    err_norm: float
    accepted: bool
    switched_u: bool
    switched_v: bool
    nfev: int


def make_nonlinear_nonstiff_schrodinger_ode(size: int):
    """Same problem generation as schrodinger_global_error/problem_generation.py."""
    A = sps.diags([1, 1], [-1, 1], shape=(size, size), format="csc", dtype=np.complex128)
    alpha = 0.1
    ode = NonLinearSchrodingerOde(A, alpha)

    mu1 = round(0.6 * size)
    mu2 = round(0.5 * size)
    nu1 = round(0.5 * size)
    nu2 = round(0.4 * size)
    sigma = 0.1 * size
    X0 = np.zeros((size, size), dtype=np.complex128)
    grid = np.arange(size)
    jj, kk = np.meshgrid(grid, grid, indexing="ij")
    X0 = (
        np.exp(-((jj - mu1) ** 2 + (kk - nu1) ** 2) / (sigma**2))
        + np.exp(-((jj - mu2) ** 2 + (kk - nu2) ** 2) / (sigma**2))
    ).astype(np.complex128)
    ode._use_low_rank_hadamard = True
    return ode, X0


def qdeim_with_M(U: np.ndarray):
    p, M = QDEIM(U, compute_M=True)
    return np.asarray(p, dtype=int), M


def qdeim_state(Y: SVD):
    p_u, M_u = qdeim_with_M(Y.U)
    p_v, M_v = qdeim_with_M(Y.V)
    return p_u, M_u, p_v, M_v


def same_indices(a: np.ndarray | None, b: np.ndarray) -> bool:
    return a is not None and a.shape == b.shape and np.array_equal(a, b)


def combine_and_truncate(Y: SVD, h: float, coeffs: np.ndarray, ks: list[SVD], rank: int) -> SVD:
    terms = [Y]
    for coeff, k in zip(coeffs, ks):
        if coeff != 0:
            terms.append((h * coeff) * k)
    return SVD.multi_add(terms, truncate=False).truncate(rank)


def dense_relative_error(Y: SVD, X_ref: np.ndarray) -> float:
    return np.linalg.norm(Y.todense() - X_ref, "fro") / np.linalg.norm(X_ref, "fro")


class AdaptiveRK45QDEIM:
    """Dormand-Prince 5(4) PRK-DEIM stepper with QDEIM switching logs."""

    # SciPy RK45 / Dormand-Prince tableau.
    C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], dtype=float)
    A = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=float)
    B4 = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
        dtype=float,
    )

    def __init__(
        self,
        ode,
        rank: int,
        rtol: float,
        atol: float,
        h0: float,
        h_min: float,
        h_max: float,
        safety: float = 0.9,
    ):
        self.ode = ode
        self.rank = rank
        self.rtol = rtol
        self.atol = atol
        self.h0 = h0
        self.h_min = h_min
        self.h_max = h_max
        self.safety = safety
        self.records: list[StepRecord] = []
        self.accepted_states: list[SVD] = []
        self.accepted_times: list[float] = []
        self.switch_events: list[dict] = []
        self.stage_switch_events: list[dict] = []

    def projected_field(self, t: float, Y: SVD, p_u: np.ndarray, M_u, p_v: np.ndarray, M_v) -> SVD:
        return self.ode.DEIM_tangent_space_ode_F(
            t,
            Y,
            U_indexes=p_u,
            M_u=M_u,
            V_indexes=p_v,
            M_v=M_v,
            truncate=False,
        )

    def step_attempt(self, t: float, Y: SVD, h: float):
        ks: list[SVD] = []
        stage_indices = []

        for j, c_j in enumerate(self.C):
            if j == 0:
                eta = Y
            else:
                eta = combine_and_truncate(Y, h, np.asarray(self.A[j], dtype=float), ks, self.rank)

            p_u, M_u, p_v, M_v = qdeim_state(eta)
            stage_indices.append((p_u, p_v))
            ks.append(self.projected_field(t + c_j * h, eta, p_u, M_u, p_v, M_v))

        Y5 = combine_and_truncate(Y, h, self.B5, ks, self.rank)
        Y4 = combine_and_truncate(Y, h, self.B4, ks, self.rank)
        diff = SVD.multi_add([Y5, -Y4], truncate=False)
        scale = self.atol + self.rtol * max(Y.norm("fro"), Y5.norm("fro"))
        err_norm = diff.norm("fro") / scale
        return Y5, err_norm, stage_indices

    def solve(self, t_span: tuple[float, float], Y0: SVD):
        t, tf = t_span
        Y = Y0
        h = min(self.h0, self.h_max, tf - t)
        self.accepted_times = [t]
        self.accepted_states = [Y]
        prev_u, _, prev_v, _ = qdeim_state(Y)

        while t < tf:
            h = min(h, tf - t, self.h_max)
            if h < self.h_min:
                raise RuntimeError(f"Step size underflow at t={t:.16e}, h={h:.3e}.")

            Y_candidate, err_norm, stage_indices = self.step_attempt(t, Y, h)
            accepted = err_norm <= 1.0

            next_u, _, next_v, _ = qdeim_state(Y_candidate)
            switched_u = accepted and not same_indices(prev_u, next_u)
            switched_v = accepted and not same_indices(prev_v, next_v)

            record = StepRecord(
                t0=t,
                t1=t + h if accepted else t,
                h=h,
                err_norm=float(err_norm),
                accepted=accepted,
                switched_u=switched_u,
                switched_v=switched_v,
                nfev=len(self.C),
            )
            self.records.append(record)

            for stage, (p_u, p_v) in enumerate(stage_indices):
                if not same_indices(prev_u, p_u) or not same_indices(prev_v, p_v):
                    self.stage_switch_events.append(
                        {
                            "attempt_t0": t,
                            "attempt_h": h,
                            "stage": stage,
                            "stage_time": t + self.C[stage] * h,
                            "u_changed_from_step_start": not same_indices(prev_u, p_u),
                            "v_changed_from_step_start": not same_indices(prev_v, p_v),
                        }
                    )

            if accepted:
                old_u, old_v = prev_u, prev_v
                t = t + h
                Y = Y_candidate
                self.accepted_times.append(t)
                self.accepted_states.append(Y)
                if switched_u or switched_v:
                    self.switch_events.append(
                        {
                            "t": t,
                            "h": h,
                            "switched_u": switched_u,
                            "switched_v": switched_v,
                            "old_u": old_u.tolist(),
                            "new_u": next_u.tolist(),
                            "old_v": old_v.tolist(),
                            "new_v": next_v.tolist(),
                        }
                    )
                prev_u, prev_v = next_u, next_v

            if err_norm == 0:
                factor = 5.0
            else:
                factor = self.safety * err_norm ** (-1 / 5)
                factor = min(5.0, max(0.2, factor))

            h *= factor

        return Y


def run_fixed_global_error(ode, X0, X_ref_final, rank: int, t_span, nb_steps: int, nb_substeps):
    Y0 = SVD.truncated_svd(X0, rank)
    t_eval = np.linspace(t_span[0], t_span[1], nb_steps + 1)
    rows = []
    for nb in nb_substeps:
        for order in (1, 2, 3):
            kwargs = {
                "nb_substeps": int(nb),
                "order": order,
                "deim_method": "qdeim",
                "deim_kwargs": {},
            }
            start = time.time()
            sol = solve_dlra_deim(
                ode,
                t_span,
                Y0,
                dlra_deim_solver=ProjectedRungeKuttaDeim,
                t_eval=t_eval,
                dense_output=False,
                monitor=False,
                dlra_deim_kwargs=kwargs,
            )
            elapsed = time.time() - start
            err = dense_relative_error(sol.Xs[-1], X_ref_final)
            rows.append(
                {
                    "method": f"PRK{order}-QDEIM",
                    "nb_substeps_per_output": int(nb),
                    "h": (t_span[1] - t_span[0]) / (nb_steps * int(nb)),
                    "relative_error": err,
                    "time_sec": elapsed,
                }
            )
            print(f"fixed PRK{order}-QDEIM nb={nb}: relerr={err:.3e}, time={elapsed:.2f}s")
            sys.stdout.flush()
    return pd.DataFrame(rows)


def plot_steps(step_df: pd.DataFrame, switch_df: pd.DataFrame, stage_switch_df: pd.DataFrame, output_path: Path):
    accepted = step_df[step_df["accepted"]]
    rejected = step_df[~step_df["accepted"]]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, constrained_layout=True)
    ax0.step(accepted["t1"], accepted["h"], where="post", label="accepted RK45 step", color="#1f77b4")
    if not rejected.empty:
        ax0.scatter(rejected["t0"], rejected["h"], marker="x", color="#d62728", label="rejected attempts")
    for _, row in switch_df.iterrows():
        ax0.axvline(row["t"], color="#2ca02c", alpha=0.35, linewidth=1.2)
    ax0.set_yscale("log")
    ax0.set_ylabel("step size h")
    ax0.legend(loc="best")
    ax0.set_title("Adaptive RK45-QDEIM: step sizes and accepted QDEIM switches")

    ax1.semilogy(accepted["t1"], np.maximum(accepted["err_norm"], 1e-16), ".", color="#1f77b4")
    if not stage_switch_df.empty:
        ax1.vlines(stage_switch_df["stage_time"], 1e-16, 1.0, color="#ff7f0e", alpha=0.10, linewidth=0.8)
    for _, row in switch_df.iterrows():
        ax1.axvline(row["t"], color="#2ca02c", alpha=0.35, linewidth=1.2)
    ax1.axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("embedded error")
    ax1.set_xlabel("time")
    ax1.set_ylim(bottom=1e-16)
    ax1.set_title("Orange: within-step QDEIM stage changes; green: accepted state change")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_global_errors(fixed_df: pd.DataFrame, adaptive_error: float, output_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for method, group in fixed_df.groupby("method"):
        group = group.sort_values("h")
        ax.loglog(group["h"], group["relative_error"], "o-", label=method)
    ax.axhline(adaptive_error, color="#d62728", linestyle="--", label="adaptive RK45-QDEIM")
    ax.set_xlabel("fixed step size h")
    ax.set_ylabel("relative error at T")
    ax.invert_xaxis()
    ax.legend(loc="best")
    ax.set_title("Schrodinger global error with QDEIM")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_substeps(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--rank", type=int, default=9)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=0.01)
    parser.add_argument("--nb-steps", type=int, default=10)
    parser.add_argument("--substeps", type=parse_substeps, default=parse_substeps("1,3,10,30"))
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--h0", type=float, default=1e-3)
    parser.add_argument("--h-min", type=float, default=1e-10)
    parser.add_argument("--h-max", type=float, default=5e-2)
    parser.add_argument("--reference-rtol", type=float, default=1e-10)
    parser.add_argument("--reference-atol", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or (
        script_dir
        / "data"
        / "adaptive_rk45_qdeim"
        / f"size_{args.size}_rank_{args.rank}_rtol_{args.rtol:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up Schrodinger problem...")
    ode, X0_raw = make_nonlinear_nonstiff_schrodinger_ode(args.size)
    print(f"Warmup full solution on [0, {args.warmup}]...")
    X0 = solve_matrix_ivp(
        ode,
        (0.0, args.warmup),
        X0_raw,
        dense_output=True,
        rtol=args.reference_rtol,
        atol=args.reference_atol,
    )

    t_span = (0.0, args.t_final)
    print("Computing full reference solution...")
    X_ref_final = solve_matrix_ivp(
        ode,
        t_span,
        X0,
        dense_output=True,
        rtol=args.reference_rtol,
        atol=args.reference_atol,
    )

    print("Running fixed-step global error with QDEIM...")
    fixed_df = run_fixed_global_error(
        ode,
        X0,
        X_ref_final,
        args.rank,
        t_span,
        args.nb_steps,
        args.substeps,
    )

    print("Running adaptive RK45-QDEIM...")
    Y0 = SVD.truncated_svd(X0, args.rank)
    solver = AdaptiveRK45QDEIM(
        ode,
        rank=args.rank,
        rtol=args.rtol,
        atol=args.atol,
        h0=args.h0,
        h_min=args.h_min,
        h_max=args.h_max,
    )
    start = time.time()
    Y_adapt = solver.solve(t_span, Y0)
    adaptive_time = time.time() - start
    adaptive_error = dense_relative_error(Y_adapt, X_ref_final)

    step_df = pd.DataFrame([asdict(record) for record in solver.records])
    switch_df = pd.DataFrame(solver.switch_events)
    stage_switch_df = pd.DataFrame(solver.stage_switch_events)

    fixed_df.to_csv(output_dir / "fixed_global_errors_qdeim.csv", index=False)
    step_df.to_csv(output_dir / "adaptive_steps_qdeim.csv", index=False)
    switch_df.to_csv(output_dir / "accepted_qdeim_switches.csv", index=False)
    stage_switch_df.to_csv(output_dir / "stage_qdeim_switches.csv", index=False)

    summary = {
        "size": args.size,
        "rank": args.rank,
        "t_final": args.t_final,
        "warmup": args.warmup,
        "rtol": args.rtol,
        "atol": args.atol,
        "h0": args.h0,
        "h_max": args.h_max,
        "accepted_steps": int(step_df["accepted"].sum()),
        "rejected_steps": int((~step_df["accepted"]).sum()),
        "accepted_switches": int(len(switch_df)),
        "stage_switch_events": int(len(stage_switch_df)),
        "adaptive_relative_error": float(adaptive_error),
        "adaptive_time_sec": float(adaptive_time),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_steps(step_df, switch_df, stage_switch_df, output_dir / "adaptive_steps_vs_qdeim_switches.png")
    plot_global_errors(fixed_df, adaptive_error, output_dir / "global_error_qdeim.png")

    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
