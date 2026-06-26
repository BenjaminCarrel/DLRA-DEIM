"""Adaptive RK45-QDEIM experiment for the Allen-Cahn problem.

This reuses the adaptive Dormand-Prince PRK-DEIM stepper from the Schrodinger
experiment and applies it to the Allen-Cahn setup from this folder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from low_rank_toolbox import SVD
from matrix_ode_toolbox import SylvesterLikeOde
from matrix_ode_toolbox.integrate import solve_matrix_ivp
from matrix_ode_toolbox.utils import laplacian_1d_dx2


def load_adaptive_common():
    common_path = Path(__file__).resolve().parents[1] / "schrodinger_global_error" / "adaptive_rk45_qdeim.py"
    spec = importlib.util.spec_from_file_location("adaptive_rk45_qdeim_common", common_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_adaptive_common()
BaseAdaptiveRK45QDEIM = common.AdaptiveRK45QDEIM
dense_relative_error = common.dense_relative_error
plot_steps = common.plot_steps
parse_substeps = common.parse_substeps


def combine_untruncated(Y: SVD, h: float, coeffs: np.ndarray, ks: list[SVD]) -> SVD:
    terms = [Y]
    for coeff, k in zip(coeffs, ks):
        if coeff != 0:
            terms.append((h * coeff) * k)
    return SVD.multi_add(terms, truncate=False)


class AdaptiveRK45QDEIMInternalProjection(BaseAdaptiveRK45QDEIM):
    """Dormand-Prince 5(4) with PRK-style internal retractions.

    The stage accumulator Z_j is formed without truncation.  The vector field
    is evaluated at R(Z_j), with QDEIM pivots selected from R(Z_j), and the
    two embedded outputs are retracted at the end.
    """

    def step_attempt(self, t: float, Y: SVD, h: float):
        ks: list[SVD] = []
        stage_indices = []

        for j, c_j in enumerate(self.C):
            if j == 0:
                eta = Y
            else:
                Z_j = combine_untruncated(Y, h, np.asarray(self.A[j], dtype=float), ks)
                eta = Z_j.truncate(self.rank)

            p_u, M_u, p_v, M_v = common.qdeim_state(eta)
            stage_indices.append((p_u, p_v))
            ks.append(self.projected_field(t + c_j * h, eta, p_u, M_u, p_v, M_v))

        Z5 = combine_untruncated(Y, h, self.B5, ks)
        Z4 = combine_untruncated(Y, h, self.B4, ks)
        Y5 = Z5.truncate(self.rank)
        Y4 = Z4.truncate(self.rank)
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
        prev_u, _, prev_v, _ = common.qdeim_state(Y)
        max_attempts = getattr(self, "max_attempts", None)
        progress_every = getattr(self, "progress_every", 0)
        attempts = 0

        while t < tf:
            attempts += 1
            if max_attempts is not None and attempts > max_attempts:
                raise RuntimeError(
                    f"Maximum attempts exceeded at t={t:.16e}, h={h:.3e}; "
                    f"accepted={len(self.accepted_times) - 1}, total_attempts={attempts - 1}."
                )
            h = min(h, tf - t, self.h_max)
            if h < self.h_min:
                raise RuntimeError(f"Step size underflow at t={t:.16e}, h={h:.3e}.")

            Y_candidate, err_norm, stage_indices = self.step_attempt(t, Y, h)
            accepted = err_norm <= 1.0

            next_u, _, next_v, _ = common.qdeim_state(Y_candidate)
            switched_u = accepted and not common.same_indices(prev_u, next_u)
            switched_v = accepted and not common.same_indices(prev_v, next_v)

            record = common.StepRecord(
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
                if not common.same_indices(prev_u, p_u) or not common.same_indices(prev_v, p_v):
                    self.stage_switch_events.append(
                        {
                            "attempt_t0": t,
                            "attempt_h": h,
                            "stage": stage,
                            "stage_time": t + self.C[stage] * h,
                            "u_changed_from_step_start": not common.same_indices(prev_u, p_u),
                            "v_changed_from_step_start": not common.same_indices(prev_v, p_v),
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

            if progress_every and attempts % progress_every == 0:
                print(
                    f"attempt={attempts}, t={t:.6e}, h={h:.3e}, "
                    f"err={err_norm:.3e}, accepted={accepted}"
                )
                sys.stdout.flush()

            if err_norm == 0:
                factor = 5.0
            else:
                factor = self.safety * err_norm ** (-1 / 5)
                factor = min(5.0, max(0.2, factor))
            h *= factor

        return Y


class AdaptiveRK45OrthogonalInternalProjection(BaseAdaptiveRK45QDEIM):
    """Dormand-Prince 5(4) for orthogonal DLRA with internal retractions."""

    def step_attempt(self, t: float, Y: SVD, h: float):
        ks: list[SVD] = []

        for j, c_j in enumerate(self.C):
            if j == 0:
                eta = Y
            else:
                Z_j = combine_untruncated(Y, h, np.asarray(self.A[j], dtype=float), ks)
                eta = Z_j.truncate(self.rank)
            ks.append(self.ode.tangent_space_ode_F(t + c_j * h, eta, truncate=False))

        Z5 = combine_untruncated(Y, h, self.B5, ks)
        Z4 = combine_untruncated(Y, h, self.B4, ks)
        Y5 = Z5.truncate(self.rank)
        Y4 = Z4.truncate(self.rank)
        diff = SVD.multi_add([Y5, -Y4], truncate=False)
        scale = self.atol + self.rtol * max(Y.norm("fro"), Y5.norm("fro"))
        err_norm = diff.norm("fro") / scale
        return Y5, err_norm

    def solve(self, t_span: tuple[float, float], Y0: SVD):
        t, tf = t_span
        Y = Y0
        h = min(self.h0, self.h_max, tf - t)
        self.accepted_times = [t]
        self.accepted_states = [Y]
        max_attempts = getattr(self, "max_attempts", None)
        progress_every = getattr(self, "progress_every", 0)
        attempts = 0

        while t < tf:
            attempts += 1
            if max_attempts is not None and attempts > max_attempts:
                raise RuntimeError(
                    f"Maximum attempts exceeded at t={t:.16e}, h={h:.3e}; "
                    f"accepted={len(self.accepted_times) - 1}, total_attempts={attempts - 1}."
                )
            h = min(h, tf - t, self.h_max)
            if h < self.h_min:
                raise RuntimeError(f"Step size underflow at t={t:.16e}, h={h:.3e}.")

            Y_candidate, err_norm = self.step_attempt(t, Y, h)
            accepted = err_norm <= 1.0
            self.records.append(
                common.StepRecord(
                    t0=t,
                    t1=t + h if accepted else t,
                    h=h,
                    err_norm=float(err_norm),
                    accepted=accepted,
                    switched_u=False,
                    switched_v=False,
                    nfev=len(self.C),
                )
            )

            if accepted:
                t = t + h
                Y = Y_candidate
                self.accepted_times.append(t)
                self.accepted_states.append(Y)

            if progress_every and attempts % progress_every == 0:
                print(
                    f"orthogonal attempt={attempts}, t={t:.6e}, h={h:.3e}, "
                    f"err={err_norm:.3e}, accepted={accepted}"
                )
                sys.stdout.flush()

            if err_norm == 0:
                factor = 5.0
            else:
                factor = self.safety * err_norm ** (-1 / 5)
                factor = min(5.0, max(0.2, factor))
            h *= factor

        return Y


def make_allen_cahn(size: int, epsilon: float = 0.01):
    """Same Allen-Cahn problem as problem_generation.py, parameterized by size."""
    dx = 2 * np.pi / (size + 1)
    xs = np.linspace(dx, 2 * np.pi - dx, size)
    ys = np.linspace(dx, 2 * np.pi - dx, size)

    A = epsilon * laplacian_1d_dx2(size, dx=dx, periodic=True)

    def G(t, X, rows: list = None, cols: list = None):
        if rows is not None and cols is not None:
            return X[rows, :][:, cols] - X[rows, :][:, cols] ** 3
        if rows is not None:
            return X[rows, :] - X[rows, :] ** 3
        if cols is not None:
            return X[:, cols] - X[:, cols] ** 3
        return X - X.hadamard(X.hadamard(X)) if hasattr(X, "hadamard") else X - X**3

    ode = SylvesterLikeOde(A, A, G)

    def u(x, y):
        return (
            (np.exp(-np.tan(x) ** 2) + np.exp(-np.tan(y) ** 2))
            * np.sin(x)
            * np.sin(y)
            / (1 + np.exp(np.abs(1 / np.sin(-x / 2))) + np.exp(np.abs(1 / np.sin(-y / 2))))
        )

    X0 = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            X0[i, j] = u(xs[i], ys[-j])

    ode._use_low_rank_hadamard = True
    return ode, X0


def run_fixed_global_error(ode, X0, X_ref_final, rank: int, t_span, nb_steps: int, nb_substeps):
    """Fixed-step PRK-QDEIM baseline; deliberately non-exponential to match RK45."""
    from matrix_ode_toolbox.dlra_deim import ProjectedRungeKuttaDeim, solve_dlra_deim

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
    ax.set_title("Allen-Cahn global error with QDEIM")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_solution_snapshots(X0, X_ref_final, Y_final, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    data = [X0, X_ref_final, Y_final.todense()]
    titles = ["after warmup", "reference at T", "adaptive RK45-QDEIM at T"]
    vmin = min(np.min(np.real(X)) for X in data)
    vmax = max(np.max(np.real(X)) for X in data)
    for ax, X, title in zip(axes, data, titles):
        im = ax.imshow(np.real(X), origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.75)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_adaptive_comparison(qdeim_df: pd.DataFrame, orth_df: pd.DataFrame, output_path: Path):
    q_acc = qdeim_df[qdeim_df["accepted"]]
    q_rej = qdeim_df[~qdeim_df["accepted"]]
    o_acc = orth_df[orth_df["accepted"]]
    o_rej = orth_df[~orth_df["accepted"]]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, constrained_layout=True)
    ax0.step(q_acc["t1"], q_acc["h"], where="post", label="QDEIM accepted", color="#1f77b4")
    ax0.step(o_acc["t1"], o_acc["h"], where="post", label="orthogonal accepted", color="#2ca02c")
    if not q_rej.empty:
        ax0.scatter(q_rej["t0"], q_rej["h"], marker="x", color="#d62728", label="QDEIM rejected")
    if not o_rej.empty:
        ax0.scatter(o_rej["t0"], o_rej["h"], marker="+", color="#9467bd", label="orthogonal rejected")
    ax0.set_yscale("log")
    ax0.set_ylabel("step size h")
    ax0.legend(loc="best")
    ax0.set_title("Adaptive RK45: QDEIM vs orthogonal DLRA")

    ax1.semilogy(q_acc["t1"], np.maximum(q_acc["err_norm"], 1e-16), ".", color="#1f77b4", label="QDEIM accepted")
    ax1.semilogy(o_acc["t1"], np.maximum(o_acc["err_norm"], 1e-16), ".", color="#2ca02c", label="orthogonal accepted")
    if not q_rej.empty:
        ax1.semilogy(q_rej["t0"], np.maximum(q_rej["err_norm"], 1e-16), "x", color="#d62728", label="QDEIM rejected")
    if not o_rej.empty:
        ax1.semilogy(o_rej["t0"], np.maximum(o_rej["err_norm"], 1e-16), "+", color="#9467bd", label="orthogonal rejected")
    ax1.axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("embedded error")
    ax1.set_xlabel("time")
    ax1.set_ylim(bottom=1e-16)
    ax1.legend(loc="best")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--nb-steps", type=int, default=10)
    parser.add_argument("--substeps", type=parse_substeps, default=parse_substeps("10,30,100"))
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--h0", type=float, default=1e-4)
    parser.add_argument("--h-min", type=float, default=1e-12)
    parser.add_argument("--h-max", type=float, default=5e-2)
    parser.add_argument("--reference-rtol", type=float, default=1e-9)
    parser.add_argument("--reference-atol", type=float, default=1e-11)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--compare-orthogonal", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or (
        script_dir
        / "data"
        / "adaptive_rk45_qdeim"
        / f"size_{args.size}_rank_{args.rank}_eps_{args.epsilon:g}_rtol_{args.rtol:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up Allen-Cahn problem...")
    ode, X0_raw = make_allen_cahn(args.size, args.epsilon)
    start_time = args.warmup if args.start_time is None else args.start_time
    print(f"Computing full initial value on [0, {start_time}]...")
    X0 = solve_matrix_ivp(
        ode,
        (0.0, start_time),
        X0_raw,
        dense_output=True,
        scipy_method="RK45",
        rtol=args.reference_rtol,
        atol=args.reference_atol,
    )

    t_span = (start_time, args.t_final) if args.start_time is not None else (0.0, args.t_final)
    print("Computing full reference solution...")
    X_ref_final = solve_matrix_ivp(
        ode,
        t_span,
        X0,
        dense_output=True,
        scipy_method="RK45",
        rtol=args.reference_rtol,
        atol=args.reference_atol,
    )

    if args.skip_fixed:
        fixed_df = pd.DataFrame(columns=["method", "h", "relative_error"])
    else:
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
    solver = AdaptiveRK45QDEIMInternalProjection(
        ode,
        rank=args.rank,
        rtol=args.rtol,
        atol=args.atol,
        h0=args.h0,
        h_min=args.h_min,
        h_max=args.h_max,
    )
    solver.max_attempts = args.max_attempts
    solver.progress_every = args.progress_every
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

    orth_summary = None
    orth_df = pd.DataFrame()
    orth_error = None
    if args.compare_orthogonal:
        print("Running adaptive RK45 orthogonal DLRA...")
        orth_solver = AdaptiveRK45OrthogonalInternalProjection(
            ode,
            rank=args.rank,
            rtol=args.rtol,
            atol=args.atol,
            h0=args.h0,
            h_min=args.h_min,
            h_max=args.h_max,
        )
        orth_solver.max_attempts = args.max_attempts
        orth_solver.progress_every = args.progress_every
        start = time.time()
        Y_orth = orth_solver.solve(t_span, Y0)
        orth_time = time.time() - start
        orth_error = dense_relative_error(Y_orth, X_ref_final)
        orth_df = pd.DataFrame([asdict(record) for record in orth_solver.records])
        orth_df.to_csv(output_dir / "adaptive_steps_orthogonal.csv", index=False)
        orth_summary = {
            "adaptive_method": "RK45 orthogonal DLRA with PRK internal projection",
            "accepted_steps": int(orth_df["accepted"].sum()),
            "rejected_steps": int((~orth_df["accepted"]).sum()),
            "adaptive_relative_error": float(orth_error),
            "adaptive_time_sec": float(orth_time),
        }

    summary = {
        "problem": "Allen-Cahn",
        "adaptive_method": "RK45-QDEIM with PRK internal projection",
        "stage_formula": "Z_j untruncated; evaluate DEIM tangent field at R(Z_j); retract embedded outputs",
        "size": args.size,
        "rank": args.rank,
        "epsilon": args.epsilon,
        "start_time": start_time,
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
    if orth_summary is not None:
        summary["orthogonal"] = orth_summary
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_steps(step_df, switch_df, stage_switch_df, output_dir / "adaptive_steps_vs_qdeim_switches.png")
    if not fixed_df.empty:
        plot_global_errors(fixed_df, adaptive_error, output_dir / "global_error_qdeim.png")
    plot_solution_snapshots(X0, X_ref_final, Y_adapt, output_dir / "solution_snapshots.png")
    if args.compare_orthogonal:
        plot_adaptive_comparison(step_df, orth_df, output_dir / "adaptive_qdeim_vs_orthogonal.png")

    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
