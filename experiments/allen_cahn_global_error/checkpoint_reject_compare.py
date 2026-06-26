"""Checkpointed Allen-Cahn adaptive RK45 comparison.

This experiment first follows the QDEIM-DLRA trajectory and saves the low-rank
state every fixed checkpoint interval.  It then detects the first checkpoint
interval with many rejected QDEIM attempts and restarts both QDEIM-DLRA and
orthogonal DLRA from the latest saved low-rank state before that interval.
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
from matrix_ode_toolbox.integrate import solve_matrix_ivp


def load_allen_common():
    common_path = Path(__file__).resolve().parent / "adaptive_rk45_qdeim.py"
    spec = importlib.util.spec_from_file_location("allen_adaptive_common", common_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_allen_common()


def save_svd(Y: SVD, path: Path):
    np.savez(path, U=Y.U, s=Y.sing_vals, V=Y.V)


def make_solver(cls, ode, args):
    solver = cls(
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
    return solver


def run_segmented_qdeim(ode, Y0: SVD, checkpoints: np.ndarray, args, state_dir: Path):
    Y = Y0
    all_records = []
    all_switches = []
    all_stage_switches = []
    checkpoint_rows = []
    segment_rows = []

    save_svd(Y, state_dir / "checkpoint_000.npz")
    checkpoint_rows.append({"checkpoint": float(checkpoints[0]), "state_file": "checkpoint_000.npz"})

    for k, (ta, tb) in enumerate(zip(checkpoints[:-1], checkpoints[1:]), start=1):
        solver = make_solver(common.AdaptiveRK45QDEIMInternalProjection, ode, args)
        start = time.time()
        Y = solver.solve((float(ta), float(tb)), Y)
        elapsed = time.time() - start

        records = pd.DataFrame([asdict(record) for record in solver.records])
        switches = pd.DataFrame(solver.switch_events)
        stage_switches = pd.DataFrame(solver.stage_switch_events)
        all_records.append(records)
        all_switches.append(switches)
        all_stage_switches.append(stage_switches)

        rejected = int((~records["accepted"]).sum()) if not records.empty else 0
        accepted = int(records["accepted"].sum()) if not records.empty else 0
        segment_rows.append(
            {
                "segment_start": float(ta),
                "segment_end": float(tb),
                "accepted_steps": accepted,
                "rejected_steps": rejected,
                "accepted_switches": int(len(switches)),
                "stage_switch_events": int(len(stage_switches)),
                "time_sec": elapsed,
            }
        )

        filename = f"checkpoint_{k:03d}.npz"
        save_svd(Y, state_dir / filename)
        checkpoint_rows.append({"checkpoint": float(tb), "state_file": filename})

    records_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    switches_df = pd.concat(all_switches, ignore_index=True) if all_switches else pd.DataFrame()
    stage_switches_df = pd.concat(all_stage_switches, ignore_index=True) if all_stage_switches else pd.DataFrame()
    segments_df = pd.DataFrame(segment_rows)
    checkpoints_df = pd.DataFrame(checkpoint_rows)
    return Y, records_df, switches_df, stage_switches_df, segments_df, checkpoints_df


def run_solver(cls, ode, Y0: SVD, t_span: tuple[float, float], args):
    solver = make_solver(cls, ode, args)
    start = time.time()
    Y = solver.solve(t_span, Y0)
    elapsed = time.time() - start
    records = pd.DataFrame([asdict(record) for record in solver.records])
    switches = pd.DataFrame(getattr(solver, "switch_events", []))
    stage_switches = pd.DataFrame(getattr(solver, "stage_switch_events", []))
    return Y, records, switches, stage_switches, elapsed


def plot_checkpointed(records_df, switches_df, segments_df, output_path: Path):
    accepted = records_df[records_df["accepted"]]
    rejected = records_df[~records_df["accepted"]]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, constrained_layout=True)
    ax0.step(accepted["t1"], accepted["h"], where="post", label="accepted RK45 step", color="#1f77b4")
    if not rejected.empty:
        ax0.scatter(rejected["t0"], rejected["h"], marker="x", color="#d62728", label="rejected attempts")
    if not switches_df.empty:
        for _, row in switches_df.iterrows():
            ax0.axvline(row["t"], color="#2ca02c", alpha=0.25, linewidth=1.0)
    for t in segments_df["segment_end"].iloc[:-1]:
        ax0.axvline(t, color="0.25", alpha=0.35, linestyle="--", linewidth=0.8)
    ax0.set_yscale("log")
    ax0.set_ylabel("step size h")
    ax0.set_title("Checkpointed QDEIM-DLRA run")
    ax0.legend(loc="best")

    ax1.semilogy(accepted["t1"], np.maximum(accepted["err_norm"], 1e-16), ".", color="#1f77b4")
    if not rejected.empty:
        ax1.semilogy(rejected["t0"], np.maximum(rejected["err_norm"], 1e-16), "x", color="#d62728")
    ax1.axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    for t in segments_df["segment_end"].iloc[:-1]:
        ax1.axvline(t, color="0.25", alpha=0.35, linestyle="--", linewidth=0.8)
    ax1.set_ylabel("embedded error")
    ax1.set_xlabel("relative time")
    ax1.set_ylim(bottom=1e-16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_restart_comparison(qdeim_df, orth_df, switches_df, output_path: Path):
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
    if not switches_df.empty:
        switch_times = switches_df.loc[
            switches_df["switched_u"].astype(bool) | switches_df["switched_v"].astype(bool),
            "t",
        ]
        for k, t in enumerate(switch_times):
            ax0.axvline(
                t,
                color="#ff7f0e",
                alpha=0.22,
                linewidth=0.8,
                label="QDEIM index-set change" if k == 0 else None,
            )
    ax0.set_yscale("log")
    ax0.set_ylabel("step size h")
    ax0.set_title("Restart from saved QDEIM state: QDEIM vs orthogonal DLRA")
    ax0.legend(loc="best")

    ax1.semilogy(q_acc["t1"], np.maximum(q_acc["err_norm"], 1e-16), ".", color="#1f77b4", label="QDEIM accepted")
    if not q_rej.empty:
        ax1.semilogy(q_rej["t0"], np.maximum(q_rej["err_norm"], 1e-16), "x", color="#d62728", label="QDEIM rejected")
    ax1.semilogy(o_acc["t1"], np.maximum(o_acc["err_norm"], 1e-16), ".", color="#2ca02c", label="orthogonal accepted")
    if not o_rej.empty:
        ax1.semilogy(o_rej["t0"], np.maximum(o_rej["err_norm"], 1e-16), "+", color="#9467bd", label="orthogonal rejected")
    if not switches_df.empty:
        switch_times = switches_df.loc[
            switches_df["switched_u"].astype(bool) | switches_df["switched_v"].astype(bool),
            "t",
        ]
        for t in switch_times:
            ax1.axvline(t, color="#ff7f0e", alpha=0.22, linewidth=0.8)
    ax1.axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("embedded error")
    ax1.set_xlabel("relative time")
    ax1.set_ylim(bottom=1e-16)
    ax1.legend(loc="best")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_restart_steps(qdeim_df, orth_df, switches_df, output_path: Path):
    q_acc = qdeim_df[qdeim_df["accepted"]]
    q_rej = qdeim_df[~qdeim_df["accepted"]]
    o_acc = orth_df[orth_df["accepted"]]
    o_rej = orth_df[~orth_df["accepted"]]

    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    ax.step(q_acc["t1"], q_acc["h"], where="post", label="QDEIM accepted", color="#1f77b4")
    ax.step(o_acc["t1"], o_acc["h"], where="post", label="orthogonal accepted", color="#2ca02c")
    if not q_rej.empty:
        ax.scatter(q_rej["t0"], q_rej["h"], marker="x", color="#d62728", label="QDEIM rejected")
    if not o_rej.empty:
        ax.scatter(o_rej["t0"], o_rej["h"], marker="+", color="#9467bd", label="orthogonal rejected")
    if not switches_df.empty:
        switch_times = switches_df.loc[
            switches_df["switched_u"].astype(bool) | switches_df["switched_v"].astype(bool),
            "t",
        ]
        for k, t in enumerate(switch_times):
            ax.axvline(
                t,
                color="#ff7f0e",
                alpha=0.22,
                linewidth=0.8,
                label="QDEIM index-set change" if k == 0 else None,
            )
    ax.set_yscale("log")
    ax.set_ylabel("step size h")
    ax.set_xlabel("relative time")
    ax.set_title("Restart from saved QDEIM state: QDEIM vs orthogonal DLRA")
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--t-final", type=float, default=0.25)
    parser.add_argument("--checkpoint-dt", type=float, default=0.05)
    parser.add_argument("--reject-threshold", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--h0", type=float, default=1e-4)
    parser.add_argument("--h-min", type=float, default=1e-12)
    parser.add_argument("--h-max", type=float, default=5e-1)
    parser.add_argument("--reference-rtol", type=float, default=1e-12)
    parser.add_argument("--reference-atol", type=float, default=1e-14)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or (
        script_dir
        / "data"
        / f"checkpoint_compare_size_{args.size}_rank_{args.rank}_tol_{args.rtol:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "states"
    state_dir.mkdir(exist_ok=True)

    print("Setting up Allen-Cahn problem...")
    ode, X0_raw = common.make_allen_cahn(args.size, args.epsilon)
    print(f"Computing full initial value on [0, {args.warmup}]...")
    X0 = solve_matrix_ivp(
        ode,
        (0.0, args.warmup),
        X0_raw,
        dense_output=True,
        scipy_method="RK45",
        rtol=args.reference_rtol,
        atol=args.reference_atol,
    )
    Y0 = SVD.truncated_svd(X0, args.rank)

    checkpoints = np.arange(0.0, args.t_final + 0.5 * args.checkpoint_dt, args.checkpoint_dt)
    if checkpoints[-1] < args.t_final:
        checkpoints = np.append(checkpoints, args.t_final)
    checkpoints[-1] = args.t_final

    print("Running checkpointed QDEIM-DLRA trajectory...")
    Y_final, records_df, switches_df, stage_switches_df, segments_df, checkpoints_df = run_segmented_qdeim(
        ode, Y0, checkpoints, args, state_dir
    )

    records_df.to_csv(output_dir / "checkpointed_qdeim_steps.csv", index=False)
    switches_df.to_csv(output_dir / "checkpointed_qdeim_switches.csv", index=False)
    stage_switches_df.to_csv(output_dir / "checkpointed_qdeim_stage_switches.csv", index=False)
    segments_df.to_csv(output_dir / "checkpoint_segments.csv", index=False)
    checkpoints_df.to_csv(output_dir / "checkpoints.csv", index=False)

    heavy = segments_df[segments_df["rejected_steps"] >= args.reject_threshold]
    if heavy.empty:
        detected = segments_df.loc[segments_df["rejected_steps"].idxmax()]
    else:
        detected = heavy.iloc[0]
    restart_time = float(detected["segment_start"])
    restart_index = int(np.where(np.isclose(checkpoints, restart_time))[0][0])
    restart_state_path = state_dir / f"checkpoint_{restart_index:03d}.npz"
    print(
        f"Detected reject-heavy interval [{detected['segment_start']}, {detected['segment_end']}], "
        f"restarting from checkpoint {restart_time}."
    )

    data = np.load(restart_state_path)
    Y_restart = SVD(data["U"], data["s"], data["V"])

    print("Restarting QDEIM-DLRA from saved state...")
    Y_q_restart, q_restart_df, q_switch_df, q_stage_df, q_time = run_solver(
        common.AdaptiveRK45QDEIMInternalProjection, ode, Y_restart, (restart_time, args.t_final), args
    )
    print("Restarting orthogonal DLRA from same saved state...")
    Y_o_restart, o_restart_df, _, _, o_time = run_solver(
        common.AdaptiveRK45OrthogonalInternalProjection, ode, Y_restart, (restart_time, args.t_final), args
    )

    q_restart_df.to_csv(output_dir / "restart_qdeim_steps.csv", index=False)
    q_switch_df.to_csv(output_dir / "restart_qdeim_switches.csv", index=False)
    q_stage_df.to_csv(output_dir / "restart_qdeim_stage_switches.csv", index=False)
    o_restart_df.to_csv(output_dir / "restart_orthogonal_steps.csv", index=False)

    print("Computing full reference for final error...")
    X_ref_final = solve_matrix_ivp(
        ode,
        (0.0, args.t_final),
        X0,
        dense_output=True,
        scipy_method="RK45",
        rtol=args.reference_rtol,
        atol=args.reference_atol,
    )

    summary = {
        "problem": "Allen-Cahn",
        "size": args.size,
        "rank": args.rank,
        "epsilon": args.epsilon,
        "warmup": args.warmup,
        "t_final": args.t_final,
        "checkpoint_dt": args.checkpoint_dt,
        "reject_threshold": args.reject_threshold,
        "rtol": args.rtol,
        "atol": args.atol,
        "detected_interval": {
            "start": float(detected["segment_start"]),
            "end": float(detected["segment_end"]),
            "rejected_steps": int(detected["rejected_steps"]),
            "accepted_steps": int(detected["accepted_steps"]),
        },
        "checkpointed_qdeim": {
            "accepted_steps": int(records_df["accepted"].sum()),
            "rejected_steps": int((~records_df["accepted"]).sum()),
            "accepted_switches": int(len(switches_df)),
            "stage_switch_events": int(len(stage_switches_df)),
        },
        "restart_qdeim": {
            "start": restart_time,
            "accepted_steps": int(q_restart_df["accepted"].sum()),
            "rejected_steps": int((~q_restart_df["accepted"]).sum()),
            "accepted_switches": int(len(q_switch_df)),
            "stage_switch_events": int(len(q_stage_df)),
            "relative_error": float(common.dense_relative_error(Y_q_restart, X_ref_final)),
            "time_sec": q_time,
        },
        "restart_orthogonal": {
            "start": restart_time,
            "accepted_steps": int(o_restart_df["accepted"].sum()),
            "rejected_steps": int((~o_restart_df["accepted"]).sum()),
            "relative_error": float(common.dense_relative_error(Y_o_restart, X_ref_final)),
            "time_sec": o_time,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_checkpointed(records_df, switches_df, segments_df, output_dir / "checkpointed_qdeim_steps.png")
    plot_restart_comparison(q_restart_df, o_restart_df, q_switch_df, output_dir / "restart_qdeim_vs_orthogonal.png")
    plot_restart_steps(q_restart_df, o_restart_df, q_switch_df, output_dir / "restart_qdeim_steps_only.png")

    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
