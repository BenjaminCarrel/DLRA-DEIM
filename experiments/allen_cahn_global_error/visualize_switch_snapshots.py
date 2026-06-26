"""Visualize Allen-Cahn DLRA states around the QDEIM switching burst."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from low_rank_toolbox import SVD

import checkpoint_reject_compare as checkpoint


def load_svd(path: Path) -> SVD:
    data = np.load(path)
    return SVD(data["U"], data["s"], data["V"])


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


def solve_targets(cls, ode, Y0: SVD, start_time: float, targets: list[float], args):
    snapshots = []
    records = []
    switches = []
    Y = Y0
    t0 = start_time

    for target in targets:
        solver = make_solver(cls, ode, args)
        Y = solver.solve((t0, target), Y)
        snapshots.append(Y)
        records.extend(solver.records)
        switches.extend(getattr(solver, "switch_events", []))
        t0 = target

    return snapshots, records, switches


def plot_qdeim_only(times, qdeim_states, output_path: Path):
    mats = [np.real(Y.todense()) for Y in qdeim_states]
    vmin = min(float(np.min(X)) for X in mats)
    vmax = max(float(np.max(X)) for X in mats)

    fig, axes = plt.subplots(1, len(times), figsize=(9, 3.0), constrained_layout=True)
    for ax, t, X in zip(axes, times, mats):
        im = ax.imshow(X, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"QDEIM, t={t:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.78)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_qdeim_vs_orthogonal(times, qdeim_states, orthogonal_states, output_path: Path):
    q_mats = [np.real(Y.todense()) for Y in qdeim_states]
    o_mats = [np.real(Y.todense()) for Y in orthogonal_states]
    all_mats = q_mats + o_mats
    vmin = min(float(np.min(X)) for X in all_mats)
    vmax = max(float(np.max(X)) for X in all_mats)

    fig, axes = plt.subplots(2, len(times), figsize=(9, 5.0), constrained_layout=True)
    for col, t in enumerate(times):
        im = axes[0, col].imshow(q_mats[col], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"QDEIM, t={t:.3f}")
        axes[1, col].imshow(o_mats[col], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f"orthogonal, t={t:.3f}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.82)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_difference(times, qdeim_states, orthogonal_states, output_path: Path):
    diffs = [
        np.real(Yq.todense() - Yo.todense())
        for Yq, Yo in zip(qdeim_states, orthogonal_states)
    ]
    vmax = max(float(np.max(np.abs(X))) for X in diffs)

    fig, axes = plt.subplots(1, len(times), figsize=(9, 3.0), constrained_layout=True)
    for ax, t, X in zip(axes, times, diffs):
        im = ax.imshow(X, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"QDEIM - orth., t={t:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.78)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--start-time", type=float, default=0.2)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.238, 0.240, 0.242])
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--h0", type=float, default=1e-4)
    parser.add_argument("--h-min", type=float, default=1e-12)
    parser.add_argument("--h-max", type=float, default=5e-1)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--input-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ode, _ = checkpoint.common.make_allen_cahn(args.size, args.epsilon)
    Y0 = load_svd(args.input_state)
    targets = sorted(args.targets)

    print(f"Integrating QDEIM snapshots to {targets}...")
    qdeim_states, q_records, q_switches = solve_targets(
        checkpoint.common.AdaptiveRK45QDEIMInternalProjection,
        ode,
        Y0,
        args.start_time,
        targets,
        args,
    )
    print(f"Integrating orthogonal snapshots to {targets}...")
    orthogonal_states, o_records, _ = solve_targets(
        checkpoint.common.AdaptiveRK45OrthogonalInternalProjection,
        ode,
        Y0,
        args.start_time,
        targets,
        args,
    )

    for t, Yq, Yo in zip(targets, qdeim_states, orthogonal_states):
        tag = f"{t:.3f}".replace(".", "p")
        np.savez(args.output_dir / f"snapshot_qdeim_t{tag}.npz", U=Yq.U, s=Yq.sing_vals, V=Yq.V)
        np.savez(args.output_dir / f"snapshot_orthogonal_t{tag}.npz", U=Yo.U, s=Yo.sing_vals, V=Yo.V)

    plot_qdeim_only(targets, qdeim_states, args.output_dir / "snapshots_qdeim_around_0p24.png")
    plot_qdeim_vs_orthogonal(
        targets,
        qdeim_states,
        orthogonal_states,
        args.output_dir / "snapshots_qdeim_vs_orthogonal_around_0p24.png",
    )
    plot_difference(targets, qdeim_states, orthogonal_states, args.output_dir / "snapshots_difference_around_0p24.png")

    print(f"QDEIM accepted/rejected: {sum(r.accepted for r in q_records)}/{sum(not r.accepted for r in q_records)}")
    print(f"QDEIM accepted index-set switches: {len(q_switches)}")
    print(f"Orthogonal accepted/rejected: {sum(r.accepted for r in o_records)}/{sum(not r.accepted for r in o_records)}")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
