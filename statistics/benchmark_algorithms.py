"""
Benchmark script for greedy chordal search algorithms.

Runs all four algorithms across a grid of (k, g) parameters,
measures execution time and resulting graph size, and enforces
a per-run timeout to skip configurations that take too long.
"""

import os
import sys
import time
import csv
import multiprocessing as mp
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from greedy_chordal_search.algorithms.backtracking_with_forward_check import (
    backtracking_with_forward_check_hamiltonian_kg,
)
from greedy_chordal_search.algorithms.farthest_first_heuristic import (
    farthest_first_hamiltonian_kg,
)
from greedy_chordal_search.algorithms.freedom_based_greedy import (
    freedom_based_greedy_hamiltonian_kg,
)
from greedy_chordal_search.algorithms.randomized_edge_selection import (
    randomized_edge_selection_hamiltonian_kg,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALGORITHMS = {
    "farthest_first": farthest_first_hamiltonian_kg,
    "freedom_based_greedy": freedom_based_greedy_hamiltonian_kg,
    "randomized_edge_selection": randomized_edge_selection_hamiltonian_kg,
}

# ── Parameters to sweep ─────────────────────────────────────────────
K_VALUES = [3, 4]
G_VALUES = [5, 6, 7, 8]
TIMEOUT_SECONDS = 60 * 10


def _run_algorithm(conn, algo_func, k, g):
    """Worker target: run one algorithm and send results back via a Pipe."""
    try:
        start = time.perf_counter()
        graph = algo_func(k, g)
        elapsed = time.perf_counter() - start
        conn.send({
            "vertices": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "time": elapsed,
        })
    except Exception as e:
        conn.send({"error": str(e)})
    finally:
        conn.close()


def run_with_timeout(algo_func, k, g, timeout):
    """Run algo_func(k, g) in a subprocess; kill it if it exceeds *timeout* seconds."""
    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(target=_run_algorithm, args=(child_conn, algo_func, k, g))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {"timeout": True, "time": timeout}

    if parent_conn.poll():
        return parent_conn.recv()

    return {"error": "No result received from worker"}


def moore_bound(k, g):
    if g % 2 == 1:
        return 1 + (k * ((k - 1) ** ((g - 1) // 2) - 1)) // (k - 2)
    else:
        return (2 * ((k - 1) ** (g // 2) - 1)) // (k - 2)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SCRIPT_DIR, f"benchmark_results_{timestamp}.csv")

    fieldnames = [
        "algorithm", "k", "g", "moore_bound",
        "vertices", "edges", "time_seconds", "status",
    ]

    results = []

    header = f"{'Algorithm':<30} {'k':>3} {'g':>3} {'Moore':>7} {'|':>1} {'Vertices':>10} {'Edges':>8} {'Time (s)':>12} {'Status':<10}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for k in K_VALUES:
        for g in G_VALUES:
            mb = moore_bound(k, g)
            for algo_name, algo_func in ALGORITHMS.items():
                sys.stdout.flush()

                result = run_with_timeout(algo_func, k, g, TIMEOUT_SECONDS)

                row = {
                    "algorithm": algo_name,
                    "k": k,
                    "g": g,
                    "moore_bound": mb,
                }

                if result.get("timeout"):
                    row.update(vertices="—", edges="—",
                               time_seconds=f">{TIMEOUT_SECONDS}", status="TIMEOUT")
                elif result.get("error"):
                    row.update(vertices="—", edges="—",
                               time_seconds="—", status=f"ERROR: {result['error']}")
                else:
                    row.update(
                        vertices=result["vertices"],
                        edges=result["edges"],
                        time_seconds=f"{result['time']:.4f}",
                        status="OK",
                    )

                results.append(row)

                print(
                    f"{row['algorithm']:<30} {row['k']:>3} {row['g']:>3} {row['moore_bound']:>7} | "
                    f"{str(row['vertices']):>10} {str(row['edges']):>8} {str(row['time_seconds']):>12} {row['status']:<10}"
                )

    print(sep)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_path}")

    plot_results(results, timestamp)


def plot_results(results, timestamp):
    """Generate comparison plots from benchmark results."""
    ok_results = [r for r in results if r["status"] == "OK"]
    if not ok_results:
        print("No successful runs to plot.")
        return

    algo_names = list(ALGORITHMS.keys())
    k_values = sorted({r["k"] for r in ok_results})

    MARKERS = ["o", "s", "D", "^", "v", "P", "*"]
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    style_map = {}
    for i, name in enumerate(algo_names):
        style_map[name] = {
            "color": COLORS[i % len(COLORS)],
            "marker": MARKERS[i % len(MARKERS)],
        }

    # ── Plot 1: Execution time vs g (one subplot per k) ─────────────
    fig1, axes1 = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5),
                                squeeze=False)
    fig1.suptitle("Execution Time vs Girth", fontsize=14, fontweight="bold")

    for col, k in enumerate(k_values):
        ax = axes1[0][col]
        for algo in algo_names:
            pts = sorted(
                [(r["g"], float(r["time_seconds"]))
                 for r in ok_results if r["algorithm"] == algo and r["k"] == k],
                key=lambda p: p[0],
            )
            if pts:
                gs, ts = zip(*pts)
                ax.plot(gs, ts, label=algo,
                        color=style_map[algo]["color"],
                        marker=style_map[algo]["marker"],
                        linewidth=1.5, markersize=6)

        ax.set_title(f"k = {k}")
        ax.set_xlabel("Girth (g)")
        ax.set_ylabel("Time (seconds)")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig1.tight_layout()
    path1 = os.path.join(SCRIPT_DIR, f"plot_time_vs_girth_{timestamp}.png")
    fig1.savefig(path1, dpi=150)
    print(f"Saved: {path1}")

    # ── Plot 2: Number of vertices vs g (one subplot per k) ─────────
    fig2, axes2 = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5),
                                squeeze=False)
    fig2.suptitle("Graph Size (Vertices) vs Girth", fontsize=14, fontweight="bold")

    for col, k in enumerate(k_values):
        ax = axes2[0][col]

        all_gs = sorted({r["g"] for r in ok_results if r["k"] == k})
        if all_gs:
            mb_gs = all_gs
            mb_vals = [moore_bound(k, g) for g in mb_gs]
            ax.plot(mb_gs, mb_vals, label="Moore bound", color="gray",
                    linestyle="--", linewidth=1.5, marker="x", markersize=5)

        for algo in algo_names:
            pts = sorted(
                [(r["g"], int(r["vertices"]))
                 for r in ok_results if r["algorithm"] == algo and r["k"] == k],
                key=lambda p: p[0],
            )
            if pts:
                gs, vs = zip(*pts)
                ax.plot(gs, vs, label=algo,
                        color=style_map[algo]["color"],
                        marker=style_map[algo]["marker"],
                        linewidth=1.5, markersize=6)

        ax.set_title(f"k = {k}")
        ax.set_xlabel("Girth (g)")
        ax.set_ylabel("Vertices")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig2.tight_layout()
    path2 = os.path.join(SCRIPT_DIR, f"plot_vertices_vs_girth_{timestamp}.png")
    fig2.savefig(path2, dpi=150)
    print(f"Saved: {path2}")

    # ── Plot 3: Vertex ratio (actual / Moore bound) vs g ────────────
    fig3, axes3 = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5),
                                squeeze=False)
    fig3.suptitle("Vertex Overhead (Actual / Moore Bound) vs Girth",
                  fontsize=14, fontweight="bold")

    for col, k in enumerate(k_values):
        ax = axes3[0][col]
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Moore bound")

        for algo in algo_names:
            pts = sorted(
                [(r["g"], int(r["vertices"]) / r["moore_bound"])
                 for r in ok_results if r["algorithm"] == algo and r["k"] == k],
                key=lambda p: p[0],
            )
            if pts:
                gs, ratios = zip(*pts)
                ax.plot(gs, ratios, label=algo,
                        color=style_map[algo]["color"],
                        marker=style_map[algo]["marker"],
                        linewidth=1.5, markersize=6)

        ax.set_title(f"k = {k}")
        ax.set_xlabel("Girth (g)")
        ax.set_ylabel("Vertices / Moore Bound")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig3.tight_layout()
    path3 = os.path.join(SCRIPT_DIR, f"plot_vertex_ratio_{timestamp}.png")
    fig3.savefig(path3, dpi=150)
    print(f"Saved: {path3}")

    plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
