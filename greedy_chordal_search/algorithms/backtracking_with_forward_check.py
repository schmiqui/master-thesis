import networkx as nx
import time
import matplotlib.pyplot as plt


def get_legal_options(G, u, k, g, unsaturated):
    """Finds all vertices v that can legally connect to u."""
    options = []
    # Pre-calculate distances from u
    dist = nx.single_source_shortest_path_length(G, u)

    for v in unsaturated:
        if v == u or G.has_edge(u, v) or G.degree(v) >= k:
            continue
        # Girth constraint: adding (u,v) must not create cycle < g
        if dist.get(v, float('inf')) >= g - 1:
            options.append((v, dist.get(v, float('inf'))))
    return options


def forward_check(G, k, g, unsaturated):
    """
    Looks ahead: For every unsaturated vertex, does it still have
    enough legal options to reach degree k?
    """
    for v in unsaturated:
        needed = k - G.degree(v)
        if needed <= 0: continue

        # Count how many legal partners are left for v
        options = get_legal_options(G, v, k, g, unsaturated)
        if len(options) < needed:
            return False  # Future failure detected!
    return True


def solve_kg(G, k, g):
    """Recursive backtracking function."""
    unsaturated = [v for v in G.nodes() if G.degree(v) < k]

    if not unsaturated:
        return G  # Success! Every vertex has degree k

    # 1. Most Constrained Variable: Pick u with fewest legal options
    # This prioritizes the "hardest" part of the graph
    options_map = {v: get_legal_options(G, v, k, g, unsaturated) for v in unsaturated}
    u = min(unsaturated, key=lambda v: (len(options_map[v]), -G.degree(v)))

    # 2. Try candidates for u (Least Constraining Value: farthest first)
    candidates = sorted(options_map[u], key=lambda x: x[1], reverse=True)

    for v, d in candidates:
        G.add_edge(u, v)

        # 3. Forward Checking
        if forward_check(G, k, g, unsaturated):
            result = solve_kg(G, k, g)
            if result is not None:
                return result

        # Backtrack
        G.remove_edge(u, v)

    return None


def backtracking_with_forward_check_hamiltonian_kg(k, g):
    # Start with Moore Bound
    n = 1 + (k * ((k - 1) ** ((g - 1) // 2) - 1)) // (k - 2) if g % 2 == 1 else (2 * ((k - 1) ** (g // 2) - 1)) // (
                k - 2)
    if k % 2 == 1 and n % 2 == 1: n += 1

    while True:
        print(n)
        G = nx.cycle_graph(n)  # Hamiltonian backbone

        final_graph = solve_kg(G, k, g)
        if final_graph:
            return final_graph

        n += 2 if k % 2 == 1 else 1


# Example Usage
if __name__ == '__main__':
    print("Backtracking with forward check Algorithm")
    start = time.perf_counter()
    # FIXME set values
    k_val = 3
    g_val = 9
    graph = backtracking_with_forward_check_hamiltonian_kg(k_val, g_val)
    end = time.perf_counter()
    print(f"Time: {end - start:.6f} seconds")
    print(f"Vertices: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Optional: Draw the graph
    nx.draw_circular(graph, with_labels=True)
    plt.show()
