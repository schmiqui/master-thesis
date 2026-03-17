import networkx as nx
import time
import matplotlib.pyplot as plt


def get_legal_neighbors(G, u, g, unsaturated):
    """Returns a list of vertices from 'unsaturated' that u can legally connect to."""
    lengths = nx.single_source_shortest_path_length(G, u)
    legal = []
    for v in unsaturated:
        if u == v or G.has_edge(u, v):
            continue
        # Girth check: dist must be >= g-1
        if lengths.get(v, float('inf')) >= g - 1:
            legal.append((v, lengths.get(v, float('inf'))))
    return legal


def get_best_edge_freedom_heuristic(G, k, g):
    unsaturated = [v for v in G.nodes() if G.degree(v) < k]
    if not unsaturated:
        return None

    # Map each unsaturated vertex to its list of legal candidates
    freedom_map = {}
    for u in unsaturated:
        freedom_map[u] = get_legal_neighbors(G, u, g, unsaturated)

    # 1. MOST CONSTRAINED: Pick u with the fewest legal options
    # Filter out vertices that have 0 options (they are already dead ends)
    valid_u = [u for u in unsaturated if len(freedom_map[u]) > 0]

    if not valid_u:
        return None  # Stuck!

    # Priority: 1. Fewest options, 2. Lowest current degree
    u = min(valid_u, key=lambda x: (len(freedom_map[x]), G.degree(x)))

    # 2. LEAST CONSTRAINING: From u's options, pick the one farthest away
    # This keeps the graph as "open" as possible
    best_v, dist = max(freedom_map[u], key=lambda item: item[1])

    return (u, best_v)


def freedom_based_greedy_hamiltonian_kg(k, g):
    n = 1 + (k * ((k - 1) ** ((g - 1) // 2) - 1)) // (k - 2) if g % 2 == 1 else (2 * ((k - 1) ** (g // 2) - 1)) // (
                k - 2)
    if k % 2 == 1 and n % 2 == 1: n += 1

    while True:
        print(f'n = ${n}')
        G = nx.cycle_graph(n)  # Hamiltonian backbone

        while True:
            edge = get_best_edge_freedom_heuristic(G, k, g)
            if edge:
                G.add_edge(*edge)
            else:
                break

        # Check if successful
        if all(d == k for _, d in G.degree()):
            return G
        else:
            n += 2 if k % 2 == 1 else 1


# --- Run ---

if __name__ == "__main__":
    print("Freedom based Greedy Algorithm")
    start = time.perf_counter()
    # FIXME set values
    k_val = 3
    g_val = 14
    graph = freedom_based_greedy_hamiltonian_kg(k_val, g_val)
    end = time.perf_counter()
    print(f"Time: {end - start:.6f} seconds")
    print(f"Vertices: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Optional: Draw the graph
    nx.draw_circular(graph, with_labels=True)
    plt.show()

