import networkx as nx
import time
import matplotlib.pyplot as plt


def calculate_moore_bound(k, g):
    if g % 2 == 1:
        return 1 + (k * ((k - 1) ** ((g - 1) // 2) - 1)) // (k - 2)
    else:
        return (2 * ((k - 1) ** (g // 2) - 1)) // (k - 2)


def get_best_edge(G, k, g):
    """
    Finds the 'smartest' edge to add using the Farthest-First heuristic.
    """
    # 1. Find vertices that still need edges, sorted by degree (lowest degree first)
    unsaturated = [v for v in G.nodes() if G.degree(v) < k]
    if not unsaturated:
        return None

    # Pick the vertex with the lowest degree to work on it first
    u = min(unsaturated, key=lambda v: G.degree(v))

    # Calculate distances from u to all other unsaturated vertices
    # nx.single_source_shortest_path_length returns a dict {target: distance}
    distances = nx.single_source_shortest_path_length(G, u)

    best_v = None
    max_dist = -1

    # 2. Look for the candidate v that is farthest from u
    for v in unsaturated:
        if u == v or G.has_edge(u, v):
            continue

        dist = distances.get(v, float('inf'))  # inf if no path (unlikely in Hamiltonian)

        # Girth check: dist must be >= g-1
        if dist >= g - 1:
            # Heuristic: Pick the one that is farthest away
            if dist > max_dist:
                max_dist = dist
                best_v = v

    if best_v is not None:
        return (u, best_v)
    return None


def farthest_first_hamiltonian_kg(k, g):
    n = calculate_moore_bound(k, g)
    if k % 2 == 1 and n % 2 == 1:
        n += 1

    while True:
        G = nx.cycle_graph(n)  # The Hamiltonian backbone

        while True:
            edge = get_best_edge(G, k, g)
            if edge:
                G.add_edge(*edge)
            else:
                break

        # Check if we reached k-regularity
        if all(d == k for _, d in G.degree()):
            return G
        else:
            n += 2 if k % 2 == 1 else 1


# --- Execution ---
if __name__ == "__main__":
    print("Farthest First Hamiltonian Algorithm")
    start = time.perf_counter()
    # FIXME set values
    k_val = 3
    g_val = 14
    graph = farthest_first_hamiltonian_kg(k_val, g_val)
    end = time.perf_counter()
    print(f"Time: {end - start:.6f} seconds")
    print(f"Vertices: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Optional: Draw the graph
    nx.draw_circular(graph, with_labels=True)
    plt.show()