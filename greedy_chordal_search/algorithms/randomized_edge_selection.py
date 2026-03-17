# 1. Compute the Moore bound:
#    minimum possible number of vertices n for degree k and girth g.

# 2. Fix parity if needed:
#    if k is odd, n must be even.

# 3. Build a Hamiltonian base:
#    start with a cycle 0-1-2-...-(n-1)-0,
#    so the graph is Hamiltonian from the start.

# 4. Track unsaturated vertices:
#    these are vertices with current degree < k.

# 5. Add random chords:
#    pick random pairs (u, v) from unsaturated vertices
#    to increase degrees and avoid repetitive patterns.

# 6. Check girth before adding an edge:
#    find the shortest path distance d between u and v.
#    adding (u, v) creates a cycle of length d + 1.
#    only add the edge if d >= g - 1.

# 7. Detect deadlocks:
#    if no valid edge can be found after many tries,
#    the current construction is considered stuck.

# 8. Increase n and restart:
#    if stuck, grow the graph size and try again.
#    larger n usually makes a valid (k, g)-graph easier to find.

# 9. Final verification:
#    when all vertices reach degree k, confirm the graph is k-regular.
#    since we started with a cycle and only added valid edges,
#    the result is a Hamiltonian (k, g)-graph.

import networkx as nx
import random
import time
import matplotlib.pyplot as plt

def calculate_moore_bound(k, g):
    """Calculates the lower bound for the number of vertices in a (k,g) graph."""
    if g % 2 == 1:
        return 1 + (k * ((k - 1) ** ((g - 1) // 2) - 1)) // (k - 2)
    else:
        return (2 * ((k - 1) ** (g // 2) - 1)) // (k - 2)

def randomized_edge_selection_hamiltonian_kg(k, g, max_attempts=100):
    if k < 2:
        raise ValueError("k must be at least 2")
    if g < 3:
        raise ValueError("girth must be at least 3")

    n = calculate_moore_bound(k, g)

    # If k is odd, n must be even to satisfy the Handshaking Lemma
    if k % 2 == 1 and n % 2 == 1:
        n += 1

    while True:
        # 1. Create a cycle (Hamiltonian backbone)
        G = nx.cycle_graph(n)

        # 2. Greedy edge addition
        attempts = 0
        while attempts < max_attempts:
            # Find vertices that still need edges
            unsaturated = [v for v in G.nodes() if G.degree(v) < k]

            if not unsaturated:
                break

            random.shuffle(unsaturated)
            found_edge = False

            for i in range(len(unsaturated)):
                for j in range(i + 1, len(unsaturated)):
                    u, v = unsaturated[i], unsaturated[j]

                    if G.has_edge(u, v):
                        continue

                    # 3. Girth Check: Shortest path must be >= g-1
                    # so that adding (u,v) makes a cycle of length >= g
                    try:
                        path_len = nx.shortest_path_length(G, source=u, target=v)
                    except nx.NetworkXNoPath:
                        path_len = float('inf')

                    if path_len >= g - 1:
                        G.add_edge(u, v)
                        found_edge = True
                        break
                if found_edge: break

            if not found_edge:
                attempts += 1

        # Check if we successfully made it k-regular
        degrees = [d for n, d in G.degree()]
        if all(d == k for d in degrees):
            return G
        else:
            # If we failed, increment n and try again
            if k % 2 == 1:
                n += 2
            else:
                n += 1

# --- Usage ---
if __name__ == "__main__":
    print("Randomized Edge Selection Algorithm")
    start = time.perf_counter()
    # FIXME set values
    k_val = 3
    g_val = 10
    graph = randomized_edge_selection_hamiltonian_kg(k_val, g_val)
    end = time.perf_counter()
    print(f"Time: {end - start:.6f} seconds")
    print(f"Vertices: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Optional: Draw the graph
    nx.draw_circular(graph, with_labels=True)
    plt.show()