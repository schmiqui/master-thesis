import networkx as nx

def random_sparse_regular_graph_with_report(n, tries_per_k=20, max_k=8, seed=None):
    """
    Prefer smaller regularity and larger girth.

    Strategy:
    - try smaller k first
    - for each feasible k, generate several random k-regular graphs
    - keep the graph with the largest girth
    - stop as soon as we find a decent graph for the smallest possible k

    Returns:
        best_G
    """
    if n < 4:
        raise ValueError("n must be at least 4")

    best_G = None
    best_k = None
    best_g = -1

    current_seed = seed if isinstance(seed, int) else None

    # Prefer smaller regularity
    for k in range(3, min(max_k, n - 1) + 1):
        if (n * k) % 2 != 0:
            continue

        local_best_G = None
        local_best_g = -1

        for _ in range(tries_per_k):
            G = nx.random_regular_graph(k, n, seed=current_seed)
            g = nx.girth(G)

            if g > local_best_g:
                local_best_G = G
                local_best_g = g

            if current_seed is not None:
                current_seed += 1

        # Since we prefer smaller k, return immediately when we find
        # the best sample for the first feasible small k
        if local_best_G is not None:
            print("regularity =", k)
            print("girth      =", local_best_g)
            return local_best_G

        # fallback global best
        if local_best_g > best_g:
            best_G = local_best_G
            best_k = k
            best_g = local_best_g

    if best_G is None:
        raise ValueError("Could not generate a feasible regular graph")

    print("regularity =", best_k)
    print("girth      =", best_g)
    return best_G