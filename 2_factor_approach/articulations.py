import networkx as nx
import matplotlib.pyplot as plt
import collections


def get_girth(G):
    girth = float('inf')
    for node in G.nodes():
        distances = {node: 0}
        parent = {node: None}
        queue = collections.deque([node])
        while queue:
            u = queue.popleft()
            for v in G.neighbors(u):
                if v not in distances:
                    distances[v] = distances[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    girth = min(girth, distances[u] + distances[v] + 1)
                    if girth == 3: return 3
    return girth if girth != float('inf') else 0


def verify_hamiltonian(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    if n == 0: return False
    start_node = nodes[0]
    path = [start_node]
    visited = {start_node}

    def backtrack(curr):
        if len(path) == n:
            return G.has_edge(curr, start_node)
        for neighbor in G.neighbors(curr):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                if backtrack(neighbor): return True
                path.pop()
                visited.remove(neighbor)
        return False

    return backtrack(start_node)


def lift_to_hamiltonian_final(G):
    n = G.number_of_nodes()
    # Identifikácia artikulácií
    articulations = list(nx.articulation_points(G))

    # edge_types: 0 = parallel, 1 = cross
    edge_types = {tuple(sorted(e)): 0 for e in G.edges()}

    if articulations:
        # Pre každú artikuláciu vynútime nepárne kríženie do každého komponentu
        for art in articulations:
            # Komponenty po odstránení artikulácie
            G_temp = G.copy()
            G_temp.remove_node(art)
            components = list(nx.connected_components(G_temp))

            for comp in components:
                # Hrany z artikulácie do tohto komponentu
                edges_to_comp = [tuple(sorted((art, v))) for v in G.neighbors(art) if v in comp]
                if edges_to_comp:
                    # Nastavíme práve jednu hranu ako cross (nepárna parita)
                    edge_types[edges_to_comp[0]] = 1

    # Pre ostatné hrany (ktoré nie sú pri artikulácii) môžeme použiť náhodu
    # alebo ich nechať paralelné. Skúsime miernu náhodu pre zvýšenie šance.
    for edge, etype in edge_types.items():
        if etype == 0 and random.random() < 0.2:
            edge_types[edge] = 1

    H = nx.Graph()
    for (u, v), etype in edge_types.items():
        if etype == 0:
            H.add_edge(u, v);
            H.add_edge(u + n, v + n)
        else:
            H.add_edge(u, v + n);
            H.add_edge(v, u + n)
    return H


import random

# --- Váš graf (13 vrcholov, k=4) ---
adj_text = """
1: 2 3 4 5
2: 1 6 10 11
3: 1 7 10 11
4: 1 8 12 13
5: 1 9 12 13
6: 2 7 10 11
7: 3 6 10 11
8: 4 9 12 13
9: 5 8 12 13
10: 2 3 6 7
11: 2 3 6 7
12: 4 5 8 9
13: 4 5 8 9
"""


def parse(text):
    G = nx.Graph()
    for line in text.strip().splitlines():
        u_p, n_p = line.split(":")
        u = int(u_p)
        for v in n_p.split(): G.add_edge(u, int(v))
    return G


G_base = parse(adj_text)

# Skúsime niekoľko pokusov (kvôli náhodnosti v ne-artikulačných hranách)
for attempt in range(100):
    H = lift_to_hamiltonian_final(G_base)
    if verify_hamiltonian(H):
        print(f"Úspech na pokus č. {attempt + 1}!")
        print(f"Uzly: {H.number_of_nodes()} (2n)")
        print(f"Hamiltonovský: Áno")
        print(f"Girth: {get_girth(H)} (Pôvodný: {get_girth(G_base)})")
        nx.draw_circular(H, with_labels=True, node_color='orange')
        plt.show()
        break