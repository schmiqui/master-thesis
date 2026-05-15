import csv

import matplotlib.pyplot as plt
import networkx as nx

from lift_support import (
    double_cover_from_cross_mask,
    has_hamiltonian_cycle,
    normalized_edge,
    parse_adjacency_list_text,
    relabel_sorted_integers,
)

HAMILTONICITY_VERTEX_CUTOFF = 24

DEMO_ADJACENCY_TEXT = """
1: 2 3 4
2: 1 11 14
3: 1 12 16
4: 1 13 15
5: 8 9 10
6: 7 9 10
7: 6 8 12
8: 5 7 13
9: 5 6 11
10: 5 6 11
11: 2 9 10
12: 3 7 15
13: 4 8 16
14: 2 15 16
15: 4 12 14
16: 3 13 14
"""


def two_factor_edges_complement_of_max_matching(graph):
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    tf = nx.Graph()
    tf.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        if (u, v) not in matching and (v, u) not in matching:
            tf.add_edge(u, v)
    return tf, matching


def two_factor_edges_complement_of_perfect_matching_or_none(graph):
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    if len(matching) * 2 != graph.number_of_nodes():
        return None, None
    tf = nx.Graph()
    tf.add_nodes_from(graph.nodes())
    matching_edges = {normalized_edge(u, v) for u, v in matching}
    for u, v in graph.edges():
        if normalized_edge(u, v) not in matching_edges:
            tf.add_edge(u, v)
    return tf, matching


def relabel_in_list_order(graph, node_order):
    mapping = {node: i for i, node in enumerate(node_order)}
    return nx.relabel_nodes(graph, mapping), len(node_order)


def lift_double_cover_max_matching_mst_arbitrary_node_order(graph):
    mapped, n = relabel_in_list_order(graph, list(graph.nodes()))
    tf, matching = two_factor_edges_complement_of_max_matching(mapped)
    cycles = [list(c) for c in nx.connected_components(tf)]
    lifted = {normalized_edge(u, v): False for u, v in mapped.edges()}
    for cycle in cycles:
        u = cycle[0]
        v = next(iter(tf.neighbors(u)))
        lifted[normalized_edge(u, v)] = True
    if len(cycles) > 1:
        meta = nx.Graph()
        for u, v in matching:
            cu = next(i for i, c in enumerate(cycles) if u in c)
            cv = next(i for i, c in enumerate(cycles) if v in c)
            if cu != cv:
                meta.add_edge(cu, cv, edge=(u, v))
        tree = nx.minimum_spanning_tree(meta)
        for _, _, data in tree.edges(data=True):
            ug, vg = data["edge"]
            lifted[normalized_edge(ug, vg)] = True
    return double_cover_from_cross_mask(n, lifted)


def lift_double_cover_max_matching_mst_sorted_labels(graph):
    mapped, n = relabel_sorted_integers(graph)
    tf, matching = two_factor_edges_complement_of_max_matching(mapped)
    cycles = [sorted(c) for c in nx.connected_components(tf)]
    cross = {normalized_edge(u, v): False for u, v in mapped.edges()}
    for cycle in cycles:
        u = cycle[0]
        v = next(iter(tf.neighbors(u)))
        cross[normalized_edge(u, v)] = True
    if len(cycles) > 1:
        meta = nx.Graph()
        for u, v in matching:
            cu = next(i for i, c in enumerate(cycles) if u in c)
            cv = next(i for i, c in enumerate(cycles) if v in c)
            if cu != cv:
                meta.add_edge(cu, cv, base_edge=(u, v))
        tree = nx.minimum_spanning_tree(meta)
        for _, _, data in tree.edges(data=True):
            cross[normalized_edge(*data["base_edge"])] = True
    return double_cover_from_cross_mask(n, cross)


def lift_double_cover_matching_glue_first_cycle_then_matching(graph):
    mapped, n = relabel_sorted_integers(graph)
    tf, matching_edges = two_factor_edges_complement_of_max_matching(mapped)
    cycle_comps = [list(c) for c in nx.connected_components(tf)]
    node_to_cycle = {node: i for i, cycle in enumerate(cycle_comps) for node in cycle}
    cross = {normalized_edge(u, v): False for u, v in mapped.edges()}
    first = cycle_comps[0]
    u0, v0 = first[0], next(iter(tf.neighbors(first[0])))
    cross[normalized_edge(u0, v0)] = True
    connected = {0}
    remaining = set(range(1, len(cycle_comps)))
    while remaining:
        found = False
        for u, v in matching_edges:
            c1, c2 = node_to_cycle[u], node_to_cycle[v]
            if (c1 in connected and c2 in remaining) or (c2 in connected and c1 in remaining):
                cross[normalized_edge(u, v)] = True
                new_cycle = c2 if c1 in connected else c1
                connected.add(new_cycle)
                remaining.remove(new_cycle)
                found = True
                break
        if not found:
            break
    return double_cover_from_cross_mask(n, cross)


def lift_double_cover_algebraic_matching_tree(graph):
    n = graph.number_of_nodes()
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    matching_edges = {normalized_edge(u, v) for u, v in matching}
    tf = nx.Graph()
    for u, v in graph.edges():
        if normalized_edge(u, v) not in matching_edges:
            tf.add_edge(u, v)
    cycles = [list(c) for c in nx.connected_components(tf)]
    node_to_cycle = {node: i for i, c in enumerate(cycles) for node in c}
    cross = {normalized_edge(u, v): False for u, v in graph.edges()}
    for cycle in cycles:
        u, v = cycle[0], next(iter(tf.neighbors(cycle[0])))
        cross[normalized_edge(u, v)] = True
    meta = nx.Graph()
    for u, v in matching_edges:
        c1, c2 = node_to_cycle[u], node_to_cycle[v]
        if c1 != c2:
            meta.add_edge(c1, c2, edge=(u, v))
    tree = nx.minimum_spanning_tree(meta)
    for _, _, data in tree.edges(data=True):
        u_g, v_g = data["edge"]
        cross[normalized_edge(u_g, v_g)] = True
    return double_cover_from_cross_mask(n, cross)


def lift_double_cover_matching_mst_with_connectivity_repair(graph):
    n = graph.number_of_nodes()
    mapped, _ = relabel_sorted_integers(graph)
    tf, matching = two_factor_edges_complement_of_perfect_matching_or_none(mapped)
    if tf is None:
        return None
    cycles = [list(c) for c in nx.connected_components(tf)]
    node_to_cycle = {node: i for i, c in enumerate(cycles) for node in c}
    cross = {normalized_edge(u, v): False for u, v in mapped.edges()}
    u0, v0 = cycles[0][0], next(iter(tf.neighbors(cycles[0][0])))
    cross[normalized_edge(u0, v0)] = True
    if len(cycles) > 1:
        meta = nx.Graph()
        for u, v in matching:
            c1, c2 = node_to_cycle[u], node_to_cycle[v]
            if c1 != c2:
                meta.add_edge(c1, c2, edge=(u, v))
        tree = nx.minimum_spanning_tree(meta)
        for _, _, data in tree.edges(data=True):
            cross[normalized_edge(*data["edge"])] = True

    def build_from_cross(edge_cross):
        return double_cover_from_cross_mask(
            n, {e: bool(v) for e, v in edge_cross.items()}
        )

    lifted = build_from_cross(cross)
    if not nx.is_connected(lifted):
        components = list(nx.connected_components(lifted))
        for u, v in matching:
            if (u in components[0] and v in components[1]) or (
                v in components[0] and u in components[1]
            ):
                key = normalized_edge(u, v)
                cross[key] = not cross[key]
                lifted = build_from_cross(cross)
                break
    return lifted


def csv_row_for_lift_experiment(index, G_base, H_result, hamiltonian_cutoff=HAMILTONICITY_VERTEX_CUTOFF):
    n = G_base.number_of_nodes()
    m = G_base.number_of_edges()
    k = int(2 * m / n)
    g = nx.girth(G_base)
    h_girth = nx.girth(H_result)
    articulation_count = len(list(nx.articulation_points(G_base)))
    hamiltonian = (
        has_hamiltonian_cycle(H_result) if n <= hamiltonian_cutoff else None
    )
    return {
        "id": index,
        "k": k,
        "original_nodes": n,
        "original_edges": m,
        "original_girth": g,
        "original_vertex_connectivity": nx.node_connectivity(G_base),
        "original_edge_connectivity": nx.edge_connectivity(G_base),
        "result_nodes": H_result.number_of_nodes(),
        "result_girth": h_girth,
        "result_vertex_connectivity": nx.node_connectivity(H_result),
        "result_edge_connectivity": nx.edge_connectivity(H_result),
        "has_articulation": articulation_count > 0,
        "girth_preserved": h_girth == g,
        "is_k_regular": all(deg == k for _, deg in H_result.degree()),
        "hamiltonian": hamiltonian,
        "original_graph6": nx.to_graph6_bytes(G_base, header=False).decode().strip(),
        "result_graph6": nx.to_graph6_bytes(H_result, header=False).decode().strip(),
    }


def write_lift_experiment_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_lift_experiment_csv(graphs, path="tested.csv", lift=lift_double_cover_max_matching_mst_arbitrary_node_order):
    rows = []
    for i, G_base in enumerate(graphs):
        k = int(2 * G_base.number_of_edges() / G_base.number_of_nodes())
        print(f"{i} k={k}, g={nx.girth(G_base)}")
        H_result = lift(G_base)
        rows.append(csv_row_for_lift_experiment(i, G_base, H_result))
    write_lift_experiment_csv(rows, path)
    return rows


def draw_double_cover_layers(graph, original_n=None):
    if original_n is None:
        original_n = graph.number_of_nodes() // 2
    colors = ["lightblue" if node < original_n else "orange" for node in graph.nodes()]
    edge_colors = []
    for u, v in graph.edges():
        same_layer = (u < original_n and v < original_n) or (u >= original_n and v >= original_n)
        edge_colors.append("gray" if same_layer else "red")
    pos = nx.spring_layout(graph, seed=42)
    for node in graph.nodes():
        if node >= original_n:
            pos[node][0] += 2.0
    plt.figure(figsize=(10, 6))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=colors,
        edge_color=edge_colors,
        node_size=300,
        font_size=8,
    )
    plt.show()


def main():
    G_base = parse_adjacency_list_text(DEMO_ADJACENCY_TEXT)
    k = 2 * G_base.number_of_edges() / G_base.number_of_nodes()
    H_result = lift_double_cover_algebraic_matching_tree(G_base)
    print(f"Original Nodes: {G_base.number_of_nodes()}")
    print(f"Original Girth: {nx.girth(G_base)}")
    print(f"Result Nodes:   {H_result.number_of_nodes()}")
    print(f"Result Girth:   {nx.girth(H_result)}")
    print("=" * 30)
    print("STABILITY")
    print(f"Girth:          {nx.girth(H_result) == nx.girth(G_base)}")
    print(f"Regularity:     {all(deg == k for _, deg in H_result.degree())}")
    print(f"Hamiltonian:    {has_hamiltonian_cycle(H_result)}")
    print("=" * 30)
    nx.draw(G_base, with_labels=True, node_color="orange")
    plt.show()


# https://houseofgraphs.org/graphs/981
if __name__ == "__main__":
    main()
