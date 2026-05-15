import itertools

import matplotlib.pyplot as plt
import networkx as nx

from lift_support import (
    double_cover_from_cross_mask,
    has_hamiltonian_cycle,
    normalized_edge,
    parse_graph_text,
    relabel_sorted_integers,
)
from sample_graph6 import SAMPLE_GRAPH6_STRINGS


def two_factor_via_hopcroft_karp(graph):
    nodes = sorted(graph.nodes())
    bipartite = nx.Graph()
    for u in nodes:
        bipartite.add_node(f"{u}in", bipartite=0)
        bipartite.add_node(f"{u}out", bipartite=1)
    for u, v in graph.edges():
        bipartite.add_edge(f"{u}in", f"{v}out")
        bipartite.add_edge(f"{v}in", f"{u}out")
    matching = nx.bipartite.matching.hopcroft_karp_matching(
        bipartite, top_nodes=[f"{u}in" for u in nodes]
    )
    if len(matching) < 2 * len(nodes):
        return None
    tf = nx.Graph()
    tf.add_nodes_from(nodes)
    for u_in, v_out in matching.items():
        if not u_in.endswith("in"):
            continue
        u_raw, v_raw = u_in[:-2], v_out[:-3]
        try:
            tf.add_edge(int(u_raw), int(v_raw))
        except ValueError:
            tf.add_edge(u_raw, v_raw)
    return tf


def meta_graph_with_parallel_edges(tf, mapped, node_to_cycle):
    remaining = [e for e in mapped.edges() if not tf.has_edge(*e)]
    meta = nx.Graph()
    for u, v in remaining:
        c1, c2 = node_to_cycle[u], node_to_cycle[v]
        if c1 == c2:
            continue
        if not meta.has_edge(c1, c2):
            meta.add_edge(c1, c2, possible_edges=[])
        meta[c1][c2]["possible_edges"].append(normalized_edge(u, v))
    return meta


def try_hamiltonian_lift_by_enumerating_meta_trees(graph):
    n_count = graph.number_of_nodes()
    mapped, _ = relabel_sorted_integers(graph)
    tf = two_factor_via_hopcroft_karp(mapped)
    if tf is None:
        return None
    cycles = sorted([sorted(c) for c in nx.connected_components(tf)])
    node_to_cycle = {node: i for i, c in enumerate(cycles) for node in c}
    base_cross = {normalized_edge(u, v): False for u, v in mapped.edges()}
    for cycle in cycles:
        u = cycle[0]
        v = sorted(tf.neighbors(u))[0]
        base_cross[normalized_edge(u, v)] = True
    if len(cycles) == 1:
        candidate = double_cover_from_cross_mask(n_count, base_cross)
        return candidate if has_hamiltonian_cycle(candidate) else None
    meta = meta_graph_with_parallel_edges(tf, mapped, node_to_cycle)
    if not nx.is_connected(meta):
        print("Metagraph is not connected; cannot join all cycles.")
        return None
    all_meta_edges = []
    for u_m, v_m, data in meta.edges(data=True):
        for real_edge in data["possible_edges"]:
            all_meta_edges.append({"meta": (u_m, v_m), "real": real_edge})
    num_cycles = len(cycles)
    print(f"The chosen 2-factor contains {num_cycles} disjoint cycles.")
    tried = 0
    for combo in itertools.combinations(all_meta_edges, num_cycles - 1):
        test_meta = nx.Graph()
        for item in combo:
            test_meta.add_edge(*item["meta"])
        if test_meta.number_of_nodes() != num_cycles or not nx.is_connected(test_meta):
            continue
        tried += 1
        current = dict(base_cross)
        for item in combo:
            current[item["real"]] = True
        candidate = double_cover_from_cross_mask(n_count, current)
        if has_hamiltonian_cycle(candidate):
            print(f"Spanning tree tried {tried}")
            return candidate
        if tried % 100 == 0:
            print(f"{tried} spanning trees tried...")
    print("Exhausted spanning trees; no Hamiltonian lift found.")
    return None


def save_result_plot(graph, graph_id, g6_label):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="orange",
        edge_color="gray",
        node_size=500,
        font_size=9,
    )
    plt.title(f"Graph ID: {graph_id}\nGraph6: {g6_label}", fontsize=10)
    plt.savefig(f"{graph_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    save_plot = True
    for index, g6 in enumerate(SAMPLE_GRAPH6_STRINGS, start=1):
        G_base = parse_graph_text(g6)
        print(f"GRAPH ID = {index}")
        print(f"Graph6: {g6}")
        print(f"Graph: {G_base}")
        if not G_base:
            continue
        H_result = try_hamiltonian_lift_by_enumerating_meta_trees(G_base)
        if not H_result:
            continue
        print(f"Original Nodes: {G_base.number_of_nodes()}")
        print(f"Original Girth: {nx.girth(G_base)}")
        print(f"Result Nodes:   {H_result.number_of_nodes()}")
        print(f"Result Girth:   {nx.girth(H_result)}")
        print(f"Hamiltonian:    {has_hamiltonian_cycle(H_result)}")
        print("-" * 30)
        print(H_result.edges())
        if save_plot:
            save_result_plot(H_result, index, g6)


if __name__ == "__main__":
    main()
