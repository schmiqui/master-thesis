import networkx as nx


def normalized_edge(u, v):
    return tuple(sorted((u, v)))


def double_cover_from_cross_mask(n, edge_is_cross):
    H = nx.Graph()
    for (u, v), is_cross in edge_is_cross.items():
        if is_cross:
            H.add_edge(u, v + n)
            H.add_edge(v, u + n)
        else:
            H.add_edge(u, v)
            H.add_edge(u + n, v + n)
    return H


def has_hamiltonian_cycle(graph):
    n = graph.number_of_nodes()
    if n == 0:
        return False
    nodes = list(graph.nodes())
    start = nodes[0]
    path = [start]
    visited = {start}

    def extend(curr):
        if len(path) == n:
            return graph.has_edge(curr, start)
        for nbr in graph.neighbors(curr):
            if nbr not in visited:
                visited.add(nbr)
                path.append(nbr)
                if extend(nbr):
                    return True
                path.pop()
                visited.remove(nbr)
        return False

    return extend(start)


def relabel_sorted_integers(graph):
    nodes_sorted = sorted(graph.nodes())
    n = len(nodes_sorted)
    mapping = {node: i for i, node in enumerate(nodes_sorted)}
    return nx.relabel_nodes(graph, mapping), n


def _first_line(text):
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split("\n", 1)[0].strip()


def _looks_like_graph6(text):
    first = _first_line(text)
    if not first:
        return False
    if first.startswith(":"):
        return True
    body = text.strip()
    if "\n" not in body and ":" not in body and len(body) > 2:
        return True
    if len(first) > 2 and not any(ch.isdigit() for ch in first):
        return True
    if len(first) > 2 and first[0] in "QhG":
        return True
    return False


def parse_graph_text(text):
    data = text.strip()
    if not data:
        return nx.Graph()
    if _looks_like_graph6(data):
        try:
            return nx.from_graph6_bytes(data.encode())
        except (ValueError, UnicodeEncodeError, OSError):
            pass
    graph = nx.Graph()
    for line in data.splitlines():
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        u = int(left.strip())
        for token in right.strip().split():
            graph.add_edge(u, int(token))
    return graph


def parse_adjacency_list_text(text):
    return parse_graph_text(text)


def read_adjacency_blocks(path):
    with open(path, encoding="utf-8") as handle:
        content = handle.read().strip()
    graphs = []
    for block in content.split("\n\n"):
        if block.strip():
            graphs.append(parse_adjacency_list_text(block))
    return graphs
