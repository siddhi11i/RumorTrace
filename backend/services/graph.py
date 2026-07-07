"""Synthetic rumor diffusion graph generation with NetworkX + Louvain."""

import hashlib
import random
from typing import Any

import networkx as nx
from networkx.algorithms import community

PLATFORM_TYPES = ["whatsapp", "twitter", "facebook", "telegram", "instagram"]
COMMUNITY_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generate_diffusion_graph(text: str, misinfo_score: float, num_nodes: int = 45) -> nx.Graph:
    rng = random.Random(_seed_from_text(text))
    n = max(20, min(num_nodes, 80))
    spread_factor = 0.3 + misinfo_score * 0.7

    G = nx.barabasi_albert_graph(n, max(2, int(3 * spread_factor)), seed=rng.randint(0, 2**31))

    for node in G.nodes():
        platform = rng.choice(PLATFORM_TYPES)
        followers = int(rng.lognormvariate(3.5, 1.2) * (1 + misinfo_score))
        influence = rng.betavariate(2 + misinfo_score * 3, 5 - misinfo_score * 2)
        G.nodes[node]["label"] = f"User_{node}"
        G.nodes[node]["platform"] = platform
        G.nodes[node]["followers"] = followers
        G.nodes[node]["influence"] = round(influence, 3)
        G.nodes[node]["is_bot"] = rng.random() < 0.08 * spread_factor

    for u, v in G.edges():
        weight = rng.uniform(0.3, 1.0) * spread_factor
        G.edges[u, v]["weight"] = round(weight, 3)
        G.edges[u, v]["shares"] = int(rng.paretovariate(1.5) * 10 * spread_factor)

    return G


def detect_communities(G: nx.Graph) -> dict[int, int]:
    partition = community.louvain_communities(G, weight="weight", seed=42)
    node_to_community: dict[int, int] = {}
    for comm_id, nodes in enumerate(partition):
        for node in nodes:
            node_to_community[node] = comm_id
    return node_to_community


def graph_to_json(G: nx.Graph, communities: dict[int, int]) -> dict[str, Any]:
    nodes = []
    for node in G.nodes():
        comm = communities.get(node, 0)
        nodes.append(
            {
                "id": str(node),
                "label": G.nodes[node]["label"],
                "platform": G.nodes[node]["platform"],
                "followers": G.nodes[node]["followers"],
                "influence": G.nodes[node]["influence"],
                "is_bot": G.nodes[node]["is_bot"],
                "community": comm,
                "color": COMMUNITY_COLORS[comm % len(COMMUNITY_COLORS)],
            }
        )

    links = [
        {
            "source": str(u),
            "target": str(v),
            "weight": G.edges[u, v]["weight"],
            "shares": G.edges[u, v]["shares"],
        }
        for u, v in G.edges()
    ]

    return {"nodes": nodes, "links": links}


def build_graph_payload(text: str, misinfo_score: float) -> dict[str, Any]:
    G = generate_diffusion_graph(text, misinfo_score)
    communities = detect_communities(G)
    graph_json = graph_to_json(G, communities)

    comm_sizes: dict[int, int] = {}
    for node, comm in communities.items():
        comm_sizes[comm] = comm_sizes.get(comm, 0) + 1

    return {
        "graph": graph_json,
        "stats": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "community_count": len(set(communities.values())),
            "community_sizes": comm_sizes,
            "density": round(nx.density(G), 4),
        },
        "networkx_graph": G,
        "communities": communities,
    }
