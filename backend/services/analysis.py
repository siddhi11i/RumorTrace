"""Risk scoring, spreader detection, and intervention suggestions."""

from typing import Any

import networkx as nx


def compute_centrality_scores(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Simulate GNN node importance using NetworkX centrality metrics."""
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight")
    eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
    pagerank = nx.pagerank(G, weight="weight")

    scores: dict[str, dict[str, float]] = {}
    for node in G.nodes():
        nid = str(node)
        combined = (
            0.25 * degree[node]
            + 0.30 * betweenness[node]
            + 0.20 * eigenvector[node]
            + 0.25 * pagerank[node]
        )
        influence = G.nodes[node].get("influence", 0.5)
        followers = G.nodes[node].get("followers", 100)
        follower_boost = min(followers / 10000, 0.3)
        gnn_simulated = min(combined + influence * 0.2 + follower_boost, 1.0)

        scores[nid] = {
            "degree": round(degree[node], 4),
            "betweenness": round(betweenness[node], 4),
            "eigenvector": round(eigenvector[node], 4),
            "pagerank": round(pagerank[node], 4),
            "gnn_score": round(gnn_simulated, 4),
        }
    return scores


def compute_risk_score(
    misinfo_score: float,
    G: nx.Graph,
    centrality_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    avg_gnn = sum(s["gnn_score"] for s in centrality_scores.values()) / max(len(centrality_scores), 1)
    bot_count = sum(1 for n in G.nodes() if G.nodes[n].get("is_bot"))
    bot_ratio = bot_count / max(G.number_of_nodes(), 1)
    density = nx.density(G)

    risk = (
        0.45 * misinfo_score
        + 0.25 * avg_gnn
        + 0.15 * bot_ratio
        + 0.15 * min(density * 5, 1.0)
    )
    risk = min(max(risk, 0.0), 1.0)

    if risk >= 0.75:
        level = "critical"
    elif risk >= 0.55:
        level = "high"
    elif risk >= 0.35:
        level = "moderate"
    else:
        level = "low"

    return {
        "score": round(risk, 3),
        "level": level,
        "factors": {
            "content_misinfo_score": round(misinfo_score, 3),
            "network_amplification": round(avg_gnn, 3),
            "bot_activity_ratio": round(bot_ratio, 3),
            "graph_density": round(density, 4),
        },
    }


def top_spreaders(
    G: nx.Graph,
    centrality_scores: dict[str, dict[str, float]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(
        centrality_scores.items(),
        key=lambda x: x[1]["gnn_score"],
        reverse=True,
    )[:top_k]

    spreaders = []
    for nid, scores in ranked:
        node = int(nid)
        spreaders.append(
            {
                "id": nid,
                "label": G.nodes[node]["label"],
                "platform": G.nodes[node]["platform"],
                "followers": G.nodes[node]["followers"],
                "is_bot": G.nodes[node]["is_bot"],
                "gnn_score": scores["gnn_score"],
                "betweenness": scores["betweenness"],
                "pagerank": scores["pagerank"],
            }
        )
    return spreaders


def suggest_interventions(
    G: nx.Graph,
    spreaders: list[dict[str, Any]],
    risk: dict[str, Any],
    communities: dict[int, int],
    centrality_scores: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    if risk["level"] in ("critical", "high"):
        suggestions.append(
            {
                "priority": "high",
                "action": "Fact-check amplification",
                "detail": "Publish verified counter-narrative on platforms where top spreaders are active.",
            }
        )

    bridge_nodes = [
        nid
        for nid, _ in sorted(
            centrality_scores.items(),
            key=lambda x: x[1]["betweenness"],
            reverse=True,
        )[:3]
    ]

    if bridge_nodes:
        suggestions.append(
            {
                "priority": "high",
                "action": "Target bridge nodes",
                "detail": f"Engage or limit reach of bridge nodes {', '.join(bridge_nodes)} to disrupt cross-community spread.",
            }
        )

    bot_spreaders = [s for s in spreaders if s["is_bot"]]
    if bot_spreaders:
        suggestions.append(
            {
                "priority": "medium",
                "action": "Bot mitigation",
                "detail": f"Flag {len(bot_spreaders)} suspected bot account(s) for platform review.",
            }
        )

    comm_sizes: dict[int, int] = {}
    for _, comm in communities.items():
        comm_sizes[comm] = comm_sizes.get(comm, 0) + 1
    largest_comm = max(comm_sizes, key=comm_sizes.get) if comm_sizes else 0

    suggestions.append(
        {
            "priority": "medium",
            "action": "Community-level intervention",
            "detail": f"Largest Louvain community (ID {largest_comm}, {comm_sizes.get(largest_comm, 0)} nodes) should receive targeted inoculation messaging.",
        }
    )

    if risk["factors"]["graph_density"] > 0.15:
        suggestions.append(
            {
                "priority": "low",
                "action": "Reduce resharing velocity",
                "detail": "High graph density suggests rapid resharing — add friction (e.g., share prompts, delay timers).",
            }
        )
    else:
        suggestions.append(
            {
                "priority": "low",
                "action": "Monitor growth",
                "detail": "Network is sparse; continue monitoring for new high-centrality nodes.",
            }
        )

    return suggestions
