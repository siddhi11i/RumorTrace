import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from services.analysis import (
    compute_centrality_scores,
    compute_risk_score,
    suggest_interventions,
    top_spreaders,
)
from services.classifier import classify_text
from services.graph import build_graph_payload

load_dotenv()

app = Flask(__name__)

cors_origins = os.getenv("CORS_ORIGINS", "*")
if cors_origins == "*":
    CORS(app)
else:
    CORS(app, origins=[o.strip() for o in cors_origins.split(",")])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "RumorTrace API"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text or not text.strip():
        return jsonify({"error": "Text is required"}), 400

    classification = classify_text(text)
    misinfo_score = classification["misinformation_score"]

    graph_payload = build_graph_payload(text, misinfo_score)
    G = graph_payload.pop("networkx_graph")
    communities = graph_payload.pop("communities")

    centrality = compute_centrality_scores(G)
    risk = compute_risk_score(misinfo_score, G, centrality)
    spreaders = top_spreaders(G, centrality)
    interventions = suggest_interventions(G, spreaders, risk, communities, centrality)

    for node in graph_payload["graph"]["nodes"]:
        nid = node["id"]
        if nid in centrality:
            node["gnn_score"] = centrality[nid]["gnn_score"]

    return jsonify(
        {
            "classification": classification,
            "risk": risk,
            "top_spreaders": spreaders,
            "interventions": interventions,
            "graph": graph_payload["graph"],
            "stats": graph_payload["stats"],
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
