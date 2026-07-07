"""Misinformation classification via HuggingFace Inference API (ai4bharat/indic-bert)."""

import os
import re
from typing import Any

import numpy as np
import requests

HF_MODEL = "ai4bharat/indic-bert"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

RUMOR_SEEDS = [
    "यह गुप्त सूचना है कि सरकार ने तुरंत नया कानून लागू किया है",
    "breaking urgent share before deleted viral message spreading fast",
    "अफवाह फैल रही है कि बैंक बंद हो जाएगा",
    "secret leaked document proves conspiracy everyone must forward",
]

FACT_SEEDS = [
    "According to official sources the ministry confirmed the policy update",
    "सरकारी प्रेस विज्ञप्ति के अनुसार यह जानकारी सत्यापित है",
    "The report cites verified data from published research",
    "स्थानीय प्रशासन ने आधिकारिक बयान जारी किया है",
]

MISINFO_PATTERNS = [
    r"\b(breaking|urgent|viral|share\s+now|forward\s+immediately)\b",
    r"\b(गुप्त|अफवाह|तुरंत\s+शेयर|फॉरवर्ड\s+करें|viral)\b",
    r"!{2,}",
    r"\b(unverified|anonymous\s+source|secret\s+leak)\b",
    r"\b(बिना\s+स्रोत|अनाधिकारिक)\b",
]

_seed_cache: dict[str, np.ndarray] = {}


def _headers() -> dict[str, str]:
    token = os.getenv("HF_API_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _call_hf(payload: dict[str, Any]) -> list | None:
    try:
        response = requests.post(HF_API_URL, headers=_headers(), json=payload, timeout=30)
        if response.status_code == 503:
            # Model loading — retry once after brief wait
            import time
            time.sleep(5)
            response = requests.post(HF_API_URL, headers=_headers(), json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _mean_pool(embedding: list) -> np.ndarray:
    arr = np.array(embedding, dtype=float)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=0)


def _get_embedding(text: str) -> np.ndarray | None:
    if text in _seed_cache:
        return _seed_cache[text]
    result = _call_hf({"inputs": text})
    if not result:
        return None
    emb = _mean_pool(result)
    _seed_cache[text] = emb
    return emb


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _heuristic_score(text: str) -> float:
    score = 0.0
    lower = text.lower()
    for pattern in MISINFO_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            score += 0.12
    if len(text) > 0 and sum(1 for c in text if c.isupper()) / len(text) > 0.35:
        score += 0.1
    if re.search(r"http[s]?://", lower):
        score += 0.05
    return min(score, 0.5)


def _seed_similarity(text_emb: np.ndarray, seeds: list[str]) -> float:
    sims = []
    for seed in seeds:
        seed_emb = _get_embedding(seed)
        if seed_emb is not None:
            sims.append(_cosine(text_emb, seed_emb))
    return max(sims) if sims else 0.0


def classify_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "misinformation_score": 0.0,
            "model": HF_MODEL,
            "method": "indic-bert + heuristic",
        }

    heuristic = _heuristic_score(text)
    embedding = _get_embedding(text)

    if embedding is not None:
        rumor_sim = _seed_similarity(embedding, RUMOR_SEEDS)
        fact_sim = _seed_similarity(embedding, FACT_SEEDS)
        embedding_score = max(0.0, rumor_sim - fact_sim + 0.5)
        embedding_score = min(max(embedding_score, 0.0), 1.0)
        misinfo_score = 0.6 * embedding_score + 0.4 * (heuristic * 2)
    else:
        # Fallback when HF API unavailable
        misinfo_score = min(heuristic * 2.5, 1.0)

    misinfo_score = min(max(misinfo_score, 0.0), 1.0)
    label = "misinformation" if misinfo_score >= 0.5 else "likely_factual"
    confidence = misinfo_score if label == "misinformation" else 1.0 - misinfo_score

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "misinformation_score": round(misinfo_score, 3),
        "model": HF_MODEL,
        "method": "indic-bert embeddings + heuristic fallback",
    }
