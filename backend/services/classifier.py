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
    "doctors don't want you to know this miracle cure big pharma hiding",
    "बैंक खाता ब्लॉक हो जाएगा तुरंत यह लिंक पर क्लिक करें",
]

FACT_SEEDS = [
    "According to official sources the ministry confirmed the policy update",
    "सरकारी प्रेस विज्ञप्ति के अनुसार यह जानकारी सत्यापित है",
    "The report cites verified data from published research",
    "स्थानीय प्रशासन ने आधिकारिक बयान जारी किया है",
    "Peer reviewed clinical study published in a medical journal",
    "समाचार एजेंसी ने पुष्टि की है कि यह जानकारी सही है",
]

# Weighted pattern groups: (regex, weight, category)
# Higher weight = stronger individual signal of misinformation
MISINFO_PATTERNS: list[tuple[str, float, str]] = [
    # Urgency / forward-chain language
    (r"\b(breaking|urgent|viral)\b", 0.10, "urgency"),
    (r"\b(share\s+now|share\s+before|forward\s+immediately|forward\s+(this\s+)?to\s+everyone|send\s+to\s+all)\b", 0.15, "urgency"),
    (r"\b(गुप्त|अफवाह|तुरंत\s+शेयर|फॉरवर्ड\s+करें|सबको\s+भेजें|सबको\s+forward)\b", 0.15, "urgency_hi"),

    # Suppressed-truth / conspiracy framing
    (r"\b(don.t\s+want\s+you\s+to\s+know|they\s+don.t\s+want|hiding\s+(this|the\s+truth)|cover.?up|conspiracy)\b", 0.20, "conspiracy"),
    (r"\b(big\s+pharma|deep\s+state|mainstream\s+media\s+won.t)\b", 0.15, "conspiracy"),
    (r"\b(secret\s+leak(ed)?|anonymous\s+source|unverified\s+report)\b", 0.15, "conspiracy"),
    (r"\b(छुपा\s+रह[ेा]|सच्चाई\s+छिपा)\b", 0.15, "conspiracy_hi"),

    # Health misinformation
    (r"\b(miracle\s+cure|cures?\s+(completely|instantly|in\s+\d+\s+days?)|100%\s+effective|guaranteed\s+cure)\b", 0.20, "health"),
    (r"\b(doctors\s+hate|doctors\s+won.t\s+tell\s+you|banned\s+by\s+(doctors|fda|govt))\b", 0.20, "health"),
    (r"\b(vaccine\s+causes|cures?\s+cancer|cures?\s+diabetes|detox(ify)?\s+your\s+body)\b", 0.15, "health"),

    # Financial scam / urgency
    (r"\b(account\s+will\s+be\s+blocked|verify\s+immediately|click\s+(this\s+)?link\s+now|claim\s+your\s+prize|you\s+(have\s+)?won)\b", 0.20, "scam"),
    (r"\b(खाता\s+ब्लॉक|तुरंत\s+क्लिक|लिंक\s+पर\s+क्लिक|इनाम\s+जीता)\b", 0.20, "scam_hi"),

    # Government/institution conspiracy
    (r"\b(government\s+(is\s+)?hiding|new\s+law\s+secretly|tracking\s+(chip|device)|microchip)\b", 0.20, "gov_conspiracy"),
    (r"\b(सरकार\s+छिपा|ट्रैकिंग\s+चिप|माइक्रोचिप|नया\s+कानून\s+गुप्त)\b", 0.20, "gov_conspiracy_hi"),

    # Emotional manipulation / absolutist claims
    (r"\b(100%\s+guaranteed|never\s+told\s+you|shocking\s+truth|will\s+shock\s+you)\b", 0.10, "emotional"),
    (r"!{2,}", 0.08, "punctuation"),
    (r"!", 0.03, "punctuation"),

    # Lack of sourcing
    (r"\b(unverified|no\s+official\s+source|sources\s+say)\b", 0.08, "no_source"),
    (r"\b(बिना\s+स्रोत|अनाधिकारिक)\b", 0.08, "no_source_hi"),
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
    except requests.RequestException as e:
        print(f"[classifier] HF API error: {e}")
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


def _heuristic_score(text: str) -> tuple[float, list[str]]:
    """Returns (score, matched_categories) for transparency/debugging."""
    score = 0.0
    matched: list[str] = []
    lower = text.lower()

    for pattern, weight, category in MISINFO_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            score += weight
            matched.append(category)

    # All-caps ratio (shouting) — lowered threshold from 0.35 to 0.15
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.15:
        score += 0.08
        matched.append("caps_ratio")

    if re.search(r"http[s]?://", lower):
        score += 0.05
        matched.append("has_link")

    return min(score, 1.0), matched


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
            "matched_signals": [],
        }

    heuristic, matched = _heuristic_score(text)
    embedding = _get_embedding(text)

    if embedding is not None:
        rumor_sim = _seed_similarity(embedding, RUMOR_SEEDS)
        fact_sim = _seed_similarity(embedding, FACT_SEEDS)
        embedding_score = max(0.0, rumor_sim - fact_sim + 0.5)
        embedding_score = min(max(embedding_score, 0.0), 1.0)
        # Heuristic now weighted higher (0.5/0.5) since it's more directly interpretable
        misinfo_score = 0.5 * embedding_score + 0.5 * heuristic
        method = "indic-bert embeddings + heuristic"
    else:
        # Fallback when HF API unavailable — heuristic carries full weight
        misinfo_score = heuristic
        method = "heuristic-only (HF API unavailable — check HF_API_TOKEN / logs)"

    misinfo_score = min(max(misinfo_score, 0.0), 1.0)
    label = "misinformation" if misinfo_score >= 0.5 else "likely_factual"
    confidence = misinfo_score if label == "misinformation" else 1.0 - misinfo_score

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "misinformation_score": round(misinfo_score, 3),
        "model": HF_MODEL,
        "method": method,
        "matched_signals": matched,
    }