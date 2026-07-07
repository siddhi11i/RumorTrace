# RumorTrace

Misinformation diffusion analyzer for Hindi and English text. Classifies content with **ai4bharat/indic-bert** via the Hugging Face Inference API, generates a synthetic diffusion graph with **NetworkX + Louvain** community detection, and visualizes spread patterns with a **D3.js** force-directed graph.

## Architecture

```
User text → Flask API → Indic-BERT (HF API) → misinfo score
                      → NetworkX graph + Louvain communities
                      → Centrality-based GNN simulation
                      → Risk score + spreaders + interventions
           ← JSON ← React + D3 force graph
```

## Features

- **Bilingual classification** — Hindi and English text via Indic-BERT embeddings + heuristics
- **Synthetic diffusion graph** — Barabási–Albert model seeded from input text
- **Louvain communities** — Color-coded clusters in the force graph
- **Simulated GNN scores** — Degree, betweenness, eigenvector, PageRank combined
- **Risk dashboard** — Composite risk score with factor breakdown
- **Intervention suggestions** — Bridge nodes, bots, community targeting

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
copy .env.example .env        # add HF_API_TOKEN
python app.py
```

API runs at `http://localhost:5000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App runs at `http://localhost:5173` (proxies `/api` to Flask).

### Hugging Face Token

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set `HF_API_TOKEN` in `backend/.env`. Without a token, the API still works using heuristic-only classification.

## Deployment

### Flask on Render (free tier)

1. Push this repo to GitHub.
2. Create a **Web Service** on [Render](https://render.com) and connect the repo.
3. Use the included `render.yaml` or set:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Add environment variables:
   - `HF_API_TOKEN` — your Hugging Face token
   - `CORS_ORIGINS` — `https://YOUR_USERNAME.github.io`

### React on GitHub Pages

1. Enable **GitHub Pages** → Source: **GitHub Actions**.
2. In repo **Settings → Secrets and variables → Actions → Variables**, add:
   - `VITE_API_URL` = your Render API URL (e.g. `https://rumortrace-api.onrender.com`)
3. Push to `main` — the workflow builds and deploys automatically.

Manual deploy:

```bash
cd frontend
VITE_API_URL=https://your-api.onrender.com VITE_BASE_PATH=/rumortrace/ npm run build
npm run deploy
```

## API

### `POST /api/analyze`

```json
{ "text": "Your Hindi or English text here" }
```

Response includes `classification`, `risk`, `top_spreaders`, `interventions`, `graph`, and `stats`.

### `GET /health`

Health check endpoint.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React, Vite, D3.js |
| Backend | Flask, Flask-CORS, Gunicorn |
| ML | ai4bharat/indic-bert (Hugging Face Inference API) |
| Graph | NetworkX, Louvain community detection |
| Deploy | GitHub Pages, Render |

## License

MIT
