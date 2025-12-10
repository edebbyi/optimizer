# Optimizer Agent

Stateless FastAPI service that learns your house style from liked fashion images (global preference vector) and scores prompt structures (optimizer score). It trains a preference/reward model (binary classifier) to approximate “will this structure succeed?” n8n handles scheduling; this app only responds to requests.

## How It Works
- Train (`POST /train`): fetch images/structures from Airtable, extract features (CLIP visual, GPT‑4V attributes, text/skeleton embeddings, structure metadata), train an XGBoost preference/reward classifier, save model + scaler + preference vector to `model/`.
- Score (`POST /score_structures`): load model, build proxy features for active structures, return `predicted_success_score` per `structure_id` plus top positive/negative attributes and the global preference vector.
- Health (`GET /health`): service status and cache stats.

## Repository Structure
```
optimizer/
├── api.py                  # FastAPI app + endpoints
├── train.py                # Training pipeline (preference/reward model)
├── inference.py            # Scoring logic
├── feature_extraction.py   # CLIP, GPT‑4V, text embeddings, caching
├── airtable_client.py      # Airtable fetch + helpers
├── config.py               # Constants, env, vocab
├── utils.py                # Helpers (logging, health)
├── tests/                  # Pytest suite
├── cache/                  # Feature caches (CLIP/attributes) [gitignored]
├── model/                  # Model artifacts, preference vector [gitignored]
├── requirements.txt        # Python deps
├── Dockerfile              # Container build
└── render.yaml             # Render deployment blueprint
```

## Preference Vector vs. Optimizer Score
- **Global preference vector** (`model/global_preference_vector.json`): your learned style DNA (averaged GPT‑4V attributes from liked images). Feed this to the generator to bias skeletons/prompts toward winning attributes (e.g., clean_finish, minimalist, versatile).
- **Optimizer score** (`predicted_success_score` in `/score_structures`): the strategist-facing signal. In n8n, PATCH this into Airtable Structures as `optimizer_score` so your strategist can rank/select structures based on predicted success.

## Data Sources
- Airtable Images: `Structure ID`, `ImageUrl`/`Image` attachment, `Status` ("Outlier"=1, else 0), `Prompt`.
- Airtable Structures: `Structure ID`, `Skeleton`, `Renderer`, `outlier_count`, `usage_count`, `age_weeks`, `z_score`, `AI Score`.

## Feature Vector (1387 dims)
- 512 CLIP image embedding
- 384 prompt text embedding
- 100 GPT‑4V attribute scores
- 384 skeleton embedding
- 7 metadata: `[usage_count, outlier_count, age_weeks, z_score, ai_score, renderer_encoded, structure_id]`

## Running Locally
```bash
cp .env.example .env  # fill Airtable + OpenAI keys
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

## API
- `GET /health` → status (degraded until trained).
- `POST /train` → triggers training, writes artifacts to `model/`.
- `POST /score_structures` → scores active structures, returns `predicted_success_score` and preference vector.
- `GET /preferences` → current global preference vector.
- `GET /prompts/top` → top-performing prompts by success rate.
- `GET /structure_prompt_insights` → structure-specific prompt performance data.
- `DELETE /cache` → clear CLIP/GPT‑4V caches.

## n8n Integration
- **Retrain**: trigger `/train` when 25 new Outliers since last run.
- **Rescore**: after training (or on demand), call `/score_structures`, then PATCH Airtable Structures `optimizer_score` with returned `predicted_success_score`.
- **Generator input**: load the global preference vector (top attributes) to bias your generator toward favored attributes.

## Troubleshooting
- Expired `ImageUrl`: service falls back to `Image` attachment URL.
- Malformed GPT‑4V JSON: parser is tolerant; zero-fills on failure.
- Missing model: `/score_structures` returns default 0.5 scores with a warning.
