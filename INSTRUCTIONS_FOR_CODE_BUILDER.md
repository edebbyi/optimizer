# Instructions for Code Builder (Cline/Roo)

## Overview

You are building a **stateless ML service** called the Optimizer Agent. This is NOT an agentic system - it's a simple request → response API with no loops, no tool calling, no multi-step reasoning.

## What This Service Does

1. **Learns** from liked fashion images (labeled "Outlier" in Airtable)
2. **Extracts** features using CLIP (visual), GPT-4V (attributes), and sentence-transformers (text)
3. **Trains** an XGBoost binary classifier
4. **Scores** prompt structures based on predicted success probability
5. **Exposes** results via FastAPI endpoints

## Architecture

```
POST /train (called by n8n cron)
    → Fetch images + structures from Airtable
    → Extract features (CLIP + GPT-4V + text embeddings)
    → Train XGBoost model
    → Save model to disk
    → Return metrics

POST /score_structures (called by Strategist)
    → Load trained model
    → Fetch active structures from Airtable
    → Score each structure
    → Return ranked list with scores
```

## Key Design Decisions

### 1. No Scheduler/Background Jobs
All scheduling is handled by **n8n**. This service is purely reactive - it only does work when an endpoint is called.

### 2. Stateless
The service maintains no state between requests. Model and cache are persisted to disk, but the process itself is stateless.

### 3. Skip Missing Structure IDs
Images without a Structure ID are **skipped** during training. This ensures we can attribute success/failure to specific structures.

### 4. Label Mapping
- `Status = "Outlier"` → **y = 1** (positive/liked)
- `Status = "Pass"` or empty → **y = 0** (negative/not liked)

### 5. Feature Caching
CLIP embeddings and GPT-4V attributes are **cached** by image URL hash. This is critical for cost and performance.

## Data Sources

### Airtable Images Table
- **Table ID:** `tblwnQTC0Tp8xgTue`
- **View:** `viwZ4vi03GR4oMXGg`
- **Key Fields:**
  - `ImageUrl` - URL to fetch image from
  - `Status` - "Outlier" = positive, "Pass" = negative
  - `Prompt` - Full prompt text
  - `Structure ID` - Links to structures table

### Airtable Structures Table
- **Table ID:** `tblPPDf9vlTBv2kyl`
- **Active View:** `viwoC5FLPRgpvkjj6` (for scoring)
- **All View:** `viw0O8ywpWywfjZPt` (for training - need historical structures)
- **Key Fields:**
  - `Structure ID` - Unique identifier
  - `Skeleton` - Template text
  - `Renderer` - "ImageFX" or "Recraft"
  - `outlier_count`, `usage_count`, `age_weeks`, `z_score`, `AI Score`

## Feature Vector (1387 dimensions)

| Component | Dims | Source |
|-----------|------|--------|
| Visual embedding | 512 | CLIP ViT-B-32 on image |
| Text embedding | 384 | sentence-transformers on prompt |
| Attribute vector | 100 | GPT-4V extraction |
| Skeleton embedding | 384 | sentence-transformers on skeleton |
| Structure metadata | 7 | Airtable fields |

### Structure Metadata (7D)
```python
[
    usage_count,      # int
    outlier_count,    # int
    age_weeks,        # float
    z_score,          # float
    ai_score,         # float (parsed from "8/10")
    renderer_encoded, # int (0=ImageFX, 1=Recraft)
    structure_id      # int
]
```

## Files to Implement

All files are provided. Review and validate:

1. **config.py** - Constants, env vars, attribute vocabulary
2. **airtable_client.py** - Airtable data fetching
3. **feature_extraction.py** - CLIP, GPT-4V, text embeddings
4. **train.py** - XGBoost training pipeline
5. **inference.py** - Structure scoring
6. **utils.py** - Helpers
7. **api.py** - FastAPI endpoints

## Testing Checklist

### 1. Environment Setup
```bash
cd optimizer
cp .env.example .env
# Fill in real API keys in .env
pip install -r requirements.txt
```

### 2. Test Health Check
```bash
uvicorn api:app --reload --port 8000
curl http://localhost:8000/health
```

Expected: Returns JSON with `status: "degraded"` (no model yet)

### 3. Test Training
```bash
curl -X POST http://localhost:8000/train
```

Expected: Takes 5-10+ minutes, returns training metrics

### 4. Test Scoring
```bash
curl -X POST http://localhost:8000/score_structures
```

Expected: Returns list of structures with `predicted_success_score`

## Common Issues

### 1. Missing OPENAI_API_KEY
GPT-4V extraction will fail. Service will return zero vectors for attributes.

### 2. Airtable Rate Limits
If fetching 39K+ images, may hit rate limits. Add retry logic with exponential backoff.

### 3. CLIP Model Download
First run will download ~400MB CLIP model. Ensure disk space and network.

### 4. Memory Usage
CLIP model + XGBoost can use ~2GB RAM. Ensure sufficient memory.

## n8n Integration

### Daily Retrain Workflow
```
Trigger: Schedule (cron: 0 2 * * *)
Action: HTTP Request
  - Method: POST
  - URL: http://optimizer-host:8000/train
  - Timeout: 600 seconds
```

### Threshold Retrain (Optional)
```
Trigger: Airtable (when record updated)
Filter: Status changed to "Outlier"
Action: Count new outliers
If: count >= 25
Then: HTTP Request POST /train
```

### Strategist Integration
```
Your strategist workflow calls:
POST http://optimizer-host:8000/score_structures

Response includes:
- structures[].predicted_success_score
- global_preference_vector
```

## Deployment Options

### Local
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t optimizer .
docker run -p 8000:8000 --env-file .env optimizer
```

### Railway/Render
1. Connect GitHub repo
2. Set environment variables in dashboard
3. Ensure persistent storage for `model/` and `cache/` directories

## What NOT to Build

- ❌ No prompt generation
- ❌ No strategist logic
- ❌ No frontend
- ❌ No image storage
- ❌ No n8n automation code
- ❌ No scheduling (n8n handles this)
- ❌ No agentic loops or tool calling

## Success Criteria

1. `/health` returns service status
2. `/train` successfully trains model and saves to disk
3. `/score_structures` returns ranked structures with scores
4. Caching works (second run faster than first)
5. Model persists across restarts
