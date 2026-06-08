# Backend — Speaker Confidence Assessment API

FastAPI service that combines a trained TF-IDF classifier with a curated
confidence lexicon.

## Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # windows: .venv\Scripts\activate
pip install -r requirements.txt

# first time only — train & persist the model
python ../model/train.py

# start the server
uvicorn main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for the interactive Swagger UI.

## Endpoints

| Method | Path            | Description                               |
|-------:|-----------------|-------------------------------------------|
| GET    | `/api/info`     | Backend status + best model + metrics     |
| GET    | `/api/lexicon`  | Low / high confidence word lists          |
| POST   | `/api/predict`  | `{"text": "…"}` → hybrid confidence score |

### POST /api/predict example

```jsonc
// request
{ "text": "I think this might be right, but I'm not entirely sure." }

// response
{
  "score_percent": 32.6,
  "level": "Low",
  "rule_score": 0.28,
  "ml_score":   0.36,
  "best_model": "Gradient Boosting (TF-IDF)",
  "explanation": "…",
  "probabilities": {"1":0.12,"2":0.31,"3":0.34,"4":0.16,"5":0.07},
  "markers": {
    "low":  [{"word":"i think","start":0,"end":7}, …],
    "high": []
  },
  "stats":   {"word_count":11,"low_count":2,"high_count":0}
}
```

## Fallback mode

If `model/artifacts/best_model.pkl` doesn't exist the API still works —
it returns a pure rule-based score so the UI demo is never broken.
