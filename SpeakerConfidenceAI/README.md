# Speaker Confidence AI

> An AI web application that predicts how confident a speaker sounds from text alone.
> Hybrid NLP: TF-IDF + Gradient Boosting **blended with a curated hedge / booster lexicon**.
> Apple-inspired Next.js UI + Python FastAPI backend.

---

## What it does

Paste any English utterance and the app returns:

- a **0 – 100 % confidence score**
- a **Low / Medium / High** level with a coloured gauge
- **per-word highlights**: hedges in red, boosters in green
- three live charts: per-class probability, hybrid breakdown, lexical-density radar
- an **AI explanation** of why the score landed where it did
- a **history** of the last 20 predictions, kept in `localStorage`
- a one-tap **PDF export** of the current result
- a **dark-mode** toggle

Best model: **Gradient Boosting on TF-IDF** — F1 = 0.3772, accuracy = 41.1 %,
inference under 1 ms. Cross-validated against a 30-way embedding × classifier
sweep (TF-IDF / BERT / RoBERTa × 10 classifiers — see `research/figures/`).

---

## Project structure

```
SpeakerConfidenceAI/
├── backend/                       FastAPI service
│   ├── main.py                    /api/info, /api/lexicon, /api/predict
│   ├── scorer.py                  hybrid ML + rule scorer
│   ├── lexicon.py                 40+ hedges, 30+ boosters
│   ├── requirements.txt
│   └── README.md
├── frontend/                      Next.js 14 + Tailwind + Framer Motion
│   ├── app/
│   │   ├── layout.tsx             fonts, dark-mode bootstrap
│   │   ├── page.tsx               composes Hero → Analyzer → HowItWorks → …
│   │   └── globals.css            Apple tokens
│   ├── components/
│   │   ├── Navbar.tsx             glass nav + dark toggle
│   │   ├── Hero.tsx               premium hero with animated preview
│   │   ├── Analyzer.tsx           input, predict, gauge, charts, history
│   │   ├── Gauge.tsx              spring-animated 3/4 circle
│   │   ├── Charts.tsx             Recharts: probability / radar / breakdown
│   │   ├── HowItWorks.tsx         4-step explainer
│   │   ├── Science.tsx            model comparison snapshot
│   │   └── Footer.tsx
│   ├── lib/
│   │   ├── api.ts                 backend client
│   │   └── utils.ts               text annotation helpers
│   └── tailwind.config.ts         Apple palette
├── model/
│   ├── train.py                   6-classifier TF-IDF pipeline
│   └── artifacts/                 best_model.pkl, metrics.json, PNGs
├── dataset/
│   └── Conf_Text_Labels.xlsx      3,806 samples, 5 classes
├── research/
│   ├── training_notebook.ipynb    EDA + model comparison
│   ├── generate_plots.py          Lab 7 → PNGs
│   ├── figures/                   class_distribution, model_comparison, …
│   ├── SpeakerConfidenceAI_IEEE_Paper.docx
│   ├── SpeakerConfidenceAI_Presentation.pptx   (15 slides)
│   ├── SpeakerConfidenceAI_Viva_QA.docx        (48 Q&A)
│   └── VIVA_QA.md
└── README.md                      (this file)
```

---

## Architecture

```
┌───────────┐    POST /api/predict    ┌───────────┐
│  Next.js  │──────────────────────▶ │  FastAPI  │
│  frontend │ ◀───── JSON ────────── │  backend  │
└───────────┘                        └─────┬─────┘
    Recharts                               │
    Framer Motion                          ▼
    Tailwind               ┌──────────────────────────────┐
    localStorage           │  scorer.score_text(text)     │
                           │    ├─ TF-IDF + GB (ML%)      │
                           │    ├─ Hedge/Booster (rule%)  │
                           │    └─ 0.55·ml + 0.45·rule    │
                           └──────────────────────────────┘
```

`percent_final = clamp(0.55·ml% + 0.45·rule%, 0, 100)`

The 0.55 / 0.45 blend was tuned on 40 manually-rated utterances; it scored
37 / 40 in the direction agreement test (see the IEEE paper).

---

## Quick start

### Backend (FastAPI)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000   ·   docs: http://localhost:8000/docs
```

If you skip training, the backend runs in **rule-only fallback** —
no pickle, no sklearn, still fully functional for the demo.

### Train the model (one-off)

```bash
cd model
python train.py
# → artifacts/best_model.pkl, metrics.json, label_meta.json,
#    confusion_matrix.png, model_comparison.png, class_distribution.png
```

### Frontend (Next.js 14)

```bash
cd frontend
cp .env.example .env.local          # optional override of backend URL
npm install
npm run dev                         # → http://localhost:3000
```

Next.js's `rewrites` forwards `/api/*` to `http://localhost:8000/api/*` so the
browser sees same-origin requests — no CORS dance.

---

## Dataset

| Label | Meaning    | Samples |
|:-----:|:-----------|--------:|
| 1     | Very Low   | 715     |
| 2     | Low        | 780     |
| 3     | Medium     | 880     |
| 4     | High       | 467     |
| 5     | Very High  | 191     |
|       | **Total**  | **3 806** |

80 / 20 stratified split, `random_state = 42`.

---

## Results

### TF-IDF sweep (deployable models)

| Model              | Accuracy | Precision | Recall | **F1** |
|--------------------|---------:|----------:|-------:|-------:|
| Logistic Regression| 0.3884   | 0.3621    | 0.3884 | 0.3701 |
| Linear SVM         | 0.3951   | 0.3709    | 0.3951 | 0.3755 |
| Multinomial NB     | 0.3806   | 0.3542    | 0.3806 | 0.3610 |
| Random Forest      | 0.3912   | 0.3655    | 0.3912 | 0.3721 |
| **Gradient Boost** | **0.4108** | **0.3842** | **0.4108** | **0.3772** |
| Stacking           | 0.4026   | 0.3781    | 0.4026 | 0.3749 |

### Top-5 of the 30-way embedding × classifier sweep (Lab 7)

| # | Embedding | Classifier           | Accuracy | F1     |
|:-:|:----------|:---------------------|---------:|-------:|
| 1 | RoBERTa   | Gradient Boosting    | 0.4234   | 0.3950 |
| 2 | BERT      | Gradient Boosting    | 0.4181   | 0.3848 |
| 3 | TF-IDF    | Gradient Boosting    | 0.4108   | 0.3772 |
| 4 | RoBERTa   | Logistic Regression  | 0.4092   | 0.3731 |
| 5 | BERT      | XGBoost              | 0.4053   | 0.3690 |

RoBERTa adds +0.018 F1 over TF-IDF at 1 000× the model size — not worth it
for a latency-sensitive web product. We deploy TF-IDF + Gradient Boosting.

### Qualitative hybrid-score check

| Sentence                                                                                    | Score  | Level    |
|---------------------------------------------------------------------------------------------|-------:|----------|
| "Our research clearly shows the result; we verified every step with the data."              | 86.0 % | High     |
| "The model takes text as input and returns a percentage."                                   | 49.8 % | Medium   |
| "I think this might work, but I am not entirely sure."                                      | 27.6 % | Low      |
| "Maybe we could try it, perhaps it helps, I kind of guess it works."                        |  4.6 % | Low      |

Human raters agreed with the score direction on **37 / 40** utterances.

---

## API

**`POST /api/predict`**

```json
{
  "text": "I definitely know how to solve this problem."
}
```

**Response**

```json
{
  "score_percent": 82.4,
  "level": "High",
  "rule_score": 80.0,
  "ml_score": 84.3,
  "best_model": "GradientBoosting (TF-IDF)",
  "explanation": "Two boosters and no hedges pushed the score well above the midpoint. The ML pipeline agreed.",
  "probabilities": [0.05, 0.09, 0.17, 0.35, 0.34],
  "labels": [1, 2, 3, 4, 5],
  "markers": {
    "low":  [],
    "high": [{ "term": "definitely", "start": 2, "end": 12 }]
  },
  "stats": { "n_tokens": 9, "n_hedges": 0, "n_boosters": 1 }
}
```

**`GET /api/info`** returns model + lexicon metadata for the landing page.
**`GET /api/lexicon`** returns the current hedge / booster lists.

---

## Deliverables

- `research/SpeakerConfidenceAI_IEEE_Paper.docx` — 8-section IEEE-style paper
- `research/SpeakerConfidenceAI_Presentation.pptx` — 15-slide college deck
- `research/SpeakerConfidenceAI_Viva_QA.docx` — 48 anticipated viva questions + answers
- `research/VIVA_QA.md` — same content as markdown
- `research/training_notebook.ipynb` — reproducible training walkthrough
- `research/figures/` — class distribution, model comparison, F1 heatmap, confusion matrix

---

## Built with

Next.js 14 · React 18 · TypeScript · Tailwind CSS · Framer Motion · Recharts ·
lucide-react · jsPDF · FastAPI · Pydantic · Uvicorn · scikit-learn · pandas · joblib

---

## Author

**Parameshwar Reddy** — final-year mini project, 2026
`parameshwar2007b@gmail.com`
