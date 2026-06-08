# Frontend — Speaker Confidence AI

Next.js 14 (App Router) + Tailwind + Framer Motion + Recharts.

## Run

```bash
cd frontend
cp .env.example .env.local           # optional — points at the backend
npm install
npm run dev                          # http://localhost:3000
```

The frontend expects the FastAPI backend on `http://localhost:8000` by
default. Override with `NEXT_PUBLIC_BACKEND_URL` if you deploy elsewhere.

## Structure

```
app/
  layout.tsx     — fonts, dark-mode bootstrap, metadata
  page.tsx       — home page
  globals.css    — Apple-style tokens & helpers
components/
  Navbar.tsx     — sticky glass nav + dark mode toggle
  Hero.tsx       — premium hero with animated preview card
  Analyzer.tsx   — input, predict button, gauge, charts, history
  Charts.tsx     — Recharts wrappers (probability, radar, breakdown)
  Gauge.tsx      — animated 3/4 circle confidence gauge
  HowItWorks.tsx — explainer section
  Science.tsx    — model comparison snapshot (from Lab 7)
  Footer.tsx     — copy & quick links
lib/
  api.ts         — backend client
  utils.ts       — text annotation + helpers
```

## Features

- Live typing → score update via `Predict`
- Confidence gauge with spring-animated needle
- Recharts probability, radar (lexical density), hybrid breakdown
- Inline word highlights — red for hedges, green for boosters
- Local-storage history of the last 20 predictions
- Dark mode with persisted preference + system default
- Export report as PDF (`jspdf`)
- Copy result to clipboard
- Responsive down to mobile 375 px

## Tailwind theme

The Apple tokens live in `tailwind.config.ts` and `app/globals.css`:

- `apple.black`  `#1d1d1f`
- `apple.gray`   `#86868b`
- `apple.blue`   `#0071e3`
- `shadow-glass` subtle drop used on floating cards
- `glass`        blurred saturate glassmorphism helper
- `hero-grad`    multi-radial hero backdrop
