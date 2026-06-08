/**
 * Thin client for the FastAPI backend.
 *
 * In dev the backend is reverse-proxied via `next.config.js` so the
 * frontend can always call `/api/*` regardless of where it's deployed.
 */

export interface Marker {
  word: string;
  start: number;
  end: number;
}

export interface PredictResponse {
  text: string;
  score_percent: number;
  score_normalised: number;
  rule_score: number;
  ml_score: number | null;
  level: "Low" | "Medium" | "High";
  best_model: string;
  explanation: string;
  probabilities: Record<string, number> | null;
  labels: Record<string, string>;
  markers: { low: Marker[]; high: Marker[] };
  stats: { word_count: number; low_count: number; high_count: number };
  error?: string;
}

export async function predict(text: string): Promise<PredictResponse> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`Prediction failed (${res.status}): ${msg}`);
  }
  return res.json();
}

export async function backendInfo() {
  try {
    const r = await fetch("/api/info");
    if (!r.ok) return null;
    return r.json();
  } catch {
    return null;
  }
}

// ── Analysis API ──────────────────────────────────────────────────────────────

export interface HeatmapResponse {
  features: string[];
  matrix: number[][];
  error?: string;
}

export interface PcaResponse {
  n_components: number;
  feature_names: string[];
  explained_variance_ratio: number[];
  cumulative_variance: number[];
  loadings: number[][];
  error?: string;
}

export interface CorrelationItem {
  name: string;
  correlation: number;
  abs_correlation: number;
}

export interface FeatureTargetResponse {
  target: string;
  correlations: CorrelationItem[];
  error?: string;
}

export interface LimeItem {
  word: string;
  weight: number;
}

export interface LimeResponse {
  text: string;
  predicted_class: string;
  class_names: string[];
  probabilities: Record<string, number>;
  explanation: LimeItem[];
  num_features: number;
  error?: string;
}

export async function fetchHeatmap(): Promise<HeatmapResponse> {
  const r = await fetch("/api/analysis/correlation-heatmap");
  if (!r.ok) throw new Error(`Heatmap failed (${r.status})`);
  return r.json();
}

export async function fetchPca(n_components = 5): Promise<PcaResponse> {
  const r = await fetch(`/api/analysis/pca?n_components=${n_components}`);
  if (!r.ok) throw new Error(`PCA failed (${r.status})`);
  return r.json();
}

export async function fetchFeatureTarget(use_pca = false): Promise<FeatureTargetResponse> {
  const r = await fetch(`/api/analysis/feature-target-correlation?use_pca=${use_pca}`);
  if (!r.ok) throw new Error(`Feature-target failed (${r.status})`);
  return r.json();
}

export async function fetchLime(text: string, num_features = 10): Promise<LimeResponse> {
  const r = await fetch("/api/analysis/lime", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, num_features }),
  });
  if (!r.ok) {
    const msg = await r.text().catch(() => r.statusText);
    throw new Error(`LIME failed (${r.status}): ${msg}`);
  }
  return r.json();
}
