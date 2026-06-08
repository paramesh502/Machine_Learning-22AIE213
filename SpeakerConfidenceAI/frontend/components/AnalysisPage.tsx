"use client";

import { useEffect, useState } from "react";
import { Loader2, AlertCircle, BarChart2, Layers, Target, FlaskConical } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
  ComposedChart,
  Cell,
} from "recharts";
import {
  fetchHeatmap,
  fetchPca,
  fetchFeatureTarget,
  fetchLime,
  HeatmapResponse,
  PcaResponse,
  FeatureTargetResponse,
  LimeResponse,
} from "../lib/api";

// ── colour helpers ────────────────────────────────────────────────────────────

/** Interpolate deep-blue → white → deep-red for correlation values in [-1, 1] */
function corrColor(v: number): string {
  const clamped = Math.max(-1, Math.min(1, v));
  if (clamped >= 0) {
    // white → deep red
    const t = clamped;
    const r = Math.round(255 - t * (255 - 139));
    const g = Math.round(255 - t * 255);
    const b = Math.round(255 - t * 255);
    return `rgb(${r},${g},${b})`;
  } else {
    // deep blue → white
    const t = -clamped;
    const r = Math.round(255 - t * 255);
    const g = Math.round(255 - t * 255);
    const b = Math.round(255 - t * (255 - 139));
    return `rgb(${r},${g},${b})`;
  }
}

function textOnColor(v: number): string {
  return Math.abs(v) > 0.5 ? "#fff" : "#111";
}

// ── shared loading / error states ─────────────────────────────────────────────

function Loading() {
  return (
    <div className="flex items-center justify-center h-48 text-apple-gray gap-2">
      <Loader2 size={18} className="animate-spin" />
      <span className="text-sm">Loading…</span>
    </div>
  );
}

function ErrorMsg({ msg }: { msg: string }) {
  return (
    <div className="flex items-start gap-2 text-sm text-red-600 bg-red-50 dark:bg-red-500/10 dark:text-red-300 rounded-xl px-4 py-3">
      <AlertCircle size={16} className="mt-0.5 shrink-0" />
      {msg}
    </div>
  );
}

// ── 1. Correlation Heatmap ────────────────────────────────────────────────────

function HeatmapPanel() {
  const [data, setData] = useState<HeatmapResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ row: string; col: string; val: number } | null>(null);

  useEffect(() => {
    fetchHeatmap()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="glass rounded-3xl p-6 shadow-soft">
      <div className="flex items-center gap-2 mb-4">
        <BarChart2 size={16} className="text-apple-blue" />
        <h3 className="font-semibold text-[15px]">Feature Correlation Heatmap</h3>
      </div>
      <p className="text-[12px] text-apple-gray mb-4">
        Pearson correlation between extracted text features. Deep red = +1, white = 0, deep blue = −1.
      </p>

      {loading && <Loading />}
      {error && <ErrorMsg msg={error} />}

      {data && !loading && (
        <>
          {/* Tooltip */}
          {tooltip && (
            <div className="mb-3 text-[12px] bg-black/80 text-white rounded-lg px-3 py-1.5 inline-block">
              <span className="font-medium">{tooltip.row}</span>
              {" × "}
              <span className="font-medium">{tooltip.col}</span>
              {" = "}
              <span className="font-mono">{tooltip.val.toFixed(4)}</span>
            </div>
          )}

          {/* Scrollable heatmap */}
          <div className="overflow-x-auto">
            <table className="text-[10px] border-collapse" style={{ minWidth: data.features.length * 52 }}>
              <thead>
                <tr>
                  <th className="w-28" />
                  {data.features.map((f) => (
                    <th
                      key={f}
                      className="text-apple-gray font-normal pb-1 px-0.5"
                      style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", height: 80 }}
                    >
                      {f}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.features.map((rowF, ri) => (
                  <tr key={rowF}>
                    <td className="text-apple-gray pr-2 text-right whitespace-nowrap">{rowF}</td>
                    {data.matrix[ri].map((val, ci) => (
                      <td
                        key={ci}
                        className="w-12 h-10 text-center cursor-pointer transition-opacity hover:opacity-80"
                        style={{
                          backgroundColor: corrColor(val),
                          color: textOnColor(val),
                        }}
                        aria-label={`${rowF} × ${data.features[ci]}: ${val.toFixed(4)}`}
                        onMouseEnter={() =>
                          setTooltip({ row: rowF, col: data.features[ci], val })
                        }
                        onMouseLeave={() => setTooltip(null)}
                      >
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

// ── 2. PCA Panel ─────────────────────────────────────────────────────────────

function PcaPanel() {
  const [data, setData] = useState<PcaResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPca(5)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const chartData = data
    ? data.explained_variance_ratio.map((evr, i) => ({
        name: `PC${i + 1}`,
        variance: +(evr * 100).toFixed(2),
        cumulative: +(data.cumulative_variance[i] * 100).toFixed(2),
      }))
    : [];

  return (
    <div className="glass rounded-3xl p-6 shadow-soft">
      <div className="flex items-center gap-2 mb-4">
        <Layers size={16} className="text-apple-blue" />
        <h3 className="font-semibold text-[15px]">PCA — Explained Variance</h3>
      </div>
      <p className="text-[12px] text-apple-gray mb-4">
        Bars show variance explained per component; line shows cumulative variance.
      </p>

      {loading && <Loading />}
      {error && <ErrorMsg msg={error} />}

      {data && !loading && (
        <>
          {/* Cumulative label */}
          <div className="mb-3 text-[13px] font-medium">
            Cumulative variance ({data.n_components} components):{" "}
            <span className="text-apple-blue">
              {(data.cumulative_variance[data.n_components - 1] * 100).toFixed(1)}%
            </span>
          </div>

          {/* Bar + line chart */}
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={chartData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis
                yAxisId="left"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 11 }}
              />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar yAxisId="left" dataKey="variance" name="Variance %" fill="#0071e3" radius={[4, 4, 0, 0]} />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="cumulative"
                name="Cumulative %"
                stroke="#ff6b35"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </ComposedChart>
          </ResponsiveContainer>

          {/* Loadings heatmap */}
          <div className="mt-6">
            <div className="text-[12px] text-apple-gray mb-2 font-medium uppercase tracking-wider">
              Component Loadings
            </div>
            <div className="overflow-x-auto">
              <table className="text-[10px] border-collapse w-full">
                <thead>
                  <tr>
                    <th className="text-left text-apple-gray font-normal pb-1 pr-3 w-36">Feature</th>
                    {Array.from({ length: data.n_components }, (_, i) => (
                      <th key={i} className="text-apple-gray font-normal pb-1 px-1 text-center">
                        PC{i + 1}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.feature_names.map((fname, fi) => (
                    <tr key={fname}>
                      <td className="text-apple-gray pr-3 py-0.5 whitespace-nowrap">{fname}</td>
                      {data.loadings[fi].map((val, ci) => (
                        <td
                          key={ci}
                          className="w-12 h-8 text-center"
                          style={{
                            backgroundColor: corrColor(val),
                            color: textOnColor(val),
                          }}
                          aria-label={`${fname} on PC${ci + 1}: ${val.toFixed(4)}`}
                        >
                          {val.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── 3. Feature-Target Correlation ─────────────────────────────────────────────

function FeatureTargetPanel() {
  const [usePca, setUsePca] = useState(false);
  const [data, setData] = useState<FeatureTargetResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  function load(pca: boolean) {
    setLoading(true);
    setError(null);
    fetchFeatureTarget(pca)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load(false);
  }, []);

  function toggle() {
    const next = !usePca;
    setUsePca(next);
    load(next);
  }

  const chartData = data
    ? data.correlations.map((c) => ({
        name: c.name,
        correlation: c.correlation,
      }))
    : [];

  return (
    <div className="glass rounded-3xl p-6 shadow-soft">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <Target size={16} className="text-apple-blue" />
          <h3 className="font-semibold text-[15px]">Feature-Target Correlation</h3>
        </div>
        {/* Toggle */}
        <button
          onClick={toggle}
          aria-label={usePca ? "Show raw features" : "Show PCA components"}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-[12px] font-medium bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 transition-colors"
        >
          <span
            className={`w-7 h-4 rounded-full transition-colors relative ${
              usePca ? "bg-apple-blue" : "bg-gray-300 dark:bg-gray-600"
            }`}
          >
            <span
              className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform ${
                usePca ? "translate-x-3.5" : "translate-x-0.5"
              }`}
            />
          </span>
          {usePca ? "PCA components" : "Raw features"}
        </button>
      </div>
      <p className="text-[12px] text-apple-gray mb-4">
        Pearson correlation with <strong>{data?.target ?? "Conf Label"}</strong>. Green = positive, red = negative.
        Sorted by absolute value.
      </p>

      {loading && <Loading />}
      {error && <ErrorMsg msg={error} />}

      {data && !loading && (
        <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 32)}>
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 4, right: 24, bottom: 4, left: 120 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" horizontal={false} />
            <XAxis
              type="number"
              domain={[-0.3, 0.3]}
              tickFormatter={(v) => v.toFixed(2)}
              tick={{ fontSize: 11 }}
            />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={115} />
            <Tooltip formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={
                    entry.correlation > 0
                      ? "#34c759"
                      : entry.correlation < 0
                      ? "#ff3b30"
                      : "#8e8e93"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── 4. LIME Panel ─────────────────────────────────────────────────────────────

function LimePanel() {
  const [text, setText] = useState("");
  const [numFeatures, setNumFeatures] = useState(10);
  const [data, setData] = useState<LimeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    const input = text.trim();
    if (!input) {
      setError("Please enter some text first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetchLime(input, numFeatures);
      setData(res);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const maxAbs = data
    ? Math.max(...data.explanation.map((e) => Math.abs(e.weight)), 0.001)
    : 1;

  return (
    <div className="glass rounded-3xl p-6 shadow-soft">
      <div className="flex items-center gap-2 mb-4">
        <FlaskConical size={16} className="text-apple-blue" />
        <h3 className="font-semibold text-[15px]">LIME — Local Explanation</h3>
      </div>
      <p className="text-[12px] text-apple-gray mb-4">
        LIME explains <em>why</em> the model predicted a specific confidence class for your text.
        Green bars push the prediction toward the predicted class; red bars push against it.
      </p>

      {/* Input */}
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste a sentence to explain, e.g. I think this might be correct but I'm not sure…"
        rows={3}
        className="w-full resize-none bg-transparent outline-none text-[14px] leading-relaxed placeholder:text-apple-gray border border-black/10 dark:border-white/10 rounded-xl px-4 py-3 mb-3"
      />

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <label className="text-[12px] text-apple-gray flex items-center gap-2">
          Top features:
          <input
            type="number"
            min={1}
            max={30}
            value={numFeatures}
            onChange={(e) => setNumFeatures(Math.max(1, Math.min(30, +e.target.value)))}
            className="w-14 bg-black/5 dark:bg-white/10 rounded-lg px-2 py-1 text-center text-[12px] outline-none"
          />
        </label>
        <button
          onClick={run}
          disabled={loading}
          className="inline-flex items-center gap-2 px-5 py-2 rounded-full bg-apple-black dark:bg-white text-white dark:text-apple-black text-[13px] font-medium hover:bg-black/85 dark:hover:bg-white/85 transition-all disabled:opacity-60"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <FlaskConical size={14} />}
          {loading ? "Running LIME…" : "Explain"}
        </button>
      </div>

      {error && <ErrorMsg msg={error} />}

      {data && !loading && (
        <div className="mt-2">
          {/* Predicted class + probabilities */}
          <div className="mb-4 flex flex-wrap gap-2 items-center">
            <span className="text-[12px] text-apple-gray">Predicted class:</span>
            <span className="px-3 py-1 rounded-full bg-apple-blue text-white text-[12px] font-semibold">
              {data.predicted_class}
            </span>
            <span className="text-[12px] text-apple-gray ml-2">Probabilities:</span>
            {Object.entries(data.probabilities).map(([cls, prob]) => (
              <span
                key={cls}
                className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
                  cls === data.predicted_class
                    ? "bg-apple-blue/10 text-apple-blue"
                    : "bg-black/5 dark:bg-white/10 text-apple-gray"
                }`}
              >
                {cls}: {(prob * 100).toFixed(1)}%
              </span>
            ))}
          </div>

          {/* Word-weight bars */}
          <div className="space-y-1.5">
            {data.explanation.map((item, i) => {
              const pct = (Math.abs(item.weight) / maxAbs) * 100;
              const positive = item.weight >= 0;
              return (
                <div key={i} className="flex items-center gap-3 text-[12px]">
                  {/* word label */}
                  <span
                    className="w-32 text-right shrink-0 font-mono truncate"
                    title={item.word}
                  >
                    {item.word}
                  </span>
                  {/* bar */}
                  <div className="flex-1 h-5 bg-black/5 dark:bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: positive ? "#34c759" : "#ff3b30",
                      }}
                    />
                  </div>
                  {/* weight value */}
                  <span
                    className={`w-16 tabular-nums ${
                      positive ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
                    }`}
                  >
                    {item.weight > 0 ? "+" : ""}
                    {item.weight.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>

          <p className="mt-4 text-[11px] text-apple-gray">
            Explanation is for predicted class <strong>{data.predicted_class}</strong>.
            Positive weights support this class; negative weights oppose it.
          </p>
        </div>
      )}
    </div>
  );
}

// ── Main Analysis Page ────────────────────────────────────────────────────────

export default function AnalysisPage() {
  return (
    <section id="analysis" className="relative py-24 noise">
      <div className="max-w-6xl mx-auto px-6 relative">
        <div className="text-center mb-12">
          <p className="text-sm uppercase tracking-[0.18em] text-apple-blue font-medium">
            ML Insights
          </p>
          <h2 className="mt-2 font-display text-4xl md:text-5xl font-semibold tracking-tight">
            Analysis
          </h2>
          <p className="mt-3 text-apple-gray max-w-xl mx-auto">
            Explore feature correlations, PCA structure, and LIME explanations
            derived from the speaker confidence dataset.
          </p>
        </div>

        <div className="space-y-8">
          {/* Row 1: Heatmap (full width) */}
          <HeatmapPanel />

          {/* Row 2: PCA + Feature-Target side by side on large screens */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <PcaPanel />
            <FeatureTargetPanel />
          </div>

          {/* Row 3: LIME (full width) */}
          <LimePanel />
        </div>
      </div>
    </section>
  );
}
