"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  Brain,
  Copy,
  Download,
  FileText,
  Keyboard,
  Loader2,
  Mic,
  MicOff,
  Play,
  Trash2,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { PredictResponse, predict } from "../lib/api";
import { annotate, formatDate, levelToColor } from "../lib/utils";
import { ProbabilityChart, ScoreBreakdown, WordIndicators } from "./Charts";
import Gauge from "./Gauge";

const SAMPLES = [
  "I think this might be correct, but I'm not entirely sure. Maybe we should double-check the formula?",
  "This is absolutely correct. I am 100% certain about the answer and I can verify each step.",
  "K-Nearest Neighbours uses similarity measures such as Euclidean or Cosine distance to classify points.",
  "I don't really know. I guess it could be something like gradient descent, but I'm not sure how it fits here.",
];

interface HistoryItem {
  id: string;
  ts: number;
  text: string;
  score_percent: number;
  level: PredictResponse["level"];
  rule_score: number;
  ml_score: number | null;
  best_model: string;
}

const STORAGE_KEY = "scai.history.v1";

export default function Analyzer() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [mode, setMode] = useState<"text" | "voice">("text");
  const [listening, setListening] = useState(false);
  const [micSupported, setMicSupported] = useState(true);
  const [interim, setInterim] = useState("");
  const resultRef = useRef<HTMLDivElement | null>(null);
  const recognitionRef = useRef<any>(null);
  const finalTranscriptRef = useRef<string>("");

  // restore history
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setHistory(JSON.parse(raw));
    } catch {}
  }, []);

  // set up SpeechRecognition once on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    const SR: any =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SR) {
      setMicSupported(false);
      return;
    }
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";
    rec.onresult = (e: any) => {
      let finalPart = "";
      let interimPart = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalPart += t + " ";
        else interimPart += t;
      }
      if (finalPart) {
        finalTranscriptRef.current += finalPart;
        setText(finalTranscriptRef.current.trim());
      }
      setInterim(interimPart);
    };
    rec.onend = () => {
      setListening(false);
      setInterim("");
    };
    rec.onerror = (e: any) => {
      setListening(false);
      setInterim("");
      if (e?.error === "not-allowed" || e?.error === "service-not-allowed") {
        setError("Microphone blocked. Enable it in your browser permissions.");
      }
    };
    recognitionRef.current = rec;
    return () => {
      try {
        rec.stop();
      } catch {}
    };
  }, []);

  function toggleMic() {
    if (!micSupported) {
      setError(
        "Speech recognition isn't available in this browser — try Chrome, Edge, or Safari."
      );
      return;
    }
    const rec = recognitionRef.current;
    if (!rec) return;
    if (listening) {
      try {
        rec.stop();
      } catch {}
      setListening(false);
      return;
    }
    // start fresh
    finalTranscriptRef.current = text ? text + " " : "";
    setError(null);
    setInterim("");
    try {
      rec.start();
      setListening(true);
    } catch (e: any) {
      setError(e?.message ?? "Could not start the microphone.");
    }
  }

  // persist history
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(0, 20)));
    } catch {}
  }, [history]);

  async function onPredict() {
    const input = text.trim();
    if (!input) {
      setError("Please paste some text first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predict(input);
      if (res.error) {
        setError(res.error);
      } else {
        setResult(res);
        setHistory((prev) => [
          {
            id: crypto.randomUUID(),
            ts: Date.now(),
            text: res.text,
            score_percent: res.score_percent,
            level: res.level,
            rule_score: res.rule_score,
            ml_score: res.ml_score,
            best_model: res.best_model,
          },
          ...prev,
        ].slice(0, 20));
        setTimeout(() => {
          resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);
      }
    } catch (e: any) {
      setError(
        e?.message ??
          "Could not reach the backend. Make sure FastAPI is running on :8000."
      );
    } finally {
      setLoading(false);
    }
  }

  async function copyResult() {
    if (!result) return;
    const lines = [
      `Speaker Confidence AI — Result`,
      `Text: ${result.text}`,
      `Score: ${result.score_percent.toFixed(2)}% (${result.level})`,
      `Rule-based: ${(result.rule_score * 100).toFixed(2)}% | ML: ${
        result.ml_score === null ? "n/a" : (result.ml_score * 100).toFixed(2) + "%"
      }`,
      `Model: ${result.best_model}`,
      `Explanation: ${result.explanation}`,
    ].join("\n");
    await navigator.clipboard.writeText(lines);
  }

  async function exportPdf() {
    if (!result) return;
    const jsPDF = (await import("jspdf")).default;
    const doc = new jsPDF({ unit: "pt", format: "a4" });

    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.text("Speaker Confidence AI", 56, 72);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(11);
    doc.setTextColor(120);
    doc.text(
      `Generated · ${new Date().toLocaleString()}`,
      56,
      90
    );

    doc.setTextColor(30);
    doc.setFontSize(13);
    doc.text("Analysed text", 56, 130);
    doc.setFontSize(11);
    const wrap = doc.splitTextToSize(result.text, 480);
    doc.text(wrap, 56, 150);

    let y = 150 + wrap.length * 14 + 20;
    doc.setFontSize(13);
    doc.text("Results", 56, y);
    doc.setFontSize(11);
    y += 20;
    const lines = [
      `Confidence score : ${result.score_percent.toFixed(2)} %`,
      `Confidence level : ${result.level}`,
      `Rule-based score : ${(result.rule_score * 100).toFixed(2)} %`,
      `ML model score   : ${
        result.ml_score === null ? "n/a" : (result.ml_score * 100).toFixed(2) + " %"
      }`,
      `Best model       : ${result.best_model}`,
      `Hedges detected  : ${result.stats.low_count}`,
      `Boosters detected: ${result.stats.high_count}`,
    ];
    for (const ln of lines) {
      doc.text(ln, 56, y);
      y += 16;
    }

    y += 10;
    doc.setFontSize(13);
    doc.text("Explanation", 56, y);
    doc.setFontSize(11);
    const exp = doc.splitTextToSize(result.explanation, 480);
    doc.text(exp, 56, y + 18);

    doc.save(`confidence-report-${Date.now()}.pdf`);
  }

  const segments =
    result != null
      ? annotate(result.text, result.markers.low, result.markers.high)
      : [];

  return (
    <section id="analyze" className="relative py-24 noise">
      <div className="max-w-6xl mx-auto px-6 relative">
        <div className="text-center mb-10">
          <p className="text-sm uppercase tracking-[0.18em] text-apple-blue font-medium">
            Analyze
          </p>
          <h2 className="mt-2 font-display text-4xl md:text-5xl font-semibold tracking-tight">
            Paste your text. Get a score.
          </h2>
          <p className="mt-3 text-apple-gray max-w-xl mx-auto">
            Our model returns a 0–100 confidence score, an explanation, and
            highlights every hedging or booster word it finds.
          </p>
        </div>

        {/* input card */}
        <div className="glass rounded-3xl p-6 md:p-8 shadow-soft">
          {/* mode switcher */}
          <div className="mb-4 inline-flex p-1 rounded-full bg-black/5 dark:bg-white/10 text-[12px] font-medium">
            <button
              onClick={() => {
                if (listening) toggleMic();
                setMode("text");
              }}
              className={`inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full transition-all ${
                mode === "text"
                  ? "bg-white dark:bg-white/20 shadow-sm text-apple-black dark:text-white"
                  : "text-apple-gray hover:text-apple-black dark:hover:text-white"
              }`}
            >
              <Keyboard size={13} /> Type
            </button>
            <button
              onClick={() => setMode("voice")}
              className={`inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full transition-all ${
                mode === "voice"
                  ? "bg-white dark:bg-white/20 shadow-sm text-apple-black dark:text-white"
                  : "text-apple-gray hover:text-apple-black dark:hover:text-white"
              }`}
            >
              <Mic size={13} /> Speak
            </button>
          </div>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={
              mode === "voice"
                ? "Tap “Start talking” — your words will appear here as you speak…"
                : "e.g. I think this might be the correct answer, but I'm not entirely sure…"
            }
            rows={6}
            className="w-full resize-none bg-transparent outline-none text-[15px] leading-relaxed placeholder:text-apple-gray"
          />
          {listening && interim ? (
            <div className="mt-2 text-[13px] italic text-apple-gray line-clamp-1">
              <span className="text-red-500 mr-2">●</span>
              {interim}
            </div>
          ) : null}

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2 text-[12px] text-apple-gray">
              {mode === "text" ? (
                <>
                  <span>Try a sample:</span>
                  {SAMPLES.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => setText(s)}
                      className="px-3 py-1 rounded-full bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 transition-colors"
                    >
                      sample {i + 1}
                    </button>
                  ))}
                </>
              ) : (
                <span className="flex items-center gap-2">
                  {listening ? (
                    <>
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
                      </span>
                      Listening… speak clearly. Tap mic to stop.
                    </>
                  ) : micSupported ? (
                    "Tap the mic to start dictating."
                  ) : (
                    "Speech recognition needs Chrome, Edge, or Safari."
                  )}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  if (listening) toggleMic();
                  setText("");
                  finalTranscriptRef.current = "";
                }}
                className="inline-flex items-center gap-1 px-4 py-2 rounded-full text-[13px] hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              >
                <Trash2 size={14} /> Clear
              </button>
              {mode === "voice" ? (
                <button
                  onClick={toggleMic}
                  disabled={!micSupported}
                  aria-pressed={listening}
                  className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-full font-medium transition-all disabled:opacity-50 ${
                    listening
                      ? "bg-red-500 text-white hover:bg-red-600 shadow-lg shadow-red-500/25"
                      : "bg-apple-blue text-white hover:bg-apple-blueHover"
                  }`}
                >
                  {listening ? <MicOff size={15} /> : <Mic size={15} />}
                  {listening ? "Stop" : "Start talking"}
                </button>
              ) : null}
              <button
                onClick={onPredict}
                disabled={loading}
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-full bg-apple-black dark:bg-white text-white dark:text-apple-black font-medium hover:bg-black/85 dark:hover:bg-white/85 transition-all disabled:opacity-60"
              >
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={14} />}
                {loading ? "Analysing…" : "Predict"}
              </button>
            </div>
          </div>
          {error ? (
            <div className="mt-4 text-sm text-red-600 bg-red-50 dark:bg-red-500/10 dark:text-red-300 rounded-xl px-4 py-3">
              {error}
            </div>
          ) : null}
        </div>

        {/* result */}
        <AnimatePresence mode="wait">
          {result ? (
            <motion.div
              key={result.text + result.score_percent}
              ref={resultRef}
              initial={{ opacity: 0, y: 32 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -24 }}
              transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
              {/* gauge + key figures */}
              <div className="glass rounded-3xl p-6 lg:col-span-1 flex flex-col justify-between shadow-soft">
                <Gauge value={result.score_percent} level={result.level} />
                <div className="mt-10 grid grid-cols-2 gap-4 text-[12px]">
                  <KV label="Rule-based" value={`${(result.rule_score * 100).toFixed(1)}%`} />
                  <KV
                    label="ML model"
                    value={
                      result.ml_score === null
                        ? "n/a"
                        : `${(result.ml_score * 100).toFixed(1)}%`
                    }
                  />
                  <KV label="Hedges" value={String(result.stats.low_count)} />
                  <KV label="Boosters" value={String(result.stats.high_count)} />
                </div>
                <div className="mt-6 flex flex-wrap gap-2">
                  <button
                    onClick={copyResult}
                    className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-[12px] bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 transition-colors"
                  >
                    <Copy size={12} /> Copy
                  </button>
                  <button
                    onClick={exportPdf}
                    className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-[12px] bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 transition-colors"
                  >
                    <Download size={12} /> PDF
                  </button>
                </div>
              </div>

              {/* annotated text + explanation */}
              <div className="glass rounded-3xl p-6 lg:col-span-2 shadow-soft">
                <div className="flex items-center gap-2 text-[12px] uppercase tracking-wider text-apple-gray">
                  <FileText size={14} /> Annotated text
                </div>
                <p className="mt-3 text-[16px] leading-relaxed">
                  {segments.map((s, i) =>
                    s.kind === "low" ? (
                      <span
                        key={i}
                        className="bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300 rounded px-1"
                        title="low-confidence marker"
                      >
                        {s.value}
                      </span>
                    ) : s.kind === "high" ? (
                      <span
                        key={i}
                        className="bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300 rounded px-1"
                        title="high-confidence marker"
                      >
                        {s.value}
                      </span>
                    ) : (
                      <span key={i}>{s.value}</span>
                    )
                  )}
                </p>

                <div className="mt-6 flex items-center gap-2 text-[12px] uppercase tracking-wider text-apple-gray">
                  <Brain size={14} /> AI explanation
                </div>
                <p className="mt-2 text-[14px] leading-relaxed text-apple-black/90 dark:text-white/80">
                  {result.explanation}
                </p>

                <div className="mt-4 inline-flex items-center gap-2 text-[11px] text-apple-gray">
                  <Activity size={12} />
                  Best model: <span className="font-medium">{result.best_model}</span>
                </div>
              </div>

              {/* charts row */}
              <div className="glass rounded-3xl p-6 shadow-soft lg:col-span-1">
                <div className="text-[12px] uppercase tracking-wider text-apple-gray mb-2">
                  Word indicators
                </div>
                <WordIndicators
                  low={result.stats.low_count}
                  high={result.stats.high_count}
                  wordCount={result.stats.word_count}
                />
              </div>
              <div className="glass rounded-3xl p-6 shadow-soft lg:col-span-1">
                <div className="text-[12px] uppercase tracking-wider text-apple-gray mb-2">
                  Hybrid breakdown
                </div>
                <ScoreBreakdown rule={result.rule_score} ml={result.ml_score} />
              </div>
              <div className="glass rounded-3xl p-6 shadow-soft lg:col-span-1">
                <div className="text-[12px] uppercase tracking-wider text-apple-gray mb-2">
                  Per-class probability
                </div>
                {result.probabilities ? (
                  <ProbabilityChart
                    probs={result.probabilities}
                    labels={result.labels}
                  />
                ) : (
                  <div className="h-64 flex flex-col items-center justify-center text-center text-[13px] text-apple-gray px-4">
                    Train the TF-IDF classifier (<code>python model/train.py</code>) to
                    unlock per-class probabilities.
                  </div>
                )}
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {/* history */}
        <div id="history" className="mt-20">
          <div className="flex items-end justify-between mb-4">
            <div>
              <p className="text-sm uppercase tracking-[0.18em] text-apple-blue font-medium">
                History
              </p>
              <h3 className="font-display text-3xl font-semibold tracking-tight">
                Your previous predictions
              </h3>
            </div>
            {history.length > 0 ? (
              <button
                onClick={() => setHistory([])}
                className="inline-flex items-center gap-1 text-[13px] text-apple-gray hover:text-apple-black dark:hover:text-white transition-colors"
              >
                <Trash2 size={13} /> Clear
              </button>
            ) : null}
          </div>
          {history.length === 0 ? (
            <div className="glass rounded-2xl p-8 text-center text-apple-gray">
              Nothing here yet. Run your first prediction above.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {history.map((h) => (
                <div
                  key={h.id}
                  className="glass rounded-2xl p-4 flex items-start gap-4 shadow-soft cursor-pointer hover:-translate-y-0.5 transition-transform"
                  onClick={() => setText(h.text)}
                >
                  <div
                    className="shrink-0 w-10 h-10 rounded-full flex items-center justify-center font-semibold text-white text-xs"
                    style={{ background: levelToColor(h.level) }}
                  >
                    {Math.round(h.score_percent)}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-[13px] line-clamp-2">{h.text}</p>
                    <div className="mt-1 text-[11px] text-apple-gray">
                      {formatDate(h.ts)} · {h.level} · {h.best_model}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="uppercase tracking-wider text-apple-gray">{label}</div>
      <div className="mt-0.5 text-[15px] font-semibold tabular-nums">{value}</div>
    </div>
  );
}
