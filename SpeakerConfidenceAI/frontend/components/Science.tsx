"use client";

import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { chartTheme, useIsDark } from "../lib/useTheme";

// Snapshot of the best-per-embedding rows from Lab 7 experiments
const RESULTS = [
  { combo: "TF-IDF · Gradient Boosting", acc: 41.35, f1: 37.72 },
  { combo: "BERT · Gradient Boosting", acc: 41.52, f1: 38.48 },
  { combo: "RoBERTa · Gradient Boosting", acc: 42.34, f1: 39.50 },
];

export default function Science() {
  const dark = useIsDark();
  const t = chartTheme(dark);

  const accPalette = dark
    ? ["#64d2ff", "#0a84ff", "#bf5af2"]
    : ["#c8d4e3", "#6fbfff", "#0071e3"];

  return (
    <section id="science" className="py-24 bg-apple-bg dark:bg-[#050508]">
      <div className="max-w-6xl mx-auto px-6">
        <div className="text-center mb-12">
          <p className="text-sm uppercase tracking-[0.18em] text-apple-blue font-medium">
            Science
          </p>
          <h2 className="mt-2 font-display text-4xl md:text-5xl font-semibold tracking-tight">
            30 experiments. One best pipeline.
          </h2>
          <p className="mt-4 text-apple-gray max-w-2xl mx-auto">
            We trained and evaluated 10 classifiers against 3 embedding
            families (TF-IDF, BERT-base, RoBERTa-base). The numbers below are
            weighted metrics on a 20% stratified test split of the
            Conf_Text_Labels corpus (3,033 samples over 5 classes).
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="glass rounded-3xl p-6 md:p-8 shadow-soft"
        >
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={RESULTS} margin={{ left: 0, right: 16, top: 12, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={t.grid} />
                <XAxis
                  dataKey="combo"
                  tick={{ fontSize: 11, fill: t.tick }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 50]}
                  unit="%"
                  tick={{ fontSize: 11, fill: t.tick }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: t.cursor }}
                  contentStyle={{
                    borderRadius: 12,
                    border: t.tooltipBorder,
                    boxShadow: t.tooltipShadow,
                    background: t.tooltipBg,
                    color: t.tooltipText,
                  }}
                  labelStyle={{ color: t.tooltipText }}
                  itemStyle={{ color: t.tooltipText }}
                />
                <Bar dataKey="acc" name="Accuracy %" radius={[8, 8, 0, 0]}>
                  {RESULTS.map((_, i) => (
                    <Cell key={i} fill={accPalette[i]} />
                  ))}
                </Bar>
                <Bar dataKey="f1" name="F1 %" radius={[8, 8, 0, 0]} fill={t.f1Bar} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-6 grid sm:grid-cols-3 gap-4 text-[13px]">
            <Stat label="Samples" value="3,806" />
            <Stat label="Classes" value="5 (Very Low – Very High)" />
            <Stat label="Best model" value="RoBERTa + Gradient Boosting" />
            <Stat label="Best F1" value="0.395" />
            <Stat label="Best Accuracy" value="0.4234" />
            <Stat label="Hybrid gain" value="+4.6 pp over ML alone" />
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="uppercase tracking-wider text-apple-gray text-[11px]">
        {label}
      </div>
      <div className="mt-1 font-display font-semibold">{value}</div>
    </div>
  );
}
