"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
} from "recharts";

import { chartTheme, useIsDark } from "../lib/useTheme";

interface ProbChartProps {
  probs: Record<string, number>;
  labels: Record<string, string>;
}

export function ProbabilityChart({ probs, labels }: ProbChartProps) {
  const dark = useIsDark();
  const t = chartTheme(dark);

  const data = Object.entries(probs).map(([k, v]) => ({
    label: labels[k] ?? k,
    value: Number((v * 100).toFixed(2)),
    raw: k,
  }));

  const palette = dark
    ? ["#ff453a", "#ff7a45", "#ff9f0a", "#64d2ff", "#30d158"]
    : ["#ff3b30", "#ff6b3b", "#ff9f0a", "#8fc3f6", "#34c759"];

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 12, right: 12, left: -8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={t.grid} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: t.tick }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 11, fill: t.tick }}
            axisLine={false}
            tickLine={false}
            domain={[0, 100]}
            unit="%"
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
            formatter={(v: number) => [`${v.toFixed(2)}%`, "probability"]}
          />
          <Bar dataKey="value" radius={[8, 8, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={palette[i % palette.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface WordsProps {
  low: number;
  high: number;
  wordCount: number;
}

export function WordIndicators({ low, high, wordCount }: WordsProps) {
  const dark = useIsDark();
  const t = chartTheme(dark);

  const neutral = Math.max(wordCount - low - high, 0);
  const data = [
    { name: "Hedges", value: low, fullMark: Math.max(5, wordCount) },
    { name: "Boosters", value: high, fullMark: Math.max(5, wordCount) },
    { name: "Neutral", value: neutral, fullMark: Math.max(5, wordCount) },
    { name: "Total", value: wordCount, fullMark: Math.max(5, wordCount) },
  ];
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} outerRadius="72%">
          <PolarGrid stroke={t.grid} />
          <PolarAngleAxis dataKey="name" tick={{ fontSize: 11, fill: t.tick }} />
          <Radar
            name="lexical"
            dataKey="value"
            stroke={t.radar}
            fill={t.radar}
            fillOpacity={dark ? 0.35 : 0.25}
          />
          <Tooltip
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
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface CompareProps {
  rule: number;
  ml: number | null;
}

export function ScoreBreakdown({ rule, ml }: CompareProps) {
  const dark = useIsDark();
  const t = chartTheme(dark);

  const data = [
    {
      name: "Rule-based",
      value: Number((rule * 100).toFixed(1)),
      color: dark ? "#d08cff" : "#bf5af2",
    },
    {
      name: "ML model",
      value: ml === null ? 0 : Number((ml * 100).toFixed(1)),
      color: dark ? "#0a84ff" : "#0071e3",
    },
    {
      name: "Hybrid (final)",
      value: Number((((ml ?? rule) * 0.55 + rule * 0.45) * 100).toFixed(1)),
      color: dark ? "#30d158" : "#34c759",
    },
  ];
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 16, right: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={t.grid} />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fontSize: 11, fill: t.tick }}
            axisLine={false}
            tickLine={false}
            unit="%"
          />
          <YAxis
            dataKey="name"
            type="category"
            tick={{ fontSize: 12, fill: t.tick }}
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
            formatter={(v: number) => [`${v}%`, "score"]}
          />
          <Bar dataKey="value" radius={[0, 8, 8, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
