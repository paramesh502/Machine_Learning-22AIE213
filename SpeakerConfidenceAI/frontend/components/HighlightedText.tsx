"use client";

import { motion } from "framer-motion";
import { annotate } from "@/lib/utils";
import type { Marker } from "@/lib/api";

interface Props {
  text: string;
  low: Marker[];
  high: Marker[];
}

export default function HighlightedText({ text, low, high }: Props) {
  const segs = annotate(text, low, high);
  return (
    <div className="text-[15px] leading-relaxed whitespace-pre-wrap">
      {segs.map((s, i) => {
        if (s.kind === "plain") return <span key={i}>{s.value}</span>;
        const cls =
          s.kind === "low"
            ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200"
            : "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-200";
        return (
          <motion.span
            key={i}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: i * 0.02 }}
            className={`inline-block px-1 py-0.5 rounded-md mx-0.5 ${cls}`}
            title={
              s.kind === "low" ? "Hedging / low-confidence marker" : "Booster / high-confidence marker"
            }
          >
            {s.value}
          </motion.span>
        );
      })}
    </div>
  );
}
