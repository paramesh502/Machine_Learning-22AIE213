"use client";

import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useEffect } from "react";

interface Props {
  value: number; // 0 – 100
  level: "Low" | "Medium" | "High";
}

const LEVEL_COLOR: Record<Props["level"], string> = {
  Low: "#ff3b30",
  Medium: "#ff9f0a",
  High: "#34c759",
};

export default function Gauge({ value, level }: Props) {
  const size = 260;
  const stroke = 16;
  const r = (size - stroke) / 2;
  const C = 2 * Math.PI * r;
  // only draw the top 3/4 of the circle (pleasant gauge shape)
  const arc = C * 0.75;

  const mv = useMotionValue(0);
  const spring = useSpring(mv, { stiffness: 60, damping: 18 });
  useEffect(() => {
    mv.set(value);
  }, [value, mv]);

  const dashOffset = useTransform(spring, (v: number) => arc - (arc * v) / 100);
  const text = useTransform(spring, (v: number) => v.toFixed(1) + "%");
  const color = LEVEL_COLOR[level];

  return (
    <div className="relative w-full flex flex-col items-center">
      <svg
        width={size}
        height={size * 0.78}
        viewBox={`0 0 ${size} ${size}`}
        className="-mb-10 overflow-visible"
      >
        <g transform={`rotate(135 ${size / 2} ${size / 2})`}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            style={{ stroke: "var(--gauge-track)" }}
            strokeWidth={stroke}
            strokeDasharray={`${arc} ${C}`}
            strokeLinecap="round"
          />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={`${arc} ${C}`}
            style={{ strokeDashoffset: dashOffset }}
            strokeLinecap="round"
            initial={{ strokeDashoffset: arc }}
          />
        </g>
      </svg>

      <div className="-mt-28 flex flex-col items-center">
        <motion.div
          className="font-display font-semibold tabular-nums leading-none tracking-tight text-[44px] md:text-[48px]"
          style={{ color }}
        >
          <motion.span>{text}</motion.span>
        </motion.div>
        <div className="mt-2 text-[11px] uppercase tracking-[0.18em] text-apple-gray">
          Confidence · <span style={{ color }}>{level}</span>
        </div>
      </div>
    </div>
  );
}
