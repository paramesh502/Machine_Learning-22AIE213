"use client";

import { useEffect, useState } from "react";

/**
 * Reactively tracks whether <html> has the `.dark` class.
 * Recharts uses SVG attributes that can't resolve CSS variables, so
 * we expose the booleans here and the chart components pick colours
 * from the CSS-var palette below.
 */
export function useIsDark() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    const read = () => setDark(root.classList.contains("dark"));
    read();
    const mo = new MutationObserver(read);
    mo.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => mo.disconnect();
  }, []);

  return dark;
}

export function chartTheme(dark: boolean) {
  return {
    grid: dark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
    tick: dark ? "#9f9fa6" : "#86868b",
    cursor: dark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)",
    tooltipBg: dark ? "rgba(22,22,26,0.96)" : "#ffffff",
    tooltipText: dark ? "#f5f5f7" : "#1d1d1f",
    tooltipShadow: dark
      ? "0 10px 30px rgba(0,0,0,0.6)"
      : "0 10px 30px rgba(0,0,0,0.08)",
    tooltipBorder: dark ? "1px solid rgba(255,255,255,0.08)" : "none",
    f1Bar: dark ? "#e5e5e7" : "#1d1d1f",
    radar: dark ? "#0a84ff" : "#0071e3",
  };
}
