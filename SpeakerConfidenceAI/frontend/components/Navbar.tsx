"use client";

import { motion } from "framer-motion";
import { Moon, Sun, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

export default function Navbar() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const isDark = document.documentElement.classList.contains("dark");
    setDark(isDark);
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    if (next) {
      document.documentElement.classList.add("dark");
      document.body.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      document.body.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }

  return (
    <motion.nav
      initial={{ y: -30, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="fixed top-0 inset-x-0 z-50 glass"
      style={{ borderBottomWidth: 0 }}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between h-14 px-6">
        <a
          href="#top"
          className="flex items-center gap-2 font-display font-semibold tracking-tight text-[15px]"
        >
          <span className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-apple-black dark:bg-white text-white dark:text-apple-black">
            <Sparkles size={14} strokeWidth={2.4} />
          </span>
          Speaker Confidence AI
        </a>
        <div className="hidden md:flex items-center gap-8 text-[13px] text-apple-gray">
          <a href="#analyze" className="hover:text-apple-black dark:hover:text-white transition-colors">Analyze</a>
          <a href="#analysis" className="hover:text-apple-black dark:hover:text-white transition-colors">Analysis</a>
          <a href="#how" className="hover:text-apple-black dark:hover:text-white transition-colors">How it works</a>
          <a href="#science" className="hover:text-apple-black dark:hover:text-white transition-colors">Science</a>
          <a href="#history" className="hover:text-apple-black dark:hover:text-white transition-colors">History</a>
        </div>
        <button
          onClick={toggle}
          aria-label="Toggle dark mode"
          className="inline-flex items-center justify-center w-9 h-9 rounded-full hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
        >
          {dark ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </motion.nav>
  );
}
