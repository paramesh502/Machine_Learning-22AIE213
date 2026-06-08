"use client";

import { motion } from "framer-motion";
import { ArrowDown } from "lucide-react";

const FADE = {
  hidden: { opacity: 0, y: 24 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.8, ease: [0.16, 1, 0.3, 1] },
  }),
};

export default function Hero() {
  return (
    <section id="top" className="relative overflow-hidden hero-grad">
      <div className="max-w-5xl mx-auto px-6 pt-32 pb-24 text-center">
        <motion.span
          custom={0}
          initial="hidden"
          animate="show"
          variants={FADE}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-[12px] font-medium text-apple-blue bg-white/70 dark:bg-white/10 border border-black/5 dark:border-white/10 backdrop-blur"
        >
          Hybrid NLP · TF-IDF + Transformers · Rule-based Lexicon
        </motion.span>

        <motion.h1
          custom={1}
          initial="hidden"
          animate="show"
          variants={FADE}
          className="mt-6 font-display text-5xl md:text-7xl font-semibold leading-[1.05] tracking-tight"
        >
          Speaker Confidence<br className="hidden md:block" />
          <span className="bg-gradient-to-r from-apple-blue via-[#7d5fff] to-[#ff2d55] bg-clip-text text-transparent">
            Assessment AI
          </span>
        </motion.h1>

        <motion.p
          custom={2}
          initial="hidden"
          animate="show"
          variants={FADE}
          className="mt-6 text-lg md:text-xl text-apple-gray max-w-2xl mx-auto"
        >
          Paste any sentence, transcript, or answer. Our hybrid model analyses
          lexical cues, hedges, boosters and contextual patterns to predict how
          confident the speaker sounds — instantly.
        </motion.p>

        <motion.div
          custom={3}
          initial="hidden"
          animate="show"
          variants={FADE}
          className="mt-10 flex items-center justify-center gap-3"
        >
          <a
            href="#analyze"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-apple-black dark:bg-white text-white dark:text-apple-black font-medium hover:bg-black/85 dark:hover:bg-white/85 transition-all hover:-translate-y-0.5 active:translate-y-0"
          >
            Try it now <ArrowDown size={16} />
          </a>
          <a
            href="#science"
            className="inline-flex items-center px-6 py-3 rounded-full bg-transparent border border-apple-black/20 dark:border-white/20 hover:bg-black/5 dark:hover:bg-white/10 transition-all"
          >
            See the science
          </a>
        </motion.div>

        {/* floating preview mockup */}
        <motion.div
          custom={4}
          initial="hidden"
          animate="show"
          variants={FADE}
          className="relative mt-20 mx-auto max-w-2xl"
        >
          <div className="absolute -inset-10 bg-gradient-to-br from-apple-blue/20 via-purple-400/10 to-pink-400/20 blur-3xl -z-10" />
          <div className="glass rounded-3xl px-8 py-7 text-left shadow-glass">
            <div className="flex items-center gap-1.5 mb-5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]" />
              <span className="w-2.5 h-2.5 rounded-full bg-[#febc2e]" />
              <span className="w-2.5 h-2.5 rounded-full bg-[#28c840]" />
              <span className="ml-3 text-[11px] text-apple-gray">
                speakerconfidence.ai
              </span>
            </div>
            <p className="text-[15px] leading-relaxed">
              "I <span className="bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300 px-1 rounded">think</span> the
              answer <span className="bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300 px-1 rounded">might</span> be
              correct, but I'm <span className="bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300 px-1 rounded">not
              sure</span>. It's <span className="bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300 px-1 rounded">definitely</span>
              similarity-based though."
            </p>
            <div className="mt-6 flex items-center justify-between">
              <div>
                <div className="text-xs uppercase tracking-wider text-apple-gray">confidence</div>
                <div className="text-3xl font-display font-semibold">42%</div>
              </div>
              <div className="inline-flex items-center px-3 py-1.5 rounded-full bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300 text-xs font-medium">
                Medium
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
