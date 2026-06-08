"use client";

import { motion } from "framer-motion";
import { Brain, Database, Layers, Zap } from "lucide-react";

const STEPS = [
  {
    icon: Database,
    title: "1. Clean & vectorise",
    body: "Text is lower-cased, URLs and punctuation stripped, and converted to a TF-IDF representation with 1–2 grams. Stop-words filtered for signal.",
  },
  {
    icon: Brain,
    title: "2. Hybrid model",
    body: "A Gradient-Boosting / Random-Forest / Stacking ensemble trained on 3,806 human-labelled utterances produces a per-class distribution over five confidence levels.",
  },
  {
    icon: Layers,
    title: "3. Lexical overlay",
    body: "A curated lexicon of 60+ hedges and boosters scores the same text. Negation-aware, phrase-level matching — 'I think' and 'not sure' count twice.",
  },
  {
    icon: Zap,
    title: "4. Hybrid score",
    body: "Final = 0.55 · ML + 0.45 · Rule, squashed to a 0–100 scale and labelled Low · Medium · High. This fusion beat either signal alone in offline tests.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="py-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="text-center mb-16">
          <p className="text-sm uppercase tracking-[0.18em] text-apple-blue font-medium">
            How it works
          </p>
          <h2 className="mt-2 font-display text-4xl md:text-5xl font-semibold tracking-tight">
            Two signals, one score
          </h2>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
          {STEPS.map((s, i) => (
            <motion.div
              key={s.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              className="glass rounded-3xl p-6 shadow-soft"
            >
              <div className="w-10 h-10 rounded-2xl bg-apple-black dark:bg-white text-white dark:text-apple-black flex items-center justify-center">
                <s.icon size={16} />
              </div>
              <h3 className="mt-5 font-display text-lg font-semibold">{s.title}</h3>
              <p className="mt-2 text-[13.5px] leading-relaxed text-apple-gray">
                {s.body}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
