export default function Footer() {
  return (
    <footer className="py-10 border-t border-black/5 dark:border-white/10">
      <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4 text-[12px] text-apple-gray">
        <div>
          © {new Date().getFullYear()} Speaker Confidence AI · Built with Next.js,
          FastAPI & scikit-learn.
        </div>
        <div className="flex items-center gap-5">
          <a href="#top" className="hover:text-apple-black dark:hover:text-white transition-colors">Top</a>
          <a href="#analyze" className="hover:text-apple-black dark:hover:text-white transition-colors">Analyze</a>
          <a href="#science" className="hover:text-apple-black dark:hover:text-white transition-colors">Research</a>
        </div>
      </div>
    </footer>
  );
}
