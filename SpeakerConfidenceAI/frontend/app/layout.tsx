import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Speaker Confidence AI — Assess confidence from text",
  description:
    "A premium AI web app that predicts how confident a speaker sounds based only on text. Built with Next.js, FastAPI and hybrid NLP.",
  keywords: ["NLP", "Confidence Assessment", "AI", "Speech", "Text"],
  viewport: "width=device-width, initial-scale=1, maximum-scale=1",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
        <script
          // set theme before paint to avoid flash
          dangerouslySetInnerHTML={{
            __html: `
              (function(){
                try{
                  const t = localStorage.getItem('theme');
                  if(t === 'dark' || (!t && window.matchMedia('(prefers-color-scheme: dark)').matches)){
                    document.documentElement.classList.add('dark');
                    document.body && document.body.classList.add('dark');
                  }
                }catch(e){}
              })();
            `,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
