import Analyzer from "../components/Analyzer";
import AnalysisPage from "../components/AnalysisPage";
import Footer from "../components/Footer";
import Hero from "../components/Hero";
import HowItWorks from "../components/HowItWorks";
import Navbar from "../components/Navbar";
import Science from "../components/Science";

export default function Page() {
  return (
    <main className="min-h-screen">
      <Navbar />
      <Hero />
      <Analyzer />
      <AnalysisPage />
      <HowItWorks />
      <Science />
      <Footer />
    </main>
  );
}
