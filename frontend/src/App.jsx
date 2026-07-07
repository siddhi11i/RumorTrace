import { useState } from 'react';
import { analyzeText } from './api';
import TextInput from './components/TextInput';
import RiskPanel from './components/RiskPanel';
import ForceGraph from './components/ForceGraph';
import SpreadersList from './components/SpreadersList';
import InterventionSuggestions from './components/InterventionSuggestions';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleAnalyze = async (text) => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeText(text);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>RumorTrace</h1>
        <p>
          Trace misinformation diffusion with Indic-BERT classification,
          NetworkX graph simulation, and D3 force visualization
        </p>
      </header>

      <div className="layout">
        <aside>
          <TextInput onAnalyze={handleAnalyze} loading={loading} />
          {error && <div className="error-banner">{error}</div>}
          <RiskPanel
            classification={result?.classification}
            risk={result?.risk}
            stats={result?.stats}
          />
        </aside>

        <main className="main-content">
          <ForceGraph graph={result?.graph} loading={loading} />
          {result && (
            <div className="bottom-grid">
              <SpreadersList spreaders={result.top_spreaders} />
              <InterventionSuggestions interventions={result.interventions} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
