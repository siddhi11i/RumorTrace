import { useState } from 'react';

const SAMPLE_TEXTS = [
  {
    label: 'Hindi rumor sample',
    text: 'गुप्त सूचना! तुरंत शेयर करें — बैंक कल से बंद हो जाएगा। सभी को फॉरवर्ड करें!!!',
  },
  {
    label: 'English rumor sample',
    text: 'BREAKING URGENT: Share before deleted! Secret leaked document proves massive conspiracy. Forward immediately!!!',
  },
  {
    label: 'Factual sample',
    text: 'According to the official ministry press release, the policy update will take effect next month with verified guidelines.',
  },
];

export default function TextInput({ onAnalyze, loading }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) onAnalyze(text);
  };

  return (
    <div className="panel text-input">
      <h2>Analyze Content</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste Hindi or English text to trace rumor diffusion…"
          disabled={loading}
        />
        <button type="submit" className="btn-primary" disabled={loading || !text.trim()}>
          {loading ? 'Analyzing…' : 'Trace Rumor'}
        </button>
      </form>
      <div style={{ marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
        {SAMPLE_TEXTS.map((sample) => (
          <button
            key={sample.label}
            type="button"
            onClick={() => setText(sample.text)}
            style={{
              background: 'var(--surface-2)',
              border: '1px solid var(--border)',
              color: 'var(--muted)',
              padding: '0.35rem 0.65rem',
              borderRadius: '6px',
              fontSize: '0.75rem',
              cursor: 'pointer',
            }}
          >
            {sample.label}
          </button>
        ))}
      </div>
    </div>
  );
}
