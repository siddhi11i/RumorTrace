const RISK_COLORS = {
  critical: '#ef4444',
  high: '#f97316',
  moderate: '#f59e0b',
  low: '#10b981',
};

export default function RiskPanel({ classification, risk, stats }) {
  if (!risk) return null;

  const color = RISK_COLORS[risk.level] || RISK_COLORS.moderate;
  const pct = Math.round(risk.score * 100);

  return (
    <div className="panel risk-panel">
      <h2>Risk Assessment</h2>
      <div className="risk-score-ring">
        <div className="risk-circle" style={{ borderColor: color }}>
          <span className="value">{pct}%</span>
          <span className="label">Risk</span>
        </div>
        <div className="risk-meta">
          <h3 style={{ color }}>{risk.level} risk</h3>
          <p>Composite score from content analysis + network amplification</p>
          {classification && (
            <span className={`classification-badge ${classification.label}`}>
              {classification.label === 'misinformation' ? '⚠' : '✓'}{' '}
              {classification.label.replace('_', ' ')} ({Math.round(classification.confidence * 100)}%)
            </span>
          )}
        </div>
      </div>

      {risk.factors && (
        <div className="factors">
          {Object.entries(risk.factors).map(([key, val]) => (
            <div key={key} className="factor-row">
              <span>{key.replace(/_/g, ' ')}</span>
              <span>{typeof val === 'number' ? val.toFixed(3) : val}</span>
            </div>
          ))}
        </div>
      )}

      {stats && (
        <div className="stats-row">
          <span className="stat-chip">Nodes: <strong>{stats.node_count}</strong></span>
          <span className="stat-chip">Edges: <strong>{stats.edge_count}</strong></span>
          <span className="stat-chip">Communities: <strong>{stats.community_count}</strong></span>
          <span className="stat-chip">Density: <strong>{stats.density}</strong></span>
        </div>
      )}
    </div>
  );
}
