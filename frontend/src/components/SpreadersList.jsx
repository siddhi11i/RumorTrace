export default function SpreadersList({ spreaders }) {
  if (!spreaders?.length) return null;

  return (
    <div className="panel">
      <h2>Top Spreader Nodes</h2>
      <p style={{ color: 'var(--muted)', fontSize: '0.8rem', marginBottom: '0.75rem' }}>
        Ranked by simulated GNN score (NetworkX centrality)
      </p>
      <ul className="spreaders-list">
        {spreaders.map((s, i) => (
          <li key={s.id} className="spreader-item">
            <div>
              <span className="name">
                #{i + 1} {s.label}
                {s.is_bot && <span className="bot-tag">BOT</span>}
              </span>
              <div className="meta">{s.platform} · {s.followers.toLocaleString()} followers</div>
            </div>
            <span className="score">{s.gnn_score.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
