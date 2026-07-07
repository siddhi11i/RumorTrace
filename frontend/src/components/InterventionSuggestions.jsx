export default function InterventionSuggestions({ interventions }) {
  if (!interventions?.length) return null;

  return (
    <div className="panel">
      <h2>Intervention Suggestions</h2>
      <ul className="interventions-list">
        {interventions.map((item, i) => (
          <li key={i} className={`intervention-item priority-${item.priority}`}>
            <span className="priority-tag">{item.priority} priority</span>
            <h4>{item.action}</h4>
            <p>{item.detail}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
