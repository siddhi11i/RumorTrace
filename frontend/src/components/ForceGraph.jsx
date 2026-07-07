import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export default function ForceGraph({ graph, loading }) {
  const svgRef = useRef(null);
  const simulationRef = useRef(null);

  useEffect(() => {
    if (!graph?.nodes?.length || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth || 800;
    const height = 520;

    const nodes = graph.nodes.map((n) => ({ ...n }));
    const links = graph.links.map((l) => ({ ...l }));

    const g = svg.append('g');

    const zoom = d3.zoom()
      .scaleExtent([0.3, 4])
      .on('zoom', (event) => g.attr('transform', event.transform));

    svg.call(zoom);

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d) => d.id).distance(60).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-180))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(14));

    simulationRef.current = simulation;

    const link = g.append('g')
      .attr('stroke', '#2a3344')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', (d) => Math.sqrt(d.weight || 1) * 1.5);

    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (d) => 6 + (d.gnn_score || 0) * 10)
      .attr('fill', (d) => d.color || '#6366f1')
      .attr('stroke', (d) => (d.is_bot ? '#ef4444' : '#fff'))
      .attr('stroke-width', (d) => (d.is_bot ? 2 : 0.5))
      .attr('stroke-opacity', 0.6)
      .style('cursor', 'grab')
      .call(drag(simulation));

    node.append('title').text(
      (d) => `${d.label}\nPlatform: ${d.platform}\nGNN score: ${(d.gnn_score || 0).toFixed(3)}\nCommunity: ${d.community}`
    );

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);

      node
        .attr('cx', (d) => d.x)
        .attr('cy', (d) => d.y);
    });

    return () => {
      simulation.stop();
    };
  }, [graph]);

  if (!graph && !loading) {
    return (
      <div className="panel graph-container">
        <h2>Diffusion Graph</h2>
        <div className="empty-state">
          Submit text to generate a synthetic rumor diffusion network
        </div>
      </div>
    );
  }

  const communities = [...new Set((graph?.nodes || []).map((n) => n.community))];

  return (
    <div className="panel graph-container">
      <h2>Diffusion Graph (Louvain Communities)</h2>
      <svg ref={svgRef} />
      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
        </div>
      )}
      {graph && (
        <div className="graph-legend">
          <span>Node size = GNN centrality score</span>
          <span>Red border = suspected bot</span>
          {communities.slice(0, 6).map((c) => {
            const node = graph.nodes.find((n) => n.community === c);
            return (
              <span key={c} className="legend-item">
                <span className="legend-dot" style={{ background: node?.color }} />
                Community {c}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function drag(simulation) {
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended);
}
