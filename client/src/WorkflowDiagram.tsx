const stages = ['Exploration', 'Preprocessing', 'Features', 'Models', 'Post processing', 'Done']

export default function WorkflowDiagram({ iterative = false }: { iterative?: boolean }) {
  const id = iterative ? 'iterative-workflow' : 'linear-workflow'
  const y = iterative ? 155 : 20
  return <figure className="workflow-diagram">
    <figcaption>{iterative ? 'What the workflow actually looked like' : 'What I wish the workflow looked like'}</figcaption>
    <div className="workflow-scroll">
      <svg viewBox={`0 0 1080 ${iterative ? 350 : 100}`} role="img" aria-labelledby={`${id}-title ${id}-description`}>
        <title id={`${id}-title`}>{iterative ? 'An iterative workflow with returns to earlier steps' : 'A linear workflow from exploration to completion'}</title>
        <desc id={`${id}-description`}>{iterative
          ? 'Exploration, preprocessing, features, preprocessing, features, models, features, models, preprocessing, post processing, models, done.'
          : 'Exploration, preprocessing, features, models, post processing, done.'}</desc>
        <defs><marker id={`${id}-arrow`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
        <g fill="none" stroke="#628573" strokeWidth="2" markerEnd={`url(#${id}-arrow)`}>
          {(iterative ? [0, 1, 2] : [0, 1, 2, 3, 4]).map(i => <path key={i} d={`M ${170 + i * 180} ${y + 28} H ${198 + i * 180}`} />)}
          {iterative && <>
            <path d="M 455 155 V 120 Q 455 110 445 110 H 285 Q 275 110 275 120 V 153" />
            <path d="M 635 211 V 250 Q 635 260 625 260 H 465 Q 455 260 455 250 V 213" />
            <path d="M 615 155 V 80 Q 615 70 605 70 H 265 Q 255 70 255 80 V 153" />
            <path d="M 295 211 V 295 Q 295 305 305 305 H 785 Q 795 305 795 295 V 213" />
            <path d="M 740 183 H 712" />
            <path d="M 655 155 V 40 Q 655 30 665 30 H 985 Q 995 30 995 40 V 153" />
          </>}
        </g>
        {iterative && <g className="workflow-label" textAnchor="middle">
          <text x="365" y="100">Revisit preprocessing</text>
          <text x="435" y="60">Adjust preprocessing again</text>
          <text x="545" y="248">Revisit features</text>
          <text x="545" y="330">Move on to post processing</text>
          <text x="815" y="20">Final model and conclusion</text>
          <text x="725" y="232">Retest</text>
        </g>}
        {stages.map((stage, i) => <g key={stage}>
          <rect x={20 + i * 180} y={y} width="150" height="56" rx="8" fill="#f6f9f7" stroke="#b8cec1" />
          <text x={95 + i * 180} y={y + 33} textAnchor="middle" className="workflow-stage">{stage}</text>
        </g>)}
      </svg>
    </div>
  </figure>
}
