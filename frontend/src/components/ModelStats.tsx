type ModelStatsProps = {
	log: string | null;
	paths: {
		graph_test: string;
		graph_cm: string;
		model_dir: string;
	} | null;
	testMetrics: any;
	images: {
		graph_test_base64?: string;
		graph_cm_base64?: string;
	} | null;
};

export default function ModelStats({ log, paths, testMetrics, images }: ModelStatsProps) {
	if (!log && !paths && !images) {
		return (
			<div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
				<h3 style={{ marginTop: 0 }}>Model Statistics</h3>
				<p style={{ opacity: 0.6 }}>No test results available. Run a test first.</p>
			</div>
		);
	}

	return (
		<div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
			<h3 style={{ marginTop: 0 }}>Model Statistics</h3>

			{/* Graphs Section */}
			{images && (images.graph_test_base64 || images.graph_cm_base64) && (
				<div style={{ marginBottom: '2rem' }}>
					<h4>Performance Graphs</h4>
					<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginTop: '1rem' }}>
						{/* Test Graph */}
						{images.graph_test_base64 && (
							<div style={{ backgroundColor: '#0a0a0a', padding: '1rem', borderRadius: '8px', border: '1px solid #333' }}>
								<h5 style={{ marginTop: 0, marginBottom: '1rem' }}>Test Performance</h5>
								<img
									src={`data:image/png;base64,${images.graph_test_base64}`}
									alt="Test Performance Graph"
									style={{ width: '100%', height: 'auto', borderRadius: '4px' }}
								/>
								{paths?.graph_test && (
									<p style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem', wordBreak: 'break-all' }}>
										{paths.graph_test}
									</p>
								)}
							</div>
						)}

						{/* Confusion Matrix */}
						{images.graph_cm_base64 && (
							<div style={{ backgroundColor: '#0a0a0a', padding: '1rem', borderRadius: '8px', border: '1px solid #333' }}>
								<h5 style={{ marginTop: 0, marginBottom: '1rem' }}>Confusion Matrix</h5>
								<img
									src={`data:image/png;base64,${images.graph_cm_base64}`}
									alt="Confusion Matrix"
									style={{ width: '100%', height: 'auto', borderRadius: '4px' }}
								/>
								{paths?.graph_cm && (
									<p style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem', wordBreak: 'break-all' }}>{paths.graph_cm}</p>
								)}
							</div>
						)}
					</div>
				</div>
			)}

			{/* Test Metrics Summary */}
			{testMetrics && (
				<div style={{ marginBottom: '2rem' }}>
					<h4>Test Metrics</h4>
					<div
						style={{
							backgroundColor: '#0a0a0a',
							padding: '1rem',
							borderRadius: '4px',
							border: '1px solid #333'
						}}
					>
						<pre style={{ margin: 0, fontSize: '0.9rem', lineHeight: '1.6' }}>{JSON.stringify(testMetrics, null, 2)}</pre>
					</div>
				</div>
			)}

			{/* Log Section */}
			{log && (
				<div>
					<h4>Detailed Log</h4>
					<pre
						style={{
							backgroundColor: '#0a0a0a',
							padding: '1rem',
							borderRadius: '4px',
							overflowX: 'auto',
							fontSize: '0.85rem',
							lineHeight: '1.5',
							border: '1px solid #333',
							maxHeight: '500px',
							overflowY: 'auto',
							margin: 0
						}}
					>
						{log}
					</pre>
				</div>
			)}
		</div>
	);
}
