import { useMemo, useState } from 'react';

type PredictionError = { individual: string; site: string; predicted: number; actual: number };

type ModelStatsProps = {
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
	debugMode?: boolean;
	predictionErrors?: PredictionError[] | null;
};

const METRIC_LABELS: Record<string, string> = {
	accuracy: 'Accuracy',
	balanced_accuracy: 'Balanced Accuracy',
	auc_macro: 'Macro AUC',
	f1_macro: 'F1 (Macro)',
	f1_weighted: 'F1 (Weighted)'
};

const SKIP_KEYS = new Set(['model', 'dataset']);

function formatValue(v: any): string {
	if (typeof v === 'number') return (v * 100).toFixed(2) + '%';
	return String(v);
}

export default function ModelStats({ paths, testMetrics, images, debugMode = false, predictionErrors }: ModelStatsProps) {
	const [rowPageIndex, setRowPageIndex] = useState(0);
	const ROWS_PER_PAGE = 10;

	const uniqueErrorSites = useMemo(() => new Set((predictionErrors ?? []).map((e) => e.site)).size, [predictionErrors]);

	const pagedErrors = useMemo(() => {
		const list = predictionErrors ?? [];
		const start = rowPageIndex * ROWS_PER_PAGE;
		return list.slice(start, start + ROWS_PER_PAGE);
	}, [predictionErrors, rowPageIndex]);

	if (!paths && !images && !testMetrics) {
		return (
			<div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
				<h3 style={{ marginTop: 0 }}>Model Statistics</h3>
				<p style={{ opacity: 0.6 }}>No test results available. Run a test first.</p>
			</div>
		);
	}

	const metricRows = testMetrics ? Object.entries(testMetrics as Record<string, any>).filter(([k]) => !SKIP_KEYS.has(k)) : [];

	return (
		<div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
			<h3 style={{ marginTop: 0 }}>Model Statistics</h3>

			{/* Summary sentence */}
			{testMetrics && (
				<p style={{ marginBottom: '1.25rem' }}>
					Model <strong>{testMetrics.model ?? '—'}</strong> was applied to dataset <strong>{testMetrics.dataset ?? '—'}</strong>.
				</p>
			)}

			{/* Metrics table */}
			{metricRows.length > 0 && (
				<div style={{ marginBottom: '2rem' }}>
					<h4 style={{ marginBottom: '0.75rem' }}>Test Metrics</h4>
					<table
						style={{
							borderCollapse: 'collapse',
							width: '100%',
							maxWidth: '480px',
							fontSize: '0.9rem'
						}}
					>
						<thead>
							<tr>
								<th
									style={{
										textAlign: 'left',
										padding: '0.5rem 1rem',
										borderBottom: '1px solid #444',
										color: '#aaa',
										fontWeight: 'normal'
									}}
								>
									Metric
								</th>
								<th
									style={{
										textAlign: 'right',
										padding: '0.5rem 1rem',
										borderBottom: '1px solid #444',
										color: '#aaa',
										fontWeight: 'normal'
									}}
								>
									Value
								</th>
							</tr>
						</thead>
						<tbody>
							{metricRows.map(([key, value]) => (
								<tr key={key}>
									<td style={{ padding: '0.4rem 1rem', borderBottom: '1px solid #2a2a2a' }}>{METRIC_LABELS[key] ?? key}</td>
									<td
										style={{
											padding: '0.4rem 1rem',
											borderBottom: '1px solid #2a2a2a',
											textAlign: 'right',
											fontVariantNumeric: 'tabular-nums'
										}}
									>
										{formatValue(value)}
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			)}

			{/* Graphs Section */}
			{images && (images.graph_test_base64 || images.graph_cm_base64) && (
				<div style={{ marginBottom: '2rem' }}>
					<h4>Performance Graphs</h4>
					<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginTop: '1rem' }}>
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

			{/* Raw JSON dump — debug only */}
			{debugMode && testMetrics && (
				<div style={{ marginBottom: '2rem' }}>
					<h4>Raw Response</h4>
					<div style={{ backgroundColor: '#0a0a0a', padding: '1rem', borderRadius: '4px', border: '1px solid #333' }}>
						<pre style={{ margin: 0, fontSize: '0.9rem', lineHeight: '1.6' }}>{JSON.stringify(testMetrics, null, 2)}</pre>
					</div>
				</div>
			)}

			{/* Prediction Error Analysis */}
			{predictionErrors != null && (
				<div style={{ marginTop: '2rem' }}>
					<h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Prediction Error Analysis</h4>

					<p style={{ marginTop: 0, opacity: 0.8, fontSize: '0.875rem' }}>
						{predictionErrors.length} error{predictionErrors.length !== 1 ? 's' : ''} across {uniqueErrorSites} site
						{uniqueErrorSites !== 1 ? 's' : ''}
						{'  ·  '}Showing {predictionErrors.length === 0 ? 0 : rowPageIndex * ROWS_PER_PAGE + 1}–
						{Math.min((rowPageIndex + 1) * ROWS_PER_PAGE, predictionErrors.length)} of {predictionErrors.length}
					</p>

					<div style={{ marginBottom: '0.5rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
						<button
							type="button"
							onClick={() => setRowPageIndex(Math.max(0, rowPageIndex - 1))}
							disabled={rowPageIndex === 0}
							style={{ padding: '0.3rem 0.7rem', cursor: 'pointer' }}
						>
							↑ Prev
						</button>
						<button
							type="button"
							onClick={() => setRowPageIndex(Math.min(rowPageIndex + 1, Math.ceil(predictionErrors.length / ROWS_PER_PAGE) - 1))}
							disabled={(rowPageIndex + 1) * ROWS_PER_PAGE >= predictionErrors.length}
							style={{ padding: '0.3rem 0.7rem', cursor: 'pointer' }}
						>
							↓ Next
						</button>
					</div>

					<PredictionErrorTable errors={pagedErrors} />
				</div>
			)}
		</div>
	);
}

function PredictionErrorTable({ errors }: { errors: PredictionError[] }) {
	return (
		<div style={{ overflowX: 'hidden', width: '100%', boxSizing: 'border-box', minHeight: '120px' }}>
			<table style={{ borderCollapse: 'collapse', width: '100%', tableLayout: 'auto' }}>
				<thead>
					<tr>
						{['Individual', 'Site', 'Predicted', 'Actual'].map((h) => (
							<th
								key={h}
								style={{
									textAlign: 'left',
									borderBottom: '1px solid #ccc',
									padding: '0.5rem',
									whiteSpace: 'nowrap',
									fontWeight: 'bold'
								}}
							>
								{h}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{errors.map((e, ridx) => (
						<tr key={ridx}>
							<td style={{ borderBottom: '1px solid #2a2a2a', padding: '0.5rem', whiteSpace: 'nowrap' }}>{e.individual}</td>
							<td style={{ borderBottom: '1px solid #2a2a2a', padding: '0.5rem', whiteSpace: 'nowrap' }}>{e.site}</td>
							<td
								style={{
									borderBottom: '1px solid #2a2a2a',
									padding: '0.5rem',
									whiteSpace: 'nowrap',
									color: '#ff6b6b',
									fontWeight: 'bold'
								}}
							>
								{e.predicted}
							</td>
							<td
								style={{
									borderBottom: '1px solid #2a2a2a',
									padding: '0.5rem',
									whiteSpace: 'nowrap',
									color: '#00aa00',
									fontWeight: 'bold'
								}}
							>
								{e.actual}
							</td>
						</tr>
					))}
					{errors.length === 0 && (
						<tr>
							<td colSpan={4} style={{ padding: '0.5rem', opacity: 0.6 }}>
								No prediction errors.
							</td>
						</tr>
					)}
				</tbody>
			</table>
		</div>
	);
}
