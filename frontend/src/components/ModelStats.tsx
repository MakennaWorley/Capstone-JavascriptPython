import { FormControlLabel, Paper, Switch, Table, TableBody, TableCell, TableContainer, TableHead, TablePagination, TableRow } from '@mui/material';
import { useTheme } from '@mui/material/styles';
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

const METRIC_DESCRIPTIONS: Record<string, string> = {
	accuracy:
		'Percentage of all genotype calls that were correct. Because most datasets are unbalanced — dosage 0 is far more common than 1 or 2 — a high accuracy can hide poor performance on rare genotypes.',
	balanced_accuracy:
		'Like accuracy, but each genotype class is weighted equally regardless of how often it appears. Since datasets are usually unbalanced, this is a more honest measure of overall performance than plain accuracy.',
	auc_macro:
		'How well the model separates each genotype class (0, 1, 2), averaged equally across all three. Unaffected by class imbalance. A score of 1.0 is perfect; 0.5 is no better than random.',
	f1_macro:
		'Balance of precision and recall, averaged equally across all classes. Because datasets are unbalanced, this penalises the model heavily if it ignores a rare genotype class entirely.',
	f1_weighted:
		'Same as F1 Macro, but classes that appear more often contribute more to the average. On unbalanced data this tends to look better than F1 Macro — use both together for a complete picture.'
};

const METRIC_DESCRIPTIONS_NERDS: Record<string, string> = {
	accuracy:
		'Standard classification accuracy = (TP + TN) / N. Reports the fraction of all SNP dosage calls that match the truth set exactly. Sensitive to class imbalance — inflated when the majority class (e.g. homozygous reference, dosage 0) dominates.',
	balanced_accuracy:
		'Mean of per-class recall (sensitivity), equivalent to macro-averaged recall. Computed as (1/K) Σ TPₖ / (TPₖ + FNₖ) across K=3 dosage classes. Corrects for imbalance by weighting each class equally regardless of support.',
	auc_macro:
		'Macro-averaged one-vs-rest AUROC. For each dosage class c ∈ {0,1,2}, computes the area under the ROC curve treating c as positive and all others as negative, then averages the three scores equally. Invariant to class priors and a reliable rank-based metric.',
	f1_macro:
		'Macro-averaged F₁ = (1/K) Σ 2·Pₖ·Rₖ / (Pₖ + Rₖ). Each class contributes equally. Stricter than accuracy under imbalance because low precision or recall on any single dosage class pulls the average down significantly.',
	f1_weighted:
		'Support-weighted F₁ = Σ (nₖ / N) · F₁ₖ. Per-class F₁ scores are averaged with weights proportional to the number of true instances of each class. Reflects real-world frequency distribution and is the closest analogue to accuracy in the F₁ family.'
};

const SKIP_KEYS = new Set(['model', 'dataset']);

function formatValue(v: any): string {
	if (typeof v === 'number') return (v * 100).toFixed(2) + '%';
	return String(v);
}

export default function ModelStats({ paths, testMetrics, images, debugMode = false, predictionErrors }: ModelStatsProps) {
	const theme = useTheme();
	const [nerdsMode, setNerdsMode] = useState(false);
	const ROWS_PER_PAGE = 10;

	const uniqueErrorSites = useMemo(() => new Set((predictionErrors ?? []).map((e) => e.site)).size, [predictionErrors]);

	if (!paths && !images && !testMetrics) {
		return (
			<div style={{ marginTop: '2rem' }}>
				<h3 style={{ marginTop: 0 }}>Model Statistics</h3>
				<p style={{ opacity: 0.7 }}>No test results available. Run a test first.</p>
			</div>
		);
	}

	const metricRows = testMetrics ? Object.entries(testMetrics as Record<string, any>).filter(([k]) => !SKIP_KEYS.has(k)) : [];

	return (
		<div style={{ marginTop: '2rem' }}>
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
					<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
						<h4 style={{ margin: 0 }}>Test Metrics</h4>
						<FormControlLabel
							control={
								<Switch
									checked={nerdsMode}
									onChange={(e) => setNerdsMode(e.target.checked)}
									size="small"
									sx={{
										'& .MuiSwitch-switchBase.Mui-checked': { color: '#452ee4' },
										'& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { backgroundColor: '#452ee4' }
									}}
								/>
							}
							label="Stats for Nerds"
							sx={{ mr: 0, '& .MuiFormControlLabel-label': { fontSize: '0.85rem', opacity: 0.8 } }}
						/>
					</div>
					<TableContainer>
						<Table size="small">
							<TableHead>
								<TableRow>
									<TableCell sx={{ fontWeight: 'normal', color: 'text.secondary' }}>Metric</TableCell>
									<TableCell align="right" sx={{ fontWeight: 'normal', color: 'text.secondary', whiteSpace: 'nowrap' }}>
										Value
									</TableCell>
									<TableCell sx={{ fontWeight: 'normal', color: 'text.secondary' }}>What it means</TableCell>
								</TableRow>
							</TableHead>
							<TableBody>
								{metricRows.map(([key, value]) => (
									<TableRow key={key} hover>
										<TableCell sx={{ fontWeight: 500, whiteSpace: 'nowrap' }}>{METRIC_LABELS[key] ?? key}</TableCell>
										<TableCell
											align="right"
											sx={{ fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap', fontWeight: 'bold' }}
										>
											{formatValue(value)}
										</TableCell>
										<TableCell sx={{ fontSize: '0.82rem', opacity: 0.8, lineHeight: '1.6' }}>
											{nerdsMode
												? (METRIC_DESCRIPTIONS_NERDS[key] ?? METRIC_DESCRIPTIONS[key] ?? '')
												: (METRIC_DESCRIPTIONS[key] ?? '')}
										</TableCell>
									</TableRow>
								))}
							</TableBody>
						</Table>
					</TableContainer>
				</div>
			)}

			{/* Graphs Section */}
			{images && (images.graph_test_base64 || images.graph_cm_base64) && (
				<div style={{ marginBottom: '2rem' }}>
					<h4>Performance Graphs</h4>
					<div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', marginTop: '1rem' }}>
						{images.graph_test_base64 && (
							<div style={{ gridColumn: 'span 2' }}>
								<h5 style={{ marginTop: 0, marginBottom: '1rem' }}>Test Performance</h5>
								<img
									src={`data:image/png;base64,${images.graph_test_base64}`}
									alt="Test Performance Graph"
									style={{
										width: '100%',
										height: 'auto',
										borderRadius: '4px',
										filter: theme.palette.mode === 'dark' ? 'invert(1) hue-rotate(180deg)' : 'none'
									}}
								/>
							</div>
						)}
						{images.graph_cm_base64 && (
							<div style={{ gridColumn: 'span 1' }}>
								<h5 style={{ marginTop: 0, marginBottom: '1rem' }}>Confusion Matrix</h5>
								<img
									src={`data:image/png;base64,${images.graph_cm_base64}`}
									alt="Confusion Matrix"
									style={{
										width: '100%',
										height: 'auto',
										borderRadius: '4px',
										filter: theme.palette.mode === 'dark' ? 'invert(1) hue-rotate(180deg)' : 'none'
									}}
								/>
							</div>
						)}
					</div>
				</div>
			)}

			{/* Raw JSON dump — debug only */}
			{debugMode && testMetrics && (
				<div style={{ marginBottom: '2rem' }}>
					<h4>Raw Response</h4>
					<div
						style={{
							padding: '1rem',
							borderRadius: '4px',
							border: `1px solid ${theme.palette.divider}`
						}}
					>
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
					</p>
					<PredictionErrorTable errors={predictionErrors} />
				</div>
			)}
		</div>
	);
}

function PredictionErrorTable({ errors }: { errors: PredictionError[] }) {
	const theme = useTheme();
	const ROWS_PER_PAGE = 100;
	const [page, setPage] = useState(0);

	const pagedErrors = errors.slice(page * ROWS_PER_PAGE, (page + 1) * ROWS_PER_PAGE);

	return (
		<>
			{errors.length > ROWS_PER_PAGE && (
				<TablePagination
					component="div"
					count={errors.length}
					page={page}
					onPageChange={(_, newPage) => setPage(newPage)}
					rowsPerPage={ROWS_PER_PAGE}
					onRowsPerPageChange={() => {}}
					rowsPerPageOptions={[]}
					labelDisplayedRows={({ from, to, count }) => `Rows ${from}–${to} of ${count}`}
				/>
			)}
			<TableContainer component={Paper} sx={{ width: '100%', maxHeight: 400, overflow: 'auto' }}>
				<Table size="small" stickyHeader sx={{ tableLayout: 'auto' }}>
					<TableHead>
						<TableRow>
							{['Individual', 'Site', 'Predicted', 'Actual'].map((h) => (
								<TableCell
									key={h}
									sx={{
										fontWeight: 'bold',
										whiteSpace: 'nowrap',
										...(h === 'Individual' && {
											position: 'sticky',
											left: 0,
											zIndex: 11
										})
									}}
								>
									{h}
								</TableCell>
							))}
						</TableRow>
					</TableHead>
					<TableBody>
						{pagedErrors.map((e, ridx) => (
							<TableRow key={ridx} hover>
								<TableCell
									sx={{
										whiteSpace: 'nowrap',
										position: 'sticky',
										left: 0,
										backgroundColor:
											theme.palette.mode === 'dark' ? theme.palette.background.paper : theme.palette.background.default,
										zIndex: 9
									}}
								>
									{e.individual}
								</TableCell>
								<TableCell sx={{ whiteSpace: 'nowrap' }}>{e.site}</TableCell>
								<TableCell
									sx={{ whiteSpace: 'nowrap', color: theme.palette.mode === 'dark' ? '#ff6b6b' : '#c62828', fontWeight: 'bold' }}
								>
									{e.predicted}
								</TableCell>
								<TableCell
									sx={{ whiteSpace: 'nowrap', color: theme.palette.mode === 'dark' ? '#00aa00' : '#2e7d32', fontWeight: 'bold' }}
								>
									{e.actual}
								</TableCell>
							</TableRow>
						))}
						{errors.length === 0 && (
							<TableRow>
								<TableCell colSpan={4} sx={{ opacity: 0.7 }}>
									No prediction errors.
								</TableCell>
							</TableRow>
						)}
					</TableBody>
				</Table>
			</TableContainer>
		</>
	);
}
