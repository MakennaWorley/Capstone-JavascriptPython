import {
	Dialog,
	DialogContent,
	DialogTitle,
	FormControlLabel,
	IconButton,
	Paper,
	Switch,
	Table,
	TableBody,
	TableCell,
	TableContainer,
	TableHead,
	TablePagination,
	TableRow
} from '@mui/material';
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
		'Percentage of all genotype calls that were correct. Because most datasets are unbalanced — dosage 0 is far more common than 1 or 2 — a high accuracy can hide poor performance on rare genotypes. For example, 92% accuracy means 92 out of every 100 predictions matched the true genotype, but a model that just guesses "0" for everyone can score deceptively high. That\'s why accuracy alone doesn\'t tell the full story — always read it alongside Balanced Accuracy.',
	balanced_accuracy:
		"Like accuracy, but each genotype class is weighted equally regardless of how often it appears. Since datasets are usually unbalanced, this is a more honest measure of overall performance than plain accuracy. The score is calculated fairly for each class (0, 1, and 2) separately, then averaged — so the model can't look great overall by nailing the common cases while ignoring the rare ones. A score above 80% means the model is genuinely performing well across the board.",
	auc_macro:
		"How well the model separates each genotype class (0, 1, 2), averaged equally across all three. Unaffected by class imbalance. A score of 1.0 is perfect; 0.5 is no better than random. Think of it as asking: how reliably does the model rank the right answer above the wrong ones? Values above 0.90 are generally considered strong; above 0.95 is excellent. This is one of the most trustworthy numbers in the table because it doesn't reward the model for simply predicting whichever class is most common.",
	f1_macro:
		'Balance of precision and recall, averaged equally across all classes. Because datasets are unbalanced, this penalises the model heavily if it ignores a rare genotype class entirely. Precision means "when it says dosage 2, it really is dosage 2" (not crying wolf); recall means "if there really are dosage 2 individuals, it finds them" (not missing cases). If the model struggles with rare dosages, this score will drop noticeably — making it a strict but fair test of overall quality.',
	f1_weighted:
		"Same as F1 Macro, but classes that appear more often contribute more to the average. On unbalanced data this tends to look better than F1 Macro — use both together for a complete picture. It reflects how well the model handles the real-world frequency distribution, so it's a realistic picture of day-to-day performance. Use F1 Macro to check for hidden weaknesses on rare classes; use F1 Weighted to see overall performance weighted by how common each class actually is."
};

const METRIC_DESCRIPTIONS_NERDS: Record<string, string> = {
	accuracy:
		'Standard classification accuracy = (TP + TN) / N. Reports the fraction of all SNP dosage calls that match the truth set exactly. Sensitive to class imbalance — inflated when the majority class (e.g. homozygous reference, dosage 0) dominates. In pedigree imputation datasets, dosage 0 typically accounts for 70–90% of all sites, so a trivial classifier that always predicts 0 can achieve misleadingly high accuracy. Never use as a standalone metric; pair with balanced accuracy or per-class recall.',
	balanced_accuracy:
		'Mean of per-class recall (sensitivity), equivalent to macro-averaged recall. Computed as (1/K) Σ TPₖ / (TPₖ + FNₖ) across K=3 dosage classes. Corrects for imbalance by weighting each class equally regardless of support. Equivalent to the arithmetic mean of the diagonal of the normalised confusion matrix. Values below 0.60 typically indicate the model is collapsing predictions toward the majority class; values above 0.85 indicate robust multi-class discrimination.',
	auc_macro:
		'Macro-averaged one-vs-rest AUROC. For each dosage class c ∈ {0,1,2}, computes the area under the ROC curve treating c as positive and all others as negative, then averages the three scores equally. Invariant to class priors and a reliable rank-based metric. Unlike F1, does not depend on a classification threshold — it evaluates the full ranking of posterior probabilities. Particularly informative for comparing probabilistic models (e.g. Bayesian, HMM) where calibrated posterior outputs differ meaningfully between architectures.',
	f1_macro:
		'Macro-averaged F₁ = (1/K) Σ 2·Pₖ·Rₖ / (Pₖ + Rₖ). Each class contributes equally. Stricter than accuracy under imbalance because low precision or recall on any single dosage class pulls the average down significantly. Undefined per class when both TP and FP (or FP and FN) are zero — scikit-learn sets F₁ to 0.0 in that case, which can artificially deflate the macro average. Check per-class F₁ in the confusion matrix when the macro score seems unexpectedly low.',
	f1_weighted:
		'Support-weighted F₁ = Σ (nₖ / N) · F₁ₖ. Per-class F₁ scores are averaged with weights proportional to the number of true instances of each class. Reflects real-world frequency distribution and is the closest analogue to accuracy in the F₁ family. Will be dominated by the majority class (dosage 0) in typical pedigree datasets — a high F1 Weighted alongside a low F1 Macro is a reliable signal that the model is underperforming on heterozygous (dosage 1) or homozygous-alt (dosage 2) calls.'
};

const SKIP_KEYS = new Set(['model', 'dataset']);

function formatValue(v: any): string {
	if (typeof v === 'number') return (v * 100).toFixed(2) + '%';
	return String(v);
}

export default function ModelStats({ paths, testMetrics, images, debugMode = false, predictionErrors }: ModelStatsProps) {
	const [rowPageIndex, setRowPageIndex] = useState(0);
	const [nerdsMode, setNerdsMode] = useState(false);
	const [modalGraph, setModalGraph] = useState<'test' | 'cm' | null>(null);
	const ROWS_PER_PAGE = 10;

	const uniqueErrorSites = useMemo(() => new Set((predictionErrors ?? []).map((e) => e.site)).size, [predictionErrors]);

	if (!paths && !images && !testMetrics) {
		return (
			<div className="section-top">
				<h2 className="heading-flush">Model Statistics</h2>
				<p className="empty-state">No test results available. Run a test first.</p>
			</div>
		);
	}

	const metricRows = testMetrics ? Object.entries(testMetrics as Record<string, any>).filter(([k]) => !SKIP_KEYS.has(k)) : [];

	return (
		<div className="section-top">
			<h2 className="heading-flush">Model Statistics</h2>

			{/* Summary sentence */}
			{testMetrics && (
				<p className="para-intro">
					Model <strong>{testMetrics.model ?? '—'}</strong> was applied to dataset <strong>{testMetrics.dataset ?? '—'}</strong>.
				</p>
			)}

			{/* Metrics table */}
			{metricRows.length > 0 && (
				<div className="section-mb-xl">
					<div className="flex-between-mb-lg">
						<h2 className="heading-flush">Test Metrics</h2>
						<FormControlLabel
							control={
								<Switch
									checked={nerdsMode}
									onChange={(e) => setNerdsMode(e.target.checked)}
									size="small"
									className="purple-switch"
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
				<div className="section-mb-xl">
					<h1>Performance Graphs</h1>
					<div className="graphs-grid">
						{images.graph_test_base64 && (
						<div className="col-span-2">
								<h2 className="section-heading">Test Performance</h2>
								{nerdsMode ? (
									<p className="description-text">
										Side-by-side one-vs-rest curves for all three dosage classes (0 = homozygous reference, 1 = heterozygous, 2 =
										homozygous alt). <strong>Left — ROC curve:</strong> True Positive Rate (sensitivity) vs False Positive Rate (1
										− specificity) as the classification threshold is swept from 1 → 0. The dashed diagonal is the random-chance
										baseline (AUC = 0.5). The legend reports per-class AUC; the macro average of these three values is the{' '}
										<em>Macro AUC</em> in the metrics table. <strong>Right — Precision-Recall curve:</strong> Precision (positive
										predictive value) vs Recall (sensitivity) at each threshold. More informative than ROC under severe class
										imbalance because it is not influenced by the large number of true negatives. A curve that stays high across
										the full recall range indicates the model is both accurate and thorough. A high-precision / low-recall curve
										means the model only predicts a class when very confident but misses many true positives.
									</p>
								) : (
									<p className="description-text"> how well the model distinguishes between dosage 0, 1, and 2 — one line
										per class. <strong>Left (ROC curves):</strong> Each line traces the trade-off between correctly identifying a
										dosage class and accidentally mislabelling others as that type. A line hugging the top-left corner is ideal;
										the dashed diagonal means no better than a coin flip. The AUC number in the legend summarises this — 1.0 is
										perfect, 0.5 is random. <strong>Right (Precision-Recall curves):</strong> Shows the balance between only
										speaking up when confident (precision) versus catching every true case (recall). A line that stays high across
										the full width means the model is both accurate and thorough for that dosage class.
									</p>
								)}
								<img
									src={`data:image/png;base64,${images.graph_test_base64}`}
									alt="Test Performance Graph"
									onClick={() => setModalGraph('test')}
									className="dark-mode-image"
								/>
							</div>
						)}
						{images.graph_cm_base64 && (
						<div>
								<h2 className="section-heading">Confusion Matrix</h2>
								{nerdsMode ? (
									<p className="description-text">
										A 3×3 matrix where rows index the true dosage class and columns index the predicted class. Diagonal cells
										(top-left → bottom-right) are correct classifications; all off-diagonal cells are errors. Cell colour
										intensity is proportional to count, making systematic biases immediately visible. Common failure modes: high
										counts in row 0 / col 1 or row 1 / col 0 indicate heterozygote confusion with homozygous reference; high
										counts in row 1 / col 2 or row 2 / col 1 indicate adjacent-dosage confusion around the heterozygous class. A
										well-calibrated model will have a strongly diagonal matrix with near-zero off-diagonal counts.
									</p>
								) : (
									<p className="description-text">
										This grid shows every combination of what the model predicted (columns) versus what was actually true (rows).
										The numbers along the diagonal (top-left to bottom-right) are correct predictions. Anything off the diagonal
										is a mistake — for example, a number in the "True: 1, Predicted: 0" cell means the model thought someone had
										no copies of the variant when they actually had one copy. The more saturated (intensely coloured) a cell is,
										the more predictions landed there. A well-performing model will have strongly coloured cells only along the
										diagonal and near-zero (faded) cells everywhere else.
									</p>
								)}
								<img
									src={`data:image/png;base64,${images.graph_cm_base64}`}
									alt="Confusion Matrix"
									onClick={() => setModalGraph('cm')}
									className="dark-mode-image"
								/>
							</div>
						)}
					</div>
				</div>
			)}

			{/* Graph modal */}
			<Dialog open={modalGraph !== null} onClose={() => setModalGraph(null)} maxWidth="lg" fullWidth>
				<DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', pb: 1 }}>
					{modalGraph === 'test' ? 'Test Performance' : 'Confusion Matrix'}
					<IconButton onClick={() => setModalGraph(null)} size="small" aria-label="close">
						✕
					</IconButton>
				</DialogTitle>
				<DialogContent dividers>
					{modalGraph === 'test' ? (
						<>
							<p className="context-text">
								{nerdsMode ? (
									<>
										Side-by-side one-vs-rest curves for all three dosage classes (0 = homozygous reference, 1 = heterozygous, 2 =
										homozygous alt). <strong>Left — ROC curve:</strong> True Positive Rate (sensitivity) vs False Positive Rate (1
										− specificity) as the classification threshold is swept from 1 → 0. The dashed diagonal is the random-chance
										baseline (AUC = 0.5). The legend reports per-class AUC; the macro average of these three values is the{' '}
										<em>Macro AUC</em> in the metrics table. <strong>Right — Precision-Recall curve:</strong> Precision (positive
										predictive value) vs Recall (sensitivity) at each threshold. More informative than ROC under severe class
										imbalance because it is not influenced by the large number of true negatives. A curve that stays high across
										the full recall range indicates the model is both accurate and thorough. A high-precision / low-recall curve
										means the model only predicts a class when very confident but misses many true positives.
									</>
								) : (
									<>
										This chart shows two ways of measuring how well the model distinguishes between dosage 0, 1, and 2 — one line
										per class. <strong>Left (ROC curves):</strong> Each line traces the trade-off between correctly identifying a
										dosage class and accidentally mislabelling others as that type. A line hugging the top-left corner is ideal;
										the dashed diagonal means no better than a coin flip. The AUC number in the legend summarises this — 1.0 is
										perfect, 0.5 is random. <strong>Right (Precision-Recall curves):</strong> Shows the balance between only
										speaking up when confident (precision) versus catching every true case (recall). A line that stays high across
										the full width means the model is both accurate and thorough for that dosage class.
									</>
								)}
							</p>
							<img
								src={`data:image/png;base64,${images?.graph_test_base64}`}
								alt="Test Performance Graph"
								className="dark-mode-image-static"
							/>
						</>
					) : (
						<>
							<p className="context-text">
								{nerdsMode ? (
									<>
										A 3×3 matrix where rows index the true dosage class and columns index the predicted class. Diagonal cells
										(top-left → bottom-right) are correct classifications; all off-diagonal cells are errors. Cell colour
										intensity is proportional to count, making systematic biases immediately visible. Common failure modes: high
										counts in row 0 / col 1 or row 1 / col 0 indicate heterozygote confusion with homozygous reference; high
										counts in row 1 / col 2 or row 2 / col 1 indicate adjacent-dosage confusion around the heterozygous class. A
										well-calibrated model will have a strongly diagonal matrix with near-zero off-diagonal counts.
									</>
								) : (
									<>
										This grid shows every combination of what the model predicted (columns) versus what was actually true (rows).
										The numbers along the diagonal (top-left to bottom-right) are correct predictions. Anything off the diagonal
										is a mistake — for example, a number in the "True: 1, Predicted: 0" cell means the model thought someone had
										no copies of the variant when they actually had one copy. The more saturated (intensely coloured) a cell is,
										the more predictions landed there. A well-performing model will have strongly coloured cells only along the
										diagonal and near-zero (faded) cells everywhere else.
									</>
								)}
							</p>
							<img
								src={`data:image/png;base64,${images?.graph_cm_base64}`}
								alt="Confusion Matrix"
								className="dark-mode-image-static"
							/>
						</>
					)}
				</DialogContent>
			</Dialog>

			{/* Raw JSON dump — debug only */}
			{debugMode && testMetrics && (
				<div className="section-mb-xl">
					<h4>Raw Response</h4>
					<div className="debug-output">
						<pre>{JSON.stringify(testMetrics, null, 2)}</pre>
					</div>
				</div>
			)}

			{/* Context: why results look the way they do */}
			{testMetrics && (
				<div className="section-top mb-sm">
					<h2 className="section-heading">Understanding the Results</h2>
					<p className="description-text">
						<strong>The Test Lab:</strong> To evaluate whether models actually work, a dataset where the ground truth is known is required
						— something impossible with real sequencing data. msprime (Baumdicker et al., <em>Genetics</em>, 2022) was used to simulate
						entire biological histories and ancestral lineages. A fraction of individuals were then masked — their genotypes hidden — to
						simulate the real-world condition of absent family members. The model you just tested was evaluated on that held-out masked
						data.
					</p>
					<p className="description-text">
						<strong>Why isn't it 100%?</strong> Reconstructing the past is a game of probability, not certainty. While the best models are
						roughly <strong>2× more accurate than random guessing</strong>, they struggle as data becomes sparse. Just as a detective
						can't solve a case with zero clues, the models lose accuracy when relatives are too far apart in the pedigree to provide a
						clear mathematical trail.
					</p>
					<p className="description-text">
						<strong>What I Figured Out:</strong> The most important takeaway is that pedigree structure contains a real, exploitable
						signal — even when significant portions of ancestors are missing, the surrounding relatives provide enough context for models
						to reconstruct genotypes better than guessing. However, performance degrades sharply as dataset scale increases, pointing to a
						fundamental limit: the signal from relatives weakens as data gets sparser.
					</p>
					<p className="description-text para-flush">
						<strong>Obstacles:</strong> Accuracy dropped from <strong>~63% on the tiny dataset</strong> to as low as{' '}
						<strong>~17% (HMM)</strong> and <strong>~37% (DNN, logistic regression)</strong> on the medium dataset. The hardest case to
						predict was always the <strong>heterozygous genotype</strong> (dosage&nbsp;=&nbsp;1), because it is the most ambiguous under
						Mendelian inheritance. The Bayesian model achieved the highest accuracy but uses MCMC sampling, making it computationally
						expensive to scale — results at small and medium dataset sizes are still pending.
					</p>
				</div>
			)}

			{/* Prediction Error Analysis */}
			{predictionErrors != null && (
				<div className="section-top">
					<h2 className="section-heading">Prediction Error Analysis</h2>
					{nerdsMode ? (
						<p className="description-text">
							Exhaustive list of every site × individual pair where the model's argmax prediction did not match the ground-truth dosage
							label. Each row shows the individual ID, the genomic site identifier, the predicted dosage class (0, 1, or 2), and the
							true dosage. Errors are recorded after Phase 3 evaluation on the held-out test split only — training and validation errors
							are not included. Use this table to identify systematic failure patterns: e.g. if most errors cluster on a specific site,
							that site may have low feature signal or high missingness in the relative neighbourhood; if errors concentrate on a
							specific individual, that individual may have an unusually sparse pedigree context. Total error count and unique site
							count are shown below.
						</p>
					) : (
						<p className="description-text">
							This table lists every single genotype call where the model got it wrong — the individual, the genomic site, what the
							model predicted, and what the true answer actually was. It only covers the final held-out test group, so these are
							mistakes the model made on people it had never seen before. Scrolling through this table can help spot patterns: for
							example, if the same site keeps appearing, the model may consistently struggle to infer that particular position from the
							available relatives. If the same individual appears repeatedly, they may have few close relatives in the dataset, leaving
							the model with less information to work from.
						</p>
					)}
					<p className="context-text">
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
									className="sticky-col"
									sx={{
										backgroundColor:
											theme.palette.mode === 'dark' ? theme.palette.background.paper : theme.palette.background.default,
									}}
								>
									{e.individual}
								</TableCell>
								<TableCell sx={{ whiteSpace: 'nowrap' }}>{e.site}</TableCell>
								<TableCell className="cell-predicted">
									{e.predicted}
								</TableCell>
								<TableCell className="cell-actual">
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
