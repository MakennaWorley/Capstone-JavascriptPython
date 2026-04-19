import CloseIcon from '@mui/icons-material/Close';
import {
	Dialog,
	DialogContent,
	DialogTitle,
	IconButton,
	Paper,
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
	nerdMode: boolean;
	onNerdModeChange: (value: boolean) => void;
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
		'Out of 100 guesses, how many did the model get right? If it says 88%, that means 88 correct out of 100. But be careful: if one answer is way more common than others, the model might look good just by guessing that common answer every time.',
	balanced_accuracy:
		"A fairer version of accuracy. If some answers are rarer than others, this treats them equally so the model can't cheat by ignoring the rare cases. A score of 85% is solid — it means the model is actually learning to spot each type of pattern, not just memorizing the common ones.",
	auc_macro:
		"Measures how good the model is at ranking the right answer above the wrong ones. Score of 1.0 is perfect, 0.5 means it's just guessing randomly. Anything above 0.85 is good. This is a trustworthy metric because the model can't fake it by picking one answer more often.",
	f1_macro:
		"A balanced score that checks if the model is both right when it makes a guess (accuracy) and finding the cases that actually exist (coverage). It penalizes the model for missing rare cases or being wrong often. Good scores are above 0.75.",
	f1_weighted:
		"Similar to the F1 above, but accounts for the fact that some answers show up way more often than others in real data. This is what you'd see in everyday use. Compare this with F1 Macro to see if the model handles rare cases well."
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

export default function ModelStats({ paths, testMetrics, images, debugMode = false, predictionErrors, nerdMode, onNerdModeChange }: ModelStatsProps) {
	const [rowPageIndex, setRowPageIndex] = useState(0);
	const theme = useTheme();
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
					<h2 className="heading-flush">Test Metrics</h2>
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
									<TableRow key={key}>
										<TableCell sx={{ fontWeight: 500, whiteSpace: 'nowrap' }}>{METRIC_LABELS[key] ?? key}</TableCell>
										<TableCell
											align="right"
											sx={{ fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap', fontWeight: 'bold' }}
										>
											{formatValue(value)}
										</TableCell>
										<TableCell sx={{ fontSize: '0.82rem', opacity: 0.8, lineHeight: '1.6' }}>
											{nerdMode
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

			{/* Brief explanation of metrics */}
			{metricRows.length > 0 && (
				<div className="section-mb-xl">
					<h3 className="section-heading">Reading These Numbers</h3>
					{nerdMode ? (
						<>
							<p className="context-text">
								The metrics above summarize performance across the held-out test set. Accuracy is inflated under class imbalance; Balanced Accuracy and Macro AUC are more trustworthy. F1 Macro penalizes the model for missing rare dosage classes; F1 Weighted reflects real-world frequency distribution. All values are between 0 and 1, reported as percentages.
							</p>
							<p className="context-text">
								The model predicts unordered allele dosage — how many copies of the variant each individual carries. This is why there are exactly three possibilities: 0, 1, or 2 copies. If order mattered — distinguishing which allele came from which parent — there would be six permutations; but dosage ignores parentage order. Random guessing on three equiprobable classes achieves about 33% accuracy.
							</p>
							<p className="context-text">
								Our best models roughly double this baseline, reaching 60–70% accuracy. A well-performing model typically achieves ≥85% Balanced Accuracy and ≥0.85 Macro AUC. Performance degrades sharply as dataset scale increases: the signal from relatives weakens as pedigree connectivity becomes sparse. A score of 40–70% is still a success — it means the model exploits real signal from pedigree structure, not just memorising the modal class.
							</p>
						</>
					) : (
						<>
							<p className="context-text">
								These numbers measure how well the model did on people and sites it had never seen before. Accuracy tells you the percentage right, but can be misleading if some answers are way more common. Balanced Accuracy and Macro AUC are fairer — they treat all three possible answers equally.
							</p>
							<p className="context-text">
								Reconstructing someone's genetics from distant relatives is genuinely hard. The model guesses one of three possible answers: 0, 1, or 2 copies of a genetic variant. Random guessing gets about 1 out of 3 (33%) right. Our best models roughly double that, reaching 60–70% accuracy.
							</p>
							<p className="context-text">
								They struggle when the data becomes sparse, like a detective who can't solve a case with zero clues. A score of 40–70% is still a success — it means the model is finding real patterns in the pedigree, not just cheating by always guessing the most common answer.
							</p>
						</>
					)}
				</div>
			)}


			{/* Graphs Section */}
			{images && (images.graph_test_base64 || images.graph_cm_base64) && (
				<div className="section-mb-xl">
					<h1>Performance Graphs</h1>
					<div className="graphs-grid">
						{images.graph_test_base64 && (
							<div>
								<h2 className="section-heading">Test Performance</h2>
							{nerdMode ? (
									<>
										<p className="context-text">
											Side-by-side one-vs-rest curves for all three dosage classes (0 = homozygous reference, 1 = heterozygous, 2 = homozygous alt).
										</p>
										<p className="context-text">
											<strong>Left — ROC curve:</strong> True Positive Rate (sensitivity) vs False Positive Rate (1 − specificity) as the classification threshold is swept from 1 → 0. The dashed diagonal is the random-chance baseline (AUC = 0.5). The legend reports per-class AUC; the macro average of these three values is the <em>Macro AUC</em> in the metrics table.
										</p>
										<p className="context-text">
											<strong>Right — Precision-Recall curve:</strong> Precision (positive predictive value) vs Recall (sensitivity) at each threshold. More informative than ROC under severe class imbalance because it is not influenced by the large number of true negatives. A curve that stays high across the full recall range indicates the model is both accurate and thorough. A high-precision / low-recall curve means the model only predicts a class when very confident but misses many true positives.
										</p>
									</>
								) : (
									<>
										<p className="context-text">Two charts showing how well the model ranks dosage 0, 1, and 2. One chart per dosage class.</p>
										<p className="context-text"><strong>Left (ROC curve):</strong> Each line traces a trade-off — how many correct predictions the model gets (vertical) versus how many false alarms (horizontal). A line that hugs the top-left corner is ideal. The dashed diagonal line is what random guessing would look like. The AUC number (0.5 = random, 1.0 = perfect) summarizes how far the line is from that diagonal.</p>
										<p className="context-text"><strong>Right (Precision-Recall curve):</strong> Shows the balance between being precise when you speak (precision) and catching all the real cases (recall). A line that stays high across the whole width means the model is both accurate and thorough for that dosage.</p>
									</>
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
							{nerdMode ? (
									<>
										<p className="context-text">
											A 3×3 matrix where rows index the true dosage class and columns index the predicted class. Diagonal cells
											(top-left → bottom-right) are correct classifications; all off-diagonal cells are errors. Cell color
											intensity (saturation) is proportional to count, making systematic biases immediately visible.
										</p>
										<p className="context-text">
											Common failure modes: high counts in row 0 / col 1 or row 1 / col 0 indicate heterozygote confusion with homozygous reference; high
											counts in row 1 / col 2 or row 2 / col 1 indicate adjacent-dosage confusion around the heterozygous class. A
											well-calibrated model will have a strongly diagonal matrix with near-zero off-diagonal counts.
										</p>
									</>
								) : (
								<p className="context-text">
									This grid shows what the model predicted (columns) versus what was actually true (rows). Numbers along the diagonal (top-left to bottom-right) are correct predictions. Off-diagonal numbers are mistakes. The more contrast with the background (more saturated) a cell, the more predictions landed there. A good model will have very saturated cells only along the diagonal and faded cells everywhere else.
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
			<Dialog open={modalGraph !== null} onClose={() => setModalGraph(null)} maxWidth="lg" fullWidth className="graph-modal">
				<DialogTitle className="graph-modal-title">
					{modalGraph === 'test' ? 'Test Performance' : 'Confusion Matrix'}
					<IconButton onClick={() => setModalGraph(null)} aria-label="close" className="graph-modal-close">
						<CloseIcon />
					</IconButton>
				</DialogTitle>
				<DialogContent dividers>
					{modalGraph === 'test' ? (
						<>
							{nerdMode ? (
								<>
									<p className="context-text">
										Side-by-side one-vs-rest curves for all three dosage classes (0 = homozygous reference, 1 = heterozygous, 2 = homozygous alt).
									</p>
									<p className="context-text">
										<strong>Left — ROC curve:</strong> True Positive Rate (sensitivity) vs False Positive Rate (1 − specificity) as the classification threshold is swept from 1 → 0. The dashed diagonal is the random-chance baseline (AUC = 0.5). The legend reports per-class AUC; the macro average of these three values is the <em>Macro AUC</em> in the metrics table.
									</p>
									<p className="context-text">
										<strong>Right — Precision-Recall curve:</strong> Precision (positive predictive value) vs Recall (sensitivity) at each threshold. More informative than ROC under severe class imbalance because it is not influenced by the large number of true negatives. A curve that stays high across the full recall range indicates the model is both accurate and thorough. A high-precision / low-recall curve means the model only predicts a class when very confident but misses many true positives.
									</p>
								</>
							) : (
								<>
									<p className="context-text">Two charts showing how well the model ranks dosage 0, 1, and 2. One chart per dosage class.</p>
									<p className="context-text"><strong>Left (ROC curve):</strong> Each line traces a trade-off — how many correct predictions the model gets (vertical) versus how many false alarms (horizontal). A line that hugs the top-left corner is ideal. The dashed diagonal line is what random guessing would look like. The AUC number (0.5 = random, 1.0 = perfect) summarizes how far the line is from that diagonal.</p>
									<p className="context-text"><strong>Right (Precision-Recall curve):</strong> Shows the balance between being precise when you speak (precision) and catching all the real cases (recall). A line that stays high across the whole width means the model is both accurate and thorough for that dosage.</p>
								</>
							)}
							<img
								src={`data:image/png;base64,${images?.graph_test_base64}`}
								alt="Test Performance Graph"
							className="dark-mode-image-static graph-image-fullwidth"
							/>
						</>
					) : (
						<>
							{nerdMode ? (
								<>
									<p className="context-text">
										A 3×3 matrix where rows index the true dosage class and columns index the predicted class. Diagonal cells
									(top-left → bottom-right) are correct classifications; all off-diagonal cells are errors. Cell color
										intensity (saturation) is proportional to count, making systematic biases immediately visible.
									</p>
									<p className="context-text">
										Common failure modes: high counts in row 0 / col 1 or row 1 / col 0 indicate heterozygote confusion with homozygous reference; high
										counts in row 1 / col 2 or row 2 / col 1 indicate adjacent-dosage confusion around the heterozygous class. A
										well-calibrated model will have a strongly diagonal matrix with near-zero off-diagonal counts.
									</p>
								</>
							) : (
								<p className="context-text">
									This grid shows what the model predicted (columns) versus what was actually true (rows). Numbers along the diagonal (top-left to bottom-right) are correct predictions. Off-diagonal numbers are mistakes. The darker (more saturated) a cell, the more predictions landed there. A good model will have dark cells only along the diagonal and faded cells everywhere else.
								</p>
							)}
								<img src={`data:image/png;base64,${images?.graph_cm_base64}`} alt="Confusion Matrix" className="dark-mode-image-static graph-image-fullwidth" />
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
					{nerdMode ? (
						<>
							<p className="context-text">
								<strong>Methodology:</strong> Test set evaluation uses held-out individuals from the simulated population with genotypes masked at evaluation time. The model reconstructs dosage values conditioned on pedigree structure and other individuals' genotypes, effectively performing approximate Bayesian inference on the haploid space. This setup isolates the contribution of relational information (kinship coefficients) independent of allele frequency priors.
							</p>
							<p className="context-text">
								<strong>Information-Theoretic Interpretation:</strong> Model performance significantly exceeds null baseline (33% accuracy), confirming that identity-by-descent (IBD) sharing patterns contain substantial mutual information about genotypes. Accuracy degradation with dataset size reflects decreasing per-individual information density — larger populations increase sparsity in the pedigree context relative to parameter count, reducing inference capacity.
							</p>
							<p className="context-text para-flush">
								<strong>Heterozygote Classification Challenge:</strong> Dosage = 1 exhibits highest misclassification rate due to symmetric decision boundaries in latent space relative to homozygous classes. Marginal accuracy ranges from ~63% (smallest dataset) to 17–37% (larger datasets). Bayesian models achieve best heterozygote recall but suffer from exponential computational complexity; neural architectures provide better scalability via learned sufficient statistics.
							</p>
						</>
					) : (
						<>
							<p className="context-text">
								<strong>How We Tested It:</strong> We created a simulated dataset where we knew the true answers. We hid some people's genotypes (pretending they were missing family members) and asked the model to figure out what they should be based on their relatives' DNA. This setup mimics the real-world problem: reconstructing someone's genetics from distant relatives.
							</p>
							<p className="context-text">
								<strong>The Big Picture:</strong> The fact that the model performs better than random guessing shows that pedigree structure contains real, usable information. Even when many ancestors are missing, the remaining relatives provide enough clues for the model to make educated guesses. But as the dataset gets larger and more complex, there's less information per person — and accuracy drops.
							</p>
							<p className="context-text para-flush">
								<strong>What Was Hardest:</strong> The heterozygous genotype (dosage = 1) was always the toughest to predict because it looks similar to nearby genotypes. On the smallest dataset, accuracy hovered around 63%, but it fell to 17–37% on larger datasets. The Bayesian model did best but is very slow to compute.
							</p>
						</>
					)}
				</div>
			)}

			{/* Prediction Error Analysis */}
			{predictionErrors != null && (
				<div className="section-top">
					<h2 className="section-heading">Prediction Error Analysis</h2>
					{nerdMode ? (
						<>
							<p className="context-text">
								False positives and false negatives from the holdout test set. Errors are stratified across the three dosage classes — homozygous classes (0, 2) typically exhibit lower error rates due to imbalance-driven learning pressure, while heterozygous errors (dosage = 1) are systematically overrepresented. Each row reports [individual ID, genomic site, predicted dosage, true dosage].
							</p>
							<p className="context-text">
								Patterns indicate failure modes: (1) Site-level clustering suggests a genomic region with poor local pedigree context or high linkage disequilibrium that disrupts imputation; (2) Individual-level clustering indicates sparse kinship (few relatives), weak signal-to-noise ratio, or potential upstream data quality issues (e.g., Mendelian inconsistencies). Cross-reference errors against kinship matrices and site-level annotations to distinguish systematic bias from stochastic misclassification.
							</p>
						</>
					) : (
						<>
							<p className="context-text">
								This table shows every mistake the model made on the test data.
							</p>
							<p className="context-text">
								Look for patterns: if the same genomic location keeps appearing, the model consistently gets that position wrong. If the same person appears multiple times, they probably had fewer relatives to learn from.
							</p>
						</>
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
							<TableRow key={ridx}>
								<TableCell
									className="sticky-col"
									sx={{
										backgroundColor:
											theme.palette.mode === 'dark' ? theme.palette.background.paper : theme.palette.background.default
									}}
								>
									{e.individual}
								</TableCell>
								<TableCell sx={{ whiteSpace: 'nowrap' }}>{e.site}</TableCell>
								<TableCell className="cell-predicted">{e.predicted}</TableCell>
								<TableCell className="cell-actual">{e.actual}</TableCell>
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
