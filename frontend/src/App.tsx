import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
	Accordion,
	AccordionDetails,
	AccordionSummary,
	Alert,
	Box,
	Button,
	CssBaseline,
	Dialog,
	DialogContent,
	DialogTitle,
	IconButton,
	Snackbar,
	Typography,
	useMediaQuery
} from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { useEffect, useMemo, useRef, useState } from 'react';
import { createAppTheme } from './assets/theme/index.js';
import DatasetDashboard from './components/DatasetDisplayDashboard.js';
import DatasetModelCreationForm from './components/DatasetModelCreationForm.js';
import DatasetSelector from './components/DatasetSelector.js';
import ModelDashboard from './components/ModelDashboard.js';
import ModelSelector from './components/ModelSelector.js';
import ModelStats from './components/ModelStats.js';
import ModelTrainer from './components/ModelTrainer.js';
import Sidebar from './components/Sidebar.js';
import { useDatasetsPoll, useModelsPoll } from './hooks/useApiPolling.js';

type Model = {
	model_name: string;
	model_type: string;
};

export default function App() {
	const API_BASE = import.meta.env.VITE_API_BASE;
	const API_KEY = import.meta.env.VITE_X_API_KEY;

	const [selectedDataset, setSelectedDataset] = useState<string>('');
	const [selectedModel, setSelectedModel] = useState<Model | null>(null);
	const [testResults, setTestResults] = useState<{
		paths: unknown;
		testMetrics: unknown;
		images: unknown;
		predictionErrors: Array<{ individual: string; site: string; predicted: number; actual: number }>;
	} | null>(null);

	function handleSelectDataset(dataset: string) {
		setSelectedDataset(dataset);
		setTestResults(null);
	}

	function handleSelectModel(model: Model | null) {
		setSelectedModel(model);
		setTestResults(null);
	}
	const [debugMode, setDebugMode] = useState(false);
	const [showCreateDatasetModal, setShowCreateDatasetModal] = useState(false);
	const [snackbarOpen, setSnackbarOpen] = useState(false);
	const [snackbarMessage, setSnackbarMessage] = useState('');
	const [panelOpen, setPanelOpen] = useState({ dataset: true, model: true, test: true });
	const testAccordionRef = useRef<HTMLDivElement>(null);
	const systemPrefersDark = useMediaQuery('(prefers-color-scheme: dark)');
	const [darkModeOverride, setDarkModeOverride] = useState<boolean | null>(null);
	const darkMode = darkModeOverride !== null ? darkModeOverride : systemPrefersDark;
	const theme = useMemo(() => createAppTheme(darkMode ? 'dark' : 'light'), [darkMode]);

	useEffect(() => {
		document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
	}, [darkMode]);

	// Use RxJS observables for automatic polling (updates every 5 seconds)
	const { datasets, error: datasetsError, isLoading: datasetsLoading, refresh: refreshDatasets } = useDatasetsPoll(API_BASE, API_KEY, 5000);
	const { models, error: modelsError, isLoading: modelsLoading } = useModelsPoll(API_BASE, API_KEY, 5000);

	async function pingBackend() {
		try {
			const r = await fetch(`${API_BASE}/api/hello`);
			const j = await r.json();
			setSnackbarMessage(`Success! Message: ${j.message}`);
			setSnackbarOpen(true);
		} catch {
			setSnackbarMessage('Error contacting backend');
			setSnackbarOpen(true);
		}
	}

	function handleDatasetCreated() {
		if (!debugMode) {
			setShowCreateDatasetModal(false);
		}
		refreshDatasets();
	}

	function handleDatasetCreationSuccess(message: string) {
		setSnackbarMessage(message);
		setSnackbarOpen(true);
	}

	function handleDatasetSelect(dataset: string) {
		setSelectedDataset(dataset);
		if (dataset) {
			setSnackbarMessage('Dataset selected — scroll down to see the Dataset Dashboard!');
			setSnackbarOpen(true);
		}
	}

	function handleModelSelect(model: Model | null) {
		setSelectedModel(model);
		if (model) {
			const msg = selectedDataset
				? 'Model selected — scroll down to see the Model Dashboard and test your model!'
				: 'Model selected — scroll down to see the Model Dashboard!';
			setSnackbarMessage(msg);
			setSnackbarOpen(true);
		}
	}

	return (
		<ThemeProvider theme={theme}>
			<CssBaseline />
			<Box
				component="a"
				href="#main-content"
				className="skip-link"
				sx={{
					position: 'absolute',
					left: '-9999px',
					zIndex: 9999,
					padding: '1rem',
					background: 'var(--color-primary)',
					'&:focus': { left: 0, top: 0 }
				}}
			>
				Skip to main content
			</Box>
			<Box sx={{ display: 'flex', height: '100vh', fontFamily: 'Arial, sans-serif', bgcolor: 'background.default' }}>
				{/* Sidebar Navigation */}
				<Sidebar
					darkMode={darkMode}
					onThemeToggle={() => setDarkModeOverride(!darkMode)}
					debugMode={debugMode}
					onDebugToggle={() => setDebugMode(!debugMode)}
					onCreateDataset={() => setShowCreateDatasetModal(true)}
				/>

				{/* Main Content */}
				<Box
					component="main"
					id="main-content"
					sx={{ flex: 1, overflowY: 'auto', padding: '2rem', display: 'flex', flexDirection: 'column' }}
				>
					{/* Debug Features - Only visible when debug mode is ON */}
					{debugMode && (
						<Box sx={{ marginBottom: '0.5rem', display: 'flex', gap: '1rem', alignItems: 'center', height: 'fit-content' }}>
							<Button
								variant="contained"
								onClick={pingBackend}
								sx={{
									flex: 1,
									padding: '0.5rem 1rem',
									whiteSpace: 'nowrap'
								}}
							>
								Ping FastAPI
							</Button>

							<Box
								sx={{
									flex: 1,
									display: 'flex',
									alignItems: 'center',
									gap: '0.5rem'
								}}
							>
								<strong className="text-accent">Datasets:</strong> {datasetsLoading ? 'Loading...' : `${datasets.length}`}
								{datasetsError && <span className="text-unknown"> - Error: {datasetsError}</span>}
							</Box>

							<Box
								sx={{
									flex: 1,
									display: 'flex',
									alignItems: 'center',
									gap: '0.5rem'
								}}
							>
								<strong className="text-accent">Models:</strong> {modelsLoading ? 'Loading...' : `${models.length}`}
								{modelsError && <span className="text-unknown"> - Error: {modelsError}</span>}
							</Box>
						</Box>
					)}

					<Box sx={{ mb: 2.5 }}>
						<Typography variant="h2" fontWeight="bold" sx={{ color: 'var(--color-primary-accent)', mb: 0.75 }}>
							Probabilistic Ancestral Inference
						</Typography>
						<Typography variant="subtitle1" sx={{ color: 'text.secondary', mb: 1 }}>
							A research capstone project exploring ancestral genotype reconstruction using <strong>Bayesian Models</strong>,{' '}
							<strong>HMMs</strong>, <strong>DNNs</strong>, and <strong>GNNs</strong>.
						</Typography>
						<p className="context-text">
							<strong>Stochastic</strong> — involving random probability and unpredictability where future states cannot be precisely
							determined. In genetics, this manifests as missing or unobservable data: ancestors whose genotypes were never sequenced
							due to cost, sample degradation, or ethical constraints.
						</p>
						<p className="context-text">
							In genetics research, it is common for ancestors to be unsequenced — grandparents or earlier relatives may be deceased,
							unavailable, or too costly to sequence. These gaps in family trees limit our ability to reconstruct inheritance patterns,
							predict hereditary traits, and model population history. If we simply ignore these gaps, our analysis becomes biased and
							our understanding of family inheritance patterns falls apart. This project tackles that gap computationally.
						</p>
						<p className="context-text">
							The system uses{' '}
							<a href="https://pubmed.ncbi.nlm.nih.gov/34897427/" target="_blank" rel="noopener noreferrer" className="text-accent">
								<strong>msprime</strong>
							</a>{' '}
							to simulate realistic diploid populations with explicit multi-generational pedigrees, where every individual's true
							genotype is known. A configurable fraction of individuals are then masked — their genotypes hidden — to simulate the
							real-world condition of absent family members. Five inference architectures are trained on the visible data and tasked
							with recovering the hidden genotypes: a <strong>Bayesian Categorical Model</strong> (PyMC MCMC with hierarchical priors),
							a <strong>Hidden Markov Model</strong> (hmmlearn, treating each individual as a sequence across genomic sites), a{' '}
							<strong>Deep Neural Network</strong> (PyTorch, with batch normalization and dropout), a{' '}
							<strong>Graph Neural Network</strong> (PyTorch Geometric, using pedigree structure as the graph), and a{' '}
							<strong>Multinomial Logistic Regression</strong> baseline (scikit-learn).
						</p>
						<p className="context-text">
							Each model predicts the <strong>allele dosage</strong> (0, 1, or 2 copies of the alternate allele) for every masked
							individual at every genomic site. Predictions are then compared against the known ground truth and evaluated using
							precision, recall, F1-score, ROC and precision-recall curves, confusion matrices, and calibration plots. The goal is not
							just to build an imputation tool, but to rigorously characterize <strong>when and why</strong> each class of model
							succeeds or breaks down as data becomes increasingly sparse.
						</p>
					</Box>

					<div className="selector-grid">
						<div className="grid-2col">
							<div>
								<span className="selector-label">Dataset</span>
								<DatasetSelector datasets={datasets} selected={selectedDataset} onSelect={handleDatasetSelect} />
							</div>
							<div>
								<span className="selector-label">Model</span>
								<ModelSelector models={models} selected={selectedModel} onSelect={handleModelSelect} />
							</div>
						</div>
					</div>

					{(selectedDataset || selectedModel) && (
						<Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', mt: 2 }}>
							<Button size="small" variant="contained" onClick={() => setPanelOpen({ dataset: true, model: true, test: true })}>
								Expand All
							</Button>
							<Button size="small" variant="contained" onClick={() => setPanelOpen({ dataset: false, model: false, test: false })}>
								Collapse All
							</Button>
						</Box>
					)}

					{selectedDataset && (
						<Accordion
							expanded={panelOpen.dataset}
							onChange={(_, expanded) => setPanelOpen((p) => ({ ...p, dataset: expanded }))}
							slotProps={{ transition: { unmountOnExit: false } }}
							className="panel-accordion"
							sx={{ mt: 2 }}
						>
							<AccordionSummary
								expandIcon={<ExpandMoreIcon />}
								sx={{ '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': { transform: 'rotate(180deg)' } }}
							>
								<Typography variant="h6" fontWeight="bold">
									<h2>Dataset Dashboard</h2>
								</Typography>
							</AccordionSummary>
							<AccordionDetails sx={{ pt: 0 }}>
								<DatasetDashboard apiBase={API_BASE} xApiKey={API_KEY} selectedDataset={selectedDataset} />
							</AccordionDetails>
						</Accordion>
					)}

					{selectedModel && (
						<Accordion
							expanded={panelOpen.model}
							onChange={(_, expanded) => setPanelOpen((p) => ({ ...p, model: expanded }))}
							slotProps={{ transition: { unmountOnExit: false } }}
							className="panel-accordion"
							sx={{ mt: 2 }}
						>
							<AccordionSummary
								expandIcon={<ExpandMoreIcon />}
								sx={{ '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': { transform: 'rotate(180deg)' } }}
							>
								<Typography fontWeight="bold">
									<h2>Model Dashboard</h2>
								</Typography>
							</AccordionSummary>
							<AccordionDetails sx={{ pt: 0 }}>
								<ModelDashboard model={selectedModel} />
							</AccordionDetails>
						</Accordion>
					)}

					{selectedDataset && selectedModel && (
						<Accordion
							ref={testAccordionRef}
							expanded={panelOpen.test}
							onChange={(_, expanded) => {
								setPanelOpen((p) => ({ ...p, test: expanded }));
								if (expanded && testAccordionRef.current) {
									setTimeout(() => {
										const rect = testAccordionRef.current?.getBoundingClientRect();
										if (rect) {
											window.scrollTo({
												top: window.scrollY + rect.top - 100,
												behavior: 'smooth'
											});
										}
									}, 500);
								}
							}}
							slotProps={{ transition: { unmountOnExit: false } }}
							className="panel-accordion"
							sx={{ mt: 2 }}
						>
							<AccordionSummary
								expandIcon={<ExpandMoreIcon />}
								sx={{ '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': { transform: 'rotate(180deg)' } }}
							>
								<Typography fontWeight="bold">
									<h2>Test Model</h2>
								</Typography>
							</AccordionSummary>
							<AccordionDetails sx={{ pt: 0 }}>
								<ModelTrainer
									apiBase={API_BASE}
									xApiKey={API_KEY}
									selectedDataset={selectedDataset}
									selectedModel={selectedModel}
									onTestComplete={setTestResults}
								/>
								<ModelStats
									paths={(testResults?.paths as any) || null}
									testMetrics={testResults?.testMetrics || null}
									images={testResults?.images || null}
									predictionErrors={testResults?.predictionErrors ?? null}
									debugMode={debugMode}
								/>
							</AccordionDetails>
						</Accordion>
					)}
					<Box
						component="footer"
						sx={{
							mt: 'auto',
							pt: 4,
							pb: 1,
							textAlign: 'center',
							borderTop: '1px solid',
							borderColor: 'divider'
						}}
					>
						<Typography variant="body2" sx={{ color: 'text.secondary' }}>
							&copy; {new Date().getFullYear()}{' '}
							<Box
								component="a"
								href="https://makennaworley.com"
								target="_blank"
								rel="noopener noreferrer"
								sx={{
									color: darkMode ? '#9d91f5' : '#452ee4',
									textDecoration: 'none',
									'&:hover': { textDecoration: 'underline' }
								}}
							>
								Makenna Worley
							</Box>
						</Typography>
					</Box>
				</Box>

				{/* Snackbar for Ping FastAPI feedback */}
				<Snackbar
					open={snackbarOpen}
					autoHideDuration={5000}
					onClose={() => setSnackbarOpen(false)}
					anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
				>
					<Alert
						onClose={() => setSnackbarOpen(false)}
						severity={snackbarMessage.includes('Error') ? 'error' : 'success'}
						sx={{ width: '100%' }}
					>
						{snackbarMessage}
					</Alert>
				</Snackbar>

				{/* Create Dataset Modal */}
				<Dialog
					open={showCreateDatasetModal}
					onClose={() => setShowCreateDatasetModal(false)}
					maxWidth="sm"
					fullWidth
					PaperProps={{
						className: 'create-dialog'
					}}
				>
					<DialogTitle className="create-dialog-title">
						Create Dataset
						<IconButton onClick={() => setShowCreateDatasetModal(false)} aria-label="Close" className="create-dialog-close">
							<CloseIcon />
						</IconButton>
					</DialogTitle>
					<DialogContent sx={{ p: 3 }}>
						<DatasetModelCreationForm
							apiBase={API_BASE}
							xApiKey={API_KEY}
							debugMode={debugMode}
							onSuccess={handleDatasetCreated}
							onSuccessNotification={handleDatasetCreationSuccess}
						/>
					</DialogContent>
				</Dialog>
			</Box>
		</ThemeProvider>
	);
}
