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
import { useMemo, useState } from 'react';
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

	const [msg, setMsg] = useState<string | null>(null);
	const [status, setStatus] = useState('');
	const [selectedDataset, setSelectedDataset] = useState<string>('');
	const [selectedModel, setSelectedModel] = useState<Model | null>(null);
	const [testResults, setTestResults] = useState<{
		paths: unknown;
		testMetrics: unknown;
		images: unknown;
		predictionErrors: Array<{ individual: string; site: string; predicted: number; actual: number }>;
	} | null>(null);
	const [debugMode, setDebugMode] = useState(false);
	const [showCreateDatasetModal, setShowCreateDatasetModal] = useState(false);
	const [snackbarOpen, setSnackbarOpen] = useState(false);
	const [snackbarMessage, setSnackbarMessage] = useState('');
	const [panelOpen, setPanelOpen] = useState({ dataset: true, model: true, test: true });
	const systemPrefersDark = useMediaQuery('(prefers-color-scheme: dark)');
	const [darkModeOverride, setDarkModeOverride] = useState<boolean | null>(null);
	const darkMode = darkModeOverride !== null ? darkModeOverride : systemPrefersDark;
	const theme = useMemo(() => createAppTheme(darkMode ? 'dark' : 'light'), [darkMode]);

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

	return (
		<ThemeProvider theme={theme}>
			<CssBaseline />
			<Box
				component="a"
				href="#main-content"
				sx={{
					position: 'absolute',
					left: '-9999px',
					zIndex: 9999,
					padding: '1rem',
					background: '#452ee4',
					color: '#fff',
					textDecoration: 'none',
					fontWeight: 'bold',
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
				<Box component="main" id="main-content" sx={{ flex: 1, overflowY: 'auto', padding: '2rem' }}>
					{/* Debug Features - Only visible when debug mode is ON */}
					{debugMode && (
						<Box sx={{ marginBottom: '0.5rem', display: 'flex', gap: '1rem', alignItems: 'center', height: 'fit-content' }}>
							<Button
								variant="contained"
								onClick={pingBackend}
								sx={{
									backgroundColor: '#452ee4',
									'&:hover': { backgroundColor: '#241291' },
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
								<strong style={{ color: darkMode ? '#7c6bf0' : '#452ee4' }}>Datasets:</strong>{' '}
								{datasetsLoading ? 'Loading...' : `${datasets.length}`}
								{datasetsError && <span style={{ color: darkMode ? '#ff6b6b' : '#c62828' }}> - Error: {datasetsError}</span>}
							</Box>

							<Box
								sx={{
									flex: 1,
									display: 'flex',
									alignItems: 'center',
									gap: '0.5rem'
								}}
							>
								<strong style={{ color: darkMode ? '#7c6bf0' : '#452ee4' }}>Models:</strong>{' '}
								{modelsLoading ? 'Loading...' : `${models.length}`}
								{modelsError && <span style={{ color: darkMode ? '#ff6b6b' : '#c62828' }}> - Error: {modelsError}</span>}
							</Box>
						</Box>
					)}

					<Box sx={{ mb: 2.5 }}>
						<Typography variant="h2" fontWeight="bold" sx={{ color: darkMode ? '#9d91f5' : '#452ee4', mb: 0.75 }}>
							Probabilistic Ancestral Inference
						</Typography>
						<Typography variant="subtitle1" sx={{ color: 'text.secondary', mb: 1 }}>
							A research capstone project exploring ancestral genotype reconstruction using <strong>Bayesian Models</strong>, <strong>HMMs</strong>, <strong>DNNs</strong>, and <strong>GNNs</strong>.
						</Typography>
						<p style={{ margin: '0.5rem 0 0', color: 'inherit', opacity: 0.75, fontSize: '0.95rem', lineHeight: 1.7 }}>
							In genetics research, it is common for ancestors to be unsequenced — grandparents or earlier relatives may be deceased,
							unavailable, or too costly to sequence. Without that data, downstream analyses like disease risk prediction, inheritance
							tracing, and population history reconstruction become incomplete or impossible. This project tackles that gap computationally.
						</p>
						<p style={{ margin: '0.75rem 0 0', color: 'inherit', opacity: 0.75, fontSize: '0.95rem', lineHeight: 1.7 }}>
							The system uses{' '}
						<a href="https://pubmed.ncbi.nlm.nih.gov/34897427/" target="_blank" rel="noopener noreferrer" style={{ color: darkMode ? '#9d91f5' : '#452ee4' }}>
							<strong>msprime</strong>
						</a>{' '}
						to simulate realistic diploid populations with explicit multi-generational pedigrees,
							where every individual's true genotype is known. A configurable fraction of individuals are then masked — their genotypes
							hidden — to simulate the real-world condition of absent family members. Five inference architectures are trained on the
							visible data and tasked with recovering the hidden genotypes: a <strong>Bayesian Categorical Model</strong> (PyMC MCMC with
							hierarchical priors), a <strong>Hidden Markov Model</strong> (hmmlearn, treating each individual as a sequence across
							genomic sites), a <strong>Deep Neural Network</strong> (PyTorch, with batch normalization and dropout), a{' '}
							<strong>Graph Neural Network</strong> (PyTorch Geometric, using pedigree structure as the graph), and a{' '}
							<strong>Multinomial Logistic Regression</strong> baseline (scikit-learn).
						</p>
						<p style={{ margin: '0.75rem 0 0', color: 'inherit', opacity: 0.75, fontSize: '0.95rem', lineHeight: 1.7 }}>
							Each model predicts the <strong>allele dosage</strong> (0, 1, or 2 copies of the alternate allele) for every masked
							individual at every genomic site. Predictions are then compared against the known ground truth and evaluated using
							precision, recall, F1-score, ROC and precision-recall curves, confusion matrices, and calibration plots. The goal is not
							just to build an imputation tool, but to rigorously characterize <strong>when and why</strong> each class of model
							succeeds or breaks down as data becomes increasingly sparse.
						</p>
					</Box>

					<div style={{ marginTop: '1.25rem' }}>
						<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
							<div>
								<span style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Dataset</span>
								<DatasetSelector datasets={datasets} selected={selectedDataset} onSelect={setSelectedDataset} />
							</div>
							<div>
								<span style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Model</span>
								<ModelSelector models={models} selected={selectedModel} onSelect={setSelectedModel} />
							</div>
						</div>
					</div>

					{(selectedDataset || selectedModel) && (
						<Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', mt: 2 }}>
							<Button
								size="small"
								variant="contained"
								onClick={() => setPanelOpen({ dataset: true, model: true, test: true })}
								sx={{ backgroundColor: '#452ee4', '&:hover': { backgroundColor: '#241291' } }}
							>
								Expand All
							</Button>
							<Button
								size="small"
								variant="contained"
								onClick={() => setPanelOpen({ dataset: false, model: false, test: false })}
								sx={{ backgroundColor: '#452ee4', '&:hover': { backgroundColor: '#241291' } }}
							>
								Collapse All
							</Button>
						</Box>
					)}

					{selectedDataset && (
						<Accordion
							expanded={panelOpen.dataset}
							onChange={(_, expanded) => setPanelOpen((p) => ({ ...p, dataset: expanded }))}
							slotProps={{ transition: { unmountOnExit: false } }}
							sx={{
								mt: 2,
								'&:before': { display: 'none' },
								border: `1px solid`,
								borderColor: 'divider',
								borderRadius: '8px !important',
								boxShadow: 'none'
							}}
						>
							<AccordionSummary
								expandIcon={<ExpandMoreIcon sx={{ color: '#452ee4' }} />}
								sx={{ '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': { transform: 'rotate(180deg)' } }}
							>
								<Typography fontWeight="bold"><h2>Dataset Dashboard</h2></Typography>
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
							sx={{
								mt: 2,
								'&:before': { display: 'none' },
								border: `1px solid`,
								borderColor: 'divider',
								borderRadius: '8px !important',
								boxShadow: 'none'
							}}
						>
							<AccordionSummary
								expandIcon={<ExpandMoreIcon sx={{ color: '#452ee4' }} />}
								sx={{ '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': { transform: 'rotate(180deg)' } }}
							>
								<Typography fontWeight="bold"><h2>Model Dashboard</h2></Typography>
							</AccordionSummary>
							<AccordionDetails sx={{ pt: 0 }}>
								<ModelDashboard model={selectedModel} />
							</AccordionDetails>
						</Accordion>
					)}

					{selectedDataset && selectedModel && (
						<Accordion
							expanded={panelOpen.test}
							onChange={(_, expanded) => setPanelOpen((p) => ({ ...p, test: expanded }))}
							slotProps={{ transition: { unmountOnExit: false } }}
							sx={{
								mt: 2,
								'&:before': { display: 'none' },
								border: `1px solid`,
								borderColor: 'divider',
								borderRadius: '8px !important',
								boxShadow: 'none'
							}}
						>
							<AccordionSummary
								expandIcon={<ExpandMoreIcon sx={{ color: '#452ee4' }} />}
								sx={{ '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': { transform: 'rotate(180deg)' } }}
							>
								<Typography fontWeight="bold"><h2>Test Model</h2></Typography>
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
				</Box>

				{/* Snackbar for Ping FastAPI feedback */}
				<Snackbar
					open={snackbarOpen}
					autoHideDuration={3000}
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
						sx: {
							bgcolor: 'background.paper',
							borderRadius: '8px',
							border: '2px solid #452ee4'
						}
					}}
				>
					<DialogTitle
						sx={{
							display: 'flex',
							justifyContent: 'space-between',
							alignItems: 'center',
							bgcolor: 'background.paper',
							borderBottom: '2px solid #452ee4',
							color: 'text.primary'
						}}
					>
						Create Dataset
						<IconButton
							onClick={() => setShowCreateDatasetModal(false)}
							aria-label="Close"
							sx={{
								color: '#452ee4',
								'&:hover': {
									backgroundColor: 'rgba(69, 46, 228, 0.1)'
								}
							}}
						>
							<CloseIcon />
						</IconButton>
					</DialogTitle>
					<DialogContent sx={{ p: 3, bgcolor: 'background.paper' }}>
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
