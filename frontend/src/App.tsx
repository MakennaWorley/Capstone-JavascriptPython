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

					<div style={{ marginTop: '1.25rem' }}>
						<h2>Selection</h2>
						<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
							<div>
								<span style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Dataset</span>
								<DatasetSelector datasets={datasets} selected={selectedDataset} onSelect={handleSelectDataset} />
							</div>
							<div>
								<span style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Model</span>
								<ModelSelector models={models} selected={selectedModel} onSelect={handleSelectModel} />
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
								<Typography fontWeight="bold">Dataset Dashboard</Typography>
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
								<Typography fontWeight="bold">Model Dashboard</Typography>
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
								<Typography fontWeight="bold">Test Model</Typography>
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
