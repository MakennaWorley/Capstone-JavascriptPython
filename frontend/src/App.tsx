import CloseIcon from '@mui/icons-material/Close';
import { Alert, Box, Button, Dialog, DialogContent, DialogTitle, IconButton, Snackbar } from '@mui/material';
import { useState } from 'react';
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

	// Use RxJS observables for automatic polling (updates every 5 seconds)
	const { datasets, error: datasetsError, isLoading: datasetsLoading } = useDatasetsPoll(API_BASE, API_KEY, 5000);
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

	async function refreshDatasets() {
		try {
			await fetch(`${API_BASE}/api/datasets/list`, {
				method: 'GET',
				headers: { 'x-api-key': API_KEY }
			});
		} catch (err) {
			console.error('Failed to refresh datasets:', err);
		}
	}

	function handleDatasetCreated() {
		setShowCreateDatasetModal(false);
		refreshDatasets();
	}

	function handleDatasetCreationSuccess(message: string) {
		setSnackbarMessage(message);
		setSnackbarOpen(true);
	}

	return (
		<Box sx={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>
			{/* Sidebar Navigation */}
			<Sidebar debugMode={debugMode} onDebugToggle={() => setDebugMode(!debugMode)} onCreateDataset={() => setShowCreateDatasetModal(true)} />

			{/* Main Content */}
			<Box sx={{ flex: 1, overflowY: 'auto', padding: '2rem' }}>
				{/* Debug Features - Only visible when debug mode is ON */}
				{debugMode && (
					<Box sx={{ marginBottom: '0.5rem', display: 'flex', gap: '1rem', alignItems: 'center', height: 'fit-content' }}>
						<Button
							variant="contained"
							onClick={pingBackend}
							sx={{
								backgroundColor: '#646cff',
								'&:hover': { backgroundColor: '#747eff' },
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
								gap: '0.5rem',
								color: '#fff'
							}}
						>
							<strong style={{ color: '#646cff' }}>Datasets:</strong> {datasetsLoading ? 'Loading...' : `${datasets.length}`}
							{datasetsError && <span style={{ color: '#ff6b6b' }}> - Error: {datasetsError}</span>}
						</Box>

						<Box
							sx={{
								flex: 1,
								display: 'flex',
								alignItems: 'center',
								gap: '0.5rem',
								color: '#fff'
							}}
						>
							<strong style={{ color: '#646cff' }}>Models:</strong> {modelsLoading ? 'Loading...' : `${models.length}`}
							{modelsError && <span style={{ color: '#ff6b6b' }}> - Error: {modelsError}</span>}
						</Box>
					</Box>
				)}

				<div style={{ marginTop: '1.25rem' }}>
					<h3>Selection</h3>
					<div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
						<div>
							<label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Dataset</label>
							<DatasetSelector datasets={datasets} selected={selectedDataset} onSelect={setSelectedDataset} />
						</div>
						<div>
							<label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>Model</label>
							<ModelSelector models={models} selected={selectedModel} onSelect={setSelectedModel} />
						</div>
					</div>
				</div>

				{selectedDataset && (
					<>
						<div style={{ marginTop: '1.25rem' }}>
							<h3>Dataset Dashboard</h3>
						</div>

						<DatasetDashboard apiBase={API_BASE} xApiKey={API_KEY} selectedDataset={selectedDataset} />
					</>
				)}

				{selectedModel && <ModelDashboard model={selectedModel} />}

				{selectedDataset && selectedModel && (
					<div style={{ marginTop: '1.25rem' }}>
						<h3>Test Model</h3>

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
					</div>
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
						backgroundColor: '#000',
						borderRadius: '8px',
						border: '1px solid #646cff'
					}
				}}
			>
				<DialogTitle
					sx={{
						display: 'flex',
						justifyContent: 'space-between',
						alignItems: 'center',
						backgroundColor: '#1a1a1a',
						borderBottom: '1px solid #646cff',
						color: '#fff'
					}}
				>
					Create Dataset
					<IconButton
						onClick={() => setShowCreateDatasetModal(false)}
						sx={{
							color: '#646cff',
							'&:hover': {
								backgroundColor: 'rgba(100, 108, 255, 0.1)'
							}
						}}
					>
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
	);
}
