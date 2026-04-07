import { useState } from 'react';
import DatasetDashboard from './components/DatasetDisplayDashboard.js';
import DatasetModelCreationForm from './components/DatasetModelCreationForm.js';
import DatasetSelector from './components/DatasetSelector.js';
import ModelDashboard from './components/ModelDashboard.js';
import ModelSelector from './components/ModelSelector.js';
import ModelStats from './components/ModelStats.js';
import ModelTrainer from './components/ModelTrainer.js';
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
	} | null>(null);
	const [debugMode, setDebugMode] = useState(false);
	const [showCreateDatasetModal, setShowCreateDatasetModal] = useState(false);

	// Use RxJS observables for automatic polling (updates every 5 seconds)
	const { datasets, error: datasetsError, isLoading: datasetsLoading } = useDatasetsPoll(API_BASE, API_KEY, 5000);
	const { models, error: modelsError, isLoading: modelsLoading } = useModelsPoll(API_BASE, API_KEY, 5000);

	async function pingBackend() {
		try {
			setStatus('Pinging...');
			const r = await fetch(`${API_BASE}/api/hello`);
			const j = await r.json();
			setMsg(j.message);
			setStatus('Success!');
		} catch {
			setStatus('Error contacting backend');
		}
	}

	return (
		<div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>
			{/* Side Menu */}
			<div
				style={{
					width: '60px',
					backgroundColor: '#1a1a1a',
					borderRight: '1px solid #646cff',
					display: 'flex',
					flexDirection: 'column',
					alignItems: 'center',
					paddingTop: '1rem',
					gap: '1rem',
					position: 'fixed',
					left: 0,
					top: 0,
					height: '100vh',
					zIndex: 1000,
					justifyContent: 'flex-start'
				}}
			>
				{/* Plus Icon - Create Dataset */}
				<button
					type="button"
					onClick={() => setShowCreateDatasetModal(true)}
					title="Create new dataset"
					style={{
						width: '40px',
						height: '40px',
						borderRadius: '8px',
						backgroundColor: '#646cff',
						color: 'white',
						border: 'none',
						cursor: 'pointer',
						fontSize: '24px',
						display: 'flex',
						alignItems: 'center',
						justifyContent: 'center',
						transition: 'background-color 0.2s',
						flexShrink: 0
					}}
					onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#747eff')}
					onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#646cff')}
				>
					+
				</button>

				{/* Spacer */}
				<div style={{ flex: 1 }} />

				{/* Bug Icon - Debug Toggle */}
				<button
					type="button"
					onClick={() => setDebugMode(!debugMode)}
					title={`Debug: ${debugMode ? 'ON' : 'OFF'}`}
					style={{
						width: '40px',
						height: '40px',
						borderRadius: '8px',
						backgroundColor: debugMode ? '#4caf50' : '#555',
						color: 'white',
						border: 'none',
						cursor: 'pointer',
						display: 'flex',
						alignItems: 'center',
						justifyContent: 'center',
						transition: 'background-color 0.2s',
						flexShrink: 0,
						marginBottom: '1rem',
						padding: 0
					}}
					onMouseOver={(e) => (e.currentTarget.style.backgroundColor = debugMode ? '#45a049' : '#666')}
					onMouseOut={(e) => (e.currentTarget.style.backgroundColor = debugMode ? '#4caf50' : '#555')}
				>
					<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640" width="22" height="22" fill="currentColor">
						<path d="M224 160C224 107 267 64 320 64C373 64 416 107 416 160L416 163.6C416 179.3 403.3 192 387.6 192L252.5 192C236.8 192 224.1 179.3 224.1 163.6L224.1 160zM569.6 172.8C580.2 186.9 577.3 207 563.2 217.6L465.4 290.9C470.7 299.8 474.7 309.6 477.2 320L576 320C593.7 320 608 334.3 608 352C608 369.7 593.7 384 576 384L480 384L480 416C480 418.6 479.9 421.3 479.8 423.9L563.2 486.4C577.3 497 580.2 517.1 569.6 531.2C559 545.3 538.9 548.2 524.8 537.6L461.7 490.3C438.5 534.5 395.2 566.5 344 574.2L344 344C344 330.7 333.3 320 320 320C306.7 320 296 330.7 296 344L296 574.2C244.8 566.5 201.5 534.5 178.3 490.3L115.2 537.6C101.1 548.2 81 545.3 70.4 531.2C59.8 517.1 62.7 497 76.8 486.4L160.2 423.9C160.1 421.3 160 418.7 160 416L160 384L64 384C46.3 384 32 369.7 32 352C32 334.3 46.3 320 64 320L162.8 320C165.3 309.6 169.3 299.8 174.6 290.9L76.8 217.6C62.7 207 59.8 186.9 70.4 172.8C81 158.7 101.1 155.8 115.2 166.4L224 248C236.3 242.9 249.8 240 264 240L376 240C390.2 240 403.7 242.8 416 248L524.8 166.4C538.9 155.8 559 158.7 569.6 172.8z" />
					</svg>
				</button>
			</div>

			{/* Main Content */}
			<div style={{ marginLeft: '60px', flex: 1, overflowY: 'auto', padding: '2rem' }}>
				{/* Debug Features - Only visible when debug mode is ON */}
				{debugMode && (
					<div style={{ marginBottom: '1.5rem' }}>
						<div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
							<button type="button" onClick={pingBackend} style={{ padding: '0.5rem 1rem' }}>
								Ping FastAPI
							</button>

							{status && <p style={{ margin: 0 }}>{status}</p>}
							{msg && <p style={{ margin: 0 }}>Message: {msg}</p>}

							{/* Dataset Status */}
							<div
								style={{
									padding: '0.75rem',
									backgroundColor: '#1a1a1a',
									borderRadius: '4px',
									border: '1px solid #646cff',
									minWidth: '250px'
								}}
							>
								<small style={{ color: 'rgba(255, 255, 255, 0.87)' }}>
									<strong style={{ color: '#646cff' }}>Datasets:</strong>{' '}
									{datasetsLoading ? 'Loading...' : `${datasets.length} datasets`}
									{datasetsError && <span style={{ color: '#ff6b6b' }}> - Error: {datasetsError}</span>}
									<br />
									<span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>(Auto-updating every 5 seconds)</span>
								</small>
							</div>

							{/* Model Status */}
							<div
								style={{
									padding: '0.75rem',
									backgroundColor: '#1a1a1a',
									borderRadius: '4px',
									border: '1px solid #646cff',
									minWidth: '250px'
								}}
							>
								<small style={{ color: 'rgba(255, 255, 255, 0.87)' }}>
									<strong style={{ color: '#646cff' }}>Models:</strong> {modelsLoading ? 'Loading...' : `${models.length} models`}
									{modelsError && <span style={{ color: '#ff6b6b' }}> - Error: {modelsError}</span>}
									<br />
									<span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>(Auto-updating every 5 seconds)</span>
								</small>
							</div>
						</div>
					</div>
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
							paths={testResults?.paths || null}
							testMetrics={testResults?.testMetrics || null}
							images={testResults?.images || null}
							debugMode={debugMode}
						/>
					</div>
				)}
			</div>

			{/* Create Dataset Modal */}
			{showCreateDatasetModal && (
				<div
					style={{
						position: 'fixed',
						inset: 0,
						backgroundColor: 'rgba(0, 0, 0, 0.7)',
						display: 'flex',
						alignItems: 'center',
						justifyContent: 'center',
						zIndex: 2000
					}}
					onClick={(e) => {
						if (e.target === e.currentTarget) {
							setShowCreateDatasetModal(false);
						}
					}}
				>
					<div
						style={{
							backgroundColor: '#000',
							borderRadius: '8px',
							padding: '2rem',
							maxWidth: '600px',
							width: '90%',
							maxHeight: '90vh',
							overflowY: 'auto',
							border: '1px solid #646cff'
						}}
					>
						<div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
							<h2>Create Dataset</h2>
							<button
								type="button"
								onClick={() => setShowCreateDatasetModal(false)}
								style={{
									background: 'none',
									border: 'none',
									fontSize: '24px',
									cursor: 'pointer',
									color: '#fff'
								}}
							>
								×
							</button>
						</div>

						<DatasetModelCreationForm apiBase={API_BASE} xApiKey={API_KEY} debugMode={debugMode} />
					</div>
				</div>
			)}
		</div>
	);
}
