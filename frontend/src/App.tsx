import { useState } from 'react';
import DatasetDashboard from './components/DatasetDisplayDashboard.js';
import DatasetModelCreationForm from './components/DatasetModelCreationForm.js';
import DatasetSelector from './components/DatasetSelector.js';
import ModelSelector from './components/ModelSelector.js';
import ModelStats from './components/ModelStats.js';
import ModelTrainer from './components/ModelTrainer.js';
import { useDatasetsPoll, useModelsPoll } from './hooks/useApiPolling';

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
		log: string;
		paths: unknown;
		testMetrics: unknown;
		images: unknown;
	} | null>(null);

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
		<div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
			<button type="button" onClick={pingBackend}>
				Ping FastAPI
			</button>

			{status && <p>{status}</p>}
			{msg && <p>Message: {msg}</p>}

			{/* Dataset Status */}
			<div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#1a1a1a', borderRadius: '4px', border: '1px solid #646cff' }}>
				<small style={{ color: 'rgba(255, 255, 255, 0.87)' }}>
					<strong style={{ color: '#646cff' }}>Datasets:</strong> {datasetsLoading ? 'Loading...' : `${datasets.length} datasets`}
					{datasetsError && <span style={{ color: '#ff6b6b' }}> - Error: {datasetsError}</span>}
					<br />
					<span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>(Auto-updating every 5 seconds)</span>
				</small>
			</div>

			{/* Model Status */}
			<div style={{ marginTop: '0.5rem', padding: '0.75rem', backgroundColor: '#1a1a1a', borderRadius: '4px', border: '1px solid #646cff' }}>
				<small style={{ color: 'rgba(255, 255, 255, 0.87)' }}>
					<strong style={{ color: '#646cff' }}>Models:</strong> {modelsLoading ? 'Loading...' : `${models.length} models`}
					{modelsError && <span style={{ color: '#ff6b6b' }}> - Error: {modelsError}</span>}
					<br />
					<span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>(Auto-updating every 5 seconds)</span>
				</small>
			</div>

			<div style={{ marginTop: '1.25rem' }}>
				<h3>Dataset Dashboard</h3>
				<DatasetSelector datasets={datasets} selected={selectedDataset} onSelect={setSelectedDataset} />
			</div>

			<DatasetDashboard apiBase={API_BASE} xApiKey={API_KEY} selectedDataset={selectedDataset} />

			<div style={{ marginTop: '1.25rem' }}>
				<h3>Model Dashboard</h3>
				<ModelSelector models={models} selected={selectedModel} onSelect={setSelectedModel} />
				{selectedModel && (
					<div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'black', borderRadius: '4px' }}>
						<h4>Selected Model</h4>
						<p>
							<strong>Model Name:</strong> {selectedModel.model_name}
						</p>
						<p>
							<strong>Model Type:</strong> {selectedModel.model_type}
						</p>
					</div>
				)}
			</div>

			<DatasetModelCreationForm apiBase={API_BASE} xApiKey={API_KEY} />

			<ModelTrainer
				apiBase={API_BASE}
				xApiKey={API_KEY}
				selectedDataset={selectedDataset}
				selectedModel={selectedModel}
				onTestComplete={setTestResults}
			/>

			<ModelStats
				log={testResults?.log || null}
				paths={testResults?.paths || null}
				testMetrics={testResults?.testMetrics || null}
				images={testResults?.images || null}
			/>
		</div>
	);
}
