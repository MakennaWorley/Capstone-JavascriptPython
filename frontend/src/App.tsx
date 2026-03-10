import { useState } from 'react';
import DatasetDashboard from './components/DatasetDisplayDashboard.js';
import DatasetModelCreationForm from './components/DatasetModelCreationForm.js';
import DatasetSelector from './components/DatasetSelector.js';
import ModelSelector from './components/ModelSelector.js';
import ModelStats from './components/ModelStats.js';
import ModelTrainer from './components/ModelTrainer.js';

type Model = {
	model_name: string;
	model_type: string;
};

type ApiSuccessDatasets = {
	status: 'success';
	message: string;
	data: {
		datasets: string[];
		count: number;
	};
};

type ApiSuccessModels = {
	status: 'success';
	message: string;
	data: {
		models: Model[];
		count: number;
	};
};

type ApiError = {
	status: 'error';
	code?: string;
	message: string;
};

export default function App() {
	const API_BASE = import.meta.env.VITE_API_BASE;
	const API_KEY = import.meta.env.VITE_X_API_KEY;

	const [msg, setMsg] = useState<string | null>(null);
	const [status, setStatus] = useState('');
	const [datasets, setDatasets] = useState<string[]>([]);
	const [selectedDataset, setSelectedDataset] = useState<string>('');
	const [models, setModels] = useState<Model[]>([]);
	const [selectedModel, setSelectedModel] = useState<Model | null>(null);
	const [testResults, setTestResults] = useState<{
		log: string;
		paths: unknown;
		testMetrics: unknown;
		images: unknown;
	} | null>(null);

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

	async function fetchDatasets() {
		try {
			setStatus('Fetching datasets...');

			const r = await fetch(`${API_BASE}/api/datasets/list`, {
				method: 'GET',
				headers: { 'x-api-key': API_KEY }
			});

			const j: ApiSuccessDatasets | ApiError = await r.json();

			if (j.status === 'success') {
				setDatasets(j.data.datasets);
				setMsg(`Loaded ${j.data.count} datasets`);
				setStatus('Success!');
			} else {
				setDatasets([]);
				setStatus(j.message || 'Failed to load datasets');
			}
		} catch {
			setDatasets([]);
			setStatus('Error fetching datasets');
		}
	}

	async function fetchModels() {
		try {
			setStatus('Fetching models...');

			const r = await fetch(`${API_BASE}/api/models/list`, {
				method: 'GET',
				headers: { 'x-api-key': API_KEY }
			});

			const j: ApiSuccessModels | ApiError = await r.json();

			if (j.status === 'success') {
				setModels(j.data.models);
				setMsg(`Loaded ${j.data.count} models`);
				setStatus('Success!');
			} else {
				setModels([]);
				setStatus(j.message || 'Failed to load models');
			}
		} catch {
			setModels([]);
			setStatus('Error fetching models');
		}
	}

	return (
		<div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
			<button type="button" onClick={pingBackend}>
				Ping FastAPI
			</button>
			<button type="button" onClick={fetchDatasets} style={{ marginLeft: '1rem' }}>
				List Datasets
			</button>
			<button type="button" onClick={fetchModels} style={{ marginLeft: '1rem' }}>
				List Models
			</button>

			{status && <p>{status}</p>}
			{msg && <p>Message: {msg}</p>}

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
